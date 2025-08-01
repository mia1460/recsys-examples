# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) python3 benchmark/real_inference_benchmark.py --gin-config-file kuairand_1k_ranking_debug.gin --time_interval_ms 3600000
import torch
import pandas as pd
import sys
import types
from dataclasses import dataclass
from typing import List, Tuple, cast


def mock_megatron_for_inference():
    import sys
    import types

    dummy_class = type("Dummy", (), {})
    def dummy_function(*args, **kwargs):
        pass

    def fake_module(name, attrs=None):
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        return mod

    # 构造所有需要的模块
    modules_to_mock = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),

        "megatron.core.parallel_state": fake_module("megatron.core.parallel_state"),
        "megatron.core.tensor_parallel": fake_module("megatron.core.tensor_parallel"),

        "megatron.core.distributed": fake_module("megatron.core.distributed", {
            "DistributedDataParallel": dummy_class,
            "finalize_model_grads": dummy_function,
        }),
        "megatron.core.distributed.distributed_data_parallel": fake_module(
            "megatron.core.distributed.distributed_data_parallel", {
                "DistributedDataParallel": dummy_class,
            }
        ),

        "megatron.core.optimizer": fake_module("megatron.core.optimizer", {
            "MegatronOptimizer": dummy_class,
        }),

        "megatron.core.transformer": types.ModuleType("megatron.core.transformer"),
        "megatron.core.transformer.module": fake_module("megatron.core.transformer.module", {
            "Float16Module": dummy_class,
            "MegatronModule": dummy_class,
        }),
    }

    # 必须标记为包，才能向下 import
    modules_to_mock["megatron.core"].__path__ = []
    modules_to_mock["megatron.core.transformer"].__path__ = []
    modules_to_mock["megatron.core.distributed"].__path__ = []

    sys.modules.update(modules_to_mock)

mock_megatron_for_inference()
import argparse
import gin

from training import (
    get_dataset_and_embedding_args,
    TrainerArgs,
    NetworkArgs,
    TensorModelParallelArgs,
    OptimizerArgs,
    maybe_load_ckpts,
)

def extract_table_dims_from_embedding_args(embedding_args) -> dict:
    table_dims = {}
    for e in embedding_args:
        if hasattr(e, "sharding_type") and e.sharding_type == "data_parallel":
            table_dims[e.table_name] = e.item_vocab_size_or_capacity
    return table_dims


def split_and_assign_concatenated_embeddings(state_dict, model, table_dims):
    concat_key = "_embedding_collection._data_parallel_embedding_collection.embeddings.user_active_degree/follow_user_num_range/fans_user_num_range/friend_user_num_range/register_days_range/action_weights_weights"
    if concat_key not in state_dict:
        return {}

    concat_tensor = state_dict[concat_key]
    embedding_dim = 512
    offset = 0
    result = {}
    for name, size in table_dims.items():
        length = size * embedding_dim
        key = f"_embedding_collection._nondynamic_embedding_collection.embeddings.{name}.weight"
        if key in model.state_dict():
            result[key] = concat_tensor[offset: offset + length].reshape(size, embedding_dim)
        offset += length
    return result


def load_training_checkpoint_to_inference_model(ckpt_path, infer_model, table_dims):
    import os
    import torch

    rank = 0
    ckpt_file = os.path.join(ckpt_path, "torch_module", f"model.{rank}.pth")
    assert os.path.exists(ckpt_file), f"Checkpoint file not found: {ckpt_file}"

    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    new_state_dict = {}
    model_keys = infer_model.state_dict().keys()

    mapping_rules = [
        ("_hstu_block._attention_layers", "_hstu_block._attention_layers"),
        ("_embedding_collection._data_parallel_embedding_collection.embeddings", "_embedding_collection._nondynamic_embedding_collection.embeddings"),
        ("_embedding_collection._model_parallel_embedding_collection.embeddings", "_embedding_collection._dynamic_embedding_collection._embedding_tables"),
        ("_mlp", "_dense_module"),
    ]

    for k, v in state_dict.items():
        for train_prefix, infer_prefix in mapping_rules:
            if k.startswith(train_prefix):
                infer_key = k.replace(train_prefix, infer_prefix, 1)
                if infer_key in model_keys and infer_model.state_dict()[infer_key].shape == v.shape:
                    new_state_dict[infer_key] = v
                break

    # 拆分concat嵌入
    split_embeddings = split_and_assign_concatenated_embeddings(state_dict, infer_model, table_dims)
    new_state_dict.update(split_embeddings)

    # 加载位置编码
    pos_key = "_hstu_block._positional_encoder._position_embeddings_weight"
    if pos_key in state_dict and pos_key in model_keys:
        new_state_dict[pos_key] = state_dict[pos_key]

    # 加载 state_dict
    missing_keys, unexpected_keys = infer_model.load_state_dict(new_state_dict, strict=False)
    print(f"[INFO] Loaded keys: {len(new_state_dict)}")
    print(f"[INFO] Missing keys: {missing_keys}")
    print(f"[INFO] Unexpected keys: {unexpected_keys}")

@gin.configurable
@dataclass
class RankingArgs:
    prediction_head_arch: List[int] = cast(List[int], None)
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True
    num_tasks: int = 1
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch for ranking model"
        if isinstance(self.prediction_head_act_type, str):
            assert self.prediction_head_act_type.lower() in [
                "relu",
                "gelu",
            ], "prediction_head_act_type should be in ['relu', 'gelu']"

from configs import (
    InferenceEmbeddingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
    PositionEncodingConfig,
)
sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR

# -----------------------------
# CLI for gin-config-file
# -----------------------------
parser = argparse.ArgumentParser(description="GR Inference Args", allow_abbrev=False)
parser.add_argument("--gin-config-file", type=str, required=True)
parser.add_argument("--max_window_slide_cnt", type=int, default=-1,
                    help="Maximum number of time windows to slide through. Default is -1, which means no limit.")
parser.add_argument("--time_interval_ms", type=int, default=360000,
                    help="Time interval in milliseconds for sliding window. Default is 360000 ms.")
args = parser.parse_args()

# -----------------------------
# Load gin config
# -----------------------------
gin.parse_config_file(args.gin_config_file)


def get_batch_feature_ids(batch, feature_name: str) -> List[int]:
    jt = batch.features[feature_name]
    return jt.values()


def run_sequence_inference_with_config():
    # -------------------------------
    # Load dataset, embedding and network args
    # -------------------------------
    trainer_args = TrainerArgs()
    network_args = NetworkArgs()
    dataset_args, embedding_args = get_dataset_and_embedding_args()

    from dataset.sequence_inference_dataset import get_dataset

    # Load dataset
    train_dataset, _ = get_dataset(
        dataset_name=dataset_args.dataset_name,
        max_sequence_length=dataset_args.max_sequence_length,
        # max_sequence_length=32,
        max_num_candidates=dataset_args.max_num_candidates,
        num_tasks=0,
        batch_size=trainer_args.eval_batch_size,
        rank=0,
        world_size=1,
        shuffle=False,
        random_seed=trainer_args.seed,
        eval_batch_size=None,
        use_time_segment=True,
        time_interval_ms=args.time_interval_ms,
    )

    # Feature and sequence settings
    item_fea_name = train_dataset._item_feature_name
    action_fea_name = train_dataset._action_feature_name
    contextual_feature_names = train_dataset._contextual_feature_names
    max_seqlen = dataset_args.max_sequence_length
    max_num_candidates = dataset_args.max_num_candidates
    batch_size = trainer_args.eval_batch_size

    # -------------------------------
    # Build inference model config
    # -------------------------------
    hidden_dim = network_args.hidden_size
    num_heads = network_args.num_attention_heads
    num_layers = network_args.num_layers
    head_dim = network_args.kv_channels
    dtype_str = network_args.dtype_str
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=network_args.num_position_buckets,
        num_time_buckets=2048,
        use_time_encoding=False,
    )
    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        position_encoding_config=position_encoding_config,
    )

    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=10240,
        page_size=256,
        offload_chunksize=8192,
        max_batch_size=batch_size,
        max_seq_len=max_seqlen,
        # max_seq_len=32
    )

    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=e.feature_names,
            table_name=e.table_name,
            vocab_size=e.item_vocab_size_or_capacity,
            dim=hidden_dim,
            use_dynamicemb=(
                hasattr(e, "item_vocab_gpu_capacity_ratio") and e.item_vocab_gpu_capacity_ratio > 0
            ),
            # use_dynamicemb=False,  # [yinj] disable dynamic embedding for test
        )
        for e in embedding_args
    ]

    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )
    # print(ranking_args.prediction_head_arch)

    model = InferenceRankingGR(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config,
        task_config=task_config,
        use_cudagraph=False,
    #     cudagraph_configs={
    #         "batch_size": [1, 2, 4, 8],
    #         "length_per_sequence": [i * 256 for i in range(2, 18)],
    #     },
    )
    print(model)
    table_dims = extract_table_dims_from_embedding_args(embedding_args)
    load_training_checkpoint_to_inference_model(trainer_args.ckpt_load_dir, model, table_dims)
    model.to(dtype=dtype).eval()

    # -------------------------------
    # Inference over time windows
    # -------------------------------
    results = []
    step = 0
    window_slide_cnt = 0
    whole_batch = 0
    while True:
        print(f"=========running step {step} ============")
        print(f"train_dataset.current_time is {train_dataset.current_time_ms}, users cnt is {train_dataset._num_samples}")
        total_batches = 0
        total_time_ms = 0.0
        iterator = iter(train_dataset)
        for batch in iterator:
            print(f"=========running batch {total_batches}/whole: {whole_batch} ============")
            # Extract uids if available
            if "user_id" in batch.features.keys():
                uids = get_batch_feature_ids(batch, "user_id")
            else:
                uids = list(range(batch.batch_size))
            cached_start_pos, cached_len = model.get_user_kvdata_info(uids)
            truncate_start_pos = cached_start_pos + cached_len
            ts_start = torch.cuda.Event(enable_timing=True)
            ts_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            ts_start.record()
            print(f"batch.batch_size is {batch.batch_size}")
            # print(f"batch.features is {batch.features}")
            # print(f"batch.features.lenghts is {batch.features.lengths()}")
            model.forward(batch, uids, truncate_start_pos)
            ts_end.record()
            torch.cuda.synchronize()
            elapsed = ts_start.elapsed_time(ts_end)
            total_batches += 1
            whole_batch += 1
            total_time_ms += elapsed
        print(f"[Time window {step}] batches={total_batches}, time={total_time_ms:.2f} ms")
        results.append({
            "Time Window": step,
            "Batches": total_batches,
            "Total Time (ms)": round(total_time_ms, 2),
            "Avg Time per Batch (ms)": round(total_time_ms / total_batches, 2) if total_batches > 0 else 0.0,
        })
        step += 1
        if not train_dataset.slide_window():
            print("Finished all time windows.")
            break
        window_slide_cnt += 1
        if args.max_window_slide_cnt > 0 and window_slide_cnt >= args.max_window_slide_cnt:
            print(f"Reached maximum window slide count: {args.max_window_slide_cnt}. Stopping.")
            break
        

    # -------------------------------
    # Output results
    # -------------------------------
    df = pd.DataFrame(results)
    print("\n==== Inference Time Summary ====\n")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    run_sequence_inference_with_config()
