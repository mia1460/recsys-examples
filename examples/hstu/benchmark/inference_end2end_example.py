# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3 ./benchmark/inference_end2end_example.py > /workspace/recsys-separate_data/bug_fix/test_e2e/fix_noCudagraph.log 2>&1
# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import logging
import argparse
import sys
import time

from commons.utils.stringify import stringify_dict
from dataclasses import dataclass
import gin
import torch
import math
import os

# print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])

# python benchmark/inference_end2end_example.py --max_batch_ctr 2 >/workspace/recsys-examples/examples/hstu/bug_report/pin_gpu_cmp_1.log
# python benchmark/inference_end2end_example.py --max_batch_ctr 2 >/workspace/recsys-examples/examples/hstu/bug_report/pin_gpu_cmp_1.log

# python benchmark/inference_end2end_example.py --max_batch_ctr 2 > /workspace/recsys-separate_data/bug_fix/check_kvcache/kvcache_diff/new_code/unstable_results/heihei/init.log
# python /workspace/recsys-shared_data/scripts/debug/check_qkv.py --nums 134,156
# python /workspace/recsys-shared_data/scripts/debug/check_qkv.py --nums 134,156 --prefix output


# seed = 123
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.use_deterministic_algorithms(True)


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# print(logger.handlers)
# for handler in logger.handlers[:]:
#     logger.removeHandler(handler)

# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.DEBUG)
# logger.addHandler(console_handler)

from configs import (
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from dataset import get_data_loader
from dataset.inference_dataset import InferenceDataset
from dataset.random_inference_dataset import RandomInferenceDataGenerator
from dataset.utils import FeatureConfig
from modules.metrics import get_multi_event_metric_module
from preprocessor import get_common_preprocessors
from typing import List, Tuple, cast

sys.path.append("./model/")
from inference_ranking_gr import InferenceRankingGR
sys.path.append("./training/")
from gin_config_args import TrainerArgs, EmbeddingArgs, DatasetArgs, NetworkArgs, OptimizerArgs

#duplicate
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

def get_inference_dataset_and_embedding_configs(
    disable_contextual_features: bool = False
):
    dataset_args = DatasetArgs()
    embedding_dim = NetworkArgs().hidden_size
    HASH_SIZE = 10_000_000
    if dataset_args.dataset_name == "kuairand-1k":
        embedding_configs = [
            InferenceEmbeddingConfig(
                feature_names=["user_id"],
                table_name="user_id",
                vocab_size=1000,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["user_active_degree"],
                table_name="user_active_degree",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["follow_user_num_range"],
                table_name="follow_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["fans_user_num_range"],
                table_name="fans_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["friend_user_num_range"],
                table_name="friend_user_num_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["register_days_range"],
                table_name="register_days_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["video_id"],
                table_name="video_id",
                vocab_size=HASH_SIZE,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["action_weights"],
                table_name="action_weights",
                vocab_size=233,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
        ]
        return dataset_args, embedding_configs if not disable_contextual_features else embedding_configs[-2:]
    
    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")

def check_logit(baseline_list, kvcache_list, baseline_uid_list, kvcache_uid_list, baseline_num_tokens_list, baseline_cached_len_list, kvcache_num_tokens_list, kvcache_cached_len_list, atol=0, rtol=0, eps = 0):
    """
    Compare two logit lists and print user-friendly results.

    Args:
        baseline_list (list[Tensor]): baseline logits list
        kvcache_list (list[Tensor]): kvcache logits list
        atol (float): absolute tolerance
        rtol (float): relative tolerance
        baseline_uid_list (list[Tensor]): baseline uid list
        kvcache_uid_list (list[Tensor]): kvcache uid list
    """
    if len(baseline_list) != len(kvcache_list):
        print(f"❌ Different logits list lengths: baseline={len(baseline_list)}, kvcache={len(kvcache_list)}")
        return
    if len(baseline_uid_list) != len(kvcache_uid_list):
        print(f"❌ Different uid list lengths: baseline={len(baseline_uid_list)}, kvcache={len(kvcache_uid_list)}")
        return

    all_match = True
    for i, (base, cache) in enumerate(zip(baseline_list, kvcache_list)):
        base_uid = baseline_uid_list[i]
        if not torch.allclose(base, cache, atol=atol, rtol=rtol):
            diff = torch.abs(base - cache).max().item() # abs_diff
            rel_diff = (torch.abs(base - cache) / (torch.abs(base) + eps)).max().item()
            kv_uid = kvcache_uid_list[i]
            print(f"⚠️ Batch {i} mismatch (max rel_diff difference: {rel_diff:.2e}), base_uid: {base_uid}, kv_uid: {kv_uid}, base: {baseline_cached_len_list[i].tolist()} + {baseline_num_tokens_list[i]}, kvcache: {kvcache_cached_len_list[i].tolist()} + {kvcache_num_tokens_list[i]}")
            all_match = False
        else:
            print(f"✅ Batch {i} match, uid: {base_uid}, base: {baseline_cached_len_list[i].tolist()} + {baseline_num_tokens_list[i]}, kvcache: {kvcache_cached_len_list[i].tolist()} + {kvcache_num_tokens_list[i]}")

    if all_match:
        print("\n🎉 All batches match within tolerance")
    else:
        print("\n❌ Some batches differ, please check model output or cache logic.")

def run_dataset_with_kvcache_option(
    max_batch_ctr: int,
    dataloader_iter,
    dataset,
    model_predict, 
    num_contextual_features,
    eval_module,
    use_kvcache: bool = True,
    ret_logit_uid: bool = True,
):
    if use_kvcache:
        print(f"==================fwd with kvcache==================")
    else:
        print(f"==================fwd without kvcache==================")
    model_predict.eval()
    model_predict.clear_kv_cache()
    cur_date = None
    if ret_logit_uid:
        logit_list = list()
        uid_list = list()
        num_tokens_list = list()
        cached_len_list = list()
    num_batches_ctr = 0
    num_skipped_ctr = 0
    while True:
        try:
            uids, dates, seq_endptrs = next(dataloader_iter)
            print(f"--------------running iter {num_batches_ctr}--------------")
            print(f"dates[0] is {dates[0]}, cur_date is {cur_date}, dates is {dates}")
            if not use_kvcache:
                print(f"clear with use_kvcache = {use_kvcache}")
                model_predict.clear_kv_cache()
            if dates[0] != cur_date:
                if cur_date is not None:
                    eval_metric_dict = eval_module.compute()
                    print(
                        f"[eval]:\n    "
                        + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
                    )
                if use_kvcache:
                    model_predict.clear_kv_cache()
                cur_date = dates[0]
            cached_start_pos, cached_len = model_predict.get_user_kvdata_info(uids, dbg_print=True)
            new_cache_start_pos = cached_start_pos + cached_len
            non_contextual_mask = new_cache_start_pos >= num_contextual_features
            # print(f"non_contextual_mask is {non_contextual_mask}")
            contextual_mask = torch.logical_not(non_contextual_mask)
            seq_startptrs = (torch.clip(new_cache_start_pos - num_contextual_features, 0) / 2).int()
            print(f"uid: {uids.tolist()}, cached_start_pos: {cached_start_pos.tolist()}, cached_len: {cached_len.tolist()}, new_cache_start_pos: {new_cache_start_pos.tolist()}")
            batch_0 = dataset.get_input_batch(
                uids[non_contextual_mask],
                dates[non_contextual_mask],
                seq_endptrs[non_contextual_mask],
                seq_startptrs[non_contextual_mask], 
                with_contextual_features = True, 
                with_ranking_labels = True)
            # print(f"uids[non_contextual_mask] is {uids[non_contextual_mask]}, dates[non_contextual_mask] is {dates[non_contextual_mask]}, seq_endptrs[non_contextual_mask] is {seq_endptrs[non_contextual_mask]}, seq_startptrs[non_contextual_mask] is {seq_startptrs[non_contextual_mask]}")
            if batch_0 is not None:
                num_tokens = batch_0.features.values().shape[0]
                logits = model_predict.forward(batch_0, uids[non_contextual_mask].int(), new_cache_start_pos[non_contextual_mask])
                eval_module(logits, batch_0.labels)
                num_batches_ctr += 1
                if ret_logit_uid:
                    logit_list.append(logits.cpu())
                    uid_list.append(uids)
                    num_tokens_list.append(num_tokens)
                    cached_len_list.append(new_cache_start_pos.cpu())
            else:
                num_skipped_ctr += 1
                print(f"skip batch with 0 valid users")
            if False and "test with contextual ftrs":
                batch_1 = dataset.get_input_batch(
                    uids[contextual_mask],
                    dates[contextual_mask],
                    seq_endptrs[contextual_mask],
                    seq_startptrs[contextual_mask], 
                    with_contextual_features = True, 
                    with_ranking_labels = True)
                if batch_1 is not None:
                    logits = model_predict.forward(batch_1, uids[contextual_mask].int(), new_cache_start_pos[contextual_mask])
                    eval_module(logits, batch_1.labels)
            if num_batches_ctr == max_batch_ctr:
                break
            print(f".", end="", flush=True)
            # if num_batches_ctr % 100 == 0:
            #     print(f"")
            if num_batches_ctr % 100 == 0 and num_batches_ctr > 0:
                print(f"\n{num_batches_ctr} iters", flush=True)
        except StopIteration:
            break
    if ret_logit_uid:
        return logit_list, uid_list, num_tokens_list, cached_len_list

def models_equal(m1, m2):
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        if k1 != k2 or not torch.equal(v1, v2):
            return False
    return True

def run_ranking_gr_inference(
    checkpoint_dir: str,
    check_auc: bool,
    check_kvcache: bool,
    max_batch_ctr: int,
    disable_contextual_features: bool = True
):
    logger.info("Start run_ranking_gr_inference, begin with making configs...")

    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(disable_contextual_features)
    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    elif network_args.dtype_str == "float16":
        inference_dtype = torch.float16
    else:
        raise ValueError(f"Inference data type {network_args.dtype_str} is not supported")

    # TODO: move dataset_path to args
    dataset_path = "/workspace/recsys-shared_data/used_data"
    dataproc = get_common_preprocessors(dataset_path)[dataset_args.dataset_name]
    dataproc._batching_file = dataproc._output_file.replace("processed_seqs", "processed_batches")
    num_contextual_features = len(dataproc._contextual_feature_names) if not disable_contextual_features else 0

    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
    )

    hstu_config = get_inference_hstu_config(
        hidden_size=network_args.hidden_size,
        num_layers=network_args.num_layers,
        num_attention_heads=network_args.num_attention_heads,
        head_dim=network_args.kv_channels,
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=num_contextual_features,
    )

    kvcache_args = {
        "blocks_in_primary_pool": 512,
        "page_size": 32,
        "offload_chunksize": 32,
        "max_batch_size": 1,
        "max_seq_len": math.ceil(total_max_seqlen / 32) * 32,
    }
    kv_cache_config = get_kvcache_config(**kvcache_args)

    # TODO: need to init & setup
    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )

    hstu_cudagraph_configs = {
        "batch_size": [1],
        "length_per_sequence": [128] + [i * 256 for i in range(1, 34)],
    }

    logger.info("Finish making configs, begin with inference...")
    with torch.inference_mode():
        model_predict = InferenceRankingGR(
            hstu_config=hstu_config,
            kvcache_config=kv_cache_config,
            task_config=task_config,
            use_cudagraph=True,
            cudagraph_configs=hstu_cudagraph_configs,
        )
        if hstu_config.bf16:
            model_predict.bfloat16()
        elif hstu_config.fp16:
            model_predict.half()
        model_predict.load_checkpoint(checkpoint_dir)

        # print(f"model is equal: {models_equal(model_predict, model_predict_cmp)}")
        
        model_predict.eval()

        if check_auc:
            eval_module = get_multi_event_metric_module(
                num_classes=task_config.prediction_head_arch[-1],
                num_tasks=task_config.num_tasks,
                metric_types=task_config.eval_metrics,
            )

        logger.info("Begin making dataset and dataloader...")
        dataset = InferenceDataset(
            seq_logs_file=dataproc._output_file,
            batch_logs_file=dataproc._batching_file,
            batch_size=kvcache_args["max_batch_size"],
            max_seqlen=dataset_args.max_sequence_length,
            item_feature_name=dataproc._item_feature_name,
            contextual_feature_names=dataproc._contextual_feature_names if not disable_contextual_features else [],
            action_feature_name=dataproc._action_feature_name,
            max_num_candidates=dataset_args.max_num_candidates,

            item_vocab_size=10_000_000,
            userid_name='user_id',
            date_name='date',
            sequence_endptr_name='interval_indptr',
            timestamp_names=['date', 'interval_end_ts'],
        )

        dataloader = get_data_loader(dataset=dataset)
        dataloader_iter = iter(dataloader)

        num_batches_ctr = 0
        num_skipped_ctr = 0
        seq_lengths = set()
        seqlen_histogram = dict()

        # ts_start, ts_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start_time = time.time()
        cur_date = None
        logger.info("Begin iterating dataloader...")
        
        if check_kvcache:
            cmp_dataloader = get_data_loader(dataset=dataset)
            cmp_dataloader_iter = iter(cmp_dataloader)

            baseline_logit_list, baseline_uid_list, baseline_num_tokens_list, baseline_cached_len_list = run_dataset_with_kvcache_option(
                use_kvcache = False,
                ret_logit_uid = True,
                max_batch_ctr = max_batch_ctr,
                dataloader_iter = cmp_dataloader_iter,
                dataset = dataset,
                model_predict = model_predict,
                num_contextual_features = num_contextual_features,
                eval_module = eval_module,
            )

        kvcache_logit_list, kvcache_uid_list, kvcache_num_tokens_list, kvcache_cached_len_list = run_dataset_with_kvcache_option(
            use_kvcache = True,
            ret_logit_uid = True,
            max_batch_ctr = max_batch_ctr,
            dataloader_iter = dataloader_iter,
            dataset = dataset,
            model_predict = model_predict,
            num_contextual_features = num_contextual_features,
            eval_module = eval_module,
        )

        if check_kvcache:
            print(f"==================check kvcache accuracy==================")
            check_logit(baseline_logit_list, kvcache_logit_list, baseline_uid_list, kvcache_uid_list, baseline_num_tokens_list, baseline_cached_len_list, kvcache_num_tokens_list, kvcache_cached_len_list, atol=0, rtol=0)      
        

if __name__ == "__main__":
    # TODO: change to args (argsparser)
    parser = argparse.ArgumentParser(description="Run GR ranking inference.")
    parser.add_argument(
        "--config",
        type=str,
        default="./kuairand_1k_inference_ranking.gin",
        help="Path to gin config file",
    )
    parser.add_argument(
        "--check_auc",
        action="store_true",
        default=True,
        help="Enable AUC checking",
    )
    parser.add_argument(
        "--check_kvcache",
        action="store_true",
        default=True,
        help="Enable kv-cache consistency checking",
    )

    parser.add_argument(
        "--max_batch_ctr",
        type=int,
        default=200,
        help="Maximum number of iterations to process",
    )

    args = parser.parse_args()

    # parse gin config
    gin.parse_config_file(args.config)

    checkpoint_dir = '/workspace/recsys-shared_data/ckpts/kuairand-1k/ranking/2025_08_31-02_20_07/final-iter1000/'

    # run inference with flags
    run_ranking_gr_inference(
        checkpoint_dir=checkpoint_dir,
        check_auc=args.check_auc,
        check_kvcache=args.check_kvcache,
        max_batch_ctr=args.max_batch_ctr,
    )