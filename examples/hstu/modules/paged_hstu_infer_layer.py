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
import hstu_attn
import paged_kvcache_ops
import torch
import torch.nn.functional as F
from configs import InferenceHSTUConfig, KVCacheConfig
from modules.jagged_data import JaggedData


class PagedHSTUInferLayer(torch.nn.Module):
    """
    x = ln(x)
    u,v,q,k = silu(linear_bias(x))
    attn_output = hstu_attn.hstu_attn_varlen_func(q,k,v,offsets,max_seqlen)
    normed_out = ln_mul_dropout(attn_output)
    out = linear_residual(normed_out)

    One basic unit of PagedHSTUBlock. Input and output are all JaggedData.
    """

    def __init__(
        self,
        config: InferenceHSTUConfig,
        kv_cache_config: KVCacheConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._embedding_dim: int = config.hidden_size
        # per head dim;
        self._linear_dim_per_head: int = config.head_dim
        self._attention_dim_per_head: int = config.head_dim

        self._num_heads: int = config.num_heads

        self._eps = config.layernorm_epsilon
        self._is_causal = config.is_causal
        self._target_group_size = config.target_group_size
        self._alpha = 1.0 / (self._attention_dim_per_head**0.5)
        self._residual = config.residual

        self._split_arg_list = [
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
        ]
        self._max_seqlen = kv_cache_config.max_seq_len

        dtype = (
            torch.bfloat16
            if config.bf16
            else torch.float16
            if config.fp16
            else torch.float32
        )
        device = torch.cuda.current_device()

        # linear_uvqk
        self._linear_uvqk = torch.nn.Linear(
            self._embedding_dim,
            (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
            * self._num_heads,
            bias=True,
            dtype=dtype,
            device=device,
        )
        for param in self._linear_uvqk.parameters():
            param.requires_grad = False
            param.copy_(torch.empty_like(param).uniform_(-0.5, 0.5))
        self._linear_uvqk_weight = self._linear_uvqk.weight.T.contiguous()
        self._linear_uvqk_bias = self._linear_uvqk.bias.T.contiguous()

        # input norm
        if config.learnable_input_layernorm:
            self._input_layernorm_weight = torch.nn.Parameter(
                torch.ones(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
            self._input_layernorm_bias = torch.nn.Parameter(
                torch.zeros(self._embedding_dim, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self._input_layernorm_weight = None
            self._input_layernorm_bias = None

        # output norm
        self._output_layernorm_weight = torch.nn.Parameter(
            torch.ones(
                self._num_heads * self._linear_dim_per_head, dtype=dtype, device=device
            ),
            requires_grad=False,
        )
        self._output_layernorm_bias = torch.nn.Parameter(
            torch.zeros(
                self._num_heads * self._linear_dim_per_head, dtype=dtype, device=device
            ),
            requires_grad=False,
        )

        # linear_proj
        self._linear_proj = torch.nn.Linear(
            self._linear_dim_per_head * self._num_heads,
            self._embedding_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )

        for param in self._linear_proj.parameters():
            param.requires_grad = False
            param.copy_(torch.randn_like(param))

        # output buffer
        max_num_tokens = kv_cache_config.max_batch_size * kv_cache_config.max_seq_len
        self.output_buffer_ = torch.empty(
            (max_num_tokens, config.hidden_size),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        self.uvqk_buffer_ = torch.empty(
            (
                max_num_tokens,
                (self._linear_dim_per_head * 2 + self._attention_dim_per_head * 2)
                * self._num_heads,
            ),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

    def layer_output(num_tokens):
        return self.output_buffer_[:num_tokens, ...]

    def load_variable(
        self,
    ):
        pass

    @torch.inference_mode()
    def forward_naive(
        self,
        batch_size: int,
        num_tokens: int,
        layer_input: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
        layer_idx: int,
    ) -> JaggedData:
        normed_input = F.layer_norm(
            layer_input,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=self._input_layernorm_bias,
            eps=self._eps,
        )
        # print(f"normed_input.shape is {normed_input.shape}, layer_idx is {layer_idx}")

        lineared_uvqk = self._linear_uvqk(normed_input)
        
        # mixed_uvqk = F.silu(self._linear_uvqk(normed_input))
        mixed_uvqk = F.silu(lineared_uvqk)
        # print(f"mixed_uvqk.shape is {mixed_uvqk.shape}, layer_idx is {layer_idx}")
        (user, value, query, key) = torch.split(
            mixed_uvqk,
            self._split_arg_list,
            dim=-1,
        )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        query = query.view(-1, self._num_heads, self._attention_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
        (paged_k_cache, paged_v_cache) = kv_cache_table.unbind(dim=1)
        paged_kvcache_ops.append_kvcache(
            key,
            value,
            kv_cache_metadata.batch_indices,
            kv_cache_metadata.position,
            jd.num_candidates_offsets,
            kv_cache_metadata.new_history_nnz_cuda,
            num_tokens,  # kv_cache_metadata.new_history_nnz
            paged_k_cache,
            paged_v_cache,
            kv_cache_metadata.kv_indices,
            kv_cache_metadata.kv_indptr,
            kv_cache_metadata.kv_last_page_len,
            0,  # NHD layout
        )

        kv_cache_metadata.onload_history_kv_events[self.layer_idx].wait(
            torch.cuda.current_stream()
        )

        small_token = 118 # 3 # 122 # 118 # 112 # 134 # 268 # 270 # 290 # 280 #  134 # 112 # 
        large_token = 174 # 15 # 134 # 174 # 130 # 156 # 460 # 428 # 312 # 452 # 156 # 130 # 
        save_qkv = True
        save_out = True
        dump_dir_prefix = "/workspace/recsys-separate_data/bug_fix/check_kvcache/kvcache_diff/new_code/mismatch/dump_dir/new_code_0902/u867"
        if (num_tokens == small_token or num_tokens == large_token) and save_qkv:
            dump_dir = dump_dir_prefix
            if num_tokens == large_token:
                torch.save({"layer_input": layer_input[-small_token:].detach().cpu(),
                    "layer_norm_input.weight": self._input_layernorm_weight.cpu(),
                    "layer_norm_input.bias": self._input_layernorm_bias.cpu(),
                    "normed_input": normed_input[-small_token:].detach().cpu(),
                    "linear_uvqk.weight": self._linear_uvqk.weight.cpu(),
                    "linear_uvqk.bias": self._linear_uvqk.bias.cpu(),
                    "lineared_uvqk": lineared_uvqk[-small_token:].detach().cpu(),
                    "mixed_uvqk": mixed_uvqk[-small_token:].detach().cpu(),
                    "key": key[-small_token:].detach().cpu(),
                    "value": value[-small_token:].detach().cpu(),
                    "query": query[-small_token:].detach().cpu()},
                   f"{dump_dir}/dump_qkv_{num_tokens}_layer_{layer_idx}.pt")
            else:
                torch.save({"layer_input": layer_input.detach().cpu(),
                    "layer_norm_input.weight": self._input_layernorm_weight.cpu(),
                    "layer_norm_input.bias": self._input_layernorm_bias.cpu(),
                    "normed_input": normed_input.detach().cpu(),
                    "linear_uvqk.weight": self._linear_uvqk.weight.cpu(),
                    "linear_uvqk.bias": self._linear_uvqk.bias.cpu(),
                    "lineared_uvqk": lineared_uvqk.detach().cpu(),
                    "mixed_uvqk": mixed_uvqk.detach().cpu(),
                    "key": key.detach().cpu(),
                    "value": value.detach().cpu(),
                    "query": query.detach().cpu()},
                   f"{dump_dir}/dump_qkv_{num_tokens}_layer_{layer_idx}.pt")
            print(f"save dump_qkv_{num_tokens}_layer_{layer_idx}.pt done")

        jagged_attn_output = hstu_attn.hstu_attn_varlen_func(
            query,
            key,
            value,
            jd.seqlen_offsets,
            kv_cache_metadata.total_history_offsets[: batch_size + 1],
            self._max_seqlen,
            self._max_seqlen,
            num_contexts=jd.contextual_seqlen if jd.contextual_max_seqlen > 0 else None,
            num_targets=jd.num_candidates,
            target_group_size=1,
            window_size=(-1, 0),
            alpha=self._alpha,
            rab=None,
            has_drab=False,
            is_delta_q=jd.contextual_max_seqlen == 0,
            kv_cache=kv_cache_table,
            page_offsets=kv_cache_metadata.kv_indptr,
            page_ids=kv_cache_metadata.kv_indices,
            last_page_lens=kv_cache_metadata.kv_last_page_len,
            seq_offsets_t=jd.num_candidates_offsets,
        )

        jagged_attn_output = jagged_attn_output.view(
            -1, self._num_heads * self._linear_dim_per_head
        )
        parallel_input = user * F.layer_norm(
            jagged_attn_output,
            normalized_shape=[self._num_heads * self._linear_dim_per_head],
            weight=self._output_layernorm_weight,
            bias=self._output_layernorm_bias,
            eps=self._eps,
        )

        layer_output = self._linear_proj(parallel_input)

        layer_output_before_residual = layer_output.clone()

        if self._residual:
            torch.add(layer_output, layer_input, out=layer_output)

        if (num_tokens == small_token or num_tokens == large_token) and save_out:
            dump_dir = dump_dir_prefix
            # proj_full_then_slice = self._linear_proj(parallel_input)[-small_token:]          # 先 452x 再切 280
            # proj_slice_direct   = self._linear_proj(parallel_input[-small_token:])           # 直接 280x
            # print("Δ(proj_full_then_slice vs proj_slice_direct) L∞ =",
            #       (proj_full_then_slice - proj_slice_direct).abs().max().item())
            if "save" and True:
                if num_tokens == large_token:
                    torch.save({"jagged_attn_output": jagged_attn_output[-small_token:].detach().cpu(),
                        "parallel_input": parallel_input[-small_token:].detach().cpu(),
                        "layer_output_before_residual": layer_output_before_residual[-small_token:].detach().cpu(),
                        "layer_output": layer_output[-small_token:].detach().cpu()},
                       f"{dump_dir}/dump_output_{num_tokens}_layer_{layer_idx}.pt")
                else:
                    torch.save({"jagged_attn_output": jagged_attn_output.detach().cpu(),
                        "parallel_input": parallel_input.detach().cpu(),
                        "layer_output_before_residual": layer_output_before_residual.detach().cpu(),
                        "layer_output": layer_output.detach().cpu()},
                       f"{dump_dir}/dump_output_{num_tokens}_layer_{layer_idx}.pt")
                print(f"save dump_output_{num_tokens}_layer_{layer_idx}.pt done")

        return layer_output

    @torch.inference_mode()
    def forward_input(
        self,
        batch_size: int,
        num_tokens: int,
        input_buffer: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> JaggedData:
        input_tensor = input_buffer[:num_tokens, ...]
        normed_input = F.layer_norm(
            input_tensor,
            normalized_shape=[self._embedding_dim],
            weight=self._input_layernorm_weight,
            bias=self._input_layernorm_bias,
            eps=self._eps,
        )

        torch.addmm(
            self._linear_uvqk.bias,
            normed_input,
            self._linear_uvqk_weight,
            out=self.uvqk_buffer_[:num_tokens, ...],
        )
        F.silu(self.uvqk_buffer_[:num_tokens, ...], inplace=True)
        (user, value, query, key) = torch.split(
            self.uvqk_buffer_[:num_tokens, ...],
            self._split_arg_list,
            dim=-1,
        )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
        (paged_k_cache, paged_v_cache) = kv_cache_table.unbind(dim=1)
        paged_kvcache_ops.append_kvcache(
            key,
            value,
            kv_cache_metadata.batch_indices,
            kv_cache_metadata.position,
            jd.num_candidates_offsets[: batch_size + 1],
            kv_cache_metadata.new_history_nnz_cuda,
            num_tokens,  # kv_cache_metadata.new_history_nnz
            paged_k_cache,
            paged_v_cache,
            kv_cache_metadata.kv_indices,
            kv_cache_metadata.kv_indptr,
            kv_cache_metadata.kv_last_page_len,
            0,  # NHD layout
        )

        return self.uvqk_buffer_[:num_tokens, ...]

    @torch.inference_mode()
    def forward_output(
        self,
        batch_size: int,
        num_tokens: int,
        input_buffer: torch.Tensor,
        jd: JaggedData,
        kv_cache_metadata,
    ) -> JaggedData:
        (user, value, query, key) = torch.split(
            self.uvqk_buffer_[:num_tokens, ...],
            self._split_arg_list,
            dim=-1,
        )

        value = value.view(-1, self._num_heads, self._linear_dim_per_head)
        query = query.view(-1, self._num_heads, self._attention_dim_per_head)
        key = key.view(-1, self._num_heads, self._attention_dim_per_head)

        kv_cache_table = kv_cache_metadata.kv_cache_table[self.layer_idx]
        jagged_attn_output = hstu_attn.hstu_attn_varlen_func(
            query,
            key,
            value,
            jd.seqlen_offsets[: batch_size + 1],
            kv_cache_metadata.total_history_offsets[: batch_size + 1],
            self._max_seqlen,
            self._max_seqlen,
            num_contexts=jd.contextual_seqlen[:batch_size] if jd.contextual_max_seqlen > 0 else None,
            num_targets=jd.num_candidates[:batch_size],
            target_group_size=1,
            window_size=(-1, 0),
            alpha=self._alpha,
            rab=None,
            has_drab=False,
            is_delta_q=jd.contextual_max_seqlen == 0,
            kv_cache=kv_cache_table,
            page_offsets=kv_cache_metadata.kv_indptr,
            page_ids=kv_cache_metadata.kv_indices,
            last_page_lens=kv_cache_metadata.kv_last_page_len,
            seq_offsets_t=jd.num_candidates_offsets[: batch_size + 1],
        )

        jagged_attn_output = jagged_attn_output.view(
            -1, self._num_heads * self._linear_dim_per_head
        )
        parallel_input = user * F.layer_norm(
            jagged_attn_output,
            normalized_shape=[self._num_heads * self._linear_dim_per_head],
            weight=self._output_layernorm_weight,
            bias=self._output_layernorm_bias,
            eps=self._eps,
        )

        if self._residual:
            torch.add(
                self._linear_proj(parallel_input),
                input_buffer[:num_tokens, ...],
                out=self.output_buffer_[:num_tokens, ...],
            )
        else:
            self.output_buffer_[:num_tokens, ...] = self._linear_proj(parallel_input)

        return self.output_buffer_[:num_tokens, ...]
