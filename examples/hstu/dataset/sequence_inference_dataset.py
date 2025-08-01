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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from dataset.utils import Batch, RankingBatch, RetrievalBatch
from preprocessor import get_common_preprocessors
from torch.utils.data.dataset import IterableDataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def load_seq(x: str):
    if isinstance(x, str):
        y = json.loads(x)
    else:
        y = x
    return y


def maybe_truncate_seq(
    y: List[int],
    max_seq_len: int,
) -> List[int]:
    y_len = len(y)
    if y_len > max_seq_len:
        print(f"[yinj] Truncating sequence from {y_len} to {max_seq_len}")
        y = y[:max_seq_len]
    return y


class SequenceInferenceDataset(IterableDataset[Batch]):
    """
    SequenceInferenceDataset is an iterable dataset designed for distributed recommendation systems.
    It handles loading, shuffling, and batching of sequence data for training models.

    Args:
        seq_logs_file (str): Path to the sequence logs file.
        batch_size (int): The batch size.
        max_seqlen (int): The maximum sequence length.
        item_feature_name (str): The name of the item feature.
        contextual_feature_names (list[str], optional): List of contextual feature names. Defaults to [].
        action_feature_name (str, optional): The name of the action feature. Defaults to None.
        max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
        num_tasks (int, optional): The number of tasks. Defaults to 0.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        shuffle (bool): Whether to shuffle the data.
        random_seed (int): The random seed for shuffling.
        is_train_dataset (bool): Whether this dataset is for training.
        nrows (int, optional): The number of rows to read from the file. Defaults to None, meaning all rows are read.
    """

    def __init__(
        self,
        seq_logs_file: str,
        batch_size: int,
        max_seqlen: int,
        item_feature_name: str,
        contextual_feature_names: List[str],
        action_feature_name: str,
        max_num_candidates: int = 0,
        num_tasks: int = 0,
        *,
        rank: int,
        world_size: int,
        shuffle: bool,
        random_seed: int,
        is_train_dataset: bool,
        nrows: Optional[int] = None,
        use_time_segment: bool = False,
        time_interval_ms: int = 3600000,
        current_time_ms: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._device = torch.cpu.current_device()

        if nrows is None:
            self._seq_logs_frame = pd.read_csv(seq_logs_file, delimiter=",")
        else:
            self._seq_logs_frame = pd.read_csv(
                seq_logs_file, delimiter=",", nrows=nrows
            )
        num_total_samples = len(self._seq_logs_frame)
        train_samples = int(num_total_samples * 0.7)
        test_samples = num_total_samples - train_samples
        if is_train_dataset:
            self._seq_logs_frame = self._seq_logs_frame.head(train_samples)
        else:
            self._seq_logs_frame = self._seq_logs_frame.tail(test_samples)

        self._max_seqlen = max_seqlen

        self._batch_size = batch_size
        self._global_batch_size = batch_size * world_size
        self._shuffle = shuffle
        self._random_seed = random_seed

        self._contextual_feature_names = contextual_feature_names
        if self._max_seqlen <= len(self._contextual_feature_names):
            raise ValueError(
                f"max_seqlen is too small. should > {len(self._contextual_feature_names)}"
            )
        self._item_feature_name = item_feature_name
        self._action_feature_name = action_feature_name
        self._max_num_candidates = max_num_candidates
        self._num_tasks = num_tasks

        self._rank = rank
        self._world_size = world_size
        self._global_batch = batch_size * self._world_size

        self.use_time_segment = use_time_segment
        self.time_interval_ms = time_interval_ms
        self.current_time_ms = current_time_ms

        # self._window_frame = None

        if self.use_time_segment:
            self.time_column_name = None
            for candidate in ["time_ms", "unix_timestamp"]:
                if candidate in self._seq_logs_frame.columns:
                    self.time_column_name = candidate
                    break
            if self.time_column_name is None:
                raise ValueError("use_time_segment=True but no suitable time column ('time_ms' or 'unix_timestamp') found.")
            # Apply json.loads to get _list version
            self._seq_logs_frame[f"{self.time_column_name}_list"] = self._seq_logs_frame[self.time_column_name].apply(json.loads)
            self._earliest_time = min(min(x) for x in self._seq_logs_frame[f"{self.time_column_name}_list"])
            self._latest_time = max(max(x) for x in self._seq_logs_frame[f"{self.time_column_name}_list"])

            self.current_time_ms = self._earliest_time if current_time_ms is None else current_time_ms
            print(f"[yinj] current_time is {self.current_time_ms}, earliest time is {self._earliest_time}")
            self._refresh_time_window()
        else:
            self._num_samples = len(self._seq_logs_frame)
            self._sample_ids = np.arange(self._num_samples)
            self._shuffle_batch()
    
    def _refresh_time_window(self):
        """
        Keep only users with interactions in the current time window,
        i.e., between current_time_ms and min(current_time_ms + time_interval_ms, latest_time).
        Truncate sequences to the current window.
        """
        valid_rows = []
        time_list_col = f"{self.time_column_name}_list"

        window_start = self.current_time_ms
        window_end = min(self.current_time_ms + self.time_interval_ms, self._latest_time)

        for _, row in self._seq_logs_frame.iterrows():
            times = row[time_list_col]
            # 找出所有时间在窗口内的索引
            valid_idx = [i for i, t in enumerate(times) if window_start <= t < window_end]

            if valid_idx:
                new_row = row.copy()
                # 按valid_idx截断所有序列字段
                for col in [self._item_feature_name, self._action_feature_name, self.time_column_name]:
                    seq = json.loads(row[col]) if isinstance(row[col], str) else row[col]
                    new_row[col] = json.dumps([seq[i] for i in valid_idx])
                    # new_row["user_id"] = new_row["user_id"] % 2 # [yinj] test
                valid_rows.append(new_row)

        self._window_frame = pd.DataFrame(valid_rows)
        # print(f"self._window_frame is {self._window_frame}")
        self._num_samples = len(self._window_frame)
        self._sample_ids = np.arange(self._num_samples)
        self._shuffle_batch()



    def slide_window(self) -> bool:
        """
        Slide to the next time window and refresh the data.
        Returns True if successfully slid to next window, False if reached the end.
        """
        next_start = self.current_time_ms + self.time_interval_ms
        if next_start > self._latest_time:
            # 到时间末尾，无法继续滑动了
            return False
        self.current_time_ms = next_start
        self._refresh_time_window()
        return True


    # We do batching in our own
    def __len__(self) -> int:
        return math.ceil(self._num_samples / self._global_batch_size)

    def __iter__(self) -> Iterator[Batch]:
        frame = self._window_frame if self.use_time_segment else self._seq_logs_frame
        for i in range(len(self)):
            local_batch_start = (
                i * self._global_batch_size + self._rank * self._batch_size
            )
            local_batch_end = min(
                i * self._global_batch_size + (self._rank + 1) * self._batch_size,
                len(self._sample_ids),
            )
            sample_ids = self._sample_ids[local_batch_start:local_batch_end]

            contextual_features: Dict[str, List[int]] = defaultdict(list)
            contextual_features_seqlen: Dict[str, List[int]] = defaultdict(list)
            item_features: List[int] = []
            item_features_seqlen: List[int] = []
            action_features: List[int] = []
            action_features_seqlen: List[int] = []
            num_candidates: List[int] = []
            # labels dtype: int
            labels: List[int] = []
            for sample_id in sample_ids:
                data = frame.iloc[sample_id]
                for contextual_feature_name in self._contextual_feature_names:
                    contextual_features[contextual_feature_name].append(
                        data[contextual_feature_name]
                    )
                    contextual_features_seqlen[contextual_feature_name].append(1)

                item_seq = load_seq(data[self._item_feature_name])
                candidate_seq = item_seq[-self._max_num_candidates :]
                item_seq = item_seq[: -self._max_num_candidates]

                item_seq = maybe_truncate_seq(
                    item_seq,
                    self._max_seqlen
                    - len(self._contextual_feature_names)
                    - self._max_num_candidates,
                )
                item_seq = item_seq + candidate_seq
                item_features.extend(item_seq)
                item_features_seqlen.append(len(item_seq))

                action_seq = load_seq(data[self._action_feature_name])
                candidate_action_seq = action_seq[-self._max_num_candidates :]
                action_seq = action_seq[: -self._max_num_candidates]
                action_seq = maybe_truncate_seq(
                    action_seq,
                    self._max_seqlen
                    - len(self._contextual_feature_names)
                    - self._max_num_candidates,
                )
                action_seq = action_seq + candidate_action_seq
                action_features.extend(action_seq)
                action_features_seqlen.append(len(action_seq))
                if self._max_num_candidates > 0:
                    num_candidate = min(self._max_num_candidates, len(item_seq))
                    num_candidates.append(num_candidate)

                if self._num_tasks > 0:
                    label = (
                        candidate_action_seq
                        if self._max_num_candidates > 0
                        else action_seq
                    )
                    labels.extend(label)
            # if len(item_features_seqlen) < self._batch_size:
            #     padded_size = self._batch_size - len(item_features_seqlen)
            #     for name in self._contextual_feature_names:
            #         contextual_features_seqlen[name] += [0 for _ in range(padded_size)]
            #     item_features_seqlen += [0 for _ in range(padded_size)]
            #     action_features_seqlen += [0 for _ in range(padded_size)]
            #     if self._max_num_candidates > 0:
            #         num_candidates += [0 for _ in range(padded_size)]
            feature_to_max_seqlen = {}
            for name in self._contextual_feature_names:
                feature_to_max_seqlen[name] = max(contextual_features_seqlen[name])
            feature_to_max_seqlen[self._item_feature_name] = max(item_features_seqlen)
            feature_to_max_seqlen[self._action_feature_name] = max(
                action_features_seqlen
            )
            contextual_features_tensor = torch.tensor(
                [contextual_features[name] for name in self._contextual_feature_names]
            ).view(-1)
            contextual_features_lengths_tensor = torch.tensor(
                [
                    contextual_features_seqlen[name]
                    for name in self._contextual_feature_names
                ]
            ).view(-1)
            features = KeyedJaggedTensor.from_lengths_sync(
                keys=self._contextual_feature_names
                + [self._item_feature_name, self._action_feature_name],
                values=torch.concat(
                    [
                        contextual_features_tensor,
                        torch.tensor(item_features, device=self._device),
                        torch.tensor(action_features, device=self._device),
                    ]
                ).long(),
                lengths=torch.concat(
                    [
                        contextual_features_lengths_tensor,
                        torch.tensor(item_features_seqlen, device=self._device),
                        torch.tensor(action_features_seqlen, device=self._device),
                    ]
                ).long(),
            )
            batch_kwargs = dict(
                features=features,
                batch_size=len(sample_ids),
                feature_to_max_seqlen=feature_to_max_seqlen,
                contextual_feature_names=self._contextual_feature_names,
                item_feature_name=self._item_feature_name,
                action_feature_name=self._action_feature_name,
                max_num_candidates=self._max_num_candidates,
                num_candidates=torch.tensor(num_candidates, device=self._device)
                if self._max_num_candidates > 0
                else None,
            )
            if self._num_tasks > 0:
                # TODO: Need to considering using float in the future.
                yield RankingBatch(
                    labels=torch.tensor(
                        labels, device=self._device, dtype=torch.int64
                    ).view(-1),
                    **batch_kwargs,
                )
            else:
                yield RetrievalBatch(**batch_kwargs)

    def _shuffle_batch(self):
        """
        currently, only inter global batch shuffle is supported
        """
        rng_state = np.random.get_state()
        np.random.seed(self._random_seed)
        if self._shuffle:
            self._sample_ids = np.random.permutation(self._sample_ids)
        # do not miss restoring the random state
        np.random.set_state(rng_state)


def get_dataset(
    dataset_name: str,
    max_sequence_length: int,
    max_num_candidates: int,
    num_tasks: int,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool,
    random_seed: int,
    eval_batch_size: Optional[int] = None,
    *,
    nrows=None,
    use_time_segment: bool = False,
    time_interval_ms: int = 3600000,
    current_time_ms: Optional[int] = None,
) -> Tuple[SequenceInferenceDataset, Optional[SequenceInferenceDataset]]:
    """
    Retrieves the training and evaluation datasets for sequence-based recommendation tasks.

    Args:
        dataset_name (str): The name of the dataset to retrieve.
        max_sequence_length (int): The maximum length of sequences in the dataset.
        max_num_candidates (int): The maximum number of candidate items.
        num_tasks (int): The number of tasks;
        batch_size (int): The batch size for training.
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes in distributed training.
        shuffle (bool): Whether to shuffle the data during training.
        random_seed (int): The random seed for shuffling.
        eval_batch_size (Optional[int], optional): The batch size for evaluation. Defaults to None.
        nrows (Optional[int], optional): The number of rows to read from the dataset file. Defaults to None, meaning all rows are read.

    Returns:
        Tuple[SequenceInferenceDataset, Optional[SequenceInferenceDataset]]: A tuple containing the training dataset and the evaluation dataset (if `eval_batch_size` is provided).
    """
    common_preprocessors = get_common_preprocessors('tmp_data/')
    if dataset_name not in common_preprocessors:
        raise ValueError(f"{dataset_name} not in preprocessors")
    dp = common_preprocessors[dataset_name]
    train_dataset = SequenceInferenceDataset(
        seq_logs_file=dp._output_file,
        batch_size=batch_size,
        max_seqlen=max_sequence_length,
        item_feature_name=dp._item_feature_name,
        contextual_feature_names=dp._contextual_feature_names,
        action_feature_name=dp._action_feature_name,
        max_num_candidates=max_num_candidates,
        num_tasks=num_tasks,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        random_seed=random_seed,
        is_train_dataset=True,
        nrows=nrows,
        use_time_segment = use_time_segment,
        time_interval_ms = time_interval_ms,
        current_time_ms = current_time_ms,
    )
    if eval_batch_size is not None:
        eval_dataset = SequenceInferenceDataset(
            seq_logs_file=dp._output_file,
            batch_size=eval_batch_size,
            max_seqlen=max_sequence_length,
            item_feature_name=dp._item_feature_name,
            contextual_feature_names=dp._contextual_feature_names,
            action_feature_name=dp._action_feature_name,
            max_num_candidates=max_num_candidates,
            num_tasks=num_tasks,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            random_seed=random_seed,
            is_train_dataset=False,
            nrows=nrows,
            use_time_segment = use_time_segment,
            time_interval_ms = time_interval_ms,
            current_time_ms = current_time_ms,
        )
    else:
        eval_dataset = None
    return train_dataset, eval_dataset
