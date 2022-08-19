# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import collections
import csv
import gc
import logging
import os
import pickle
import time
from multiprocessing import Pool
from typing import Tuple

import torch
from torch.utils.data import Dataset

#N_JOBS = cpu_count()
N_JOBS = N_USABLE_CPUS = len(os.sched_getaffinity(0))
logger = logging.getLogger(__name__)


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield si, si+(d+1 if i < r else d)


def feature_creation_worker(files, tokenizer, block_size, worker_idx):
    examples = []
    sample_client = []
    client_mapping = collections.defaultdict(list)

    user_id = -1
    start_time = time.time()
    for idx, file in enumerate(files):
        try:
            with open(file, encoding="utf-8", errors='ignore') as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            if len(tokenized_text) > 0:
                user_id += 1

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
                client_mapping[user_id].append(len(examples)-1)
                sample_client.append(user_id)
        except Exception as e:
            logging.error(f"Worker {worker_idx}: fail due to {e}")
        if idx % 100 == 0:
            logging.info(f"Worker {worker_idx}: {len(files)-idx} files left, {idx} files complete, remaining time {(time.time()-start_time)/(idx+1)*(len(files)-idx)}")
            gc.collect()

    return (examples, client_mapping, sample_client)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512, evaluate=False):

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        if evaluate:
            directory = os.path.join(args.log_path, 'metadata', args.data_set, 'test') #file_path
        else:
            directory = os.path.join(args.log_path, 'metadata', args.data_set, 'train') #file_path
        if not os.path.isdir(directory):
            os.mkdir(directory, exist_ok=True)
        cached_features_file = os.path.join(directory, args.model + "_cached_lm_" + str(block_size) + '_' + str(args.process_files_ratio))

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Rank: %s - Loading features from cached file %s", args.this_rank, cached_features_file)
            gc.disable()
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
                self.client_mapping = pickle.load(handle)
            gc.enable()
            gc.collect()
            logger.info("Rank: %s - Finished features from cached file %s", args.this_rank, cached_features_file)
        else:
            # Ahmed - only one worker (rank==1) should process the files, make sure the save location is shared by workers
            if args.this_rank == 1:
                logger.info("Creating features from dataset file at %s", directory)

                self.examples = []
                self.sample_client = []
                self.client_mapping = collections.defaultdict(list)
                user_id = -1

                files = [entry.name for entry in os.scandir(file_path) if '_cached_lm_' not in entry.name]
                # make sure files are ordered
                files = [os.path.join(file_path, x) for x in sorted(files)]

                pool_inputs = []
                pool = Pool(N_JOBS)
                worker_cnt = 0
                #for begin, end in chunks_idx(range(len(files)), N_JOBS):
                for begin, end in chunks_idx(range(int(len(files) * args.process_files_ratio)), N_JOBS):
                    logging.info(f'{begin} {end} files assigned to {worker_cnt}:{N_USABLE_CPUS}:{N_JOBS}')
                    pool_inputs.append([files[begin:end], tokenizer, block_size, worker_cnt])
                    worker_cnt += 1

                pool_outputs = pool.starmap(feature_creation_worker, pool_inputs)
                pool.close()
                pool.join()

                user_id_base = 1
                for (examples, client_mapping, sample_client) in pool_outputs:
                    self.examples += examples
                    true_sample_client = [i + user_id_base for i in sample_client]
                    self.sample_client += true_sample_client
                    for user_id, true_user_id in zip(sample_client, true_sample_client):
                        self.client_mapping[true_user_id] = client_mapping[user_id]
                    user_id_base = true_sample_client[-1] + 1

                # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should look for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.
                logger.info("Saving features into cached file %s", cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=-1)
                    pickle.dump(self.client_mapping, handle, protocol=-1)
                    pickle.dump(self.sample_client, handle, protocol=-1)

                # dump the data_mapping_file
                results = [['client_id', 'sample_path', 'label_name', 'label_id']]
                for i in range(len(self.sample_client)):
                    results.append([self.sample_client[i], i, -1, -1])

                #with open(os.path.join(file_path, '../client_data_mapping', 'result.csv'), 'w') as csvFile:
                with open(os.path.join(directory, 'result_' + str(args.process_files_ratio) + '.csv'), 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    for line in results:
                        writer.writerow(line)
            else:
                #Ahmed - other ranks sleep until the main worker (rank=1) finishes and writes the files
                while not os.path.exists(cached_features_file):
                    time.sleep(120)
                # Ahmed - load the dumped files by main worker (rank 1)
                logger.info("Rank: %s - Loading features from cached file %s", args.this_rank, cached_features_file)
                gc.disable()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                    self.client_mapping = pickle.load(handle)
                gc.enable()
                gc.collect()
                logger.info("Rank: %s - Finished loading features from cached file %s", args.this_rank, cached_features_file)

        self.data = self.examples
        self.targets = [0 for i in range(len(self.data))]
        #Ahmed - change to return the actual tokens - does not work, mask tokens expect tensors
        #_, self.targets = mask_tokens(self.data, tokenizer, args)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = os.path.join(args.data_dir, 'test') if evaluate else os.path.join(args.data_dir, 'train')
    return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size, evaluate=evaluate)

def mask_tokens(inputs, tokenizer, args, device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone().to(device=device)
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability, device=device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.tensor(torch.bernoulli(probability_matrix), dtype=torch.bool).detach().to(device=device)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.tensor(torch.bernoulli(torch.full(labels.shape, 0.8)), dtype=torch.bool, device=device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.tensor(torch.bernoulli(torch.full(labels.shape, 0.5)), dtype=torch.bool, device=device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    bool_indices_random = indices_random
    inputs[bool_indices_random] = random_words[bool_indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
