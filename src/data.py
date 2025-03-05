import logging
import os
import sys

import numpy as np
import torch
from datasets import Dataset
from transformers import Qwen2Tokenizer

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.DEBUG)


class TextCodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Qwen2Tokenizer,
        text_column_name: str = "text",
        code_column_name: str = "codes",
        code_template: str = "<|AUDIO{idx}|>",
        code_vocab_size: int = 4096,
        boa_token: str = "<|beginofaudio|>",
        eoa_token: str = "<|endofaudio|>",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.text_column_name = text_column_name
        self.code_column_name = code_column_name
        self.code_template = code_template
        self.code_vocab_size = code_vocab_size
        self.boa_token = boa_token
        self.eoa_token = eoa_token

        # Expand tokenizer if necessary
        audio_tokens = [boa_token, eoa_token] + [code_template.format(idx=idx) for idx in range(code_vocab_size)]
        vocab = tokenizer.get_vocab()
        non_vocab_tokens = [token for token in audio_tokens if token not in vocab]
        self.tokenizer.add_tokens(non_vocab_tokens, special_tokens=False)

        self.boa_token_id = self.tokenizer.convert_tokens_to_ids(boa_token)
        self.eoa_token_id = self.tokenizer.convert_tokens_to_ids(eoa_token)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_column_name]
        codes = item[self.code_column_name]

        text_tokens = self.tokenizer(text).input_ids
        code_str = self.boa_token + "".join([self.code_template.format(idx=code) for code in codes]) + self.eoa_token
        code_tokens = self.tokenizer(code_str).input_ids

        tokens = torch.LongTensor(text_tokens + code_tokens)
        labels = tokens.clone()
        labels[tokens <= self.boa_token_id] = -100
        position_ids = torch.arange(len(tokens))
        return {
            "input_ids": tokens.squeeze(),
            "labels": labels.squeeze(),
            "position_ids": position_ids.squeeze(),
        }


class SequencePackWrapper(torch.utils.data.IterableDataset):
    def __init__(self, dataset: TextCodeDataset, max_length: int = 2048, buf_size: int = 100):
        self.dataset = dataset
        self.max_length = max_length
        self.buf_size = buf_size

        self.buffer = []
        self.generator = None
        self.pad_token_id = self.dataset.tokenizer.pad_token_id

    def pytorch_worker_info(group=None):  # sourcery skip: use-contextlib-suppress
        """Return node and worker info for PyTorch and some distributed environments.

        Args:
            group (optional): The process group for distributed environments. Defaults to None.

        Returns:
            tuple: A tuple containing (rank, world_size, worker, num_workers).
        """
        rank = 0
        world_size = 1
        worker = 0
        num_workers = 1
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = group or torch.distributed.group.WORLD
                rank = torch.distributed.get_rank(group=group)
                world_size = torch.distributed.get_world_size(group=group)
        if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
            worker = int(os.environ["WORKER"])
            num_workers = int(os.environ["NUM_WORKERS"])
        else:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers

        return rank, world_size, worker, num_workers

    def enqueue(self):
        while True:
            idx = torch.randint(0, len(self.dataset), (1,), generator=self.generator).item()
            sample = self.dataset[idx]
            if sample["input_ids"].shape[0] > self.max_length:
                continue
            break
        self.buffer.append(sample)

    def get_packed_sample(self):
        np.random.shuffle(self.buffer)

        cur_length = 0
        samples = []
        idx = len(self.buffer) - 1
        while idx >= 0 and cur_length < self.max_length:
            sample = self.buffer[idx]
            length = len(sample["input_ids"])
            if cur_length + length > self.max_length:
                idx -= 1
                continue
            cur_length += length
            samples.append(sample)
            self.buffer[idx] = self.buffer[-1]
            idx -= 1

        for _ in range(len(samples)):
            self.buffer.pop()

        # concatenate samples
        sample = {
            "input_ids": torch.concat([s["input_ids"] for s in samples], dim=0),
            "labels": torch.concat([s["labels"] for s in samples], dim=0),
            "position_ids": torch.concat([s["position_ids"] for s in samples], dim=0),
        }

        # pad to max length
        pad_length = self.max_length - sample["input_ids"].shape[0]
        sample["input_ids"] = torch.cat([sample["input_ids"], torch.full((pad_length,), self.pad_token_id)], dim=0)
        sample["labels"] = torch.cat([sample["labels"], torch.full((pad_length,), -100)], dim=0)
        sample["position_ids"] = torch.cat([sample["position_ids"], torch.arange(0, pad_length)], dim=0)

        return sample

    def __iter__(self):
        if self.generator is None:
            rank, world_size, worker, num_workers = self.pytorch_worker_info()
            seed = rank * num_workers + worker
            logger.debug(
                f"Rank: {rank}, World Size: {world_size}, Worker: {worker}, Num Workers: {num_workers}, Seed: {seed}"
            )
            self.generator = torch.Generator(device="cpu")
            self.generator.manual_seed(seed)
            np.random.seed(seed)
        # Infinitely generate samples
        while True:
            if len(self.buffer) < self.buf_size:
                self.enqueue()
            else:
                yield self.get_packed_sample()
