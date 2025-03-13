from functools import partial
from itertools import chain

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.sampler import MultipackDistributedBatchSampler

boa_token = "<|beginofaudio|>"
eoa_token = "<|endofaudio|>"
code_template = "<|AUDIO{idx}|>"
code_vocab_size = 4096
use_chat_template = True
tts_instruction = "Convert the following text to audible speech.\n{text}"
max_seq_length = 4096


def tokenize_tts(item, tokenizer, boa_token_id, use_chat_template: bool = False):
    text = item["text"].strip()
    codes = item["codes"]
    code_str = boa_token + "".join([code_template.format(idx=code) for code in codes]) + eoa_token
    if use_chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": tts_instruction.format(text=text)},
            {"role": "assistant", "content": code_str},
        ]
        chat_str = tokenizer.apply_chat_template(messages, tokenize=False).strip()
        tokens = torch.LongTensor(tokenizer(chat_str).input_ids)
        labels = tokens.clone()
        labels[labels < boa_token_id] = -100  # on sft format, boa_token should be also predicted
        labels[-1] = tokens[-1]  # eos should be predicted
    else:
        text_tokens = tokenizer(text).input_ids
        code_tokens = tokenizer(code_str).input_ids
        tokens = torch.LongTensor(text_tokens + code_tokens)
        labels = tokens.clone()
        labels[tokens <= boa_token_id] = -100
    return {"input_ids": tokens.squeeze(), "labels": labels.squeeze(), "length": tokens.shape[-1]}


num_workers = 32
tokenizer = AutoTokenizer.from_pretrained("seastar105/qwen2.5-0.5b-inst-expansion")
dataset = load_dataset("seastar105/aihub-542-tokenized", split="train", num_proc=num_workers)
boa_token_id = tokenizer.convert_tokens_to_ids(boa_token)
tokenize_fn = partial(tokenize_tts, tokenizer=tokenizer, boa_token_id=boa_token_id, use_chat_template=use_chat_template)
tokenized_dataset = dataset.map(
    tokenize_fn, remove_columns=dataset.column_names, num_proc=num_workers, desc="Formatting TTS dataset"
)

lengths = np.array(tokenized_dataset["length"])
sampler = MultipackDistributedBatchSampler(max_seq_length, lengths, num_replicas=1, rank=0)

packed_dataset = []

for batch_indices in tqdm(sampler):
    batch = [tokenized_dataset[i] for i in batch_indices.tolist()]
    input_ids = list(chain(*[item["input_ids"] for item in batch]))
    labels = list(chain(*[item["labels"] for item in batch]))
    packed_dataset.append({"input_ids": input_ids, "labels": labels})

packed_dataset = Dataset.from_list(packed_dataset)
packed_dataset.push_to_hub("seastar105/aihub-542-wavtok-packed")
