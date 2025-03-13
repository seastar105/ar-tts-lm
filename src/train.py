import sys

import rootutils
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, get_token
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import SequencePackWrapper, TextCodeDataset


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    dataset_name = "seastar105/aihub-542-tokenized"
    dataset_split = "train"

    dataset = load_dataset(dataset_name, split=dataset_split, num_proc=16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = TextCodeDataset(dataset, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.train()

    # manage vocab expansion
    mean_resizing = False  # I think mean resizing does not make sense here
    model.resize_token_embeddings(len(dataset.tokenizer), mean_resizing=mean_resizing)

    output_dir = "./output/pack_aihub_1M"
    dataset.tokenizer.save_pretrained(output_dir)

    per_device_batch_size = 256
    per_device_micro_batch_size = 1
    gradient_accumulation = per_device_batch_size // per_device_micro_batch_size

    dataloader_num_workers = 4
    max_steps = 10000
    warmup_steps = 200
    learning_rate = 2e-4

    adamw_beta1 = 0.9
    adamw_beta2 = 0.999
    decay = 0.01

    max_length = 4096
    packed_dataset = SequencePackWrapper(dataset, max_length=max_length)

    trainer = Trainer(
        model=model,
        train_dataset=packed_dataset,
        args=TrainingArguments(
            do_train=True,
            do_eval=False,
            do_predict=False,
            per_device_train_batch_size=per_device_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            weight_decay=decay,
            adam_beta1=adamw_beta1,
            adam_beta2=adamw_beta2,
            max_grad_norm=1.0,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            logging_steps=20,
            logging_dir="./logs/pack_aihub_1M",
            dataloader_num_workers=dataloader_num_workers,
            bf16=True,
            gradient_checkpointing=True,
            lr_scheduler_type="cosine",
            save_strategy="no",
        ),
    )
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
