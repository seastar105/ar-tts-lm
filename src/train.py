import sys

import rootutils
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import SequencePackWrapper, TextCodeDataset


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    dataset_name = "seastar105/speech-token-dataset"
    dataset_split = "emilia_yodas_ko_wavtok"

    dataset = load_dataset(dataset_name, split=dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = TextCodeDataset(dataset, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.train()

    # manage vocab expansion
    model.resize_token_embeddings(len(tokenizer))

    total_batch_size = 32
    micro_batch_size = 8
    gradient_accumulation = total_batch_size // micro_batch_size

    dataloader_num_workers = 4
    max_steps = 5000
    warmup_steps = 200
    learning_rate = 1e-5

    adamw_beta1 = 0.9
    adamw_beta2 = 0.999
    decay = 0.01

    max_length = 2048
    packed_dataset = SequencePackWrapper(dataset, max_length=max_length)

    trainer = Trainer(
        model=model,
        train_dataset=packed_dataset,
        args=TrainingArguments(
            do_train=True,
            do_eval=False,
            do_predict=False,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            weight_decay=decay,
            adam_beta1=adamw_beta1,
            adam_beta2=adamw_beta2,
            max_grad_norm=1.0,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            logging_steps=20,
            logging_dir="./logs",
            dataloader_num_workers=dataloader_num_workers,
            bf16=True,
            gradient_checkpointing=True,
            lr_scheduler_type="cosine",
        ),
    )
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    trainer.train()
    model.save_pretrained("./output")


if __name__ == "__main__":
    main()
