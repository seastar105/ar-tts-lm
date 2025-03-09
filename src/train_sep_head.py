import sys

import rootutils
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, get_token
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Model,
    Trainer,
    TrainingArguments,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import SequencePackWrapper, TextCodeDataset


def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


class SeparateHeadModel(torch.nn.Module):
    def __init__(self, model, vocab_size, audio_code_offset):
        super().__init__()

        self.lm_model = model
        self.vocab_size = vocab_size
        self.audio_embeds = torch.nn.Embedding(vocab_size + 2, model.config.hidden_size)
        self.audio_head = torch.nn.Linear(model.config.hidden_size, vocab_size + 2)
        self.audio_code_offset = audio_code_offset

    def forward(self, input_ids, labels=None, **kwargs):
        audio_mask = input_ids >= self.audio_code_offset
        audio_ids = input_ids[audio_mask] - self.audio_code_offset

        input_ids[audio_mask] = 0
        input_embeds = self.lm_model.get_input_embeddings()(input_ids)

        audio_embeds = self.audio_embeds(audio_ids)

        input_embeds[audio_mask] = audio_embeds

        if not kwargs.get("return_dict", False):
            kwargs["return_dict"] = True
        hidden_states = self.lm_model.model(inputs_embeds=input_embeds, **kwargs).last_hidden_state
        audio_logits = self.audio_head(hidden_states)
        audio_logits = audio_logits[audio_mask]
        labels = labels[audio_mask] - self.audio_code_offset
        labels[labels < 0] = -100
        loss = ForCausalLMLoss(audio_logits, labels, self.vocab_size + 2, **kwargs)
        return (loss,)

    def gradient_checkpointing_enable(self, **kwargs):
        self.lm_model.gradient_checkpointing_enable(**kwargs)


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    dataset_name = "seastar105/speech-token-dataset"
    dataset_split = "emilia_yodas_ko_wavtok"

    # dataset = load_dataset(dataset_name, split=dataset_split)
    dataset = load_dataset("json", data_files="cache/kss_wavtok/*.json", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = TextCodeDataset(dataset, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    model = SeparateHeadModel(model, dataset.code_vocab_size, audio_code_offset=dataset.boa_token_id)
    model = model.to(dtype=torch.bfloat16)

    output_dir = "./output/pack_rand_sep_freeze"
    dataset.tokenizer.save_pretrained(output_dir)

    per_device_batch_size = 4
    per_device_micro_batch_size = 1
    gradient_accumulation = per_device_batch_size // per_device_micro_batch_size

    dataloader_num_workers = 4
    max_steps = 5000
    warmup_steps = 200
    learning_rate = 1e-4

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
            logging_dir="./logs/pack_rand_sep_freeze",
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
