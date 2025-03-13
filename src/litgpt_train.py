# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightning as L
import torch
import torch_xla.core.xla_model as xm
from datasets import load_dataset
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.strategies import XLAFSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor, measure_flops
from litgpt.model import GPT, Block, Config
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    estimate_flops,
    lazy_load,
    num_parameters,
)

from src.utils import rank_print, sequential_load_and_fsdp_wrap

eval_interval = 200
save_interval = 1000
eval_iters = 100
eval_max_new_tokens = 100
log_interval = 10
devices = XLAAccelerator.auto_device_count()
# the state of very large models will not fit on the system RAM, this flag can alleviate it by loading it on each rank
# sequentially
reduce_cpu_memory_usage_during_load = False

# Hyperparameters
learning_rate = 2e-4
global_batch_size = 64
num_nodes = 4
assert global_batch_size % (devices * num_nodes) == 0, "global batch size is not divisible by num_deivces"
batch_size = global_batch_size // devices // num_nodes
micro_batch_size = 4  # max micro batch size at single TPUv4(v4-8), with 4k seqlen,  0.5B
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
log_interval *= gradient_accumulation_iters
max_steps = 10000
warmup_steps = 200
max_iters = max_steps * gradient_accumulation_iters
weight_decay = 0.01

boa_token = "<|beginofaudio|>"
eoa_token = "<|endofaudio|>"
code_template = "<|AUDIO{idx}|>"
code_vocab_size = 4096
use_chat_template = True
tts_instruction = "Convert the following text to audible speech.\n{text}"

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(
    *,
    checkpoint_dir: Path = Path("checkpoints/tiiuae/falcon-7b"),
    out_dir: Path = Path("out/alpaca"),
    precision: str = "bf16-true",
) -> None:
    if devices > 1:
        strategy = XLAFSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="sharded",  # change to "sharded" in multi-host environments where the filesystem is not shared
            sequential_save=True,
        )
    else:
        strategy = "auto"
    logger = TensorBoardLogger(root_dir=out_dir, name="tensorboard")
    load_dataset("seastar105/aihub-542-wavtok-packed", split="train", num_proc=32)
    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger)
    rank_print(fabric, hparams)
    fabric.launch(main, checkpoint_dir, out_dir)


def main(fabric: L.Fabric, checkpoint_dir: Path, out_dir: Path) -> None:
    rank_print("Start main")
    check_valid_checkpoint_dir(checkpoint_dir)

    fabric.seed_everything(998244353)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    dataset = load_dataset("seastar105/aihub-542-wavtok-packed", split="train", num_proc=32)

    config_file = checkpoint_dir / "model_config.yaml"
    config = Config.from_file(config_file)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    rank_print(fabric, f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")

    if reduce_cpu_memory_usage_during_load:
        model = sequential_load_and_fsdp_wrap(fabric, lambda: GPT(config), checkpoint_path)
    else:
        with fabric.init_module(empty_init=False):
            model = GPT(config)
        checkpoint = lazy_load(checkpoint_path)
        # strict=False because missing keys due to adapter weights not contained in state dict
        model.load_state_dict(checkpoint, strict=False)

    model = fabric.setup_module(model)

    # these are not correct in the sharding case
    rank_print(fabric, f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    rank_print(fabric, f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)

    fabric.seed_everything(998244353 + fabric.global_rank)

    train_time = time.perf_counter()
    train(fabric, model, optimizer, dataset, checkpoint_dir, out_dir)
    rank_print(fabric, f"Training time: {(time.perf_counter()-train_time):.2f}s")

    # Save the final checkpoint at the end of training
    save_path = out_dir / "lit_model_adapter_finetuned.pth"
    save_checkpoint(fabric, model, save_path)


def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    dataset: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length = 4096
    model.max_seq_length = longest_seq_length
    # to avoid recompilation, this script is configured to pad batches to the `longest_seq_length`
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `flops_per_batch=estimated_flops` instead
        estimated_flops = estimate_flops(meta_model, training=True) * micro_batch_size
        rank_print(fabric, f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (micro_batch_size, longest_seq_length))
        forward_fn = lambda: meta_model(x)  # noqa: F821
        loss_fn = lambda y: chunked_cross_entropy(y, x, chunk_size=0)  # noqa: F821
        measured_flops = measure_flops(meta_model, forward_fn, loss_fn)
        rank_print(fabric, f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    throughput = ThroughputMonitor(fabric, window_size=50)
    step_count = 0
    total_t0 = time.perf_counter()

    xm.mark_step()
    for iter_num in range(1, max_iters + 1):
        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(fabric, dataset, longest_seq_length)

        is_accumulating = iter_num % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            xm.mark_step()
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)
        xm.mark_step()

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
        else:
            xm.mark_step()

        if iter_num % log_interval == 0:
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0,
                batches=iter_num,
                samples=iter_num * micro_batch_size,
                lengths=iter_num * micro_batch_size * longest_seq_length,
                flops=measured_flops * log_interval,
            )
            throughput.compute_and_log(step=iter_num)
            rank_print(
                fabric,
                f"iter {iter_num} step {step_count}:"
                # uncomment to print the loss. this will considerably slow down the iteration times
                + f" loss {loss.item():.6f},"
                + f" iter time: {(t1 - iter_t0) * 1000:.2f}ms"
                + (" (optimizer.step)" if not is_accumulating else ""),
            )

        #        if not is_accumulating and step_count % eval_interval == 0:
        #            t0 = time.perf_counter()
        #            val_loss = validate(fabric, model, val_data, tokenizer, longest_seq_length)
        #            t1 = time.perf_counter() - t0
        #            rank_print(fabric, f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
        #            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            save_checkpoint(fabric, model, checkpoint_path)


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, longest_seq_length: int
) -> torch.Tensor:
    rank_print(fabric, "Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    xm.mark_step()
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data, longest_seq_length)
        logits = model(input_ids)
        xm.mark_step()
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()
    model.train()
    return val_loss


def get_batch(fabric: L.Fabric, data: List[Dict], longest_seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,)).tolist()

    input_ids = [torch.LongTensor(data[i]["input_ids"]) for i in ix]
    labels = [torch.LongTensor(data[i]["labels"]) for i in ix]

    def pad_right(x, pad_id):
        # pad right using a fixed longest sequence length to avoid recompilation
        n = longest_seq_length - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-100) for x in labels])

    x, y = fabric.to_device((x, y))
    return x, y


def get_longest_seq_length(data: List[Dict]) -> int:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    return max(len(d["input_ids"]) for d in data)


def save_checkpoint(fabric: L.Fabric, model: torch.nn.Module, file_path: Path) -> None:
    rank_print(fabric, f"Saving weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model})


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(setup)
