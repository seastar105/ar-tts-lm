import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import logging
import os
import sys
import time
from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import io

import datasets
import jiwer
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import DatasetDict, load_dataset
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import HfApi, HfFileSystem, hf_hub_url, get_token
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
    FlaxAutoModelForSpeechSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    is_tensorboard_available,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import webdataset as wds
import librosa

logger = logging.getLogger(__name__)

# Training Arguments
def shift_tokens_right(label_ids: np.array, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift label ids one token to the right.
    """
    shifted_label_ids = np.zeros_like(label_ids)
    shifted_label_ids[:, 1:] = label_ids[:, :-1]
    shifted_label_ids[:, 0] = decoder_start_token_id

    return shifted_label_ids


@flax.struct.dataclass
class FlaxDataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`Wav2Vec2Processor`])
            The processor used for proccessing the data.
        decoder_start_token_id (:obj: `int`)
            The begin-of-sentence of the decoder.
        input_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned input sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        target_padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned target sequences (according to the model's padding side and padding index).
            See above for details.
        max_input_length (:obj:`float`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_target_length (:obj:`int`, `optional`):
            Maximum length of the ``labels`` of the returned list and optionally padding length (see above).
        pad_input_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the input sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        pad_target_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the target sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Any
    decoder_start_token_id: int
    input_padding: Union[bool, str] = "longest"
    target_padding: Union[bool, str] = "max_length"
    max_input_length: Optional[float] = None
    max_target_length: Optional[int] = None
    pad_input_to_multiple_of: Optional[int] = None
    pad_target_to_multiple_of: Optional[int] = None
    max_lengths = (64, 128, 192, 256, 320, 384, 448)

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]

        # dataloader returns a list of features which we convert to a dict
        input_features = {model_input_name: [feature[model_input_name] for feature in features]}
        label_features = {"input_ids": [feature["labels"] for feature in features]}

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            max_length=self.max_input_length,
            padding=self.input_padding,
            pad_to_multiple_of=self.pad_input_to_multiple_of,
            return_tensors="np",
        )

        max_len = max([len(input_ids) for input_ids in label_features["input_ids"]])
        for length in self.max_lengths:
            if length >= max_len:
                max_len = length
                break

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            max_length=max_len,
            padding=self.target_padding,
            pad_to_multiple_of=self.pad_target_to_multiple_of,
            return_tensors="np",
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        labels = labels_batch["input_ids"]
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
            labels_batch.attention_mask = labels_batch.attention_mask[:, 1:]

        decoder_input_ids = shift_tokens_right(labels, self.decoder_start_token_id)

        # replace padding with -100 to ignore correctly when computing the loss
        labels = np.ma.array(labels, mask=np.not_equal(labels_batch.attention_mask, 1))
        labels = labels.filled(fill_value=-100)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    num_train_steps: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.ndarray]:
    """Returns a linear warmup, linear_decay learning rate function."""
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

AUDIO_EXTS = ["wav", "mp3", "flac", "m4a", "ogg", "opus"]

def decode_audio(key, data):
    key = key.split(".")[-1]
    if key in AUDIO_EXTS:
        return librosa.load(io.BytesIO(data), sr=None)
    return None

# 2. Setup logging
# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Set the verbosity to info of the Transformers logger.
# We only want one process per machine to log things on the screen.
logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)

# Create repo and retrieve repo_id
repo_name = "seastar105/whisper-tpu-jax-test"
hf_token = get_token()
api = HfApi(token=hf_token)
repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

# 3. Load dataset
fs = HfFileSystem(token=hf_token)
pattern = "hf://datasets/seastar105/ksponspeech-webdataset/train/*.tar"
files = [fs.resolve_path(path) for path in fs.glob(pattern)]
urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
urls = [
    f"pipe: curl --connect-timeout 30 --retry 30 --retry-delay 2 -f -s -L -H 'Authorization:Bearer {hf_token}' {url}"
    for url in urls
]

# 5. Load pretrained model, tokenizer, and feature extractor
config = AutoConfig.from_pretrained(
    "openai/whisper-large-v3-turbo"
)
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "openai/whisper-large-v3-turbo"
)
tokenizer = AutoTokenizer.from_pretrained(
    "openai/whisper-large-v3-turbo"
)
tokenizer.set_prefix_tokens(language="ko", task="transcribe")
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo", language="ko", task="transcribe")

def input_transform(item):
    audio, sr = item["flac"]
    if sr != 16000:
        audio = librosa.resample(audio)

    input_features = feature_extractor(audio, sampling_rate=sr).input_features[0]

    if isinstance(item["json"], str):
        text = json.loads(item["json"])["text"]
    else:
        text = item["json"]["text"]
    labels = tokenizer(text).input_ids

    return {
        "input_features": input_features,
        "labels": labels
    }

pipeline = [
    wds.ResampledShards(urls),
    wds.tarfile_to_samples(handler=wds.ignore_and_continue),
    wds.shuffle(),
    wds.decode(decode_audio, handler=wds.ignore_and_continue),
    wds.map(input_transform, handler=wds.ignore_and_continue),
]
train_dataset = wds.DataPipeline(*pipeline)
print("Dataset Done")

model = FlaxAutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo",
    config=config,
    dtype=getattr(jnp, "bfloat16"),
    gradient_checkpointing=True,
)
print(model.dtype)

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
print("Model Load Done")

# 7. Preprocessing the datasets.
# We need to read the audio files as arrays and tokenize the targets.
max_label_length = model.config.max_target_positions
pad_input_to_multiple_of = None
pad_target_to_multiple_of = None
num_workers = 0
model_input_name = feature_extractor.model_input_names[0]

# 8. Load Metric
def compute_metrics(preds, labels):
    # replace padded labels by the padding token
    for idx in range(len(labels)):
        labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # we do not want to group tokens when computing the metrics
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    wer = round(jiwer.wer(label_str, pred_str) * 100, 2)
    return {"wer": wer}

# # 9. Save feature extractor, tokenizer and config
# feature_extractor.push_to_hub(repo_name, token=hf_token)
# tokenizer.push_to_hub(repo_name, token=hf_token)
# config.push_to_hub(repo_name, token=hf_token)
data_collator = FlaxDataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
    input_padding="longest",
    target_padding="max_length",
    max_target_length=max_label_length,
    pad_input_to_multiple_of=pad_input_to_multiple_of,
    pad_target_to_multiple_of=None,
)
print("Collator Done")

# Enable tensorboard only on the master node
has_tensorboard = is_tensorboard_available()
if has_tensorboard and jax.process_index() == 0:
    try:
        from flax.metrics.tensorboard import SummaryWriter

        summary_writer = SummaryWriter(log_dir=Path("./debug_run"))
    except ImportError as ie:
        has_tensorboard = False
        logger.warning(
            f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
        )
else:
    logger.warning(
        "Unable to display metrics through TensorBoard because the package is not installed: "
        "Please run pip install tensorboard to enable."
    )

print("Training prep")

# Initialize our training
rng = jax.random.PRNGKey(998244353)
rng, dropout_rng = jax.random.split(rng)

# Store some constant
total_train_steps = 5000

# Create learning rate schedule
linear_decay_lr_schedule_fn = create_learning_rate_fn(
    total_train_steps,
    500,
    1e-5,
)

# We use Optax's "masking" functionality to not apply weight decay
# to bias and LayerNorm scale parameters. decay_mask_fn returns a
# mask boolean with the same structure as the parameters.
# The mask is True for parameters that should be decayed.
def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["layer_norm", "self_attn_layer_norm", "final_layer_norm", "encoder_attn_layer_norm"]
    layer_norm_named_params = {
        layer[-2:]
        for layer_norm_name in layer_norm_candidates
        for layer in flat_params.keys()
        if layer_norm_name in "".join(layer).lower()
    }
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)

# create adam optimizer
# adamw = optax.adamw(
#     learning_rate=linear_decay_lr_schedule_fn,
#     b1=0.9,
#     b2=0.98,
#     eps=1e-6,
#     weight_decay=0.1,
#     mask=decay_mask_fn,
# )
max_grad_norm = 1.0
gradient_accumulation = 8
adamw = optax.chain(
    optax.apply_every(gradient_accumulation),
    optax.clip_by_global_norm(max_grad_norm),  # Apply gradient clipping
    optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=0.9,
        b2=0.98,
        eps=1e-6,
        weight_decay=0.1,
        mask=decay_mask_fn,
    ),
)

# Setup train state
state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw, dropout_rng=dropout_rng)

# label smoothed cross entropy
def loss_fn(logits, labels, label_smoothing_factor=0.0):
    """
    The label smoothing implementation is adapted from Flax's official example:
    https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
    """
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing_factor
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

    loss = optax.softmax_cross_entropy(logits, soft_labels)
    loss = loss - normalizing_constant

    # ignore padded tokens from loss, i.e. where labels are not set to -100
    padding_mask = labels >= 0
    loss = loss * padding_mask
    loss = loss.sum()
    num_labels = padding_mask.sum()
    return loss, num_labels

# Define gradient update step fn
def train_step(state, batch, label_smoothing_factor=0.0):
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def compute_loss(params):
        labels = batch.pop("labels")
        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        loss, num_labels = loss_fn(logits, labels, label_smoothing_factor)
        return loss, num_labels

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, num_labels), grad = grad_fn(state.params)
    num_labels = jax.lax.psum(num_labels, "batch")

    # true loss = total loss / total samples
    loss = jax.lax.psum(loss, "batch")
    loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

    # true grad = total grad / total samples
    grad = jax.lax.psum(grad, "batch")
    grad = jax.tree_util.tree_map(lambda x: x / num_labels, grad)
    new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

    metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
    return new_state, metrics

# Define eval fn
def eval_step(params, batch, label_smoothing_factor=0.0):
    labels = batch.pop("labels")
    logits = model(**batch, params=params, train=False)[0]

    loss, num_labels = loss_fn(logits, labels, label_smoothing_factor)
    num_labels = jax.lax.psum(num_labels, "batch")

    # true loss = total loss / total samples
    loss = jax.lax.psum(loss, "batch")
    loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

    metrics = {"loss": loss}
    return metrics

# Define generation function
num_beams = model.config.num_beams
gen_kwargs = {"max_length": max_label_length, "num_beams": num_beams}

def generate_step(params, batch):
    model.params = params
    output_ids = model.generate(batch[model_input_name], attention_mask=batch.get("attention_mask"), **gen_kwargs)
    return output_ids.sequences

# Create parallel version of the train and eval step
p_train_step = jax.pmap(
    partial(train_step, label_smoothing_factor=0.0), "batch", donate_argnums=(0,)
)
p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=0.0), "batch")
p_generate_step = jax.pmap(generate_step, "batch")

# Replicate the train state on each device
state = state.replicate()

num_batches = total_train_steps * gradient_accumulation
train_batch_size = 16
log_interval = 50
log_interval *= gradient_accumulation
    
train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    drop_last=True,
    collate_fn=data_collator,
    num_workers=8,
    prefetch_factor=8
)

# train
for num_step, batch in tqdm(enumerate(train_loader), desc="Training...", position=1, leave=True):
    if num_step == num_batches:
        break
    batch = shard(batch.data)
    state, train_metric = p_train_step(state, batch)
    if num_step % log_interval == 0 and jax.process_index() == 0:
        print("Step at:", num_step, unreplicate(train_metric))


# save checkpoint after each epoch and push checkpoint to the hub
if jax.process_index() == 0:
    params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
    model.save_pretrained("./debug_run", params=params)
    tokenizer.save_pretrained("./debug_run")
    api.upload_folder(
        commit_message=f"Saving weights",
        folder_path="./debug_run",
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
    )
