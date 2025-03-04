import argparse
import io
import re
import shutil
from pathlib import Path

import librosa
import ray
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, HfFileSystem, get_token
from jiwer import cer
from transformers import AutoProcessor, WhisperForConditionalGeneration
from whisper_normalizer.basic import BasicTextNormalizer


class AudioReadActor:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("seastar105/whisper-base-komixv2")

    def __call__(self, item):
        audio, sr = librosa.load(io.BytesIO(item["mp3"]), sr=None)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        input_features = self.processor.feature_extractor(audio, sampling_rate=sr, return_tensors="np").input_features[
            0
        ]
        return {
            "mp3": item["mp3"],
            "json": item["json"],
            "input_features": input_features,
        }


class STTActor:
    def __init__(self):
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "seastar105/whisper-base-komixv2", attn_implementation="flash_attention_2"
        ).to(device=self.device, dtype=self.dtype)
        self.processor = AutoProcessor.from_pretrained("seastar105/whisper-base-komixv2")

    def __call__(self, batch):
        input_features = torch.from_numpy(batch.pop("input_features")).to(device=self.device, dtype=self.dtype)
        outputs = self.model.generate(input_features)
        preds = self.processor.batch_decode(outputs, skip_special_tokens=True)
        for idx, meta in enumerate(batch["json"]):
            meta["pred"] = preds[idx]
        return batch


class Metric:
    def __init__(self):
        self.normalizer = BasicTextNormalizer()

    def __call__(self, item):
        meta = item["json"]
        ref = meta["text"]
        hyp = meta["pred"]
        score = round(cer(re.sub(r"\s+", "", ref), re.sub(r"\s+", "", hyp)) * 100, 2)
        item["json"]["cer"] = score
        return {
            "mp3": item["mp3"],
            "json": item["json"],
        }


def run(files, output_dir, threshold):
    ds = ray.data.read_webdataset(files)
    ds = (
        ds.map(AudioReadActor, concurrency=(16, 48))
        .map_batches(STTActor, concurrency=8, num_gpus=0.5, batch_size=128)
        .map(Metric, concurrency=1)
        .filter(lambda x: x["json"]["cer"] < threshold)
        .write_webdataset(output_dir, min_rows_per_file=15000)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--cache_dir", type=str, default="./cache")

    args = parser.parse_args()

    fs = HfFileSystem(token=get_token())
    tar_files = fs.glob("datasets/amphion/Emilia-Dataset/Emilia-YODAS/KO/*.tar")
    api = HfApi(token=get_token())
    num_shards = 16

    for i in range(0, len(tar_files), num_shards):
        files = tar_files[i : i + num_shards]
        local_paths = []
        for file in files:
            filename = str(Path(file).relative_to("datasets/amphion/Emilia-Dataset"))
            local_paths.append(
                api.hf_hub_download(
                    repo_id="amphion/Emilia-Dataset", filename=filename, repo_type="dataset", local_dir=args.cache_dir
                )
            )

        run(local_paths, f"Emilia-YODAS-KO-Filtered-{i}", args.threshold)
        api.upload_folder(
            repo_id="seastar105/Emilia-YODAS-KO-filtered",
            folder_path=f"Emilia-YODAS-KO-Filtered-{i}",
            path_in_repo=f"data/{i:02d}",
            repo_type="dataset",
        )

        # delete local dir
        shutil.rmtree(args.cache_dir)
        shutil.rmtree(f"Emilia-YODAS-KO-Filtered-{i}")
