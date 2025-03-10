import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import io
from concurrent.futures import ThreadPoolExecutor

import librosa
import ray
import torch
from huggingface_hub import HfApi, HfFileSystem, get_token, hf_hub_download
from tqdm.auto import tqdm

from src.wavtok.pretrained import load_large_v2

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class AudioDecoder:
    def __call__(self, item):
        audio_bytes = item["wav"]
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        audio = audio.squeeze()
        meta = item["json"]

        return {
            "audio": audio,
            **meta,
        }


class WavTokActor:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = load_large_v2().eval().to(self.device)
        self.bandwidth_id = torch.tensor([0])

    def __call__(self, item):
        audio = torch.from_numpy(item.pop("audio")).unsqueeze(0).to(self.device)
        _, codes = self.model.encode_infer(audio, bandwidth_id=self.bandwidth_id)
        item["codes"] = codes.cpu().numpy().squeeze()
        return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", type=float, default=1)
    parser.add_argument("--gpu_concurrency", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    token = get_token()
    fs = HfFileSystem(token=token)
    tar_files = sorted(fs.glob("datasets/seastar105/aihub-542-webdataset/**/*.tar"))

    # Download checkpoint before parallel execution
    load_large_v2()

    num_shards_per_job = 32
    api = HfApi(token=token)

    def download_file(tar_file):
        return hf_hub_download(
            repo_id=tar_file.repo_id, filename=tar_file.path_in_repo, local_dir="./cache", repo_type="dataset"
        )

    for i in range(0, len(tar_files), num_shards_per_job):
        shard_idx = i // num_shards_per_job
        output_dir = f"{args.output_dir}/shard_{shard_idx:03d}"
        # check if already output exists on remote repo
        if len(fs.glob(f"datasets/seastar105/aihub-542-tokenized/{output_dir}/*.json")) > 0:
            print(f"Skipping {output_dir} as it already exists")
            continue

        shards = tar_files[i : i + num_shards_per_job]
        shards = [fs.resolve_path(shard) for shard in shards]
        local_paths = []
        with ThreadPoolExecutor(max_workers=min(8, num_shards_per_job)) as executor:
            for local_path in tqdm(executor.map(download_file, shards), desc="Downloading tar files..."):
                local_paths.append(local_path)
        dataset = (
            ray.data.read_webdataset(shards, filesystem=fs)
            .map(AudioDecoder, concurrency=args.gpu_concurrency * 4)
            .map(WavTokActor, num_gpus=args.num_gpus, concurrency=args.gpu_concurrency)
        )

        dataset.write_json(output_dir, force_ascii=False, min_rows_per_file=50000)
        api.upload_folder(
            repo_id="seastar105/aihub-542-tokenized",
            path_in_repo=f"shard_{shard_idx:03d}",
            repo_type="dataset",
        )

        for local_path in local_paths:
            os.remove(local_path)
