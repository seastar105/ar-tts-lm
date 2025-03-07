import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import io
from concurrent.futures import ThreadPoolExecutor

import librosa
import ray
import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from tqdm.auto import tqdm

from src.wavtok.pretrained import load_large_v2

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class AudioDecoder:
    def __call__(self, item):
        audio_bytes = item["mp3"]
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
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    fs = HfFileSystem()
    tar_files = sorted(fs.glob("datasets/seastar105/Emilia-YODAS-KO-filtered/**/*.tar"))
    tar_files = [fs.resolve_path(f) for f in tar_files]
    local_paths = []

    def download_file(tar_file):
        return hf_hub_download(
            repo_id=tar_file.repo_id, filename=tar_file.path_in_repo, local_dir=args.cache_dir, repo_type="dataset"
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        for local_path in tqdm(
            executor.map(download_file, tar_files), desc=f"Downloading taf files to {args.cache_dir}"
        ):
            local_paths.append(local_path)

    # Download checkpoint before parallel execution
    load_large_v2()

    dataset = (
        ray.data.read_webdataset(local_paths)
        .map(AudioDecoder, concurrency=args.gpu_concurrency * 4)
        .map(WavTokActor, num_gpus=args.num_gpus, concurrency=args.gpu_concurrency)
    )
    dataset.write_json(args.output_dir, force_ascii=False, min_rows_per_file=10000)
