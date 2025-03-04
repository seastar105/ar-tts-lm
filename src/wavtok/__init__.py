from huggingface_hub import hf_hub_download

from src.wavtok.pretrained import WavTokenizer


def load_large_v2(config_path: str = "assets/wavtok_config.yaml", model_path=None):
    if model_path is None:
        print("Download model from Hugging Face model hub")
        hf_hub_download(
            repo_id="novateur/WavTokenizer-large-speech-75token",
            filename="wavtokenizer_large_speech_320_v2.ckpt",
            local_dir="assets/",
        )
        model_path = "assets/wavtokenizer_large_speech_320_v2.ckpt"
    return WavTokenizer.from_pretrained0802(config_path, model_path)
