import os
import requests
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CACHED_DIR = Path.home() / ".cache" / "audio_linguistic_adversarial"
CACHED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = CACHED_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

tts_cache_dir =CACHED_DIR / "tts_outputs"
detector_cache_dir = CACHED_DIR / "detector_outputs"

detector_cache_dir.mkdir(parents=True, exist_ok=True)
tts_cache_dir.mkdir(parents=True, exist_ok=True)


aasist2_path = MODEL_DIR / "AASIST2.pth"
rawnet2_path = MODEL_DIR / "rawnet2.pth"
clad_path = MODEL_DIR / "CLAD_150_10_2310.pth.tar"
wav2net2_path = MODEL_DIR / "xlsr2_300m.pt"
f5tts_path = MODEL_DIR

model_configs = [
    {
        "path": aasist2_path,
        "url": "https://huggingface.co/VoiceWukong/VoiceWukong/resolve/main/AASIST2.pth?download=true"
    },
    {
        "path": rawnet2_path,
        "url": "https://huggingface.co/VoiceWukong/VoiceWukong/resolve/main/rawnet2.pth?download=true"
    },
    {
        "path": clad_path,
        "url": "https://huggingface.co/VoiceWukong/VoiceWukong/resolve/main/CLAD_150_10_2310.pth.tar?download=true"
    },
    {
        "path": wav2net2_path,
        "url": "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"
    }
]

def download_file(url: str, destination_path: Path):
    """Downloads a file from a URL to a specified path, with a progress bar."""
    print(f"Downloading {destination_path.name} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(destination_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong during download")
    print(f"Downloaded {destination_path.name} to {destination_path}")


for model_config in model_configs:
    model_path = model_config["path"]
    model_url = model_config["url"]
    print(model_path)

    if not model_path.exists():
        try:
            print(f"Start downloading model checkpoint: {model_path.name}")
            download_file(model_url, model_path)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {model_path.name}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while downloading {model_path.name}: {e}")
    else:
        pass