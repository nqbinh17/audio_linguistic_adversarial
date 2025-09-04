import os
import json
import time
import torch
import tomli
import struct
import hashlib
import librosa
import requests
import warnings
import torchaudio
import soundfile as sf
from TTS.api import TTS
from kokoro import KPipeline
from typing import Type, Dict
from dotenv import load_dotenv
from omegaconf import OmegaConf
from cached_path import cached_path
from importlib.resources import files
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder
)
from vars import tts_cache_dir, f5tts_path
from utils import HiddenPrints

load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')

warnings.simplefilter(action='ignore', category=FutureWarning)
# torch.serialization.add_safe_globals()

def hash_to_int32(text):
    # Create a SHA-256 hash of the text
    hash_object = hashlib.sha256(text.encode())
    
    # Get the hash digest as bytes
    hash_bytes = hash_object.digest()
    
    # Take the first 4 bytes (32 bits) of the hash and convert them to an integer
    int32_value = struct.unpack('I', hash_bytes[:4])[0]
    
    return str(int32_value)

class TTSGenerator:

    def __init__(self, voice = None):
        self.is_commercial = False
        self.output_dir = tts_cache_dir
        self.voice = voice
        if voice:
            self.output_dir = os.path.join(self.output_dir, voice)

        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        audio_files = []

        cached_files = []
        pending_files = []
        for idx, text in enumerate(texts):
            hash_value = hash_to_int32(text)
            cached_file = os.path.join(self.output_dir, f'{hash_value}.wav')

            if os.path.exists(cached_file):
                cached_files.append((idx, cached_file))
            else:
                pending_files.append((idx, cached_file, text))

        if len(pending_files) == 0:
            audio_files = [d[1] for d in cached_files]
        else:
            gen_files = self.generate_audio(texts = [d[2] for d in pending_files], 
                                            output_paths = [d[1] for d in pending_files])
            
            audio_files += cached_files
            audio_files += [(u[0], v) for u, v in zip(pending_files, gen_files)]

            audio_files = sorted(audio_files, key=lambda x: x[0])
            # Extract the second value from each tuple
            audio_files = [x[1] for x in audio_files]

        return audio_files
    
    def generate_audio(self, texts, output_paths):
        raise NotImplementedError("Subclasses must override generate_audio()")

    def update_voice(self, new_voice):
        if self.voice is None:
            self.output_dir = os.path.join(self.output_dir, new_voice)
        else:
            self.output_dir = self.output_dir.replace(self.voice, new_voice)
        os.makedirs(self.output_dir, exist_ok=True)
        self.voice = new_voice

    def get_voice_profiles(self):
        raise NotImplementedError("Subclasses must override get_voice_profiles()")

class KokoroTTS(TTSGenerator):
    def __init__(self, voice = 'af_heart', sampling_rate = 22050):
        super().__init__(voice)
        self.tts_pipeline = KPipeline(lang_code='a')
        self.voices = ['af_heart', 'am_adam', 'bf_lily', 'bm_george']
        self.voice = voice
        self.sampling_rate = sampling_rate

    def get_voice_profiles(self):
        return self.voices

    def generate_audio(self, texts, output_paths):
        if isinstance(texts, list):
            texts = "\n\n".join(texts)
        if isinstance(output_paths, str):
            output_paths = [output_paths]

        generator = self.tts_pipeline(
            texts, voice=self.voice, speed=1, split_pattern=r'\n+'
        )

        for i, (gs, ps, audio) in enumerate(generator):
            sf.write(output_paths[i], audio, self.sampling_rate)

        return output_paths


class CoquiTTS(TTSGenerator):

    def __init__(self, voice = 'DonaldTrump', base_voice = 'af_heart'):
        super().__init__(voice)
        self.voice_profiles = {
            "DonaldTrump": 'voice_profiles/DonaldTrump.mp3',
            "ElonMusk": 'voice_profiles/ElonMusk.wav',
            "TaylorSwift": 'voice_profiles/TaylorSwift.wav',
            "OprahWinfrey": 'voice_profiles/OprahWinfrey.wav',
        }
        self.update_voice(voice)
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar = False).to('cuda')

    def get_voice_profiles(self):
        return list(self.voice_profiles.keys())

    def generate_audio(self, texts, output_paths):
        with HiddenPrints():
            for text, output_path in zip(texts, output_paths):
                self.tts.tts_to_file(text=text, speaker_wav=self.speaker_wav, language="en", file_path=output_path, split_sentences = False)

        return output_paths

    def update_voice(self, voice):
        self.speaker_wav = self.voice_profiles[voice]
        super().update_voice(voice)
        self.voice = voice


class F5TTSGenerator(TTSGenerator):
    def __init__(self, voice = 'f5tts_male'):
        super().__init__(voice)
        config_path = os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml")

        config = tomli.load(open(config_path, "rb"))

        ref_audio = config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
        ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
        ref_text = config['ref_text']

        self.voice_map = {
            "f5tts_male": (ref_audio, ref_text),
            "BradPitt": ("voice_profiles/BradPitt.wav", "I had a few. I I I met some stranger who who who talked about an acting class and I went to this class and turned things, you know, pointed me in a great")
        }

        self.update_voice(voice)

        model_cls = DiT
        model_cfg = str(files("f5_tts").joinpath("configs/F5TTS_Base_train.yaml"))
        model_cfg = OmegaConf.load(model_cfg).model.arch
        ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors", cache_dir = f5tts_path))
        self.vocoder_name = "vocos"
        self.ema_model = load_model(model_cls, 
                                    model_cfg, 
                                    ckpt_file, 
                                    mel_spec_type=self.vocoder_name, 
                                    vocab_file="")
        self.ema_model.eval()

        self.vocoder = load_vocoder(vocoder_name=self.vocoder_name, 
                                    is_local=False, 
                                    local_path="../checkpoints/vocos-mel-24khz")
        self.vocoder.eval()

        self.sampling_rate = 22050

    def get_voice_profiles(self):
        return list(self.voice_map.keys())

    def update_voice(self, new_voice):
        self.ref_audio, self.ref_text = self.voice_map[new_voice]
        super().update_voice(new_voice)
        self.voice = new_voice
        return

    def single_infer(self, gen_text, wave_path):
        audio_wave, final_sample_rate, _ = infer_process(
            self.ref_audio,
            self.ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            mel_spec_type=self.vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            progress = None
        )
        

        if final_sample_rate != self.sampling_rate:
            audio_wave = librosa.resample(audio_wave, orig_sr=final_sample_rate, target_sr=self.sampling_rate)
            
        with open(wave_path, "wb") as f:
            sf.write(f.name, audio_wave, self.sampling_rate)

        return

    def generate_audio(self, texts, output_paths):
        with HiddenPrints():
            for text, output_path in zip(texts, output_paths):
                
                self.single_infer(text, output_path)

        return output_paths

def openai_tts(transcript, out_path, voice = "shimmer", model = "tts-1"):
    url = "https://api.openai.com/v1/audio/speech"

    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "input": transcript,
        "voice": voice,
        'speed': 1
    }

    r = requests.post(url, headers=headers, data=json.dumps(data))
    
    audio_content = r.content
        
    with open(out_path, 'wb') as f:
        f.write(audio_content)

    return out_path

class OpenAITTS(TTSGenerator):
    def __init__(self, voice = 'coral'):
        super().__init__(voice)
        self.voices = ['coral']
        self.voice = voice
        self.sampling_rate = 22050
        self.is_commercial = True

    def get_voice_profiles(self):
        return self.voices
    
    def generate_audio(self, texts, output_paths):
        
        for text, output_path in zip(texts, output_paths):
            openai_tts(transcript = text, out_path=output_path, voice = self.voice)
            audio, orig_sr = torchaudio.load(output_path)
            if orig_sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_sr, self.sampling_rate)
                audio = resampler(audio)

            torchaudio.save(output_path, audio, self.sampling_rate)
    
        return output_paths


TTS_MODELS_REGISTER: Dict[str, Type[TTSGenerator]] = {
    "KokoroTTS": KokoroTTS,
    "F5TTS": F5TTSGenerator,
    "CoquiTTS": CoquiTTS,
    "OpenAITTS": OpenAITTS
}