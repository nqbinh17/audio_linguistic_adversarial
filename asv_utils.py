import os
import gc
import math
import json
import yaml
import torch
import random
import librosa
import requests
import mimetypes
import torchaudio
import numpy as np
import unicodedata
from typing import Type, Dict
from pydub import AudioSegment
### Detector Models
from asv_models.aasist2 import Model
from asv_models.rawnet2 import RawNet
from asv_models import AASIST
from asv_models.CLADModel import  DownStreamLinearClassifier
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from utils import hash_to_int32
from tts_utils import TTSGenerator, KokoroTTS, CoquiTTS, F5TTSGenerator, OpenAITTS
from vars import aasist2_path, rawnet2_path, clad_path, detector_cache_dir, wav2net2_path

from dotenv import load_dotenv
load_dotenv()

import nltk
nltk.download('averaged_perceptron_tagger_eng')

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    cudnn_deterministic = True
    cudnn_benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return

class DeepfakeDetector:
    def __init__(self, tts: TTSGenerator, batch_size = 4):
        # Prepare detector
        self.rest_cnt = 0
        self.is_commercial = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
        set_random_seed(1234)
        self.tts = tts
        self.batch_size = batch_size
        self.eer_threshold = 0.5

    def load(self):
        raise NotImplementedError("Subclasses must override load()")
    
    def reload(self):
        del self.model
        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()

        self.load()


    def calibrate_bn_stats(self):
        # Stablize batch norm before eval
        if self.calibration_data is None: return
        print("Start Calibrate Batch Norm")

        self.model.train()
        self.__call__(self.calibration_data)
        self.model.eval()

    def update_voice(self, voice):
        self.tts.update_voice(voice)
    
    def prepare_detector_data(self, audio_files):
        cut = 64600  # take ~4 sec audio (64600 samples)
        x_inp = []
        _x_inp = []
        for file in audio_files:
            try:
                X, fs = librosa.load(file, sr=None)
            except:
                raise ValueError(file)
            cut = max(cut, X.size)
            _x_inp.append(X)
        for X in _x_inp:
            X_pad = pad(X, 64600)
            x_inp.append(torch.Tensor(X_pad))
        
        x_inp = torch.stack(x_inp, dim = 0).to(self.device)
        return x_inp
    
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        batch_size = self.batch_size
        probs = []
        step = math.ceil(len(texts)/ batch_size)
        for idx in range(step):
            _texts = texts[idx * batch_size:(idx+1) * batch_size]
            paths = self.tts(_texts)

            if len(paths) == 0:
                probs.append(np.array([-100]))
                continue

            detector_inputs = self.prepare_detector_data(paths)
            _probs = self.model(detector_inputs)
            test_probs = torch.softmax(_probs, dim = 1)[:, 1].data.cpu().numpy().ravel()
            probs.append(test_probs)
            self.rest_cnt += len(_texts)
            if self.rest_cnt > 50:
                self.rest_cnt = 0
                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()
            
        probs = np.concatenate(probs, 0)
        labels = probs > 0.5
        return probs, labels

class AASIST2Detector(DeepfakeDetector):
    def __init__(self, 
                 tts: TTSGenerator, 
                 calibration_data = None, 
                 batch_size = 4):
        
        super().__init__(tts=tts, batch_size = batch_size)
        self.detector_name = "aasist2"
        if isinstance(tts, F5TTSGenerator):
            calibration_data = calibration_data[:300] if calibration_data else None
        else:
            calibration_data = calibration_data[:100] if calibration_data else None
        self.calibration_data = calibration_data
        self.load()

    def load(self):
        self.model = Model(self.device, wav2net2_path)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(aasist2_path, map_location=self.device))
        self.model.eval()
        self.calibrate_bn_stats()


class RawNet2Detector(DeepfakeDetector):
    def __init__(self, 
                 tts: TTSGenerator, 
                 calibration_data = None, 
                 batch_size = 4):
        
        super().__init__(tts=tts, batch_size = batch_size)
        self.detector_name = "rawnet2"
        self.calibration_data = calibration_data
        if self.calibration_data:
            if isinstance(tts, CoquiTTS):
                self.calibration_data = None
            elif isinstance(tts, OpenAITTS):
                self.calibration_data = None
            else:
                self.calibration_data = self.calibration_data[:40]
        self.load()
        

    def load(self):
        dir_yaml = os.path.splitext('data/model_config_RawNet2')[0] + '.yaml'

        with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.load(f_yaml,Loader=yaml.FullLoader)

        self.model = RawNet(parser1['model'],  self.device).to( self.device)
        self.model.load_state_dict(torch.load(rawnet2_path, map_location= self.device))
        self.model.eval()
        self.calibrate_bn_stats()


class CLADDetector(DeepfakeDetector):
    def __init__(self, 
                 tts: TTSGenerator, 
                 calibration_data = None, 
                 batch_size = 4):
        super().__init__(tts=tts, batch_size = batch_size)
        self.detector_name = "clad"
        self.calibration_data = calibration_data
        if self.calibration_data:
            if isinstance(tts, CoquiTTS):
                self.calibration_data = self.calibration_data[:80]
            elif isinstance(tts, F5TTSGenerator):
                self.calibration_data = self.calibration_data[:100]
            else:
                self.calibration_data = self.calibration_data[:40]

        self.load()
        
    def load_model(self):
        
        with open("data/AASIST.conf", "r") as f_json:        
            aasist_config = json.loads(f_json.read())

        aasist_model_config = aasist_config["model_config"]
        aasist_encoder = AASIST.AasistEncoder(aasist_model_config).to(self.device)
        downstream_model = DownStreamLinearClassifier(aasist_encoder, input_depth=160)
        checkpoint = torch.load(clad_path, map_location=self.device)
        downstream_model.load_state_dict(checkpoint["state_dict"])
        downstream_model = downstream_model.to(self.device)
        return downstream_model

    def load(self):
        self.model = self.load_model()
        self.model.eval()
        self.calibrate_bn_stats()


### Commercial Detector

def get_audio_length(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # Duration in seconds
    return duration_seconds

class CommercialDetector:
    def __init__(self, tts: TTSGenerator = None, voice = None):
        self.cache_path = detector_cache_dir
        self.is_commercial = True
        if not hasattr(self, "detector_name"):
            self.detector_name = 'default'
        self.cache_path = os.path.join(self.cache_path, self.detector_name)
        os.makedirs(self.cache_path, exist_ok=True)
        if tts is None:
            self.tts = KokoroTTS(voice = voice)
        else:
            self.tts = tts

        self.total_processed_length = 0
        self.eer_threshold = 0.5

    def write(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f)
        return
    
    def read(self, path):
        with open(path) as f:
            data = json.load(f)

        return data

    def update_voice(self, voice):
        self.tts.update_voice(voice)

    def reload(self):
        return

    def __call__(self, texts, audio_paths = None):
        if isinstance(texts, str):
            texts = [texts]

        results = []
        if audio_paths is None:
            audio_paths = self.tts(texts)
        elif isinstance(audio_paths, str):
            audio_paths = [audio_paths]

        for audio_path in audio_paths:
            hash_value = hash_to_int32(audio_path)
            cached_path = os.path.join(self.cache_path, f"{hash_value}.json")
            if os.path.exists(cached_path):
                cached_data = self.read(cached_path)
                audio_length = cached_data['audio_length']
                results.append(cached_data['result'])
            else:
                data = self.detector(audio_path)
                audio_length = get_audio_length(audio_path)
                cached_data = {
                    "file": audio_path,
                    "audio_length": audio_length,
                    "result": data
                }
                self.write(cached_path, cached_data)
                results.append(data)
            
            self.total_processed_length += audio_length

        probs = np.array(results)[:, 1]
        labels = probs > 0.5
        return probs, labels

    def detector(self, path):
        raise NotImplementedError("Subclasses must override detector()")

    def get_total_audio_length(self):
        return f"{self.total_processed_length/3600:.2f} hours"
    
    def get_total_cost(self):
        cost = 10 * self.total_processed_length / 3600
        cost = f"${cost:.2}"
        return cost


class HiveAIDetector(CommercialDetector):
    def __init__(self, tts = None, voice = None, **kwargs):
        self.detector_name = 'the_hive_ai'
        super().__init__(tts = tts, voice = voice)
        self.token = os.getenv('HIVEAI_API')

    def detector(self, path, num_try = 0):
        url = "https://api.thehive.ai/api/v2/task/sync"
        
        headers = {
            "accept": "application/json",
            "authorization": f"token {self.token}"
        }

        files = {'media': open(path, 'rb')}
        try:
            response = requests.request("POST", url, headers=headers, files=files)
            rs = response.json()
            outputs = rs['status'][0]['response']['output'][0]['classes']
        except:
            if num_try < 3:
                return self.detector(path, num_try=num_try+1)
            else:
                raise ValueError
            
        data = [0, 0]
        for output in outputs:
            if output['class'] == 'ai_generated':
                data[0] = output['score']
            else:
                data[1] = output['score']
        
        return data


class DeepMediaAIDetector(CommercialDetector):
    def __init__(self, tts = None, voice=None, **kwargs):
        self.detector_name = 'deep_media_ai'
        super().__init__(tts = tts, voice = voice)
        self.token = os.getenv('DEEPID_API_KEY')
        self.session_token = self.request_session_token()

    def request_session_token(self):
        r = requests.get(
            "https://api.deepidentify.ai/user/token/session",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        return r.json()

    def upload_file(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, mimetypes.guess_type(file_path)[0])}
            r = requests.post(
                "https://api.deepidentify.ai/file/uploadS3",
                headers={"Authorization": f"Bearer {self.token}"},
                files=files
            )
        data = r.json()
        uploaded_filename = data.get('filename', None)
        return uploaded_filename
    
    def process_turbo_mode(self, filename, num_try = 0, local_path = None):
        """https://deepmedia.ai/documentation/deepid-api-quick-start-guide
            Interpret the probability
            Low probability (e.g., < 0.3): The media is likely authentic.

            Medium probability (e.g., 0.3 - 0.7): Exercise caution; the media might have been manipulated.

            High probability (e.g., > 0.7): Consider the media likely to be a deepfake."""
        
        r = requests.post(
            "https://api.deepidentify.ai/file/process/rt",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}"
            },
            json={
                "s3Location": filename,
                "modality": "audio"
            }
        )
        try:
            rs = r.json()['results']['audio']
        except Exception as e:
            if num_try < 3:
                return self.process_turbo_mode(
                    filename=filename, num_try=num_try+1, local_path=local_path
                )
            else:
                os.remove(local_path)
                raise ValueError
        return r.json()['results']['audio']

    def detector(self, path):
        uploaded_filename = self.upload_file(path)

        fake_prob = self.process_turbo_mode(uploaded_filename, local_path = path)
        data = [fake_prob, 1 - fake_prob]
        return data


class STTPipeline:
    def __init__(self, device = None, batch_size = 4, auto_cache = True):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3-turbo"

        stt = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        stt.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model=stt,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device)
        
        self.batch_size = batch_size
        self.auto_cache = auto_cache

    def read_text(self, path):
        with open(path) as f:
            data = f.readlines()
        
        data = "\n".join(data).strip()
        return data

    def write_text(self, path, text):
        with open(path, 'w') as f:
            f.writelines(text)

    def __call__(self, audio_paths):
        if self.auto_cache:
            ### Replace extension .wav to .txt, then save as the same place
            ext = audio_paths[0].split(".")[-1]
            cached_paths = [path.replace(ext, ".txt") for path in audio_paths]
            results = []
            pending = []
            for idx, (cached_path, audio_path) in enumerate(zip(cached_paths, audio_paths)):
                if os.path.exists(cached_path):
                    results.append((idx, self.read_text(cached_path)))
                else:
                    pending.append((idx, audio_path, cached_path))
            
            if len(pending) > 0:
                audio_files = [d[1] for d in pending]
                idx_pending = [d[0] for d in pending]
                transcripts = self.stt_pipeline(audio_files, batch_size=self.batch_size)
                transcripts = [r['text'].strip() for r in transcripts]
                results += [(idx, text) for idx, text in zip(idx_pending, transcripts)]

                for d, transcript in zip(pending, transcripts):
                    cached_file = d[2]
                    self.write_text(cached_file, transcript)

            results = sorted(results, key=lambda x: x[0])
            results = [d[1] for d in results]

        else:
            results = self.stt_pipeline(audio_paths, batch_size=self.batch_size)
            results = [r['text'].strip() for r in results]

        return results



ASV_MODELS_REGISTER: Dict[str, Type[DeepfakeDetector] | Type[CommercialDetector]] = {
    "AASIST2": AASIST2Detector,
    "CLAD": CLADDetector,
    "RAWNET2": RawNet2Detector,
    "HIVEAI": HiveAIDetector,
    "DEEPID": DeepMediaAIDetector
}