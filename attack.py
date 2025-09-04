import os
import gc
import json
import math
import time
import torch
import argparse
import numpy as np
from evaluate import load
from utils import load_dataset, get_attack_log_summary
from asv_utils import ASV_MODELS_REGISTER, STTPipeline
from tts_utils import TTS_MODELS_REGISTER, TTSGenerator, CoquiTTS

from textattack import Attacker
from textattack.datasets import Dataset
from textattack.models.wrappers import ModelWrapper
from textattack.constraints.semantics.sentence_encoders.sentence_bert import SBERT
from textattack.attack_recipes import PWWSRen2019, TextFoolerJin2019, BERTAttackLi2020, BAEGarg2019
from textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder import UniversalSentenceEncoder



class DeepfakeDetectorPipelineWrapper(ModelWrapper):
    """Transformers sentiment analysis pipeline returns a list of responses,
    like
        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]
    We need to convert that to a format TextAttack understands, like
        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, detector, device,
                 batch_size = 4):
           
        self.model = None
        self.detector = detector
        self.device = device
        self.batch_size = batch_size
        self.tts_cnt = 0

        self.start_time  = time.time()
        self.WER = load("wer")
        self.rest_cnt = 0
        # self.stt_pipeline = STTPipeline()

    def compute_wer(self, texts):
        transcripts = []
        batch_size = 4
        step = math.ceil(len(texts)/ batch_size)
        try:
            for idx in range(step):
                _texts = texts[idx * batch_size:(idx+1) * batch_size]
                audio_files = self.detector.tts(_texts)
                results = self.stt_pipeline(audio_files)
                transcripts += results

            wer = self.WER.compute(predictions=transcripts, references=texts)
            wer = f"{100 * wer:.2f}%"
        except:
            wer = "N/A"
        return wer
        
    def reset_stats(self):
        self.start_time  = time.time()
        self.tts_cnt = 0

    def get_time_cost(self):
        end_time = time.time()  # Record the end time

        # Calculate the time difference
        elapsed_time = end_time - self.start_time

        # Convert the elapsed time to hours, minutes, and seconds
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60

        r = f"{hours}h:{minutes}m:{seconds:.2f}s"

        return r

    def __call__(self, text_inputs):
        self.tts_cnt += len(text_inputs)
        probs, labels = self.detector(text_inputs)
        
        outputs = []
        for prob, label in zip(probs, labels):
            if label == True: # Detected as real
                outputs.append([min(1-prob, prob), max(1-prob, prob)])
            else: # Detected as fake
                outputs.append([max(1-prob, prob), min(1-prob, prob)])
        
        return np.array(outputs)

def save_experiment_results(model_wrapper, attacker, attacker_name, detector, file_name):
    results = attacker.attack_log_manager.results
    log_summary = get_attack_log_summary(attacker)

    items = []
    original_texts = []
    perturbed_texts = []
    eer_threshold = detector.eer_threshold

    for result in results:
        original_score = result.original_result.score
        perturbed_score = result.perturbed_result.score

        if ((original_score > eer_threshold) is False) and ((perturbed_score > eer_threshold) is True):
            original_texts.append(result.original_text())
            perturbed_texts.append(result.perturbed_text())

        items.append({
            "original_text": result.original_text(),
            "original_score": result.original_result.score,
            "perturbed_text": result.perturbed_text(),
            "perturbed_score": result.perturbed_result.score,
        })

    time_cost = model_wrapper.get_time_cost()

    # perturbed_wer = model_wrapper.compute_wer(perturbed_texts)
    # original_wer = model_wrapper.compute_wer(original_texts)

    data = {
        "log_summary": log_summary,
        "attacker": attacker_name,
        "detector": detector.detector_name,
        "tts_voice": detector.tts.voice,
        "TTS_query": model_wrapper.tts_cnt,
        "time_cost": time_cost,
        "original_wer": None,
        "perturbed_wer": None,
        "results": items,
    }

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return


def format_data(data):
    _data = [(d['text'], 0) for d in data]
    return _data

attacker_map = {
    "pwws": PWWSRen2019,
    "text_fooler": TextFoolerJin2019,
    "bae": BAEGarg2019,
    "bert_attack": BERTAttackLi2020,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attack script example')
    parser.add_argument('-d', '--detector', type=str, default = "AASIST2", help='Choose Detector')
    parser.add_argument('-t', '--tts', type=str, default = "KokoroTTS", help='Choose TTS')
    parser.add_argument('-s', '--skip', type=str, default = "True", help='Skip exists attack results')
    args = parser.parse_args()

    do_skip = args.skip == "True"
    detector_name = args.detector
    detector_cls = ASV_MODELS_REGISTER[detector_name]
    tts_name = args.tts
    tts_cls = TTS_MODELS_REGISTER[tts_name]
    tts = tts_cls()
    
    voicewukong_data = load_dataset()
    eval_data = [d for d in voicewukong_data if len(d['text'].split()) >= 10]
    calibration_data = [d['text'] for d in voicewukong_data if len(d['text'].split()) < 10]
    eval_data = format_data(eval_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = detector_cls(tts = tts, calibration_data = calibration_data)

    is_commercial_result = False
    if tts.is_commercial or detector.is_commercial:
        is_commercial_result = True
        attacker_map = {
            "text_fooler": TextFoolerJin2019
        }
        with open("data/commercial_data.txt") as f:
            eval_data = f.readlines()
            eval_data = [(line.strip(), 0) for line in eval_data if line.strip() != ""]

    if is_commercial_result is False:
        output_dir = f'experiment_results/'
    else:
        output_dir = f'commercial_results/'
    os.makedirs(output_dir, exist_ok=True)
    
    model_wrapper = DeepfakeDetectorPipelineWrapper(detector = detector, 
                                                device = device)

    voice_profiles = tts.get_voice_profiles()

    if isinstance(tts, CoquiTTS):
        voice_profiles = ["ElonMusk"]
    if is_commercial_result: # one only
        voice_profiles = voice_profiles[:1]

    for voice_profile in voice_profiles:
        detector.update_voice(voice_profile)
        detector.reload()

        for attacker_name, AttackerMethod in attacker_map.items():
            file_name = os.path.join(output_dir, f"{attacker_name}_{voice_profile}_{detector.detector_name}.json")
            if os.path.exists(file_name) and do_skip: # Skip
                print("Skip:", file_name)
                continue

            model_wrapper.reset_stats()

            recipe = AttackerMethod.build(model_wrapper)

            dataset = Dataset(
                dataset=eval_data
            )

            attacker = Attacker(recipe, dataset)
            attacker.attack_args.num_examples = -1
            # attacker.attack.goal_function.query_budget = 80

            use_idx = -1
            for idx, constraint in enumerate(attacker.attack.constraints):
                if isinstance(constraint, UniversalSentenceEncoder):
                    use_idx = idx
                    break
            
            window_size = 15
            compare_against_original = True
            if use_idx != -1:
                use_constraint = attacker.attack.constraints.pop(use_idx)
                window_size = use_constraint.window_size
                compare_against_original= use_constraint.compare_against_original

            SBERT_constraint = SBERT(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                        threshold = 0.84,
                                        window_size = window_size,
                                        metric= 'angular',
                                        compare_against_original= compare_against_original,
                                        skip_text_shorter_than_window = False)


            attacker.attack.constraints.append(SBERT_constraint)

            results = attacker.attack_dataset()

            save_experiment_results(model_wrapper, attacker, attacker_name = attacker_name, detector = detector, file_name = file_name)