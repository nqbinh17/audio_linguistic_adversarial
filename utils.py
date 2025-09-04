import os
import sys
import json
import struct
import hashlib
from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)
from textattack.metrics.quality_metrics import Perplexity, USEMetric


def hash_to_int32(text):
    # Create a SHA-256 hash of the text
    hash_object = hashlib.sha256(text.encode())
    
    # Get the hash digest as bytes
    hash_bytes = hash_object.digest()
    
    # Take the first 4 bytes (32 bits) of the hash and convert them to an integer
    int32_value = struct.unpack('I', hash_bytes[:4])[0]
    
    return str(int32_value)

def load_dataset(path = 'data/voicewukong_metadata.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_attack_log_summary(attacker):
    self = attacker.attack_log_manager
    results = attacker.attack_log_manager.results
    total_attacks = len(results)

    # Default metrics - calculated on every attack
    attack_success_stats = AttackSuccessRate().calculate(results)
    words_perturbed_stats = WordsPerturbed().calculate(results)
    attack_query_stats = AttackQueries().calculate(results)

    summary_table_rows = [
        [
            "Number of successful attacks:",
            attack_success_stats["successful_attacks"],
        ],
        ["Number of failed attacks:", attack_success_stats["failed_attacks"]],
        ["Number of skipped attacks:", attack_success_stats["skipped_attacks"]],
        [
            "Original accuracy:",
            str(attack_success_stats["original_accuracy"]) + "%",
        ],
        [
            "Accuracy under attack:",
            str(attack_success_stats["attack_accuracy_perc"]) + "%",
        ],
        [
            "Attack success rate:",
            str(attack_success_stats["attack_success_rate"]) + "%",
        ],
        [
            "Average perturbed word %:",
            str(words_perturbed_stats["avg_word_perturbed_perc"]) + "%",
        ],
        [
            "Average num. words per input:",
            words_perturbed_stats["avg_word_perturbed"],
        ],
    ]

    summary_table_rows.append(
        ["Avg num queries:", attack_query_stats["avg_num_queries"]]
    )

    for metric_name, metric in self.metrics.items():
        summary_table_rows.append([metric_name, metric.calculate(self.results)])

    if self.enable_advance_metrics:
        perplexity_stats = Perplexity().calculate(self.results)
        use_stats = USEMetric().calculate(self.results)

        summary_table_rows.append(
            [
                "Average Original Perplexity:",
                perplexity_stats["avg_original_perplexity"],
            ]
        )

        summary_table_rows.append(
            [
                "Average Attack Perplexity:",
                perplexity_stats["avg_attack_perplexity"],
            ]
        )
        summary_table_rows.append(
            ["Average Attack USE Score:", use_stats["avg_attack_use_score"]]
        )
    
    return summary_table_rows


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout