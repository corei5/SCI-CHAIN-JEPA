"""distractors.py -- adversarial flipped-finding candidate pools for F1."""
import re
import random
from typing import List, Dict
from config import CFG

_FLIP_PAIRS = [
    ("improve", "degrade"), ("increase", "decrease"), ("higher", "lower"),
    ("better", "worse"), ("reduce", "raise"), ("outperform", "underperform"),
    ("gain", "loss"), ("boost", "hurt"), ("significant", "negligible"),
    ("positive", "negative"), ("effective", "ineffective"),
    ("associated", "unassociated"), ("elevated", "reduced"),
    ("upregulated", "downregulated"),
]


def make_adversarial_distractor(result_text: str) -> str:
    text, flipped = result_text, False
    for a, b in _FLIP_PAIRS:
        if re.search(rf"\b{a}\b", text, flags=re.IGNORECASE):
            text = re.sub(rf"\b{a}\b", b, text, flags=re.IGNORECASE); flipped = True
        elif re.search(rf"\b{b}\b", text, flags=re.IGNORECASE):
            text = re.sub(rf"\b{b}\b", a, text, flags=re.IGNORECASE); flipped = True
    if not flipped:
        text = "Contrary to the findings, " + text
    return text


def build_distractor_pool(target: Dict, all_chains: List[Dict],
                          n_distractors: int) -> List[Dict]:
    pool = [{"text": target["result"], "is_true": True},
            {"text": make_adversarial_distractor(target["result"]), "is_true": False}]
    others = [c for c in all_chains if c["id"] != target["id"]]
    random.shuffle(others)
    for c in others[: max(0, n_distractors - 1)]:
        pool.append({"text": c["result"], "is_true": False})
    random.shuffle(pool)
    return pool
