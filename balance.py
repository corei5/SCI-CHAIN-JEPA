"""balance.py -- citation-imbalance-robust labels for F2."""
import numpy as np
from collections import defaultdict
from typing import List, Dict
from config import CFG
from utils import get_field


def citation_labels(chains: List[Dict]) -> np.ndarray:
    cites = np.array([c.get("citations", 0) for c in chains], dtype=float)
    if CFG.use_log_citations:
        cites = np.log1p(cites)
    labels = np.zeros(len(chains), dtype=int)
    if CFG.f2_label_strategy == "field_quartile":
        by_field = defaultdict(list)
        for idx, c in enumerate(chains):
            by_field[get_field(c)].append(idx)
        for _, idxs in by_field.items():
            if len(idxs) < 4:
                continue
            thresh = np.quantile(cites[idxs], CFG.f2_top_quantile)
            for i in idxs:
                if cites[i] >= thresh:
                    labels[i] = 1
    else:
        labels = (cites > np.median(cites)).astype(int)
    return labels
