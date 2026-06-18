"""stats.py -- paired t-test + Holm-Bonferroni."""
import numpy as np
from scipy import stats


def paired_ttest(a, b):
    if len(a) < 2:
        return float("nan")
    return float(stats.ttest_rel(a, b).pvalue)


def holm_bonferroni(pvalues, alpha=0.05):
    m = len(pvalues)
    order = np.argsort(pvalues)
    sig = [False] * m
    for rank, idx in enumerate(order):
        if pvalues[idx] <= alpha / (m - rank):
            sig[idx] = True
        else:
            break
    return sig
