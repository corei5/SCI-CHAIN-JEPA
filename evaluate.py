"""evaluate.py -- perplexity (headline) + F1 retrieval + F2 surprise."""
import os
import math
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from rank_bm25 import BM25Okapi

from config import CFG
from model import SciChainJEPA
from data_stream import load_test_chains
from distractors import build_distractor_pool
from balance import citation_labels


@torch.no_grad()
def eval_perplexity(model, test_chains):
    model.eval()
    losses = []
    for chain in test_chains[: min(CFG.perplexity_eval_examples, len(test_chains))]:
        full = " ".join(chain[s] for s in CFG.stages)
        losses.append(float(model.ntp_loss([full])))
    avg = sum(losses) / max(1, len(losses))
    return {"perplexity": math.exp(avg), "avg_nll": avg}


@torch.no_grad()
def f1_model_retrieval(model, test_chains):
    r1, r5, mrr = [], [], []
    for chain in test_chains:
        pool = build_distractor_pool(chain, test_chains, CFG.n_distractors)
        ctx = " ".join(chain[CFG.stages[k]] for k in range(4))
        pred = model.predict_stage([ctx])
        cand = model.encode([c["text"] for c in pool])
        scores = (cand @ pred.T).squeeze(1).cpu().numpy()
        ranked = [pool[i]["is_true"] for i in np.argsort(-scores)]
        rank = ranked.index(True) + 1
        r1.append(rank == 1); r5.append(rank <= 5); mrr.append(1.0 / rank)
    return {"Recall@1": float(np.mean(r1)), "Recall@5": float(np.mean(r5)),
            "MRR": float(np.mean(mrr))}


def f1_bm25_baseline(test_chains):
    r1 = []
    for chain in test_chains:
        pool = build_distractor_pool(chain, test_chains, CFG.n_distractors)
        ctx = " ".join(chain[CFG.stages[k]] for k in range(4))
        bm25 = BM25Okapi([c["text"].lower().split() for c in pool])
        ranked = [pool[i]["is_true"]
                  for i in np.argsort(-bm25.get_scores(ctx.lower().split()))]
        r1.append(ranked.index(True) == 0)
    return {"Recall@1": float(np.mean(r1))}


@torch.no_grad()
def f2_surprise(model, test_chains):
    s = []
    for chain in test_chains:
        ctx = " ".join(chain[CFG.stages[k]] for k in range(4))
        pred = model.predict_stage([ctx]); true = model.encode([chain["result"]])
        s.append(1.0 - float((pred * true).sum(-1)))
    s = np.array(s)
    labels = citation_labels(test_chains)
    cites = np.array([c.get("citations", 0) for c in test_chains], dtype=float)
    rho, p = spearmanr(s, cites)
    auroc = float("nan") if labels.min() == labels.max() else roc_auc_score(labels, s)
    return {"spearman_rho": float(rho), "spearman_p": float(p), "AUROC": float(auroc)}


def run_all(model_path=None):
    test_chains = load_test_chains()
    model = SciChainJEPA().to(CFG.device)
    if model_path:
        from transformers import AutoModelForCausalLM
        model.lm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=CFG.dtype).to(CFG.device)
    model.eval()
    res = {}
    if CFG.eval_perplexity:
        res["perplexity"] = eval_perplexity(model, test_chains)
        print(f"[eval] perplexity = {res['perplexity']['perplexity']:.3f}")
    res["F1_model"] = f1_model_retrieval(model, test_chains)
    res["F1_bm25"] = f1_bm25_baseline(test_chains)
    res["F2"] = f2_surprise(model, test_chains)
    print("[eval]", res)
    return res


if __name__ == "__main__":
    run_all()
