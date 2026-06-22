"""
experiments.py
==============
Runs the three studies and saves JSON + plots. Each study trains models by
overriding CFG fields, then evaluates perplexity (headline) + F1/F2
+ embedding-health (collapse) diagnostic.

Usage:
  python experiments.py --study lambda   --tier FT --seeds 3
  python experiments.py --study ablation --tier FT --seeds 3
  python experiments.py --study dropout  --tier FT --seeds 3
  python experiments.py --study all      --smoke
"""
import os
import json
import argparse
import numpy as np

from config import CFG, set_tier
from train import train
from evaluate import run_all
from stats import paired_ttest, holm_bonferroni


def _run_one(seed=0):
    """Train + evaluate with the CURRENT CFG. Returns merged stats+metrics."""
    CFG.seed = seed
    train_stats = train()
    metrics = run_all(train_stats.get("model_path"))
    return {**train_stats,
            "perplexity": metrics["perplexity"]["perplexity"],
            "F1_recall1": metrics["F1_model"]["Recall@1"],
            "F1_bm25_recall1": metrics["F1_bm25"]["Recall@1"],
            "F2_rho": metrics["F2"]["spearman_rho"],
            "mean_pairwise_cos": metrics["embedding_health"]["mean_pairwise_cos"],
            "effective_rank": metrics["embedding_health"]["effective_rank"]}


def _save(name, results):
    os.makedirs(CFG.output_dir, exist_ok=True)
    path = os.path.join(CFG.output_dir, f"study_{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[study] saved -> {path}")


def _plot(name, xs, ys, xlabel, ylabel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(xs, ys, "o-")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(name); plt.grid(True)
        out = os.path.join(CFG.output_dir, f"study_{name}.png")
        plt.savefig(out, bbox_inches="tight"); print(f"[study] plot -> {out}")
    except Exception as e:
        print(f"[study] plot skipped: {e}")


# ---------------------------------------------------------------------------
# STUDY 1: lambda sweep (with significance testing vs lambda=0 baseline)
# ---------------------------------------------------------------------------
def study_lambda(seeds):
    lambdas = [0.0, 0.25, 0.5, 1.0, 2.0]
    results = {}
    perp_by_lambda = {}                       # lambda -> list of per-seed ppl
    for lam in lambdas:
        CFG.lambda_jepa = lam
        CFG.jepa_enabled = (lam > 0)          # lambda=0 means no JEPA at all
        perps, cos, rank = [], [], []
        for s in seeds:
            r = _run_one(seed=s)
            perps.append(r["perplexity"])
            cos.append(r["mean_pairwise_cos"])
            rank.append(r["effective_rank"])
        perp_by_lambda[lam] = perps
        results[str(lam)] = {
            "perplexity_mean": float(np.mean(perps)),
            "perplexity_std": float(np.std(perps)),
            "perplexities": perps,
            "mean_pairwise_cos": float(np.mean(cos)),
            "effective_rank": float(np.mean(rank))}
        print(f"[lambda] lambda={lam}: ppl={np.mean(perps):.3f} "
              f"cos={np.mean(cos):.3f} eff_rank={np.mean(rank):.1f}")

    # ---- significance: each lambda>0 vs lambda=0 baseline (Holm-corrected) ----
    baseline = perp_by_lambda[0.0]
    nonzero = [l for l in lambdas if l > 0]
    pvals = [paired_ttest(perp_by_lambda[l], baseline) for l in nonzero]
    sig = holm_bonferroni([p if p == p else 1.0 for p in pvals])  # NaN->1.0
    results["_significance"] = {
        "baseline_lambda": 0.0,
        "tests": {str(l): {"p_value": pvals[i], "significant_holm": bool(sig[i]),
                           "better_than_baseline":
                               np.mean(perp_by_lambda[l]) < np.mean(baseline)}
                  for i, l in enumerate(nonzero)}}
    print(f"[lambda] significance vs baseline: {results['_significance']['tests']}")

    _save("lambda", results)
    _plot("lambda_perplexity", lambdas,
          [results[str(l)]["perplexity_mean"] for l in lambdas],
          "lambda_jepa", "perplexity (lower=better)")
    return results


# ---------------------------------------------------------------------------
# STUDY 2: ablation on design choices
# ---------------------------------------------------------------------------
def study_ablation(seeds):
    base_cfg = dict(use_pred_tokens=True, use_long_range_edge=True,
                    use_trivial_edge_filter=True, jepa_loss_type="cosine",
                    jepa_enabled=True, lambda_jepa=1.0)
    variants = {
        "full":               {},
        "no_pred_tokens":     {"use_pred_tokens": False},
        "no_long_range_edge": {"use_long_range_edge": False},
        "no_trivial_filter":  {"use_trivial_edge_filter": False},
        "mse_loss":           {"jepa_loss_type": "mse"},
        "no_jepa(baseline)":  {"jepa_enabled": False},
    }
    results = {}
    for name, override in variants.items():
        for k, v in {**base_cfg, **override}.items():
            setattr(CFG, k, v)
        perps, cos = [], []
        for s in seeds:
            r = _run_one(seed=s)
            perps.append(r["perplexity"]); cos.append(r["mean_pairwise_cos"])
        results[name] = {"perplexity_mean": float(np.mean(perps)),
                         "perplexity_std": float(np.std(perps)),
                         "mean_pairwise_cos": float(np.mean(cos))}
        print(f"[ablation] {name}: ppl={np.mean(perps):.3f} cos={np.mean(cos):.3f}")
    _save("ablation", results)
    return results


# ---------------------------------------------------------------------------
# STUDY 3: faster LLM-JEPAs via loss dropout
# ---------------------------------------------------------------------------
def study_dropout(seeds):
    dropouts = [0.0, 0.125, 0.25, 0.5, 0.75]
    CFG.jepa_enabled = True; CFG.lambda_jepa = 1.0
    results = {}
    for d in dropouts:
        CFG.loss_dropout = d
        perps, speeds, edges = [], [], []
        for s in seeds:
            r = _run_one(seed=s)
            perps.append(r["perplexity"]); speeds.append(r["steps_per_sec"])
            edges.append(r["edges_run"])
        results[str(d)] = {
            "perplexity_mean": float(np.mean(perps)),
            "steps_per_sec_mean": float(np.mean(speeds)),
            "edges_run_mean": float(np.mean(edges))}
        print(f"[dropout] d={d}: ppl={np.mean(perps):.3f} "
              f"speed={np.mean(speeds):.3f} steps/s edges={np.mean(edges):.0f}")
    _save("dropout", results)
    _plot("dropout_perplexity", dropouts,
          [results[str(d)]["perplexity_mean"] for d in dropouts],
          "loss_dropout", "perplexity")
    _plot("dropout_speed", dropouts,
          [results[str(d)]["steps_per_sec_mean"] for d in dropouts],
          "loss_dropout", "steps/sec (higher=faster)")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["lambda", "ablation", "dropout", "all"],
                    default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast run to catch errors before scaling")
    ap.add_argument("--tier", choices=["smoke", "FT", "P1", "P2"], default=None,
                    help="scale tier (FT=Phase-1 fine-tuning, P1/P2=Phase-2 scratch)")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    if args.smoke:
        set_tier("smoke")
    elif args.tier:
        set_tier(args.tier)
    seeds = list(range(args.seeds))

    if args.study in ("lambda", "all"):
        study_lambda(seeds)
    if args.study in ("ablation", "all"):
        study_ablation(seeds)
    if args.study in ("dropout", "all"):
        study_dropout(seeds)
    print("\n[done] All requested studies finished.")
