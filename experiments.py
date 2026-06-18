"""
experiments.py
==============
Runs the three studies and saves JSON + plots. Each study trains models by
overriding CFG fields, then evaluates perplexity (headline) + F1/F2.

Usage:
  python experiments.py --study lambda    [--smoke]
  python experiments.py --study ablation  [--smoke]
  python experiments.py --study dropout   [--smoke]
  python experiments.py --study all       [--smoke]
"""
import os
import json
import argparse
import numpy as np

from config import CFG, set_tier
from train import train
from evaluate import run_all


def _run_one(seed=0):
    """Train + evaluate with the CURRENT CFG. Returns merged stats+metrics."""
    CFG.seed = seed
    train_stats = train()
    metrics = run_all(train_stats.get("model_path"))
    return {**train_stats,
            "perplexity": metrics["perplexity"]["perplexity"],
            "F1_recall1": metrics["F1_model"]["Recall@1"],
            "F2_rho": metrics["F2"]["spearman_rho"]}


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
# STUDY 1: lambda sweep
# ---------------------------------------------------------------------------
def study_lambda(seeds):
    """
    Vary lambda_jepa over a grid. lambda=0 IS the pure-NTP baseline.
    We expect a U-shape: too small = no benefit, too large = NTP gets crowded out.
    """
    lambdas = [0.0, 0.25, 0.5, 1.0, 2.0]
    results = {}
    for lam in lambdas:
        CFG.lambda_jepa = lam
        CFG.jepa_enabled = (lam > 0)   # lambda=0 means no JEPA at all
        perps = []
        for s in seeds:
            # fresh shards reused; only weights change
            r = _run_one(seed=s)
            perps.append(r["perplexity"])
        results[str(lam)] = {"perplexity_mean": float(np.mean(perps)),
                             "perplexity_std": float(np.std(perps)),
                             "perplexities": perps}
        print(f"[lambda] lambda={lam}: ppl={np.mean(perps):.3f}")
    _save("lambda", results)
    _plot("lambda_perplexity", lambdas,
          [results[str(l)]["perplexity_mean"] for l in lambdas],
          "lambda_jepa", "perplexity (lower=better)")
    return results


# ---------------------------------------------------------------------------
# STUDY 2: ablation on design choices
# ---------------------------------------------------------------------------
def study_ablation(seeds):
    """
    Turn each design choice OFF (one at a time) vs the FULL model.
    Shows which pieces actually matter.
    """
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
        perps = [ _run_one(seed=s)["perplexity"] for s in seeds ]
        results[name] = {"perplexity_mean": float(np.mean(perps)),
                         "perplexity_std": float(np.std(perps))}
        print(f"[ablation] {name}: ppl={np.mean(perps):.3f}")
    _save("ablation", results)
    return results


# ---------------------------------------------------------------------------
# STUDY 3: faster LLM-JEPAs via loss dropout
# ---------------------------------------------------------------------------
def study_dropout(seeds):
    """
    Vary loss_dropout. Higher dropout = fewer edge forward-passes = FASTER,
    but possibly higher perplexity. We report BOTH perplexity and speed
    (steps/sec, edges_run) to show the speed/quality tradeoff.
    """
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
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    if args.smoke:
        set_tier("smoke")
    seeds = list(range(args.seeds))

    if args.study in ("lambda", "all"):
        study_lambda(seeds)
    if args.study in ("ablation", "all"):
        study_ablation(seeds)
    if args.study in ("dropout", "all"):
        study_dropout(seeds)
    print("\n[done] All requested studies finished.")
