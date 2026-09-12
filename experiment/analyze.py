import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def paired_summary(frame, method, baseline, repeats=10000):
    selected = frame[frame["method"].isin([method, baseline])]
    if selected.duplicated(["method", "training_seed", "scene"]).any():
        raise ValueError("duplicate evaluation episodes")
    paired = selected.pivot(index=["training_seed", "scene"], columns="method", values="success")
    if method not in paired or baseline not in paired or paired.isna().any().any():
        raise ValueError("incomplete paired evaluations")
    differences = (paired[method] - paired[baseline]).unstack("scene")
    if differences.isna().any().any():
        raise ValueError("every seed must use the same scene set")
    values = differences.to_numpy()
    rng = np.random.default_rng(481)
    draws = rng.integers(0, values.shape[1], size=(repeats, values.shape[1]))
    boot = values[:, draws].mean(axis=(0, 2))
    result = {"method": method, "baseline": baseline,
              "difference": float(values.mean()),
              "scene_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
              "training_seeds": differences.index.astype(int).tolist(),
              "scenes": int(values.shape[1]),
              "scope": "conditional on these trained checkpoints",
              "per_seed": []}
    for seed, group in paired.groupby(level="training_seed"):
        wins = int(((group[method] == 1) & (group[baseline] == 0)).sum())
        losses = int(((group[method] == 0) & (group[baseline] == 1)).sum())
        test = binomtest(min(wins, losses), wins + losses, 0.5) if (wins + losses) else None
        result["per_seed"].append({"seed": int(seed), "wins": wins, "losses": losses,
                                   "difference": float((group[method] - group[baseline]).mean()),
                                   "p_two_sided": None if test is None else float(test.pvalue)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="pullback")
    parser.add_argument("--baseline", default="endpoint")
    args = parser.parse_args()
    frames = []
    for path in args.inputs:
        rows = [json.loads(line) for line in open(path)]
        frames.append(pd.DataFrame(rows))
    frame = pd.concat(frames, ignore_index=True)
    summary = paired_summary(frame, args.method, args.baseline)
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("wrote analysis " + args.output)


if __name__ == "__main__":
    main()
