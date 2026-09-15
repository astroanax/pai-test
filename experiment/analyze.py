import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C


def paired_summary(frame, method, baseline, repeats=10000):
    selected = frame[frame["method"].isin([method, baseline])]
    if selected.duplicated(["method", "training_seed", "scene"]).any():
        raise ValueError("duplicate evaluation episodes")
    paired = selected.pivot(index=["training_seed", "scene"], columns="method",
                            values="success")
    if method not in paired or baseline not in paired or paired.isna().any().any():
        raise ValueError("incomplete paired evaluations")
    differences = (paired[method] - paired[baseline]).unstack("scene")
    if differences.isna().any().any():
        raise ValueError("every seed must use the same scene set")
    values = differences.to_numpy()
    if values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(
            "contrast is degenerate (fewer than two paired scenes or no seeds); "
            "a bootstrap over this frame is not a result")
    rng = np.random.default_rng(481)
    draws = rng.integers(0, values.shape[1], size=(repeats, values.shape[1]))
    boot = values[:, draws].mean(axis=(0, 2))
    result = {"method": method, "baseline": baseline,
              "difference": float(values.mean()),
              "scene_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
              "training_seeds": sorted(int(s) for s in differences.index.unique()),
              "scenes": int(values.shape[1]),
              "scope": "conditional on these trained checkpoints; not a training-population interval",
              "per_seed": []}
    for seed, group in paired.groupby(level="training_seed"):
        wins = int(((group[method] == 1) & (group[baseline] == 0)).sum())
        losses = int(((group[method] == 0) & (group[baseline] == 1)).sum())
        test = binomtest(min(wins, losses), wins + losses, 0.5) if (wins + losses) else None
        result["per_seed"].append({"seed": int(seed), "wins": wins, "losses": losses,
                                   "difference": float((group[method] - group[baseline]).mean()),
                                   "p_two_sided": None if test is None else float(test.pvalue)})
    return result


def descriptive_table(frame, methods, seeds, scenes):
    rows = []
    for method in methods:
        for seed in seeds:
            subset = frame[(frame["method"] == method) &
                           (frame["training_seed"] == seed)]
            if subset.empty:
                continue
            rows.append({"method": method, "seed": int(seed),
                         "n": int(len(subset)),
                         "success": float(subset["success"].mean()),
                         "score": float(subset["score"].mean()),
                         "latency_median_ms": float(subset["latency_median_ms"].median()),
                         "oor_fraction_raw": float(subset["oor_fraction_raw"].mean())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default="runs/protocol_locked.json")
    parser.add_argument("--method", default=None)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()

    protocol = C.require_protocol(args.protocol)
    expected_methods = list(protocol["modes"])
    if protocol.get("primary_contrast"):
        default_method, default_baseline = protocol["primary_contrast"]
        args.method = args.method or default_method
        args.baseline = args.baseline or default_baseline
    if not args.method or not args.baseline:
        raise ValueError("no primary contrast declared in the protocol lock")

    frames = []
    for path in args.inputs:
        if not os.path.exists(path):
            raise ValueError("missing evaluation file " + path)
        rows = [json.loads(line) for line in open(path) if line.strip()]
        if not rows:
            raise ValueError("empty evaluation file " + path)
        expected_id = protocol.get("protocol_id")
        for row in rows:
            if row.get("protocol_id") != expected_id:
                raise ValueError(f"{path}: row protocol id {row.get('protocol_id')} "
                                 f"!= locked {expected_id}")
        frames.append(pd.DataFrame(rows))
    frame = pd.concat(frames, ignore_index=True)
    for key in ("method", "training_seed", "scene", "success"):
        if key not in frame:
            raise ValueError("evaluation rows lack " + key)

    C.require_eval_metadata(frame, protocol, args.inputs, method=args.method)
    C.require_complete(frame, protocol, args.method, args.baseline)

    summary = paired_summary(frame, args.method, args.baseline)
    summary["protocol_id"] = protocol.get("protocol_id")
    summary["planned"] = {"modes": expected_methods, "seeds": protocol["seeds"],
                          "scenes": len(protocol["final_scenes"])}
    summary["descriptive"] = descriptive_table(
        frame, expected_methods, protocol["seeds"], protocol["final_scenes"])
    summary["source_hashes"] = C.source_hashes()
    summary["warning"] = ("descriptive pilot: at most "
                          f"{len(protocol['seeds'])} fine-tuning seeds and one "
                          "shared warm start; do not read as a training-population result")
    if os.path.exists(args.output):
        raise ValueError("analysis output exists, refusing to overwrite " + args.output)
    C.write_meta(args.output, summary)
    print("wrote analysis " + args.output)


if __name__ == "__main__":
    main()
