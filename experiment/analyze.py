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
              "scope": ("conditional on these trained checkpoints; not a "
                        "training-population interval"),
              "per_seed": []}
    for seed, group in paired.groupby(level="training_seed"):
        wins = int(((group[method] == 1) & (group[baseline] == 0)).sum())
        losses = int(((group[method] == 0) & (group[baseline] == 1)).sum())
        test = binomtest(min(wins, losses), wins + losses, 0.5) if (wins + losses) else None
        result["per_seed"].append({"seed": int(seed), "wins": wins, "losses": losses,
                                   "difference": float((group[method] - group[baseline]).mean()),
                                   "p_two_sided": None if test is None else float(test.pvalue)})
    return result


def descriptive_table(frame, protocol):
    """Every locked run must appear, so secondary methods cannot silently vanish.

    Declared reference measurements (teacher / native-fast) are included with
    their measured success, score, and latencies when their files are passed.
    """
    rows = []
    for method in protocol["modes"]:
        for seed in protocol["seeds"]:
            subset = frame[(frame["method"] == method) &
                           (frame["training_seed"] == int(seed))]
            if subset.empty:
                raise ValueError(f"descriptive table: locked run {method}/seed{seed} "
                                 "is missing")
            rows.append({"method": method, "seed": int(seed), "kind": "student",
                         "n": int(len(subset)),
                         "success": float(subset["success"].mean()),
                         "score": float(subset["score"].mean()),
                         "latency_median_ms": float(subset["decision_latency_median_ms"].median()),
                         "head_latency_median_ms": float(subset["action_head_latency_median_ms"].median()),
                         "raw_violation_fraction": float(subset["raw_violation_fraction"].mean())})
    return rows


def reference_table(frame, protocol):
    """Measured reference runs declared in the protocol (teacher/native-fast).

    References are required at their planned scenes when declared; their
    measured success, score, and latencies enter the report as references,
    never as arms of the primary contrast.
    """
    rows = []
    for name in ("teacher", "native_fast"):
        spec = (protocol.get("references") or {}).get(name)
        if not spec:
            continue
        subset = frame[frame["method"] == spec.get("method", name)]
        if subset.empty:
            raise ValueError(f"declared reference {name} has no evaluation rows; "
                             "pass its file or remove it from the lock")
        records = subset.to_dict("records")
        C.validate_reference_rows(records, spec, protocol)
        rows.append(dict(name=name, method=spec.get("method", name),
                         n=int(len(subset)),
                         success=float(subset["success"].mean()),
                         score=float(subset["score"].mean()),
                         latency_median_ms=float(
                             subset["decision_latency_median_ms"].median()),
                         head_latency_median_ms=float(
                             subset["action_head_latency_median_ms"].median()),
                         spec=spec))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default="runs/protocol_locked.json")
    parser.add_argument("--method", default=None)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    C.validate_config(config, args.config)
    protocol = C.load_protocol(args.protocol)
    C.verify_current_protocol(protocol, config, args.config, stage="analysis")
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
        C.verify_completed(path, role=None, protocol=protocol)
        sidecar = path.replace(".jsonl", ".meta.json")
        rows = [json.loads(line) for line in open(path) if line.strip()]
        with open(sidecar) as handle:
            meta = json.load(handle)
        if meta.get("scenes") is not None and len(meta["scenes"]) != len(rows):
            raise ValueError(f"MISSING_EPISODE: {path} holds {len(rows)} rows for sidecar scenes {meta['scenes']}; refusing to concatenate")
        if not rows:
            raise ValueError("empty evaluation file " + path)
        for row in rows:
            if row.get("protocol_id") != protocol.get("protocol_id"):
                raise ValueError(f"{path}: row protocol id {row.get('protocol_id')} "
                                 f"!= locked {protocol.get('protocol_id')}")
        frames.append(pd.DataFrame(rows))
    frame = pd.concat(frames, ignore_index=True)
    for key in ("method", "training_seed", "scene", "success", "eval_role",
                "eval_replicate", "inference_steps", "checkpoint_sha256"):
        if key not in frame:
            raise ValueError("evaluation rows lack " + key + " (pre-schema evaluation; rebuild it)")
    for row in frame.to_dict("records"):
        C.validate_eval_row_role(row, protocol)
    for key in ("method", "training_seed", "scene", "success"):
        if key not in frame:
            raise ValueError("evaluation rows lack " + key)
    C.check_binary_success(frame.to_dict("records"))
    if "split" in frame and (frame["split"] != "test").any():
        bad = sorted(frame.loc[frame["split"] != "test", "split"].unique().tolist())
        raise ValueError("final analysis accepts only test-split rows; "
                         f"development rows ({bad}) are labeled development "
                         "and can never enter a final comparison")
    metadata = C.require_eval_metadata(frame, protocol, args.inputs,
                                       method=args.method)
    if metadata["missing_runs"]:
        raise ValueError(f"locked runs still missing: {metadata['missing_runs']}")
    C.require_complete(frame, protocol, args.method, args.baseline)

    summary = paired_summary(frame, args.method, args.baseline)
    summary["protocol_id"] = protocol.get("protocol_id")
    primary = list(protocol.get("primary_contrast") or [])
    if [args.method, args.baseline] == primary:
        summary["contrast"] = "primary"
    else:
        summary["contrast"] = ("secondary: user-selected "
                               f"{args.method} vs {args.baseline}; the registered "
                               f"primary contrast is {primary}")
    summary["planned"] = {"modes": protocol["modes"], "seeds": protocol["seeds"],
                          "scenes": len(protocol["final_scenes"])}
    summary["descriptive"] = descriptive_table(frame, protocol)
    summary["references"] = reference_table(frame, protocol)
    summary["reference"] = {
        "teacher_steps": protocol.get("teacher_steps"),
        "teacher_source": protocol.get("source_distribution"),
        "protocol_verified": True}
    summary["source_hashes"] = C.source_hashes()
    summary["warning"] = ("descriptive pilot: at most "
                          f"{len(protocol['seeds'])} fine-tuning seeds from one "
                          "shared warm start; do not read as a training-population "
                          "result")
    C.write_meta(args.output, summary)
    print("wrote analysis " + args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
