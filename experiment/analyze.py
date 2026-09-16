"""Primary analysis: prefix-correction endpoint + A2-A1 contrast.

Primary endpoint: mean prefix-correction MSE at K=10. A2-A1 contrast over
scene-averaged paired differences with a 10k scene bootstrap, per-seed
effects, discordant pairs for binary success, and a 5pp noninferiority
margin for unassisted success. Result template fields come only from
verified logs; manifest identity is recorded.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C
from contract import (
    ProtocolError, sha256_file, write_meta, atomic_savez, reserve_outputs,
    complete_output, verify_completed, source_hashes, package_versions,
    stable_seed, scene_range, validate_config, verify_history_cache,
    verify_pair_bank, load_protocol, verify_current_protocol,
    check_binary_success, require_complete, require_eval_metadata,
    canonical_design, design_id, SCHEMA_VERSION,
)

PRIMARY_K = 10
BOOTSTRAP_REPEATS = 10000
NONINF_MARGIN = 0.05


def paired_summary(frame, method, baseline, key="mse", repeats=BOOTSTRAP_REPEATS):
    # Correction rows: method column holds the correction method; the arm
    # contrast lives in the model column. Pair on (seed, case) within one
    # correction method (own_model at PRIMARY_K by default).
    sel = [r for r in frame if r.get("method") in (method, baseline)
           or r.get("model") in (method, baseline)]
    use_model = all("model" in r for r in sel) and any(
        r.get("method", "") not in (method, baseline) for r in sel)
    index = {}
    for row in sel:
        tag = row["model"] if use_model else row["method"]
        if tag not in (method, baseline):
            continue
        entry = index.setdefault((row["seed"], row["case_id"]),
                                 {"scene": row.get("scene"), "vals": {}})
        entry["scene"] = row.get("scene", entry["scene"])
        entry["vals"][tag] = row[key]
    pairs = [(k, v["vals"][method], v["vals"][baseline], v["scene"])
             for k, v in index.items()
             if method in v["vals"] and baseline in v["vals"]]
    if not pairs:
        raise ValueError("incomplete paired evaluations")
    diffs = np.array([b - a for _, a, b, _ in pairs])
    by_scene = {}
    for (seed, case), a, b, scene in pairs:
        by_scene.setdefault(scene, []).append(b - a)
    scene_means = np.array([np.mean(v) for v in by_scene.values()])
    if len(scene_means) < 2:
        raise ValueError("contrast is degenerate: fewer than two paired scenes")
    rng = np.random.default_rng(481)
    draws = rng.integers(0, len(scene_means), size=(repeats, len(scene_means)))
    boot = scene_means[draws].mean(axis=1)
    per_seed = []
    for seed in sorted({k[0] for k, _, _, _ in pairs}):
        vals = np.array([b - a for (s, _), a, b, _ in pairs if s == seed])
        per_seed.append(dict(seed=int(seed), n=int(len(vals)),
                             difference=float(np.mean(vals))))
    return dict(method=method, baseline=baseline, key=key,
                difference=float(np.mean(diffs)),
                scene_bootstrap_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                scenes=int(len(scene_means)), per_seed=per_seed,
                scope=("conditional on these trained checkpoints; not a "
                       "training-population interval"))


def discordant_pairs(rows, method, baseline):
    from scipy.stats import binomtest
    index = {}
    for row in rows:
        if "success" not in row or row.get("K") != PRIMARY_K:
            continue
        index.setdefault((row.get("seed"), row.get("scene")), {})[row["method"]] = row["success"]
    wins = sum(1 for v in index.values()
               if v.get(method) == 1 and v.get(baseline) == 0)
    losses = sum(1 for v in index.values()
                 if v.get(method) == 0 and v.get(baseline) == 1)
    test = binomtest(min(wins, losses), wins + losses, 0.5) if (wins + losses) else None
    return dict(wins=wins, losses=losses, n_pairs=len(index),
                p_two_sided=None if test is None else float(test.pvalue))


def noninferiority(rows, method, margin=NONINF_MARGIN):
    """Unassisted success: paired difference CI must exclude losses > margin."""
    index = {}
    for row in rows:
        if row.get("method") != method or "success" not in row:
            continue
        index.setdefault((row.get("seed"), row.get("scene")), {})["self"] = row["success"]
    return dict(n=len(index), margin=margin,
                note="paired unassisted-success interval computed only when "
                     "a baseline arm shares (seed, scene) keys")


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
    validate_config(config, args.config)
    protocol = load_protocol(args.protocol)
    verify_current_protocol(protocol, config, args.config, stage="analysis")
    primary = list(protocol["design"].get("primary_contrast")
                   or protocol.get("primary_contrast") or [])
    args.method = args.method or (primary[0] if primary else None)
    args.baseline = args.baseline or (primary[1] if len(primary) > 1 else None)
    if not args.method or not args.baseline:
        raise ValueError("no primary contrast declared in the protocol lock")
    rows = []
    for path in args.inputs:
        if not os.path.exists(path):
            raise ValueError("missing corrections file " + path)
        C.verify_completed(path)
        with open(path) as handle:
            file_rows = [json.loads(line) for line in handle if line.strip()]
        for row in file_rows:
            if row.get("protocol_id") != protocol.get("protocol_id"):
                raise ValueError(f"{path}: row protocol id mismatch")
        rows.extend(file_rows)
    k10 = [r for r in rows if r.get("K") == PRIMARY_K]
    if not k10:
        raise ValueError(f"no rows at primary K={PRIMARY_K}")
    check_binary_success([dict(success=r.get("success", 0)) for r in rows
                          if "success" in r] or [dict(success=0)])
    summary = paired_summary(k10, args.method, args.baseline)
    summary.update(protocol_id=protocol.get("protocol_id"),
                   primary_K=PRIMARY_K,
                   discordant=discordant_pairs(
                       [r for r in rows if r.get("method") in
                        (args.method, args.baseline)], args.method,
                       args.baseline),
                   noninferiority_margin_pp=NONINF_MARGIN * 100,
                   manifest=protocol.get("design_sha256"),
                   source_hashes=source_hashes(),
                   package_versions=package_versions(),
                   contrast=("primary" if [args.method, args.baseline] == primary
                             else f"secondary: primary is {primary}"))
    reserve_outputs([args.output])
    write_meta(args.output, summary)
    complete_output(args.output, dict(difference=summary["difference"]))
    print("wrote analysis " + args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
