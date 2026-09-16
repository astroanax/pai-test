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


def paired_summary(frame, method, baseline, key="mse", repeats=BOOTSTRAP_REPEATS,
                   method_column="own_model"):
    # Item 18: the endpoint is exactly method_column at PRIMARY_K.
    # Oracle/random-search/transfer rows must never overwrite own-model
    # measurements: duplicates and missing planned tuples are errors.
    # Item 19: one aggregation/sign convention: A2 - A1 over scene means
    # for both the point estimate and the bootstrap.
    sel = [r for r in frame
           if r.get("method") == method_column and r.get("K") == PRIMARY_K
           and r.get("model") in (method, baseline)]
    index = {}
    for row in sel:
        key_tuple = (row["seed"], row["case_id"], row["model"])
        if key_tuple in index:
            raise ValueError(f"duplicate correction tuple {key_tuple}; "
                             "never deduplicate by overwriting")
        index[key_tuple] = row
    keys = {(seed, case) for (seed, case, _) in index}
    missing = []
    for (seed, case) in sorted(keys):
        for tag in (method, baseline):
            if (seed, case, tag) not in index:
                missing.append((seed, case, tag))
    if missing:
        raise ValueError(f"MISSING_EPISODE: {len(missing)} planned "
                         f"(seed, case, arm) tuples missing, first {missing[:5]}")
    if not keys:
        raise ValueError("incomplete paired evaluations")
    by_scene = {}
    for (seed, case) in keys:
        a = index[(seed, case, method)][key]
        b = index[(seed, case, baseline)][key]
        scene = index[(seed, case, method)].get("scene")
        by_scene.setdefault(scene, []).append(a - b)
    scene_means = np.array([np.mean(v) for v in by_scene.values()])
    if len(scene_means) < 2:
        raise ValueError("contrast is degenerate: fewer than two paired scenes")
    rng = np.random.default_rng(481)
    draws = rng.integers(0, len(scene_means), size=(repeats, len(scene_means)))
    boot = scene_means[draws].mean(axis=1)
    per_seed = []
    for seed in sorted({s for (s, _) in keys}):
        vals = np.array([index[(s, c, method)][key]
                         - index[(s, c, baseline)][key]
                         for (s, c) in keys if s == seed])
        per_seed.append(dict(seed=int(seed), n=int(len(vals)),
                             difference=float(np.mean(vals))))
    return dict(method=method, baseline=baseline, key=key,
                correction_method=method_column,
                difference=float(np.mean(scene_means)),
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
        tag = row.get("model", row.get("method"))
        if tag not in (method, baseline):
            continue
        key = (row.get("seed"), row.get("scene"))
        if key in index and tag in index[key]:
            raise ValueError(f"duplicate success tuple {key + (tag,)}")
        index.setdefault(key, {})[tag] = row["success"]
    wins = sum(1 for v in index.values()
               if v.get(method) == 1 and v.get(baseline) == 0)
    losses = sum(1 for v in index.values()
                 if v.get(method) == 0 and v.get(baseline) == 1)
    test = binomtest(min(wins, losses), wins + losses, 0.5) if (wins + losses) else None
    return dict(wins=wins, losses=losses, n_pairs=len(index),
                p_two_sided=None if test is None else float(test.pvalue))


def noninferiority_from_episodes(rows, method, baseline,
                                 margin=NONINF_MARGIN, repeats=BOOTSTRAP_REPEATS):
    """Item 19: unassisted success on COMPLETE episode pairs, never inferred
    from correction-MSE rows. Paired (method - baseline) success per
    (seed, scene); noninferior if the bootstrap CI lower bound exceeds
    -margin."""
    import numpy as _np
    index = {}
    for row in rows:
        if row.get("model", row.get("method")) not in (method, baseline):
            continue
        if "success" not in row:
            continue
        tag = row.get("model", row.get("method"))
        C.check_binary_success(row["success"])
        index.setdefault((row.get("seed"), row.get("scene")), {})[tag] = int(
            row["success"])
    keys = sorted(index)
    missing = [k for k in keys
               if method not in index[k] or baseline not in index[k]]
    if missing:
        raise ValueError(f"MISSING_EPISODE: {len(missing)} episode pairs "
                         f"lack an arm, first {missing[:5]}")
    if not keys:
        return dict(n=0, margin=margin, noninferior=None,
                    note="no complete episode pairs supplied")
    diffs = _np.array([index[k][method] - index[k][baseline] for k in keys])
    rng = _np.random.default_rng(481)
    draws = rng.integers(0, len(diffs), size=(repeats, len(diffs)))
    boot = diffs[draws].mean(axis=1)
    ci = _np.quantile(boot, [0.025, 0.975]).tolist()
    return dict(n=len(keys), difference=float(diffs.mean()),
                ci95=ci, margin=margin,
                noninferior=bool(ci[0] > -margin))


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
    for row in rows:
        if "success" in row:
            C.check_binary_success(row["success"])
    summary = paired_summary(rows, args.method, args.baseline)
    summary.update(protocol_id=protocol.get("protocol_id"),
                   primary_K=PRIMARY_K,
                   discordant=discordant_pairs(rows, args.method,
                                               args.baseline),
                   noninferiority=noninferiority_from_episodes(
                       rows, args.method, args.baseline),
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
