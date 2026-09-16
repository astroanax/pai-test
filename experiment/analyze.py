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
                   method_column="own_model", expected_seeds=None,
                   expected_cases=None):
    """Expected tuples come from the LOCKED eval manifest, never from
    observed rows (audit 1344ed0 item 4).

    expected_cases: {case_id: {scene, context_id, target_id,
    data_sha256}}. expected_seeds: locked seed list. Missing, extra,
    and duplicate tuples are errors in both directions; paired rows
    must share scene AND data identity (a swapped target under a
    retained ID changes data_sha256 and is rejected).
    """
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
    if expected_cases is None or expected_seeds is None:
        raise ValueError(
            "paired_summary needs the locked eval manifest (expected_cases "
            "dict + expected_seeds); observed-key completeness cannot see "
            "a deleted case or seed")
    seeds = [int(s) for s in expected_seeds]
    manifest = dict(expected_cases)
    if not seeds or not manifest:
        raise ValueError("incomplete paired evaluations: empty manifest")
    extra = [(s, c, t) for (s, c, t) in index
             if int(s) not in seeds or str(c) not in manifest]
    if extra:
        raise ValueError(f"unexpected correction tuples {extra[:5]}; "
                         "plan manifest mismatch (renamed/added case?)")
    # Normalized keys: int seeds, str case_ids (JSON round-trips turn
    # int-like IDs into strings; int(s) on a non-numeric seed raises).
    index_norm = {(int(s), str(c), t): row
                  for (s, c, t), row in index.items()}
    missing = [(s, c, t) for s in seeds for c in manifest
               for t in (method, baseline)
               if (s, str(c), t) not in index_norm]
    if missing:
        raise ValueError(f"MISSING_EPISODE: {len(missing)} planned "
                         f"(seed, case, arm) tuples missing, first {missing[:5]}")
    keys = [(s, str(c)) for s in seeds for c in manifest]
    by_scene = {}
    for (seed, case) in keys:
        a_row = index_norm[(seed, case, method)]
        b_row = index_norm[(seed, case, baseline)]
        want = manifest[case]
        for row, tag in ((a_row, method), (b_row, baseline)):
            if int(row.get("scene", -1)) != int(want["scene"]):
                raise ValueError(
                    f"paired row scene != manifest for {(seed, case, tag)}: "
                    f"{row.get('scene')} != {want['scene']}")
            got_hash = row.get("case_data_sha256")
            if got_hash is None:
                raise ValueError(
                    f"paired row lacks case_data_sha256 for "
                    f"{(seed, case, tag)}; cannot verify target identity")
            if str(got_hash) != str(want["data_sha256"]):
                raise ValueError(
                    f"paired row data divorced from manifest for "
                    f"{(seed, case, tag)}: target/condition swapped "
                    "under a retained ID?")
        a, b = a_row[key], b_row[key]
        scene = a_row.get("scene")
        by_scene.setdefault(scene, []).append(a - b)
    scene_means = np.array([np.mean(v) for v in by_scene.values()])
    if len(scene_means) < 2:
        raise ValueError("contrast is degenerate: fewer than two paired scenes")
    rng = np.random.default_rng(481)
    draws = rng.integers(0, len(scene_means), size=(repeats, len(scene_means)))
    boot = scene_means[draws].mean(axis=1)
    per_seed = []
    for seed in sorted({s for (s, _) in keys}):
        vals = np.array([index_norm[(s, c, method)][key]
                         - index_norm[(s, c, baseline)][key]
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
    # Audit 1344ed0 item 5: the exact test runs PER training seed
    # (repeated observations of one scene across fixed checkpoints are
    # dependent). The combined count is reported as pooled-descriptive
    # only, never as an independent-pair exact result. The combined
    # effect uses the scene-clustered bootstrap in
    # noninferiority_from_episodes.
    from scipy.stats import binomtest
    index = {}
    for row in rows:
        if "success" not in row:
            continue
        if "K" in row and row.get("K") != PRIMARY_K:
            continue
        tag = row.get("model", row.get("method"))
        if tag not in (method, baseline):
            continue
        key = (row.get("seed"), row.get("scene"),
               row.get("eval_replicate", 0))
        if tag in index.get(key, {}):
            raise ValueError(f"duplicate success tuple {key + (tag,)}")
        index.setdefault(key, {})[tag] = int(row["success"])
    by_seed = {}
    for (seed, _scene, _rep), pair in index.items():
        by_seed.setdefault(seed, []).append(pair)
    per_seed = []
    for seed in sorted(by_seed):
        wins = sum(1 for v in by_seed[seed]
                   if v.get(method) == 1 and v.get(baseline) == 0)
        losses = sum(1 for v in by_seed[seed]
                     if v.get(method) == 0 and v.get(baseline) == 1)
        test = (binomtest(min(wins, losses), wins + losses, 0.5)
                if (wins + losses) else None)
        per_seed.append(dict(
            seed=int(seed), wins=wins, losses=losses,
            n_pairs=len(by_seed[seed]),
            p_two_sided=None if test is None else float(test.pvalue)))
    wins = sum(1 for v in index.values()
               if v.get(method) == 1 and v.get(baseline) == 0)
    losses = sum(1 for v in index.values()
                 if v.get(method) == 0 and v.get(baseline) == 1)
    return dict(wins=wins, losses=losses, n_pairs=len(index),
                per_seed=per_seed,
                note=("pooled across training seeds and repeated scenes; "
                      "descriptive only, not an independent-pair exact "
                      "test; see per_seed exact tests and the "
                      "scene-clustered bootstrap"))


def noninferiority_from_episodes(rows, method, baseline,
                                 margin=NONINF_MARGIN, repeats=BOOTSTRAP_REPEATS,
                                 expected=None):
    """expected (audit 1344ed0 item 5): dict(seeds=[...], scenes=[...],
    replicates=[...]) — exactly the planned episode grid. Missing,
    extra, and duplicate episode tuples are errors in both directions;
    the bootstrap resamples scenes jointly."""
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
        key = (row.get("seed"), row.get("scene"),
               row.get("eval_replicate", 0))
        if tag in index.get(key, {}):
            raise ValueError(f"duplicate episode tuple {key + (tag,)}")
        index.setdefault(key, {})[tag] = int(row["success"])
    keys = sorted(index)
    missing = [k for k in keys
               if method not in index[k] or baseline not in index[k]]
    if missing:
        raise ValueError(f"MISSING_EPISODE: {len(missing)} episode pairs "
                         f"lack an arm, first {missing[:5]}")
    if expected is not None:
        exp = {(int(s), int(c), int(r), t)
               for s in expected.get("seeds", [])
               for c in expected.get("scenes", [])
               for r in expected.get("replicates", [])
               for t in (method, baseline)}
        got = {(int(s), int(c), int(r), t)
               for (s, c, r) in keys for t in (method, baseline)
               if t in index[(s, c, r)]}
        extra = sorted(got - exp)
        if extra:
            raise ValueError(f"unexpected episode tuples {extra[:5]}; "
                             "plan grid mismatch")
        absent = sorted(exp - got)
        if absent:
            raise ValueError(
                f"MISSING_EPISODE: {len(absent)} planned episode tuples "
                f"missing, first {absent[:5]}")
    if not keys:
        return dict(n=0, margin=margin, noninferior=None,
                    note="no complete episode pairs supplied")
    # Scene-clustered bootstrap: resample scenes jointly (repeated
    # scenes across fixed checkpoints stay together), not rows
    # independently.
    by_scene = {}
    for k in keys:
        by_scene.setdefault(k[1], []).append(index[k][method]
                                             - index[k][baseline])
    scene_means = _np.array([_np.mean(v) for v in by_scene.values()])
    rng = _np.random.default_rng(481)
    draws = rng.integers(0, len(scene_means),
                         size=(repeats, len(scene_means)))
    boot = scene_means[draws].mean(axis=1)
    diffs = _np.array([index[k][method] - index[k][baseline] for k in keys])
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
    parser.add_argument("--allow-unlocked", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--method", default=None)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    validate_config(config, args.config)
    protocol = load_protocol(args.protocol)
    # Audit fdb59ff item 5: shared stage identifier is "analyze".
    verify_current_protocol(protocol, config, args.config, stage="analyze")
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
    # Seeds AND cases come from the locked eval manifest (a deleted
    # planned seed or a deleted/renamed/added case is an error, never
    # silently absorbed); paired rows must also match manifest scene +
    # data identity.
    design = protocol.get("design", {})
    summary = paired_summary(
        rows, args.method, args.baseline,
        expected_seeds=(design.get("seeds") or None),
        expected_cases=(design.get("eval_cases") or None))
    summary.update(protocol_id=protocol.get("protocol_id"),
                   primary_K=PRIMARY_K,
                   discordant=discordant_pairs(rows, args.method,
                                               args.baseline),
                   noninferiority=noninferiority_from_episodes(
                       rows, args.method, args.baseline,
                       expected=dict(
                           seeds=design.get("seeds", []),
                           scenes=(design.get("scenes", {}).get("final", [])),
                           replicates=design.get("eval_replicates",
                                                 [0]))),
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
