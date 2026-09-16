"""E0 teacher/base readiness sweep.

Grid: uniform{1,4,16} + gaussian{16,32} steps x replicates {0,1}, on >=20
development scenes with deterministic noise via C.stable_seed. Per
convention: success (terminated and score >= 0.95), score, invalid-command
rate, scene coverage (success in >= 5 distinct scenes required). Gate:
success >= 0.50. Writes readiness.json. Fails loudly if none passes —
do not train students.
"""
import argparse
import json
import os
import sys

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

GRID = [
    dict(name="uniform_1", source="uniform", steps=1),
    dict(name="uniform_4", source="uniform", steps=4),
    dict(name="uniform_16", source="uniform", steps=16),
    dict(name="gaussian_16", source="gaussian", steps=16),
    dict(name="gaussian_32", source="gaussian", steps=32),
]
REPLICATES = (0, 1)
MIN_SCENES = 20
MIN_COVERAGE_SCENES = 5
GATE_SUCCESS = 0.50


def run_convention(adapter, config, device, scenes, source, steps,
                   replicate):
    """Real teacher episodes; per-scene (success, score, invalid_rate)."""
    from evaluate import rollout_episode
    rows = []
    for scene in scenes:
        result = rollout_episode(adapter, config, device, scene, source,
                                 steps, replicate, policy="teacher")
        denom = max(int(result.get("executed", 0)) * 2, 1)
        rows.append(dict(scene=int(scene),
                         success=int(result["success"]),
                         score=float(result["score"]),
                         invalid_rate=float(result.get("invalid", 0)) / denom))
    return rows


def summarize(rows):
    n = len(rows)
    success = sum(r["success"] for r in rows) / max(n, 1)
    score = sum(r["score"] for r in rows) / max(n, 1)
    invalid = sum(r["invalid_rate"] for r in rows) / max(n, 1)
    coverage = len({r["scene"] for r in rows if r["success"]})
    return dict(success=success, score=score, invalid_rate=invalid,
                coverage_scenes=coverage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=MIN_SCENES)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    validate_config(config, args.config)
    if args.scenes < MIN_SCENES:
        raise ValueError(f"readiness needs >= {MIN_SCENES} development scenes")
    dev = sorted(scene_range(config, "development"))
    if len(dev) < args.scenes:
        raise ValueError(f"only {len(dev)} development scenes configured")
    scenes = dev[:args.scenes]

    from evaluate import build_adapter
    from hri_adapter import resolve_device
    device = resolve_device(config.get("device", "cuda"))
    adapter = build_adapter(config, device)
    conventions = []
    for convention in GRID:
        for replicate in REPLICATES:
            rows = run_convention(adapter, config, device, scenes,
                                  convention["source"], convention["steps"],
                                  replicate)
            summary = summarize(rows)
            conventions.append(dict(name=convention["name"],
                                    source=convention["source"],
                                    steps=convention["steps"],
                                    replicate=replicate, scenes=scenes,
                                    n=len(rows), per_scene=rows, **summary))
    gate = dict(min_success=C.READINESS_MIN_SUCCESS,
                min_score=C.READINESS_MIN_SCORE,
                min_coverage_scenes=MIN_COVERAGE_SCENES)
    passing = [c for c in conventions
               if c["success"] >= C.READINESS_MIN_SUCCESS
               and c["coverage_scenes"] >= MIN_COVERAGE_SCENES]
    shared = C.readiness_gate(conventions, config["canonical_source"],
                              int(config["teacher_steps"]))
    passed = bool(passing) and bool(shared["passed"])
    report = dict(kind="teacher_readiness", schema_version=SCHEMA_VERSION,
                  config_sha256=sha256_file(args.config),
                  checkpoint_sha256=sha256_file(config["checkpoint"]),
                  source_hashes=source_hashes(),
                  package_versions=package_versions(),
                  conventions=conventions, gate=gate,
                  shared_gate=shared,
                  passed=passed,
                  passing=[c["name"] for c in passing],
                  compute=C.compute_env(config.get("device", "cuda")))
    reserve_outputs([args.output])
    write_meta(args.output, report)
    complete_output(args.output, dict(conventions=len(conventions)))
    if not passed:
        raise ProtocolError(
            "readiness gate failed: no convention reached success >= "
            f"{GATE_SUCCESS} with coverage >= {MIN_COVERAGE_SCENES} scenes; "
            "do not train students")
    best = max(passing, key=lambda c: (c["success"], c["score"]))
    print("readiness best: " + json.dumps({k: best[k] for k in
          ("name", "source", "steps", "success", "score", "coverage_scenes")}))
    print("wrote readiness " + args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
