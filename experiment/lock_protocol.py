import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-no-final-results", action="store_true",
                        required=True,
                        help="explicit acknowledgement that final outcomes are unseen")
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--cache", default="runs/shared_metrics.npz")
    parser.add_argument("--warm", default="runs/warm.pt")
    parser.add_argument("--smoke-report", default="runs/smoke_report.json")
    parser.add_argument("--output", default="runs/protocol_locked.json")
    parser.add_argument("--min-metric-fraction", type=float, default=0.5,
                        help="minimum fraction of the planned metric subset that "
                             "must be present; lower only for a tiny vertical test")
    parser.add_argument("--final-prefix", default="runs/final_",
                        help="directory prefix where final evaluations will land; "
                             "the lock refuses if outcomes already exist there")
    parser.add_argument("--test-episodes", type=int, default=None,
                        help="predeclared final evaluation size; defaults to the "
                             "configured test count (a smaller predeclared size "
                             "weakens precision and must be chosen before seeing "
                             "outcomes)")
    parser.add_argument("--min-support", type=int, default=8,
                        help="minimum positive-trace support per scale before "
                             "final training (0 only for a tiny vertical test)")
    parser.add_argument("--declare-teacher-ref", action="store_true",
                        help="declare the reference-teacher evaluation as a "
                             "required reference in the analysis")
    parser.add_argument("--declare-native-fast-ref", action="store_true",
                        help="declare the native-fast baseline evaluation as a "
                             "required reference in the analysis")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = json.load(handle)
    C.validate_config(config, args.config)
    if args.modes is None:
        args.modes = ["uniform", "prefix", "endpoint", "pullback", "identity",
                      "scalar"]
    if args.seeds is None:
        args.seeds = [0, 1]
    if args.updates is None:
        args.updates = config["updates"]
    # Validate every locked choice at creation: empty/duplicate modes or
    # seeds, unknown modes, nonpositive updates, bad fractions, unusable
    # contrast.
    if not args.modes or len(set(args.modes)) != len(args.modes):
        raise ValueError("modes must be nonempty with no duplicates: "
                         f"{args.modes}")
    unknown_modes = [m for m in args.modes
                     if m not in ("uniform", "prefix", "endpoint", "pullback",
                                  "identity", "scalar")]
    if unknown_modes:
        raise ValueError(f"unknown modes {unknown_modes}")
    if not args.seeds or len(set(int(s) for s in args.seeds)) != len(args.seeds):
        raise ValueError(f"seeds must be nonempty with no duplicates: {args.seeds}")
    if any(int(s) < 0 for s in args.seeds):
        raise ValueError(f"seeds must be >= 0: {args.seeds}")
    if int(args.updates) < 1:
        raise ValueError(f"updates must be >= 1: {args.updates}")
    if not 0.0 <= float(args.min_metric_fraction) <= 1.0:
        raise ValueError("min-metric-fraction must be in [0, 1]: "
                         f"{args.min_metric_fraction}")

    required = {args.config: "config", args.cache: "metric cache",
                args.warm: "warm start", config["checkpoint"]: "teacher checkpoint",
                config["normalizer"]: "normalizer",
                args.smoke_report: "smoke report"}
    missing = [f"{name} ({path})" for path, name in required.items()
               if not os.path.exists(path)]
    if missing:
        raise ValueError("missing required artifacts: " + ", ".join(missing))
    for path in (args.cache, args.warm, args.smoke_report):
        if not C.is_complete(path):
            raise ValueError(f"{path} has no completion record; rebuild it "
                             "before locking")
    allowed = C.scene_range(config, "train") | C.scene_range(config, "validation")
    verified = C.verify_label_cache(args.cache, config, expect_mode="shared",
                                    allowed_scenes=allowed, require_metric=True)
    train_rows, validation_rows = C.metric_subset(verified, config)
    required_rows = int(config["metric_contexts"]) * args.min_metric_fraction
    if len(train_rows) < required_rows:
        raise ValueError(f"metric cache holds {len(train_rows)} training contexts, "
                         f"below {args.min_metric_fraction:.0%} of the planned "
                         f"{config['metric_contexts']}")
    with open(args.smoke_report) as handle:
        smoke = json.load(handle)
    if not smoke.get("passed"):
        raise ValueError("smoke report is not a pass: " + json.dumps(
            {k: v for k, v in smoke.items() if k in ("failed",)}))
    if smoke.get("source_hashes") != C.source_hashes():
        raise ValueError("smoke report came from different experiment sources; "
                         "rerun experiment/smoke.py")
    for key in ("checkpoint", "normalizer"):
        if smoke.get("asset_hashes", {}).get(key) != C.sha256_file(config[key]):
            raise ValueError(f"smoke report asset hash mismatch for {key}")

    splits = C.validate_split_ranges(config)
    if splits["errors"]:
        raise ValueError("; ".join(splits["errors"]))
    final_scenes = sorted(C.scene_range(config, "test"))
    if args.test_episodes is not None:
        if args.test_episodes < 1 or args.test_episodes > len(final_scenes):
            raise ValueError("--test-episodes must be within the configured range")
        final_scenes = final_scenes[:args.test_episodes]
    C.validate_scene_sets({"train": sorted(C.scene_range(config, "train")),
                           "validation": sorted(C.scene_range(config, "validation")),
                           "development": sorted(C.scene_range(config, "development")),
                           "test": final_scenes})
    if args.output and os.path.exists(args.output):
        raise ValueError("protocol already locked: " + args.output)
    for mode in args.modes:
        for seed in args.seeds:
            path = f"{args.final_prefix}{mode}_seed{seed}.jsonl"
            if os.path.exists(path):
                raise ValueError("final outcomes already exist: " + path)

    primary_contrast = (["pullback", "endpoint"]
                        if "endpoint" in args.modes and "pullback" in args.modes
                        else (["pullback", "uniform"]
                              if "pullback" in args.modes and "uniform" in args.modes
                              else list(args.modes[:2])))
    if len(primary_contrast) < 2 or \
            any(m not in args.modes for m in primary_contrast):
        raise ValueError(f"unusable primary contrast {primary_contrast} "
                         f"for modes {args.modes}")

    # Sensitivity support gate BEFORE final training: every scale needs
    # substantive positive-trace support, or the comparison rests on too few
    # sensitive contexts to mean anything. Tiny vertical tests opt out
    # explicitly with --min-support 0.
    import numpy as np
    import torch
    from core import metric_scales
    metric_key = "metric_exact" if config.get("metric_mode", "exact") == "exact" \
        else "probes"
    metric_train = [c for c in C.select_split(verified, "train")
                    if c.get("has_metric", False)]
    if any(m in args.modes for m in ("endpoint", "pullback", "identity", "scalar")):
        if not metric_train:
            raise ValueError("no has_metric training rows for a metric comparison")
        _, scale_stats = metric_scales(
            torch.from_numpy(np.stack([c["jacobian_start"] for c in metric_train])).float(),
            torch.from_numpy(np.stack([c["jacobian_end"] for c in metric_train])).float(),
            torch.from_numpy(np.stack([c[metric_key] for c in metric_train])).float(),
            metric_mode=config.get("metric_mode", "exact"))
        thin = {k: v for k, v in scale_stats["support"].items()
                if v < int(args.min_support)}
        if thin:
            raise ValueError(f"sensitivity support {scale_stats['support']} below "
                             f"min-support {args.min_support}: {thin}; inspect "
                             "the cache before locking")
    else:
        scale_stats = dict(support={}, note="no metric modes locked")

    compute = C.apply_compute_env(config["device"])
    protocol_id, parts = C.protocol_id(config, args.config, args.cache, args.warm,
                                       args.modes, args.seeds, args.updates,
                                       smoke_report=args.smoke_report,
                                       final_scenes=final_scenes,
                                       primary_contrast=primary_contrast,
                                       compute=compute)
    references = {}
    if args.declare_teacher_ref:
        references["teacher"] = dict(
            method="teacher", steps=int(config["teacher_steps"]),
            source=config["source"], scenes=final_scenes)
    if args.declare_native_fast_ref:
        references["native_fast"] = dict(
            method="native_fast",
            steps=int(config["native_fast_steps"]),
            source=config["native_fast_source"], scenes=final_scenes)
    protocol = dict(
        protocol_id=protocol_id, schema_version=C.SCHEMA_VERSION,
        metric_subset=dict(train=len(train_rows), validation=len(validation_rows),
                           available=verified["n_contexts"]),
        experiment="execution-pullback pilot",
        mode="single shared warm start, independent fine-tuning seeds",
        modes=list(args.modes), seeds=[int(s) for s in args.seeds],
        updates=int(args.updates), warm_start=args.warm,
        warm_start_sha256=C.sha256_file(args.warm),
        cache=args.cache, cache_sha256=C.sha256_file(args.cache),
        primary_contrast=primary_contrast,
        final_scenes=final_scenes,
        development_scenes=sorted(C.scene_range(config, "development")),
        train_scenes=sorted(C.scene_range(config, "train")),
        validation_scenes=sorted(C.scene_range(config, "validation")),
        checkpoint=config["checkpoint"],
        checkpoint_sha256=C.sha256_file(config["checkpoint"]),
        normalizer=config["normalizer"],
        normalizer_sha256=C.sha256_file(config["normalizer"]),
        assets=C.asset_hashes(config),
        source_hashes=C.source_hashes(),
        upstream=parts["upstream"], upstream_commit=parts["upstream_commit"],
        packages=C.package_versions(),
        canonical_choices=parts["choices"],
        metric_mode=config.get("metric_mode", "exact"),
        teacher_steps=int(config["teacher_steps"]),
        source_distribution=config["source"],
        native_fast=dict(steps=int(config["native_fast_steps"]),
                         source=config["native_fast_source"]),
        references=references,
        lock_support=scale_stats,
        compute=compute,
        smoke_report=args.smoke_report,
        smoke_report_sha256=C.sha256_file(args.smoke_report),
        resolved_config=config,
        min_metric_fraction=float(args.min_metric_fraction),
        final_prefix=args.final_prefix,
        test_episodes_planned=len(final_scenes),
        no_final_results_examined=True)
    C.write_meta(args.output, protocol)
    print("locked protocol " + args.output + " id " + protocol_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
