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
    parser.add_argument("--min-metric-scenes", type=int, default=2,
                        help="minimum distinct scenes among metric-labelled "
                             "training contexts (0 only for a tiny vertical "
                             "test)")
    parser.add_argument("--declare-teacher-ref", action="store_true",
                        help="declare the reference-teacher evaluation as a "
                             "required reference in the analysis")
    parser.add_argument("--kind", choices=("integration_test", "principal_experiment"),
                        default="principal_experiment",
                        help="schema tag: tiny integration runs must never be mistaken for the principal comparison")
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
    args.seeds = C.normalize_seeds(args.seeds or [0, 1])
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
    # Coverage is counted from ACTUAL metric-labelled contexts (has_metric),
    # never from the planned subset size: metric_subset() returns the planned
    # rows, which overstates coverage when the metrics stage labelled fewer.
    metric_train = [c for c in C.select_split(verified, "train")
                    if c.get("has_metric", False)]
    metric_validation = [c for c in C.select_split(verified, "validation")
                         if c.get("has_metric", False)]
    if verified["meta"].get("profile"):
        raise ValueError("metric cache is a profile (--limit) cache; rebuild "
                         "the full cache before locking the principal protocol")
    # Cross-check the actual labels against the membership the metric
    # producer declared: a cache whose has_metric rows disagree with the
    # producer's member_keys was edited or mixed after the metrics stage.
    lineage = verified.get("lineage") or {}
    collector = lineage.get("collector")
    collector_hash = lineage.get("collector_sha256")
    if collector is None or collector_hash is None:
        raise ValueError("metric cache lineage lacks the collector checkpoint (pre-schema cache); rebuild the metrics stage")
    if collector != args.warm or collector_hash != C.sha256_file(args.warm):
        raise ValueError("metric cache was not collected by the locked warm start: collector " + repr(collector) + " vs warm " + repr(args.warm) + "; recollect from the warm checkpoint")
    declared = verified["meta"].get("member_keys")
    if declared is not None:
        def _key(context):
            return (int(context["scene"]), str(context["split"]),
                    int(context["decision"]), int(context["step"]))
        declared_keys = {tuple(k) if not isinstance(k, list)
                         else (int(k[0]), str(k[1]), int(k[2]), int(k[3]))
                         for k in declared}
        actual_keys = {_key(c) for c in metric_train + metric_validation}
        if actual_keys != declared_keys:
            raise ValueError(
                "metric-labelled contexts disagree with the producer's "
                f"declared membership: {len(actual_keys)} actual vs "
                f"{len(declared_keys)} declared; rebuild the metrics stage")
    required_rows = int(config["metric_contexts"]) * args.min_metric_fraction
    if len(metric_train) < required_rows:
        raise ValueError(f"metric cache holds {len(metric_train)} metric-labelled "
                         f"training contexts, below {args.min_metric_fraction:.0%} "
                         f"of the planned {config['metric_contexts']}")
    required_validation = int(config["validation_metric_contexts"]) * \
        args.min_metric_fraction
    if len(metric_validation) < required_validation:
        raise ValueError(f"metric cache holds {len(metric_validation)} metric-labelled "
                         f"validation contexts, below {args.min_metric_fraction:.0%} "
                         f"of the planned {config['validation_metric_contexts']}; "
                         "run the metrics stage for the validation split")
    metric_scenes = sorted({int(c["scene"]) for c in metric_train})
    if len(metric_scenes) < int(args.min_metric_scenes):
        raise ValueError(f"metric-labelled training contexts cover "
                         f"{len(metric_scenes)} distinct scenes "
                         f"({metric_scenes[:5]}), below min-metric-scenes "
                         f"{args.min_metric_scenes}")
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
    eval_noise = dict(replicate=0, key="(eval_replicate, scene, decision)",
                      source=config["source"],
                      teacher_steps=int(config["teacher_steps"]))
    references = {}
    if args.declare_teacher_ref:
        references["teacher"] = dict(
            method="teacher", role="teacher_reference",
            steps=int(config["teacher_steps"]),
            source=config["source"], scenes=final_scenes, replicate=0,
            checkpoint_sha256=C.sha256_file(config["checkpoint"]))
    if args.declare_native_fast_ref:
        references["native_fast"] = dict(
            method="native_fast", role="native_fast_reference",
            steps=int(config["native_fast_steps"]),
            source=config["native_fast_source"], scenes=final_scenes,
            replicate=0,
            checkpoint_sha256=C.sha256_file(config["checkpoint"]))
    protocol_id, parts = C.protocol_id(config, args.config, args.cache, args.warm,
                                       args.modes, args.seeds, args.updates,
                                       smoke_report=args.smoke_report,
                                       final_scenes=final_scenes,
                                       primary_contrast=primary_contrast,
                                       compute=compute, references=references,
                                       eval_noise=eval_noise, kind=args.kind)
    protocol = dict(
        protocol_id=protocol_id, schema_version=C.SCHEMA_VERSION,
        metric_subset=dict(train=len(metric_train),
                           validation=len(metric_validation),
                           metric_scenes=metric_scenes,
                           available=verified["n_contexts"],
                           n_metric_labelled=verified["n_metric"]),
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
        eval_noise=eval_noise,
        lock_support=scale_stats,
        compute=compute,
        smoke_report=args.smoke_report,
        smoke_report_sha256=C.sha256_file(args.smoke_report),
        resolved_config=config,
        experiment_kind=args.kind,
        min_metric_fraction=float(args.min_metric_fraction),
        min_metric_scenes=int(args.min_metric_scenes),
        min_support=int(args.min_support),
        final_prefix=args.final_prefix,
        test_episodes_planned=len(final_scenes),
        no_final_results_examined=True)
    C.write_meta(args.output, protocol)
    print("locked protocol " + args.output + " id " + protocol_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
