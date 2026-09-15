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
    if len(train_rows) < int(config["metric_contexts"]) * 0.5:
        raise ValueError(f"metric cache holds {len(train_rows)} training contexts, "
                         f"far below the planned {config['metric_contexts']}")
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
    C.validate_scene_sets({"train": sorted(C.scene_range(config, "train")),
                           "validation": sorted(C.scene_range(config, "validation")),
                           "development": sorted(C.scene_range(config, "development")),
                           "test": final_scenes})
    if args.output and os.path.exists(args.output):
        raise ValueError("protocol already locked: " + args.output)
    for mode in args.modes:
        for seed in args.seeds:
            path = f"runs/final_{mode}_seed{seed}.jsonl"
            if os.path.exists(path):
                raise ValueError("final outcomes already exist: " + path)

    protocol_id, parts = C.protocol_id(config, args.config, args.cache, args.warm,
                                       args.modes, args.seeds, args.updates,
                                       smoke_report=args.smoke_report)
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
        primary_contrast=["pullback", "endpoint"],
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
        smoke_report=args.smoke_report,
        smoke_report_sha256=C.sha256_file(args.smoke_report),
        resolved_config=config,
        no_final_results_examined=True)
    C.write_meta(args.output, protocol)
    print("locked protocol " + args.output + " id " + protocol_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
