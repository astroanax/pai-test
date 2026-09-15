import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-no-final-results", action="store_true", required=True)
    parser.add_argument("--modes", nargs="*",
                        default=["uniform", "prefix", "endpoint", "pullback",
                                 "identity", "scalar"])
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--repository", default=None)
    parser.add_argument("--cache", default="runs/shared_metrics.npz")
    parser.add_argument("--warm", default="runs/warm.pt")
    parser.add_argument("--smoke-report", default="runs/smoke_report.json")
    parser.add_argument("--output", default="runs/protocol_locked.json")
    args = parser.parse_args()

    with open(args.config) as handle:
        config = json.load(handle)
    C.validate_config(config, args.config)
    repository = args.repository or config["repository"]
    if args.updates is None:
        args.updates = config["updates"]

    required = [args.config, args.cache, args.warm, config["checkpoint"],
                config["normalizer"]]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise ValueError("missing required artifacts: " + ", ".join(missing))

    if not os.path.exists(args.smoke_report):
        raise ValueError("target-stack smoke report missing: run "
                         "experiment/smoke.py before locking the protocol")
    with open(args.smoke_report) as handle:
        smoke = json.load(handle)
    if not smoke.get("passed"):
        raise ValueError("smoke report is not a pass: " + json.dumps(
            {k: v for k, v in smoke.items() if k in ("failed", "checks")}))
    if smoke.get("source_hashes") != C.source_hashes():
        raise ValueError("smoke report was produced from different experiment "
                         "sources; rerun experiment/smoke.py")
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
    for name in sorted(glob_jsonl("runs/final_")):
        raise ValueError("final outcomes already exist (" + name +
                         "); refusing to lock or rerun")

    for mode in args.modes:
        for seed in args.seeds:
            path = f"runs/student_{mode}_seed{seed}.pt"
            if os.path.exists(path):
                raise ValueError("training output already exists: " + path)
    C.reserve_outputs([args.output])

    protocol_id, parts = C.protocol_id(config, args.config, args.cache,
                                       config["checkpoint"],
                                       config["normalizer"], repository)
    protocol = dict(
        protocol_id=protocol_id,
        experiment="execution-pullback pilot",
        mode="single shared warm start, two fine-tuning seeds",
        modes=args.modes, seeds=[int(s) for s in args.seeds],
        updates=int(args.updates), warm_start=args.warm,
        warm_start_sha256=C.sha256_file(args.warm),
        cache=args.cache, cache_sha256=C.sha256_file(args.cache),
        primary_contrast=["pullback", "endpoint"],
        final_scenes=final_scenes,
        development_scenes=sorted(C.scene_range(config, "development")),
        train_scenes=sorted(C.scene_range(config, "train")),
        validation_scenes=sorted(C.scene_range(config, "validation")),
        checkpoint_sha256=C.sha256_file(config["checkpoint"]),
        normalizer_sha256=C.sha256_file(config["normalizer"]),
        source_hashes=C.source_hashes(),
        upstream=parts["upstream"], upstream_commit=parts["upstream_commit"],
        packages=C.package_versions(),
        metric_mode=config.get("metric_mode", "exact"),
        teacher_steps=int(config["teacher_steps"]),
        source_distribution=config["source"],
        smoke_report_sha256=C.sha256_file(args.smoke_report),
        resolved_config=config,
        no_final_results_examined=True)
    C.write_meta(args.output, protocol)
    print("locked protocol " + args.output + " id " + protocol_id)


def glob_jsonl(prefix):
    directory = os.path.dirname(prefix) or "."
    if not os.path.isdir(directory):
        return []
    base = os.path.basename(prefix)
    return [os.path.join(directory, name) for name in sorted(os.listdir(directory))
            if name.startswith(base)]


if __name__ == "__main__":
    main()
