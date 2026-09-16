"""Immutable principal lock.

Builds ONE canonical design object via contract.canonical_design: exact
scene lists, arms, seeds, primary contrast [gad, augmented],
teacher/reference roles, source/solver conventions, latent keys,
correction targets + budgets, compute, hyperparams (beta_selected +
rho), artifact + code hashes. The design id is recomputed on every
load by verify_current_protocol. Requires a passing readiness report
(gate) plus the base-student floor (augmented dev success >= 0.30).
Refuses if final outcomes already exist under final_prefix. Tags
experiment_kind.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C

PRIMARY_CONTRAST = ["gad", "augmented"]
MIN_AUGMENTED_DEV_SUCCESS = 0.30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-no-final-results", action="store_true",
                        required=True)
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--output", default="runs/protocol_locked.json")
    parser.add_argument("--readiness-report", required=True)
    parser.add_argument("--base-report", default=None,
                        help="base-student dev report for the 0.30 floor")
    parser.add_argument("--warm", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--final-prefix", default="runs/final_")
    parser.add_argument("--kind", default="latent_controllability_principal",
                        help="experiment_kind tag")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--beta-selected", type=float, required=True)
    parser.add_argument("--rho", type=float, default=None)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    C.validate_config(config, args.config)
    arms = sorted(args.arms or list(PRIMARY_CONTRAST))
    if PRIMARY_CONTRAST[0] not in arms or PRIMARY_CONTRAST[1] not in arms:
        raise ValueError(f"arms must include primary contrast {PRIMARY_CONTRAST}")
    seeds = sorted(int(s) for s in (args.seeds if args.seeds else [0, 1]))
    updates = int(args.updates or config["updates"])
    rho = float(args.rho if args.rho is not None else config["rho"])
    if rho != float(config["rho"]):
        raise ValueError("locked rho must equal the bank rho in config")

    C.verify_completed(args.readiness_report)
    with open(args.readiness_report) as handle:
        readiness = json.load(handle)
    if not readiness.get("passed"):
        raise ValueError("readiness report is not a pass; fix the base stack")
    if readiness.get("source_hashes") != C.source_hashes():
        raise ValueError("readiness report came from different sources; rerun")
    if readiness.get("config_sha256") != C.sha256_file(args.config):
        raise ValueError("readiness report was built for a different config")
    locked_conv = [(config["canonical_source"], int(config["teacher_steps"]))]
    rows = [c for c in readiness.get("conventions", [])
            if (c.get("source"), c.get("steps")) in locked_conv]
    if not any(c.get("success", 0.0) >= 0.50 or c.get("score", 0.0) >= 0.65
               for c in rows):
        raise ValueError("no passing readiness row for the locked "
                         "source/solver convention (need success >= 0.50 "
                         "or score >= 0.65); do not train students")

    if args.base_report:
        C.verify_completed(args.base_report)
        with open(args.base_report) as handle:
            base = json.load(handle)
        aug = base.get("augmented_dev_success")
        if aug is None or float(aug) < MIN_AUGMENTED_DEV_SUCCESS:
            raise ValueError(
                f"base-student floor failed: augmented dev success {aug} < "
                f"{MIN_AUGMENTED_DEV_SUCCESS}")

    for path, role in ((args.warm, None), (args.bank, None),
                       (args.history, None)):
        C.verify_completed(path)
    bank = C.verify_pair_bank(args.bank, config)
    C.verify_history_cache(args.history, config)

    if os.path.exists(args.output):
        raise ValueError("protocol already locked: " + args.output)
    planned = ([f"{args.final_prefix}{arm}_seed{seed}.jsonl"
                for arm in arms for seed in seeds]
               + [f"{args.final_prefix}teacher.jsonl",
                  f"{args.final_prefix}corrections.jsonl"])
    for path in planned:
        if os.path.exists(path):
            raise ValueError("final outcomes already exist: " + path)

    scenes = {k: sorted(C.scene_range(config, k))
              for k in ("development", "train", "validation", "diagnostic",
                        "final")}
    correction = dict(config["correction"], primary_K=10)
    references = {
        "teacher": dict(method="teacher", role="teacher_reference",
                        source=config["canonical_source"],
                        steps=int(config["teacher_steps"]),
                        checkpoint_sha256=C.sha256_file(config["checkpoint"]),
                        replicate=0, scenes=scenes["final"]),
    }
    compute = C.compute_env(config.get("device", "cuda"))
    design = C.canonical_design(
        config, arms, seeds, updates, arms, scenes, correction,
        float(args.beta_selected), rho, compute, references,
        warm=args.warm, bank=args.bank, cache=args.history)
    protocol = dict(
        protocol_id=C.design_id(design), design=design,
        sources=C.source_hashes(), config_file=C.sha256_file(args.config),
        assets=C.asset_hashes(config), packages=C.package_versions(),
        artifacts={"warm": args.warm, "bank": args.bank,
                   "history": args.history,
                   "readiness_report": args.readiness_report},
        readiness_report=args.readiness_report,
        experiment_kind=args.kind, final_prefix=args.final_prefix,
        beta_selected=float(args.beta_selected),
        schema_version=C.SCHEMA_VERSION)
    C.reserve_outputs([args.output])
    C.write_meta(args.output, protocol)
    C.complete_output(args.output, dict(protocol_id=protocol["protocol_id"]))
    print("locked protocol " + protocol["protocol_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
