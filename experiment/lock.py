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
    parser.add_argument("--base-report", required=True,
                        help="development evaluation file (jsonl) of the "
                             "augmented base student for the 0.30 floor")
    parser.add_argument("--base-checkpoint", required=True,
                        help="the base (augmented) student checkpoint the "
                             "base report evaluates: rows are compared "
                             "against THIS hash, never the teacher "
                             "checkpoint (audit 1344ed0 item 1)")
    parser.add_argument("--eval-cases", required=True,
                        help="evaluation-case manifest file (jsonl): locks "
                             "planned case IDs + scene/context/target "
                             "identity for analysis completeness "
                             "(audit 1344ed0 item 4)")
    parser.add_argument("--lr-selection", required=True,
                        help="completed lr-selection artifact from a "
                             "tune-only correction run: hash and settings "
                             "are locked, final runs consume it without "
                             "retuning (audit 1344ed0 item 6)")
    parser.add_argument("--eval-replicates", nargs="*", type=int,
                        default=None,
                        help="planned unassisted-evaluation replicates "
                             "(default [0])")
    parser.add_argument("--warm", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--history", required=True,
                        help="warm-student history cache backing the pair "
                             "bank (warm_checkpoint -> warm_history -> bank)")
    parser.add_argument("--teacher-history", required=True,
                        help="teacher history cache the warm checkpoint was "
                             "trained on (teacher_history -> warm_checkpoint)")
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
    # Audit 1344ed0 item 7: score-only (diagnostic_pilot) acceptance
    # never authorizes the principal success-rate experiment.
    if readiness.get("acceptance_mode",
                     readiness.get("gate", {}).get("acceptance_mode")) != "principal":
        raise ValueError(
            "readiness acceptance is not principal "
            f"({readiness.get('acceptance_mode')}); a score-only pass "
            "authorizes a diagnostic pilot, not the principal lock")
    if readiness.get("source_hashes") != C.source_hashes():
        raise ValueError("readiness report came from different sources; rerun")
    if readiness.get("config_sha256") != C.sha256_file(args.config):
        raise ValueError("readiness report was built for a different config")
    locked_conv = [(config["canonical_source"], int(config["teacher_steps"]))]
    gate = C.readiness_gate(readiness.get("conventions"),
                            config["canonical_source"],
                            int(config["teacher_steps"]))
    if not gate["passed"]:
        raise ValueError(f"readiness gate failed for the locked convention: "
                         f"{gate}; do not train students")
    print(f"[lock] teacher readiness: {gate}")

    C.verify_completed(args.base_report)
    base_sidecar = (args.base_report.replace(".jsonl", ".meta.json")
                    if args.base_report.endswith(".jsonl")
                    else args.base_report + ".meta.json")
    if not os.path.exists(base_sidecar):
        raise ValueError("base report has no sidecar; rebuild the evaluation")
    with open(base_sidecar) as handle:
        base_meta = json.load(handle)
    if base_meta.get("schema_version") != C.SCHEMA_VERSION:
        raise ValueError("base report schema mismatch; rebuild the evaluation")
    base_rows = [json.loads(line) for line in open(args.base_report)
                 if line.strip()]
    aug_rows = [r for r in base_rows
                if r.get("method") == "augmented"
                and r.get("split") == "development"]
    if not aug_rows:
        raise ValueError("base report holds no development augmented rows")
    # Audit fdb59ff item 13: the base report must cover the COMPLETE
    # development set with the locked checkpoint/assets, not just any
    # matching rows.
    dev_scenes = set(sorted(C.scene_range(config, "development")))
    if {int(r.get("scene", -1)) for r in aug_rows} != dev_scenes:
        raise ValueError(
            "base report does not cover the complete development set: "
            f"{sorted({int(r.get('scene', -1)) for r in aug_rows})[:5]}")
    # Audit 1344ed0 item 1: the base gate assesses the BASE student
    # checkpoint, whose hash the report rows must carry. Comparing
    # against the teacher checkpoint rejects every legitimate
    # base-student report.
    C.verify_completed(args.base_checkpoint)
    import torch as _torch_base
    base_payload = _torch_base.load(args.base_checkpoint, map_location="cpu",
                                    weights_only=True)
    if base_payload.get("arm") != "augmented":
        raise ValueError(
            f"base checkpoint arm is {base_payload.get('arm')!r}; the base "
            "gate assesses the augmented base student")
    base_hash = C.sha256_file(args.base_checkpoint)
    for row in aug_rows:
        if row.get("checkpoint_sha256") != base_hash:
            raise ValueError(
                "base report row checkpoint differs from --base-checkpoint; "
                "the report must evaluate the checkpoint being gated, "
                "not the teacher or another student")
        if int(row.get("seed", row.get("training_seed", -1))) != int(
                base_payload.get("seed", -999)):
            raise ValueError(
                "base report row seed differs from the base checkpoint "
                "seed; refusing a mixed-seed base gate")
    aug = sum(int(r.get("success", 0)) for r in aug_rows) / len(aug_rows)
    if float(aug) < MIN_AUGMENTED_DEV_SUCCESS:
        raise ValueError(
            f"base-student floor failed: augmented dev success {aug:.3f} < "
            f"{MIN_AUGMENTED_DEV_SUCCESS}")
    print(f"[lock] base floor: augmented dev success {aug:.3f} "
          f"over {len(aug_rows)} episodes")

    # Audit fdb59ff item 9: the dependency graph is explicit:
    # teacher_history -> warm_checkpoint -> warm_history -> pair_bank
    # -> final_student. One generic history hash cannot represent both
    # histories; each edge is verified.
    for path in (args.warm, args.bank, args.history, args.teacher_history):
        C.verify_completed(path)
    teacher_hist = C.verify_history_cache(args.teacher_history, config)
    warm_hist = C.verify_history_cache(args.history, config)
    bank = C.verify_pair_bank(args.bank, config)
    # Audit fdb59ff item 10: lineage/member consistency, not just a
    # checksum: member_keys must equal the record pair_ids exactly.
    member_keys = {tuple(k) for k in
                   (bank.get("meta", {}).get("member_keys", []))}
    record_keys = {tuple(r.get("pair_id")) for r in bank.get("records", [])}
    if member_keys != record_keys:
        raise ValueError(
            "pair-bank lineage/member mismatch: "
            f"{len(member_keys)} member keys vs {len(record_keys)} records")
    # Audit 1344ed0 item 6: the lr selection is a completed,
    # hash-locked artifact. Its tune scenes must be disjoint from the
    # evaluation scenes (checked by scene AND context, not split
    # strings alone).
    C.verify_completed(args.lr_selection)
    with open(args.lr_selection) as handle:
        lr_sel = json.load(handle)
    if lr_sel.get("kind") != "lr_selection":
        raise ValueError("lr-selection artifact has wrong kind")
    locked_lr = float(lr_sel["lr"])
    lr_tune_scenes = {int(s) for s in lr_sel.get("tune_scenes", [])}
    lr_tune_contexts = {str(c) for c in lr_sel.get("tune_contexts", [])}
    # Audit 1344ed0 item 4: lock the evaluation-case manifest. Every
    # planned case_id maps to its scene/context/target identity plus a
    # data fingerprint; analysis derives expected tuples from here,
    # never from observed rows.
    # NOTE: the eval-cases file is a hand-written input manifest, not
    # a pipeline artifact, so it carries no completion sidecar; its
    # sha256 is recorded in the design instead.
    eval_cases_sha256 = C.sha256_file(args.eval_cases)
    eval_manifest = {}
    with open(args.eval_cases) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("unreachable"):
                continue
            for key in ("case_id", "context_id", "target_id", "scene",
                        "q0", "target", "condition"):
                if key not in raw:
                    raise ValueError(
                        f"eval manifest case missing {key}: "
                        f"{raw.get('case_id')}")
            case_id = str(raw["case_id"])
            q0, target, cond = C.normalize_case_arrays(
                raw["q0"], raw["target"], raw["condition"], case_id)
            entry = dict(scene=int(raw["scene"]),
                         context_id=str(raw["context_id"]),
                         target_id=str(raw["target_id"]),
                         data_sha256=C.case_data_sha256(q0, target, cond))
            if case_id in eval_manifest and eval_manifest[case_id] != entry:
                raise ValueError(
                    f"eval manifest case_id {case_id} maps to two "
                    "different identities; case IDs must be unique")
            eval_manifest[case_id] = entry
    if not eval_manifest:
        raise ValueError("eval manifest holds no reachable cases")
    eval_scenes = {e["scene"] for e in eval_manifest.values()}
    eval_contexts = {e["context_id"] for e in eval_manifest.values()}
    if lr_tune_scenes & eval_scenes:
        raise ValueError(
            "tune/eval scene overlap: "
            f"{sorted(lr_tune_scenes & eval_scenes)[:5]}; tune and "
            "evaluation must be disjoint by scene")
    if lr_tune_contexts & eval_contexts:
        raise ValueError(
            "tune/eval context overlap: "
            f"{sorted(lr_tune_contexts & eval_contexts)[:5]}")
    eval_replicates = sorted(int(r) for r in (
        args.eval_replicates if args.eval_replicates else [0]))
    import torch as _torch_lock
    warm_payload = _torch_lock.load(args.warm, map_location="cpu",
                                    weights_only=True)
    teacher_hash = C.sha256_file(args.teacher_history)
    if warm_payload.get("history_sha256") != teacher_hash:
        raise ValueError(
            "warm checkpoint was not trained on the locked teacher "
            "history "
            f"({str(warm_payload.get('history_sha256'))[:16]} != "
            f"{teacher_hash[:16]})")
    bank_parent = ((bank.get("lineage") or {}).get("parent_history_sha256"))
    if bank_parent != C.sha256_file(args.history):
        raise ValueError(
            "pair bank was not built from the locked warm history; "
            "rebuild the bank from --history")
    selected = set((bank.get("lineage") or {}).get("selected_histories", []))
    if selected:
        known = {str(h.get("history_id")) for h in warm_hist["histories"]}
        if not selected <= known:
            raise ValueError(
                "pair bank selected histories outside the locked warm "
                "history: "
                f"{sorted(selected - known)[:5]}")

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
    # Item 19: lock was calling canonical_design with a single dict
    # (TypeError: missing required args), so no principal lock could ever
    # be created, and the primary contrast was never persisted.
    design = C.canonical_design(
        config, arms, seeds, updates, arms, scenes, correction,
        float(args.beta_selected), rho, compute, references,
        warm=args.warm, bank=args.bank, cache=args.history,
        primary_contrast=PRIMARY_CONTRAST)
    design["teacher_history_sha256"] = teacher_hash
    # Audit 1344ed0 item 2: the warm init seed has its own design
    # field, checked by the init role; final seeds are checked
    # separately and never conflated with it.
    design["warm_seed"] = int(warm_payload.get("seed", -1))
    design["base_checkpoint_sha256"] = base_hash
    design["base_checkpoint"] = dict(arm=base_payload.get("arm"),
                                     seed=int(base_payload.get("seed", -1)))
    design["lr_selection"] = dict(sha256=C.sha256_file(args.lr_selection),
                                  lr=locked_lr,
                                  tune_scenes=sorted(lr_tune_scenes),
                                  tune_contexts=sorted(lr_tune_contexts))
    design["eval_cases"] = {k: eval_manifest[k]
                            for k in sorted(eval_manifest)}
    design["eval_cases_sha256"] = eval_cases_sha256
    design["eval_replicates"] = list(eval_replicates)
    design["design_id"] = C.design_id(design)
    protocol = dict(
        protocol_id=C.design_id(design), design=design,
        upstream_commit=C.upstream_commit(config.get("repository")),
        sources=C.source_hashes(), config_file=C.sha256_file(args.config),
        assets=C.asset_hashes(config), packages=C.package_versions(),
        artifacts={"warm": args.warm, "bank": args.bank,
                   "history": args.history,
                   "teacher_history": args.teacher_history,
                   "base_checkpoint": args.base_checkpoint,
                   "lr_selection": args.lr_selection,
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
