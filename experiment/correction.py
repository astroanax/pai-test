"""E2 bounded correction + edit transfer.

Objective per spec Eq.12: prefix MSE/md + gamma*||q-q0||^2/D with RMS trust
radius r=0.25. Uses gad_reference.invert_prefix (frozen weights, best
feasible iterate including q0, forward/backward counts, synchronized
latency). Validates a single lr from {0.03, 0.1, 0.3} once, then freezes it.
K in {0, 1, 5, 10, 20}, primary K=10.

Two tests:
  1. own-model inversion per model at matched budgets;
  2. teacher-edit transfer: optimize once under T, apply the same q*_T to
     all students with no student-side optimization.

Baselines: no-correction, bounded random search over a SHARED candidate
set, oracle-prefix execution. Targets: independent oracle prefixes (origin
recorded) preferred, else teacher-reachable a* = Pm T(q*, c) with a
predeclared filter + rejection rate. Unreachable targets are reported,
never silently dropped. Per-case rows carry context/target/model IDs, K,
errors, displacement, invalid counts, call counts, latency. Writes
corrections.jsonl + meta.

Cases file (JSONL, one per line): {case_id, context_id, target_id,
target_origin ("oracle" | "teacher_reachable"), q0 [[16,2]],
target_prefix [[P,2]], condition [514], reachable bool}.
Models: --students JSON {name: checkpoint} plus the frozen teacher from
the config checkpoint.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contract as C
from gad_reference import invert_prefix

K_GRID = (0, 1, 5, 10, 20)
PRIMARY_K = 10
LR_CANDIDATES = (0.03, 0.1, 0.3)
TRUST_R = 0.25
GAMMA = 0.01


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def load_student_model(path, device):
    from gad_reference import make_student
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "student" not in payload:
        raise ValueError("checkpoint lacks a student state dict: " + path)
    arch = payload.get("arch", {})
    model = make_student(514, 16, 2, int(arch.get("width", 512)))
    model.load_state_dict(payload["student"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class TeacherModel(torch.nn.Module):
    """Frozen-teacher endpoint map as a parameter-free module.

    Item 7: the forward path must stay differentiable in the LATENT.
    teacher_action wraps everything in no_grad and detaches, which
    kills the task gradient (only the regularizer's zero-at-init
    gradient survives, silently freezing teacher corrections). Here
    parameters are frozen (there are none to train) while latent
    derivatives flow through teacher_endpoint.
    """

    def __init__(self, adapter, steps, source):
        super().__init__()
        self.adapter = adapter
        self.steps = int(steps)
        self.source = str(source)

    def forward(self, latent, condition):
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__))))
        from gad_reference import teacher_endpoint
        return teacher_endpoint(self.adapter.field, latent, condition,
                                self.steps, source=self.source)


def build_teacher(config, device):
    sys.path.insert(0, os.path.join(config["repository"], "external", "models"))
    from hri_adapter import HRIAdapter, load_checkpoint, resolve_device
    device = resolve_device(str(device))
    stats = {key: {inner: np.asarray(val) for inner, val in section.items()}
             for key, section in
             np.load(config["normalizer"], allow_pickle=True)["stats"].item().items()}
    vision_encoder, noise_pred_net = load_checkpoint(config["checkpoint"], device)
    adapter = HRIAdapter(vision_encoder, noise_pred_net, stats, device,
                         legacy=config["legacy"],
                         clip_actions=config.get("clip_actions", False))
    return TeacherModel(adapter, config["teacher_steps"],
                        config["canonical_source"]).to(device).eval()


def as_tensor(array, device, name):
    t = torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)
    if not torch.isfinite(t).all():
        raise ValueError("nonfinite " + name)
    return t


def invalid_prefix_count(action_prefix):
    a = np.asarray(action_prefix, dtype=np.float64)
    return int(((a < 0.0) | (a > 512.0)).sum())


def validate_lr(models, pilot_cases, device, lrs=LR_CANDIDATES,
                gamma=GAMMA, trust_r=TRUST_R):
    """Pick one lr on the TUNE split; the caller freezes it."""
    scores = {}
    for lr in lrs:
        errs = []
        for case in pilot_cases:
            model = models[case["model"]]
            out = invert_prefix(model, case["q0_t"], case["cond_t"],
                                case["target_t"], steps=PRIMARY_K,
                                learning_rate=lr, trust_rms=trust_r,
                                gamma=gamma)
            errs.append(float(out["prefix_mse"].mean().item()))
        scores[lr] = float(np.mean(errs))
    best = min(scores, key=scores.get)
    return best, scores


def random_search_offsets(rng, shape, n, trust_r=TRUST_R):
    """SHARED offset set (fixed seeds); each case adds them to ITS q0.

    Item 8: candidates were centered at zero regardless of the case, so
    the zero candidate sat at RMS displacement 1.0 against a 0.25 trust
    ball. Offsets are bounded by construction; per-case candidates are
    q0 + offset, all inside the trust ball.
    """
    shape = tuple(shape)
    size = int(np.prod(shape))
    scale = trust_r * np.sqrt(size)
    offsets = [np.zeros(shape)]
    for _ in range(n - 1):
        step = rng.normal(size=shape)
        norm = float(np.linalg.norm(step.ravel()))
        offsets.append(step * (scale / max(norm, 1e-12)))
    return offsets


def model_prefix_mse(model, latent_np, cond_t, target_np, device):
    """Action-space prefix MSE of a latent (denormalized validity below)."""
    with torch.no_grad():
        action = model(as_tensor(latent_np, device, "candidate"),
                       cond_t).cpu().numpy()
    width = target_np.shape[1]
    return (float(np.mean((action[:, :width] - target_np) ** 2)), action)


def invalid_action_count(normalized_prefix, stats):
    """Item 8: physical validity is judged on DENORMALIZED actions."""
    from hri_adapter import decode_actions
    physical = np.asarray(decode_actions(normalized_prefix, stats),
                          dtype=np.float64)
    return int(((physical < 0.0) | (physical > 512.0)).sum())


def correct_case(case, models, teacher_outs, lr, shared_offsets, device,
                 protocol_id, stats, gamma=GAMMA, trust_r=TRUST_R,
                 budgets=None):
    # Item 8: no-correction evaluates model(q0); random search adds SHARED
    # offsets to this case's q0 (all inside the trust ball); validity is
    # judged on denormalized actions; oracle displacement is action-space.
    # Item 11: teacher edits are computed ONCE per (case, K) by the caller
    # (teacher_outs) and reused across students; transfer/random-search
    # rows measure their real forward costs.
    rows = []
    model = models[case["model"]]
    q0_t = case["q0_t"]
    cond_t = case["cond_t"]
    target_t = case["target_t"]
    target_np = case["target_np"]
    base = dict(case_id=case["case_id"], context_id=case["context_id"],
                target_id=case["target_id"],
                target_origin=case["target_origin"],
                scene=case["scene"], seed=case["seed"],
                model=case["model"], lr=lr, gamma=gamma, trust_r=trust_r,
                protocol_id=protocol_id,
                case_data_sha256=case.get("data_sha256"))
    err0, act0 = model_prefix_mse(model, case["q0_np"], cond_t, target_np,
                                  device)
    # Audit fdb59ff item 12: one budget list everywhere. Configured
    # budgets (not the module K_GRID constant) drive student rows, so
    # nondefault settings cannot produce missing-key failures or
    # incorrect metadata.
    grid = tuple(budgets) if budgets is not None else K_GRID
    for K in grid:
        row_base = dict(base, K=K)
        rows.append(dict(row_base, method="no_correction", mse=err0,
                         displacement=0.0, displacement_kind="action",
                         invalid=invalid_action_count(
                             act0[:, :target_np.shape[1]], stats),
                         forwards=1, backwards=0, latency_ms=0.0))
        start = time.perf_counter()
        sync = (lambda: torch.cuda.synchronize()) \
            if cond_t.device.type == "cuda" else (lambda: None)
        out = invert_prefix(model, q0_t, cond_t, target_t, steps=K,
                            learning_rate=lr, trust_rms=trust_r, gamma=gamma)
        sync()
        latency = (time.perf_counter() - start) * 1000.0
        best_action = out["action"].cpu().numpy()
        rows.append(dict(row_base, method="own_model",
                         mse=float(out["prefix_mse"].mean().item()),
                         displacement=float(
                             np.sqrt(np.mean((out["latent"] - q0_t
                                              ).cpu().numpy() ** 2))),
                         displacement_kind="latent_rms",
                         invalid=invalid_action_count(
                             best_action[:, :target_np.shape[1]], stats),
                         forwards=int(out["forward_calls"]),
                         backwards=int(out["backward_calls"]),
                         latency_ms=latency))
        t_out = teacher_outs[(case["context_id"], case["target_id"], K)]
        transferred = t_out["latent"].cpu().numpy()
        sync()
        _t0 = time.perf_counter()
        transfer_mse, transfer_act = model_prefix_mse(
            model, transferred, cond_t, target_np, device)
        sync()
        transfer_latency = (time.perf_counter() - _t0) * 1000.0
        # Audit 1344ed0 item 6: three costs stay separate. The
        # transfer row reports the STUDENT operation (one forward to
        # score the transferred latent + its measured latency); the
        # teacher-edit preparation cost rides along as informational
        # fields, never as the row's own forward count.
        rows.append(dict(row_base, method="teacher_transfer",
                         mse=transfer_mse,
                         displacement=float(
                             np.sqrt(np.mean((t_out["latent"] - q0_t
                                              ).cpu().numpy() ** 2))),
                         displacement_kind="latent_rms",
                         invalid=invalid_action_count(
                             transfer_act[:, :target_np.shape[1]], stats),
                         forwards=1,
                         backwards=0, latency_ms=transfer_latency,
                         teacher_edit_forwards=int(
                             t_out["forward_calls"]),
                         teacher_edit_backwards=int(
                             t_out["backward_calls"])))
        best_mse, best_disp, best_invalid = err0, 0.0, rows[-3]["invalid"]
        sync()
        _r0 = time.perf_counter()
        for offset in shared_offsets:
            cand = case["q0_np"] + offset
            mse, act = model_prefix_mse(model, cand, cond_t, target_np,
                                        device)
            if mse < best_mse:
                best_mse = mse
                best_disp = float(np.linalg.norm(offset.ravel())
                                  / np.sqrt(offset.size))
                best_invalid = invalid_action_count(
                    act[:, :target_np.shape[1]], stats)
        sync()
        search_latency = (time.perf_counter() - _r0) * 1000.0
        rows.append(dict(row_base, method="random_search",
                         mse=best_mse, displacement=best_disp,
                         displacement_kind="latent_rms",
                         invalid=best_invalid,
                         forwards=len(shared_offsets), backwards=0,
                         latency_ms=search_latency))
        # Audit fdb59ff item 12: oracle displacement is action-space
        # (target prefix vs the no-correction baseline action prefix).
        # The old code subtracted a latent q0 prefix from an action
        # target: mismatched spaces.
        oracle_disp = float(np.linalg.norm(
            (target_np - act0[:, :target_np.shape[1]]
             ).ravel()) / np.sqrt(target_np.size))
        rows.append(dict(row_base, method="oracle_prefix",
                         mse=0.0,
                         displacement=oracle_disp,
                         displacement_kind="action",
                         invalid=0, forwards=0, backwards=0, latency_ms=0.0))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--allow-unlocked", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--students", required=True,
                        help="JSON {model_name: checkpoint_path}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-selection", default=None,
                        help="completed lr-selection artifact to consume "
                             "without retuning (final runs)")
    parser.add_argument("--save-lr-selection", default=None,
                        help="write the tune-split lr selection as a "
                             "completed artifact for the lock")
    args = parser.parse_args()
    config = load_config(args.config)
    protocol_id = None
    protocol = None
    if args.protocol:
        protocol = C.load_protocol(args.protocol)
        C.verify_current_protocol(protocol, config, args.config,
                                  stage="correction")
        protocol_id = protocol.get("protocol_id")
    from hri_adapter import resolve_device
    device = resolve_device(args.device or config["device"])
    with open(args.cases) as handle:
        raw_cases = [json.loads(line) for line in handle if line.strip()]
    # Audit fdb59ff item 4: validate + normalize cases BEFORE loading
    # any model. Shapes accepted unbatched or batched and normalized
    # to q0 [1,16,2], target [1,P,2], condition [1,514]. split and
    # checkpoint identity are retained (never dropped).

    staged = []
    for raw in raw_cases:
        if raw.get("unreachable"):
            for key in ("case_id", "context_id", "target_id", "model"):
                if key not in raw:
                    raise C.ProtocolError(
                        f"unreachable case missing {key}")
            staged.append(dict(raw, unreachable=True,
                               split=str(raw.get("split", "eval"))))
            continue
        missing = [k for k in ("case_id", "context_id", "target_id",
                               "scene", "seed", "model", "checkpoint",
                               "target_origin", "q0", "target", "condition")
                   if k not in raw]
        if missing:
            raise C.ProtocolError(f"correction case missing {missing}")
        case_id = str(raw["case_id"])
        # Shared normalizer: identical arrays (and data_sha256) to the
        # lock's eval manifest for the same inputs.
        q0, target, cond = C.normalize_case_arrays(
            raw["q0"], raw["target"], raw["condition"], case_id)
        checkpoint = str(raw["checkpoint"])
        if not os.path.exists(checkpoint):
            raise C.ProtocolError(
                f"case {case_id}: checkpoint missing: {checkpoint}")
        staged.append(dict(
            case_id=case_id, context_id=str(raw["context_id"]),
            target_id=str(raw["target_id"]),
            target_origin=str(raw["target_origin"]),
            split=str(raw.get("split", "eval")),
            scene=int(raw["scene"]), seed=int(raw["seed"]),
            model=str(raw["model"]), checkpoint=checkpoint,
            checkpoint_sha256=C.sha256_file(checkpoint),
            data_sha256=C.case_data_sha256(q0, target, cond),
            q0_np=q0, target_np=target, cond_np=cond,
            q0_t=as_tensor(q0, device, "q0"),
            cond_t=as_tensor(cond, device, "condition"),
            target_t=as_tensor(target, device, "target")))
    with open(args.students) as handle:
        student_paths = json.load(handle)
    # protocol/protocol_id were loaded at main entry; reuse them here.
    # Audit 1344ed0 item 6: a correctly NAMED checkpoint must not enter
    # the wrong comparison. Every loaded model is provenance-checked
    # (arm, training seed, warm init, pair bank, updates, beta,
    # protocol identity) before it corrects anything.
    student_payloads = {}
    for name, path in student_paths.items():
        payload = torch.load(path, map_location=device, weights_only=True)
        student_payloads[name] = payload
        if protocol is not None:
            C.verify_student_provenance(payload, protocol, config, path,
                                        role="final")
    models = {name: load_student_model(path, device)
              for name, path in student_paths.items()}
    teacher = build_teacher(config, device)
    cases = []
    for case in staged:
        if case.get("unreachable"):
            cases.append(case)
            continue
        if case["model"] not in models:
            raise ValueError(
                f"case model {case['model']!r} has no checkpoint")
        if case["checkpoint"] != student_paths.get(case["model"]):
            raise C.ProtocolError(
                f"case {case['case_id']}: checkpoint "
                f"{case['checkpoint']!r} does not match the loaded "
                f"model file for {case['model']!r}")
        cases.append(case)
    # Item 25: reserve BOTH outputs before any work; on failure mark
    # both, so no dangling meta survives a failed run.
    meta_path = args.output.replace(".jsonl", ".meta.json")
    C.reserve_outputs([args.output, meta_path])
    # Item 10: lr tuned on SEPARATE validation cases, selection saved and
    # locked; final evaluation consumes it without retuning. Config/lock
    # values win over module constants.
    gamma = float(config["correction"].get("gamma", GAMMA))
    trust_r = float(config["correction"].get("trust_rms", TRUST_R))
    lrs = tuple(config["correction"].get("lrs", LR_CANDIDATES))
    budgets = tuple(config["correction"].get("budgets", K_GRID))
    if args.lr is not None and args.lr not in lrs:
        raise C.ProtocolError(f"lr {args.lr} not in {lrs}")
    lr = args.lr
    tune_cases = [c for c in cases
                  if not c.get("unreachable") and c.get("split") == "tune"]
    # Eval cases exclude tune: teacher-edit precomputation and row
    # production never touch tuning cases.
    eval_cases = [c for c in cases
                  if not c.get("unreachable") and c.get("split") != "tune"]
    tune_scenes = {c["scene"] for c in tune_cases}
    tune_contexts = {c["context_id"] for c in tune_cases}
    eval_scenes = {c["scene"] for c in eval_cases}
    eval_contexts = {c["context_id"] for c in eval_cases}
    if tune_cases and (tune_scenes & eval_scenes):
        raise C.ProtocolError(
            "tune/eval scene overlap: "
            f"{sorted(tune_scenes & eval_scenes)[:5]}")
    if tune_cases and (tune_contexts & eval_contexts):
        raise C.ProtocolError(
            "tune/eval context overlap: "
            f"{sorted(tune_contexts & eval_contexts)[:5]}")
    # Audit 1344ed0 item 6: lr comes from exactly one source. A locked
    # selection is consumed without retuning; an explicit --lr must
    # match the lock; otherwise tune once on the tune split and
    # optionally save the selection artifact for the lock.
    sel_artifact = None
    if args.lr_selection is not None:
        C.verify_completed(args.lr_selection)
        with open(args.lr_selection) as handle:
            sel_artifact = json.load(handle)
        if sel_artifact.get("kind") != "lr_selection":
            raise C.ProtocolError("lr-selection artifact has wrong kind")
        if protocol is not None:
            locked_sel = (protocol.get("design", {}).get("lr_selection")
                          or {})
            if locked_sel.get("sha256") != C.sha256_file(args.lr_selection):
                raise C.ProtocolError(
                    "lr-selection artifact differs from the locked "
                    "selection; refusing to correct off-lock")
        if lr is not None and float(lr) != float(sel_artifact["lr"]):
            raise C.ProtocolError(
                f"--lr {lr} != selection artifact {sel_artifact['lr']}")
        lr = float(sel_artifact["lr"])
        # The artifact's tune set, not this file's split strings,
        # defines separation for the consumed selection.
        tune_scenes = {int(s) for s in sel_artifact.get("tune_scenes", [])}
        tune_contexts = {str(c)
                         for c in sel_artifact.get("tune_contexts", [])}
        if tune_scenes & eval_scenes:
            raise C.ProtocolError(
                "selection-artifact tune scenes overlap eval scenes")
        if tune_contexts & eval_contexts:
            raise C.ProtocolError(
                "selection-artifact tune contexts overlap eval contexts")
        lr_scores = None
    def _run():
        if lr is None:
            if not tune_cases:
                raise C.ProtocolError(
                    "lr selection needs tune-split cases "
                    "(split='tune'); refusing to leak evaluation cases")
            chosen, lr_scores = validate_lr(
                models, tune_cases, device, lrs=lrs, gamma=gamma,
                trust_r=trust_r)
            print("frozen lr: " + json.dumps(dict(lr=chosen,
                                                  scores=lr_scores)))
            if args.save_lr_selection is not None:
                sel_path = args.save_lr_selection
                C.reserve_outputs([sel_path])
                C.write_meta(sel_path, dict(
                    kind="lr_selection", lr=chosen, scores=lr_scores,
                    tune_case_ids=sorted({c["case_id"]
                                          for c in tune_cases}),
                    tune_scenes=sorted(tune_scenes),
                    tune_contexts=sorted(tune_contexts),
                    protocol_id=protocol_id,
                    config_sha256=C.sha256_file(args.config),
                    source_hashes=C.source_hashes(),
                    package_versions=C.package_versions()))
                C.complete_output(sel_path, dict(lr=chosen))
                print("wrote lr selection " + sel_path)
        else:
            chosen, lr_scores = lr, None
        rng = np.random.default_rng(C.stable_seed("correction", 0, 0, 0, 0))
        shared_offsets = random_search_offsets(
            rng, (1, 16, 2), n=64, trust_r=trust_r)
        # Audit fdb59ff items 4, 12: teacher edits computed ONCE per
        # (context, target, K) over EVAL cases only (tuning cases
        # excluded). Arm-specific cases sharing a (context, target)
        # must carry identical q0/condition/target; otherwise they get
        # separate keys instead of silently overwriting each other.
        teacher_outs = {}
        for case in eval_cases:
            if case.get("split") == "tune":
                continue
            key_seed = (case["context_id"], case["target_id"])
            for budget in budgets:
                key = (case["context_id"], case["target_id"], budget)
                if key in teacher_outs:
                    prior = teacher_outs[key]["_identity"]
                    now = (case["q0_t"].detach().cpu().numpy().tobytes(),
                           case["cond_t"].detach().cpu().numpy().tobytes(),
                           case["target_t"].detach().cpu().numpy().tobytes())
                    if prior != now:
                        raise C.ProtocolError(
                            f"cases sharing {(key_seed, budget)} disagree "
                            "on q0/condition/target; refusing to share "
                            "one teacher edit")
                    continue
                out = invert_prefix(
                    teacher, case["q0_t"], case["cond_t"], case["target_t"],
                    steps=budget, learning_rate=chosen, trust_rms=trust_r,
                    gamma=gamma)
                out["_identity"] = (
                    case["q0_t"].detach().cpu().numpy().tobytes(),
                    case["cond_t"].detach().cpu().numpy().tobytes(),
                    case["target_t"].detach().cpu().numpy().tobytes())
                teacher_outs[key] = out
        rows, unreachable = [], 0
        for case in cases:
            if case.get("unreachable"):
                unreachable += 1
                rows.append(dict(case_id=case["case_id"],
                                 context_id=case.get("context_id"),
                                 target_id=case.get("target_id"),
                                 target_origin=case.get("target_origin"),
                                 scene=case.get("scene", 0),
                                 seed=case.get("seed", 0),
                                 model=case.get("model"), K=None,
                                 method="unreachable", mse=None,
                                 displacement=None, invalid=None,
                                 forwards=0, backwards=0, latency_ms=0.0,
                                 protocol_id=protocol_id))
                continue
            if case.get("split") == "tune":
                continue
            rows.extend(correct_case(case, models, teacher_outs, chosen,
                                     shared_offsets, device, protocol_id,
                                     teacher.adapter.stats, gamma=gamma,
                                     trust_r=trust_r, budgets=budgets))
        return rows, unreachable, chosen
    try:
        rows, unreachable, lr_used = _run()
    except Exception as error:
        C.mark_incomplete(args.output, f"{type(error).__name__}: {error}")
        C.mark_incomplete(meta_path, f"{type(error).__name__}: {error}")
        raise
    with open(args.output, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    C.write_meta(meta_path,
                 dict(kind="correction", schema_version=C.SCHEMA_VERSION,
                      lr=lr_used, K_grid=list(budgets), primary_K=PRIMARY_K,
                      gamma=gamma, trust_r=trust_r,
                      n_cases=len(cases), n_rows=len(rows),
                      unreachable=unreachable,
                      source_hashes=C.source_hashes(),
                      package_versions=C.package_versions()))
    C.complete_output(args.output, dict(rows=len(rows),
                                        unreachable=unreachable))
    C.complete_output(meta_path)
    print("wrote corrections " + args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
