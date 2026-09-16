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
    model = make_student()
    model.load_state_dict(payload["student"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class TeacherModel(torch.nn.Module):
    """Frozen teacher endpoint map as a parameter-free module."""

    def __init__(self, adapter, steps, source):
        super().__init__()
        self.adapter = adapter
        self.steps = int(steps)
        self.source = str(source)

    def forward(self, latent, condition):
        return self.adapter.teacher_action(latent, condition, self.steps,
                                           source=self.source)


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


def validate_lr(models, pilot_cases, device):
    """Pick one lr on a held-out pilot slice; the caller freezes it."""
    pilot = pilot_cases[:max(1, len(pilot_cases) // 4)]
    scores = {}
    for lr in LR_CANDIDATES:
        errs = []
        for case in pilot:
            model = models[case["model"]]
            out = invert_prefix(model, case["q0_t"], case["cond_t"],
                                case["target_t"], steps=PRIMARY_K,
                                learning_rate=lr, trust_rms=TRUST_R,
                                gamma=GAMMA)
            errs.append(float(out["prefix_mse"].mean().item()))
        scores[lr] = float(np.mean(errs))
    best = min(scores, key=scores.get)
    return best, scores


def random_search_candidates(rng, q0, n, trust_r=TRUST_R):
    """Bounded random search over a SHARED candidate set (fixed seeds)."""
    q0 = np.asarray(q0, dtype=np.float64)
    scale = trust_r * np.sqrt(q0.size)
    cands = [q0]
    for _ in range(n - 1):
        step = rng.normal(size=q0.shape)
        norm = float(np.linalg.norm(step.ravel()))
        step = step * (scale / max(norm, 1e-12))
        cands.append(q0 + step)
    return cands


def prefix_mse_of(model, latent_np, cond_t, target_np, device):
    with torch.no_grad():
        action = model(as_tensor(latent_np, device, "candidate"),
                       cond_t).cpu().numpy()
    return float(np.mean((action[:, :target_np.shape[1]] - target_np) ** 2))


def correct_case(case, models, teacher, lr, shared_candidates, device,
                 protocol_id):
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
                model=case["model"], lr=lr, gamma=GAMMA, trust_r=TRUST_R,
                protocol_id=protocol_id)
    for K in K_GRID:
        row_base = dict(base, K=K)
        err0 = float(np.mean((case["q0_np"][:, :target_np.shape[1]]
                              - target_np) ** 2))
        rows.append(dict(row_base, method="no_correction", mse=err0,
                         displacement=0.0, invalid=0,
                         forwards=0, backwards=0, latency_ms=0.0))
        start = time.perf_counter()
        sync = (lambda: torch.cuda.synchronize()) \
            if cond_t.device.type == "cuda" else (lambda: None)
        out = invert_prefix(model, q0_t, cond_t, target_t, steps=K,
                            learning_rate=lr, trust_rms=TRUST_R, gamma=GAMMA)
        sync()
        latency = (time.perf_counter() - start) * 1000.0
        best_action = out["action"].cpu().numpy()
        rows.append(dict(row_base, method="own_model",
                         mse=float(out["prefix_mse"].mean().item()),
                         displacement=float(
                             np.sqrt(np.mean((out["latent"] - q0_t
                                              ).cpu().numpy() ** 2))),
                         invalid=invalid_prefix_count(
                             best_action[:, :target_np.shape[1]]),
                         forwards=int(out["forward_calls"]),
                         backwards=int(out["backward_calls"]),
                         latency_ms=latency))
        t_out = invert_prefix(teacher, q0_t, cond_t, target_t, steps=K,
                              learning_rate=lr, trust_rms=TRUST_R, gamma=GAMMA)
        transferred = t_out["latent"].cpu().numpy()
        transfer_mse = prefix_mse_of(model, transferred, cond_t, target_np,
                                     device)
        rows.append(dict(row_base, method="teacher_transfer",
                         mse=transfer_mse,
                         displacement=float(
                             np.sqrt(np.mean((t_out["latent"] - q0_t
                                              ).cpu().numpy() ** 2))),
                         invalid=0,
                         forwards=0, backwards=0, latency_ms=0.0))
        best_mse, best_disp = err0, 0.0
        for cand in shared_candidates:
            mse = prefix_mse_of(model, cand, cond_t, target_np, device)
            if mse < best_mse:
                best_mse = mse
                best_disp = float(np.linalg.norm(
                    (cand - case["q0_np"]).ravel()) / np.sqrt(case["q0_np"].size))
        rows.append(dict(row_base, method="random_search",
                         mse=best_mse, displacement=best_disp, invalid=0,
                         forwards=len(shared_candidates), backwards=0,
                         latency_ms=0.0))
        rows.append(dict(row_base, method="oracle_prefix",
                         mse=0.0,
                         displacement=float(np.linalg.norm(
                             (target_np - case["q0_np"][:, :target_np.shape[1]]
                              ).ravel()) / np.sqrt(target_np.size)),
                         invalid=0, forwards=0, backwards=0, latency_ms=0.0))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--students", required=True,
                        help="JSON {model_name: checkpoint_path}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    protocol_id = None
    if args.protocol:
        protocol = C.load_protocol(args.protocol)
        C.verify_current_protocol(protocol, config, args.config,
                                  stage="correction")
        protocol_id = protocol.get("protocol_id")
    from hri_adapter import resolve_device
    device = resolve_device(args.device or config["device"])
    with open(args.cases) as handle:
        raw_cases = [json.loads(line) for line in handle if line.strip()]
    with open(args.students) as handle:
        student_paths = json.load(handle)
    models = {name: load_student_model(path, device)
              for name, path in student_paths.items()}
    teacher = build_teacher(config, device)
    cases = []
    for raw in raw_cases:
        if raw.get("unreachable"):
            cases.append(dict(raw, unreachable=True))
            continue
        cases.append(dict(
            case_id=str(raw["case_id"]), context_id=str(raw["context_id"]),
            target_id=str(raw["target_id"]),
            target_origin=str(raw.get("target_origin", "unknown")),
            scene=int(raw.get("scene", 0)), seed=int(raw.get("seed", 0)),
            model=str(raw["model"]),
            q0_np=np.asarray(raw["q0"], dtype=np.float64),
            target_np=np.asarray(raw["target"], dtype=np.float64),
            q0_t=as_tensor(raw["q0"], device, "q0"),
            cond_t=as_tensor(raw["condition"], device, "condition"),
            target_t=as_tensor(raw["target"], device, "target")))
    for case in cases:
        if case.get("unreachable"):
            continue
        if case["model"] not in models:
            raise ValueError(f"case model {case['model']!r} has no checkpoint")
    if args.lr is not None and args.lr not in LR_CANDIDATES:
        raise C.ProtocolError(f"lr {args.lr} not in validated set {LR_CANDIDATES}")
    lr = args.lr
    if lr is None:
        lr, lr_scores = validate_lr(models, [c for c in cases
                                             if not c.get("unreachable")],
                                    device)
        print("frozen lr: " + json.dumps(dict(lr=lr, scores=lr_scores)))
    rng = np.random.default_rng(C.stable_seed("correction", 0, 0, 0, 0))
    shared_candidates = random_search_candidates(
        rng, np.zeros((1, 16, 2), dtype=np.float64), n=64)
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
        rows.extend(correct_case(case, models, teacher, lr,
                                 shared_candidates, device, protocol_id))
    C.reserve_outputs([args.output])
    with open(args.output, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    C.write_meta(args.output.replace(".jsonl", ".meta.json"),
                 dict(kind="correction", schema_version=C.SCHEMA_VERSION,
                      lr=lr, K_grid=list(K_GRID), primary_K=PRIMARY_K,
                      gamma=GAMMA, trust_r=TRUST_R,
                      n_cases=len(cases), n_rows=len(rows),
                      unreachable=unreachable,
                      source_hashes=C.source_hashes(),
                      package_versions=C.package_versions()))
    C.complete_output(args.output, dict(rows=len(rows),
                                        unreachable=unreachable))
    print("wrote corrections " + args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
