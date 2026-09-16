"""Matched-arm training for the latent-controllability experiment.

Schedule discipline: :func:`sample_schedule` draws identical ordered
pair-record IDs per seed for ALL arms (anchor / augmented / gad share the
same RNG stream and call order). The anchor (A0) arm duplicates the anchor
input and averages the two losses so forward/backward counts match the
two-value arms.

``cmd_warm`` trains anchor-only from teacher histories (endpoint targets)
with full ``batch_size`` record batches for ``warm_updates`` and refuses to
save unless the value loss decreases. ``cmd_train`` fine-tunes one arm from
the warm init: anchor duplicates the anchor (2 forwards, averaged anchor
loss); augmented trains both pair values; gad adds ``beta * response`` via
the reference ``paired_loss``. No test-based checkpoint selection: only the
final update is saved.
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
from gad_reference import make_student, paired_loss, stable_seed


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def sample_schedule(rng, n_records, batch_size):
    """Identical ordered pair-record IDs per seed for every arm.

    All arms share this exact RNG stream: same seed and same call order
    give the same IDs, so arm differences come only from the loss, never
    from data selection.
    """
    if n_records < 1:
        raise ValueError("pair bank holds no records")
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    return rng.integers(0, n_records, size=batch_size).tolist()


def schedule_hash(schedule_ids, updates, keep=5000):
    import hashlib
    raw = repr(schedule_ids[:keep]) + f"/{updates}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _records_to_batch(records, ids, device, need_perturbed):
    keys = ["q", "t0", "condition"]
    if need_perturbed:
        keys += ["u", "t1"]
    batch = {}
    for key in keys:
        values = np.stack([np.asarray(records[i][key]) for i in ids])
        if not np.isfinite(values).all():
            raise ValueError("nonfinite bank tensor " + key)
        batch[key] = torch.from_numpy(values.astype(np.float32)).to(device)
    return batch


def _arch(config, payload=None):
    """Item 25: width always from config; a checkpoint's recorded arch
    must match it, so a bare make_student() default can never silently
    mismatch a non-default config."""
    width = int(config.get("width", 512))
    if payload is not None:
        recorded = payload.get("arch", {}).get("width", width)
        if int(recorded) != width:
            raise ValueError(
                f"checkpoint arch width {recorded} != config width {width}")
    return dict(condition_dim=514, horizon=16, action_dim=2, width=width)


def _new_student(device, config=None, payload=None):
    from gad_reference import make_student as _make
    spec = _arch(config or {}, payload)
    return _make(**spec).to(device)


def _warm_batch(histories, ids, device):
    cond = torch.from_numpy(
        np.stack([np.asarray(histories[i]["condition"],
                             dtype=np.float32) for i in ids])).to(device)
    noise = torch.from_numpy(
        np.stack([np.asarray(histories[i]["q"],
                             dtype=np.float32) for i in ids])).to(device)
    target = torch.from_numpy(
        np.stack([np.asarray(histories[i]["endpoint"],
                             dtype=np.float32) for i in ids])).to(device)
    return cond, noise, target


def cmd_warm(args, config, device):
    """Anchor-only warm start from teacher histories (endpoint targets)."""
    verified = C.verify_history_cache(args.history, config)
    # Audit fdb59ff item 7: warm consumes TRAINING histories only.
    # Membership is numeric (scene in the train range), not just the
    # split string, and targets must come from the teacher collector.
    train_scenes = set(int(s) for s in C.scene_range(config, "train"))
    histories = [h for h in verified["histories"]
                 if h.get("split") == "train"
                 and int(h.get("scene", -1)) in train_scenes]
    if not histories:
        raise ValueError("warm needs train-split histories in train scenes")
    hashes = {str(h.get("collector_hash")) for h in histories}
    if len(hashes) != 1:
        raise C.ProtocolError(
            f"warm histories come from {len(hashes)} collectors; "
            "a warm start needs one teacher collection")
    try:
        teacher_hash = C.sha256_file(config["checkpoint"])
    except (OSError, FileNotFoundError):
        teacher_hash = None
    if teacher_hash is not None and hashes != {teacher_hash}:
        raise C.ProtocolError(
            "warm targets are not teacher-collected "
            f"(collector {sorted(hashes)[0][:16]} != "
            f"checkpoint {teacher_hash[:16]}); refusing to warm-start "
            "a teacher-mimic on student-collected targets")
    # Seed BEFORE constructing the student so init is determined.
    torch.manual_seed(int(args.seed))
    student = _new_student(device, config)
    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(args.seed)
    n = len(histories)
    batch_size = config["batch_size"]
    updates = int(args.updates)
    if updates < 1:
        raise ValueError("warm updates must be >= 1")
    # Fixed diagnostic batch: loss reduction is measured on the same
    # records before/after, not on unrelated first/last minibatches.
    diag_rng = np.random.default_rng(int(args.seed) + 10 ** 6)
    diag_ids = sample_schedule(diag_rng, n, min(batch_size, n))
    student.train()
    with torch.no_grad():
        cond_d, noise_d, target_d = _warm_batch(histories, diag_ids, device)
        first_loss = float(student(noise_d, cond_d).sub(
            target_d).square().mean().item())
    log = []
    for update in range(updates):
        ids = sample_schedule(rng, n, batch_size)
        # Train on the stored generating latent, never fresh noise
        # against a stored endpoint (which would teach noise-independence).
        cond, noise, target = _warm_batch(histories, ids, device)
        pred = student(noise, cond)
        loss = (pred - target).square().mean()
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite warm loss at update {update}")
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if update % max(1, updates // 50) == 0 or update == updates - 1:
            log.append({"update": update, "loss": float(loss.item()),
                        "grad_norm": float(grad_norm)})
    student.eval()
    with torch.no_grad():
        final_loss = float(student(noise_d, cond_d).sub(
            target_d).square().mean().item())
    if not final_loss < first_loss:
        raise ValueError(
            f"warm value loss did not decrease: {first_loss} -> {final_loss}")
    for tensor in student.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite warm parameter; refusing to save")
    payload = dict(student=student.state_dict(), arm="warm", seed=args.seed,
                   updates=updates, arch=_arch(config),
                   history_sha256=C.sha256_file(args.history),
                   warm_scenes=sorted(train_scenes),
                   warm_histories=len(histories),
                   collector_hash=sorted(hashes)[0],
                   protocol_id=None,
                   hyperparams=dict(lr=config["learning_rate"],
                                    weight_decay=config["weight_decay"],
                                    batch_size=batch_size),
                   optimizer="AdamW", first_loss=first_loss,
                   final_loss=final_loss, log=log)
    C.atomic_save_torch(args.output, payload)
    C.write_meta(args.output + ".meta.json",
                 dict(kind="warm_student", seed=args.seed, updates=updates,
                      first_loss=first_loss, final_loss=final_loss,
                      source_hashes=C.source_hashes()))
    print(f"wrote warm student {args.output} loss {first_loss:.4f} -> {final_loss:.4f}")
    return dict(first_loss=first_loss, final_loss=final_loss)


def cmd_train(args, config, device):
    # Audit fdb59ff item 10: direct training performs full
    # current-protocol verification and cannot override locked beta /
    # update settings.
    protocol = C.load_protocol(args.protocol) if args.protocol else None
    if protocol is None and not getattr(args, "allow_unlocked", False):
        raise C.ProtocolError("train requires --protocol or --allow-unlocked")
    if protocol is not None:
        C.verify_current_protocol(protocol, config, args.config,
                                  stage="train")
    protocol_id = protocol.get("protocol_id") if protocol else None
    if protocol is not None:
        design = protocol.get("design", {})
        if int(args.updates) != int(design.get("updates", args.updates)):
            raise C.ProtocolError(
                f"--updates {args.updates} != locked "
                f"{design.get('updates')}; refusing to train off-lock")
        if args.arm == "gad" and args.beta is not None and float(args.beta) != float(
                design.get("beta_selected", args.beta)):
            raise C.ProtocolError(
                f"--beta {args.beta} != locked "
                f"{design.get('beta_selected')}; refusing to train off-lock")
    bank = C.verify_pair_bank(args.bank, config)
    records = bank["records"]
    # Item 5: trainers assert train-only records; validation/diagnostic/
    # final records must never reach an optimizer.
    bad = sorted({str(r.get("split")) for r in records
                  if r.get("split") != "train"})
    if bad or not records:
        raise C.ProtocolError(
            f"train consumes train-split records only; got splits {bad}")
    seed_state = torch.load(args.initial, map_location=device, weights_only=True)
    if protocol is not None:
        C.verify_student_provenance(seed_state, protocol, config,
                                    args.initial, role="init")
    init_sha256 = C.sha256_file(args.initial)
    if not isinstance(seed_state, dict) or "student" not in seed_state:
        raise C.ProtocolError("warm init checkpoint lacks a student state dict")
    student = _new_student(device, config, seed_state)
    student.load_state_dict(seed_state["student"])
    for tensor in student.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite warm-init tensor")
    design = (protocol or {}).get("design", {})
    if args.beta is not None:
        beta = float(args.beta)
    elif design.get("beta_selected") is not None:
        beta = float(design["beta_selected"])
    elif args.arm != "gad":
        beta = 0.0
    else:
        raise C.ProtocolError("gad arm needs --beta or a locked beta_selected")
    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(args.seed)
    n = len(records)
    batch_size = config["batch_size"]
    student.train()
    log = []
    grad_norms = []
    clipped = 0
    schedule_ids = []
    peak_mem = 0
    started = time.time()
    for update in range(args.updates):
        ids = sample_schedule(rng, n, batch_size)
        schedule_ids.extend(ids)
        # Audit fdb59ff item 1: load the complete pair for every arm.
        # The corrected anchor kernel ignores the perturbed target in
        # its objective, but the common loss call reads u/t1
        # unconditionally; omitting them crashed anchor with KeyError.
        batch = _records_to_batch(records, ids, device,
                                  need_perturbed=True)
        optimizer.zero_grad()
        # Item 1: the kernel returns the differentiable objective. Rebuilding
        # value + beta*response from detached logging outputs reaches
        # backward() with no graph (and conflates augmented with GAD at
        # beta=0). Detached values are for logging only.
        terms = paired_loss(
            student, batch["q"], batch["condition"], batch["u"],
            batch["t0"], batch["t1"], float(config["rho"]),
            mode=args.arm,
            beta=beta if args.arm == "gad" else 0.0,
        )
        loss = terms["loss"]
        value, response = terms["value"], terms["response"]
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at update {update}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), 1.0, error_if_nonfinite=True)
        if float(grad_norm) > 1.0:
            clipped += 1
        optimizer.step()
        grad_norm = float(grad_norm)
        grad_norms.append(grad_norm)
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device))
        if update % max(1, args.updates // 100) == 0 or update == args.updates - 1:
            with torch.no_grad():
                ratio = (float((beta * response).item() / max(float(value.item()), 1e-12))
                         if args.arm == "gad" else 0.0)
            log.append({"update": update, "loss": float(loss.item()),
                        "value": float(value.item()),
                        "response": float(response.item()) if torch.isfinite(response) else None,
                        "weighted_ratio": ratio,
                        "grad_norm": grad_norm,
                        "examples": len(ids),
                        "elapsed_s": round(time.time() - started, 1),
                        "peak_mem_bytes": peak_mem})
    for tensor in student.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite parameter after training; refusing to save")
    sched_hash = schedule_hash(schedule_ids, args.updates)
    payload = dict(student=student.state_dict(), arm=args.arm, seed=args.seed,
                   updates=args.updates, init_sha256=init_sha256,
                   arch=_arch(config),
                   pair_bank_sha256=C.sha256_file(args.bank),
                   bank_sha256=C.sha256_file(args.bank),
                   schedule_hash=sched_hash, protocol_id=protocol_id,
                   schema_version=C.SCHEMA_VERSION,
                   hyperparams=dict(beta=beta, rho=float(config["rho"]),
                                    lr=config["learning_rate"],
                                    weight_decay=config["weight_decay"],
                                    batch_size=batch_size),
                   optimizer=dict(name="AdamW", lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"]),
                   log=log, grad_norm_mean=float(np.mean(grad_norms)),
                   grad_norm_max=float(np.max(grad_norms)),
                   clip_fraction=clipped / max(args.updates, 1))
    C.atomic_save_torch(args.output, payload)
    C.write_meta(args.output + ".meta.json",
                 dict(kind="student", arm=args.arm, seed=args.seed,
                      updates=args.updates, schedule_hash=sched_hash,
                      protocol_id=protocol_id,
                      source_hashes=C.source_hashes()))
    print(f"wrote student {args.output} arm {args.arm} schedule_hash={sched_hash}")
    return dict(arm=args.arm, seed=args.seed, updates=args.updates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--allow-unlocked", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    entry = sub.add_parser("warm")
    entry.add_argument("--history", required=True)
    entry.add_argument("--output", required=True)
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--updates", type=int, default=None)
    entry.add_argument("--development", action="store_true")
    entry = sub.add_parser("train")
    entry.add_argument("--bank", required=True)
    entry.add_argument("--initial", required=True)
    entry.add_argument("--output", required=True)
    entry.add_argument("--arm", required=True,
                       choices=["anchor", "augmented", "gad"])
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--updates", type=int, default=None)
    entry.add_argument("--beta", type=float, default=None)
    entry.add_argument("--development", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    from hri_adapter import resolve_device
    device = resolve_device(args.device or config["device"])
    if getattr(args, "updates", None) is None:
        args.updates = (config["warm_updates"] if args.command == "warm"
                        else config["updates"])
    if args.updates < 1:
        raise ValueError("--updates must be >= 1")
    output = args.output
    meta_path = output + ".meta.json"
    C.reserve_outputs([output, meta_path])
    try:
        if args.command == "warm":
            cmd_warm(args, config, device)
        elif args.command == "train":
            cmd_train(args, config, device)
    except Exception as error:
        C.mark_incomplete(output, f"{type(error).__name__}: {error}")
        raise
    else:
        C.complete_output(output)
        C.complete_output(meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
