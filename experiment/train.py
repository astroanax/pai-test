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


def _new_student(device):
    from gad_reference import make_student as _make
    student = _make()
    return student.to(device)


def cmd_warm(args, config, device):
    """Anchor-only warm start from teacher histories (endpoint targets)."""
    verified = C.verify_history_cache(args.history, config)
    histories = verified["histories"]
    if not histories:
        raise ValueError("history cache holds no histories")
    student = _new_student(device)
    torch.manual_seed(int(args.seed))
    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(args.seed)
    n = len(histories)
    batch_size = config["batch_size"]
    updates = int(args.updates)
    if updates < 1:
        raise ValueError("warm updates must be >= 1")
    student.train()
    first_loss = None
    log = []
    for update in range(updates):
        ids = sample_schedule(rng, n, batch_size)
        cond = torch.from_numpy(
            np.stack([np.asarray(histories[i]["condition"],
                                 dtype=np.float32) for i in ids])).to(device)
        noise = torch.from_numpy(
            rng.standard_normal((batch_size, 16, 2)).astype(np.float32)).to(device)
        target = torch.from_numpy(
            np.stack([np.asarray(histories[i]["endpoint"],
                                 dtype=np.float32) for i in ids])).to(device)
        pred = student(noise, cond)
        loss = (pred - target).square().mean()
        if first_loss is None:
            first_loss = float(loss.detach().item())
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
    final_loss = float(log[-1]["loss"])
    if not final_loss < first_loss:
        raise ValueError(
            f"warm value loss did not decrease: {first_loss} -> {final_loss}")
    for tensor in student.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite warm parameter; refusing to save")
    payload = dict(student=student.state_dict(), arm="warm", seed=args.seed,
                   updates=updates,
                   bank_sha256=C.sha256_file(args.history),
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
    protocol = C.load_protocol(args.protocol) if args.protocol else None
    protocol_id = protocol.get("protocol_id") if protocol else None
    bank = C.verify_pair_bank(args.bank, config)
    records = bank["records"]
    seed_state = torch.load(args.initial, map_location=device, weights_only=True)
    if protocol is not None:
        C.verify_student_provenance(seed_state, protocol, config,
                                    args.initial, role="init")
    init_sha256 = C.sha256_file(args.initial)
    if not isinstance(seed_state, dict) or "student" not in seed_state:
        raise C.ProtocolError("warm init checkpoint lacks a student state dict")
    student = _new_student(device)
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
        batch = _records_to_batch(records, ids, device,
                                  need_perturbed=args.arm != "anchor")
        optimizer.zero_grad()
        if args.arm == "anchor":
            # A0: duplicate the anchor input; average the two anchor losses
            # so forward/backward counts match the two-value arms.
            pred_a = student(batch["q"], batch["condition"])
            loss_a = (pred_a - batch["t0"]).square().mean()
            pred_b = student(batch["q"], batch["condition"])
            loss_b = (pred_b - batch["t0"]).square().mean()
            value = 0.5 * (loss_a + loss_b)
            response = value.new_zeros(())
        else:
            terms = paired_loss(
                student, batch["q"], batch["condition"], batch["u"],
                batch["t0"], batch["t1"], float(config["rho"]),
                mode=("gad" if args.arm == "gad" else "augmented"),
                beta=beta)
            value, response = terms["value"], terms["response"]
        loss = value + beta * response
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
    entry = sub.add_parser("train")
    entry.add_argument("--bank", required=True)
    entry.add_argument("--initial", required=True)
    entry.add_argument("--output", required=True)
    entry.add_argument("--arm", required=True,
                       choices=["anchor", "augmented", "gad"])
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--updates", type=int, default=None)
    entry.add_argument("--beta", type=float, default=None)
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
