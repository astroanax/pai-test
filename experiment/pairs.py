"""Pair-bank producer for the latent-controllability experiment.

For each warm-student history in the verified history cache, draws
``anchors_per_history`` latent anchors ``q ~ N(0, I)`` of shape (16, 2) and
``directions_per_anchor`` unit-L2 directions ``u``; the perturbed latent is
``q' = q + rho * u`` with ``rho`` from the config (L2, RMS recorded as
equiv). Teacher endpoints ``t0 = T(q, c)`` and ``t1 = T(q', c)`` are
evaluated under ``no_grad`` through the adapter's ``teacher_action`` at
``config["teacher_steps"]``, with the source transform recorded alongside.

Determinism: every (history, anchor, direction) triple derives its RNG from
``stable_seed('pairs', scene, decision, anchor, direction)``, and the bank
caches the actual inputs, so identical inputs produce a byte-identical bank.
Any nonfinite value aborts the run.
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
from gad_reference import stable_seed, teacher_endpoint


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def build_adapter(config, device):
    """Build the environment adapter."""
    from hri_adapter import HRIAdapter, load_checkpoint
    stats = {key: {inner: np.asarray(val) for inner, val in section.items()}
             for key, section in
             np.load(config["normalizer"], allow_pickle=True)["stats"].item().items()}
    sys.path.insert(0, os.path.join(config["repository"], "external", "models"))
    vision_encoder, noise_pred_net = load_checkpoint(config["checkpoint"], device)
    return HRIAdapter(vision_encoder, noise_pred_net, stats, device,
                      legacy=config["legacy"],
                      clip_actions=config.get("clip_actions", False))


def _unit_directions(rng, count, flat_dim):
    raw = rng.standard_normal((count, flat_dim))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if not np.isfinite(raw).all() or (norms <= 0).any():
        raise ValueError("nonfinite or degenerate direction draw")
    return (raw / norms).astype(np.float32)


def cmd_pairs(args, config, device):
    verified = C.verify_history_cache(args.history, config)
    histories = [h for h in verified["histories"] if h.get("split") == "train"]
    rejected = [h for h in verified["histories"] if h.get("split") != "train"]
    if rejected:
        bad = sorted({str(h.get("split")) for h in rejected})
        raise ValueError(
            f"pair production takes train histories only; cache holds "
            f"splits {bad} (filter inputs to train and keep validation / "
            "diagnostic / final scenes out of training)")
    if not histories:
        raise ValueError("history cache holds no train histories")
    rho = float(config["rho"])
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError(f"config rho must be positive finite, got {rho!r}")
    anchors = int(config["anchors_per_history"])
    directions = int(config["directions_per_anchor"])
    teacher_steps = int(config["teacher_steps"])
    source = config["canonical_source"]

    adapter = build_adapter(config, device)
    records = []
    teacher_batches = 0
    started = time.time()
    with torch.no_grad():
        for history in histories:
            history_id = str(history["history_id"])
            scene = int(history["scene"])
            decision = int(history["decision"])
            condition = np.asarray(history["condition"], dtype=np.float32)
            cond_t = torch.from_numpy(condition).unsqueeze(0).to(device)
            for anchor in range(anchors):
                anchor_rng = np.random.default_rng(
                    stable_seed("pairs", scene, decision, anchor, 0))
                anchors_q = anchor_rng.standard_normal((1, 16, 2)).astype(np.float32)
                if not np.isfinite(anchors_q).all():
                    raise ValueError(f"nonfinite anchor at {history_id}/{anchor}")
                dir_rng = np.random.default_rng(
                    stable_seed("pairs", scene, decision, anchor, 1))
                unit = _unit_directions(dir_rng, directions, 16 * 2).reshape(
                    directions, 16, 2)
                for direction in range(directions):
                    u = unit[direction]
                    q = anchors_q[0]
                    qp = (q + rho * u).astype(np.float32)
                    q_t = torch.from_numpy(q).unsqueeze(0).to(device)
                    qp_t = torch.from_numpy(qp).unsqueeze(0).to(device)
                    t0 = adapter.teacher_action(q_t, cond_t, teacher_steps,
                                                source=source)
                    t1 = adapter.teacher_action(qp_t, cond_t, teacher_steps,
                                                source=source)
                    teacher_batches += 2
                    t0_np = t0.detach().cpu().numpy()[0].astype(np.float32)
                    t1_np = t1.detach().cpu().numpy()[0].astype(np.float32)
                    for name, value in (("q", q), ("u", u), ("qp", qp),
                                        ("t0", t0_np), ("t1", t1_np)):
                        if not np.isfinite(np.asarray(value)).all():
                            raise ValueError(
                                f"nonfinite {name} at {(history_id, anchor, direction)}")
                    rms_equiv = float(np.sqrt(np.mean((qp - q) ** 2)))
                    records.append({
                        "pair_id": [history_id, anchor, direction],
                        "history_id": history_id,
                        "anchor_id": int(anchor),
                        "direction_id": int(direction),
                        "split": "train",
                        "q": q.astype(np.float32),
                        "u": u.astype(np.float32),
                        "rho": float(rho),
                        "source": str(source),
                        "t0": t0_np,
                        "t1": t1_np,
                        "condition": condition,
                        "teacher_steps": int(teacher_steps),
                    })
    if not records:
        raise ValueError("pair production produced no records")
    meta = dict(schema_version=C.SCHEMA_VERSION, kind="pair_bank",
                rho=rho, rms_equiv=rms_equiv, source=str(source),
                teacher_steps=teacher_steps,
                member_keys=[list(r["pair_id"]) for r in records],
                anchors_per_history=anchors, directions_per_anchor=directions,
                n_records=len(records),
                teacher_endpoint_batches=teacher_batches,
                wall_s=round(time.time() - started, 1))
    lineage = dict(source_hashes=C.source_hashes(),
                   parent_history=args.history,
                   parent_history_sha256=C.sha256_file(args.history),
                   collector="pairs",
                   package_versions=C.package_versions())
    C.write_pair_bank(args.output, records, meta, config, lineage)
    print(f"wrote pair bank {args.output} records {len(records)} "
          f"teacher_endpoint_batches {teacher_batches}")
    return dict(records=len(records), teacher_endpoint_batches=teacher_batches,
                meta=meta, lineage=lineage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--allow-unlocked", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    entry = sub.add_parser("pairs")
    entry.add_argument("--history", required=True)
    entry.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    from hri_adapter import resolve_device
    device = resolve_device(args.device or config["device"])
    output = args.output
    meta_path = output + ".meta.json"
    C.reserve_outputs([output, meta_path])
    try:
        if args.command == "pairs":
            result = cmd_pairs(args, config, device)
            C.write_meta(meta_path, dict(kind="pair_bank",
                                         schema_version=C.SCHEMA_VERSION,
                                         bank=args.output,
                                         meta=result["meta"],
                                         lineage=result["lineage"]))
    except Exception as error:
        C.mark_incomplete(output, f"{type(error).__name__}: {error}")
        raise
    else:
        C.complete_output(output)
        C.complete_output(meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
