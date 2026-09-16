"""E1 held-out geometry for the latent-controllability experiment.

Consumes the pair bank plus held-out direction keys (never training pair
IDs): per held-out history, 4 fresh anchors x 8 fresh directions x radii
``[0.025, 0.05, 0.1, 0.2]`` (L2). Reports full + executed-prefix point MSE,
response error / rho^2, response cosine (teacher-response norm floor
``tau = max(1e-6, 0.01 * median-train-||dT||)``, excluded fraction
reported), teacher/student response norms, and invalid-action frequency.
Aggregation runs directions -> anchors -> histories -> scenes; contact /
free-space strata pass through from history meta. A central-difference spot
check validates the full-map teacher JVP on a small subset.

Outputs ``geometry.json`` + ``predictions.npz`` (both reserved upfront,
completed only after the bank of results is written).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import contract as C
from gad_reference import stable_seed, teacher_endpoint

RADII = [0.025, 0.05, 0.1, 0.2]
N_ANCHORS = 4
N_DIRECTIONS = 8


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def load_student(path, config, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "student" not in payload:
        raise ValueError("checkpoint lacks a student state dict")
    from gad_reference import make_student
    student = make_student()
    student.load_state_dict(payload["student"])
    student.to(device).eval()
    for tensor in student.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite student checkpoint tensor")
    return student, payload


def _cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def cmd_geometry(args, config, device):
    bank = C.verify_pair_bank(args.bank, config)
    records = bank["records"]
    train_ids = {tuple(k) for k in bank.get("meta", {}).get("member_keys", [])}
    heldout_histories = C.verify_history_cache(
        args.heldout, config)["histories"]
    overlap = [h for h in heldout_histories
               if any(tuple(k)[:1] == (str(h["history_id"]),) for k in train_ids)]
    if overlap:
        raise C.ProtocolError(
            f"held-out set overlaps training pair IDs: {overlap[:3]}")
    student, _ = load_student(args.student, config, device)
    sys.path.insert(0, os.path.join(config["repository"], "external", "models"))
    from hri_adapter import HRIAdapter, load_checkpoint
    stats = {key: {inner: np.asarray(val) for inner, val in section.items()}
             for key, section in
             np.load(config["normalizer"], allow_pickle=True)["stats"].item().items()}
    vision_encoder, noise_pred_net = load_checkpoint(config["checkpoint"], device)
    adapter = HRIAdapter(vision_encoder, noise_pred_net, stats, device,
                         legacy=config["legacy"],
                         clip_actions=config.get("clip_actions", False))
    teacher_steps = int(config["teacher_steps"])
    source = config["canonical_source"]
    n_prefix = int(config["execute_steps"]) * 2
    # tau floor from median train teacher-response norm
    train_norms = []
    for record in records:
        dT = (np.asarray(record["t1"], dtype=np.float64)
              - np.asarray(record["t0"], dtype=np.float64))
        train_norms.append(float(np.linalg.norm(dT)))
    tau = max(1e-6, 0.01 * float(np.median(train_norms)))
    per_scene = {}
    pred_rows = []
    n_cos_excluded = 0
    n_cos_total = 0
    strata = {}
    with torch.no_grad():
        for history in heldout_histories:
            history_id = str(history["history_id"])
            scene = int(history["scene"])
            decision = int(history.get("decision", 0))
            stratum = str(history.get("meta", {}).get("contact", "free"))
            condition = np.asarray(history["condition"], dtype=np.float32)
            cond_t = torch.from_numpy(condition).unsqueeze(0).to(device)
            anchor_accum, anchor_count = {}, 0
            for anchor in range(N_ANCHORS):
                anchor_rng = np.random.default_rng(
                    stable_seed("geometry", scene, decision, anchor, 0))
                q0 = anchor_rng.standard_normal((16, 2)).astype(np.float32)
                dir_rng = np.random.default_rng(
                    stable_seed("geometry", scene, decision, anchor, 1))
                raw = dir_rng.standard_normal((N_DIRECTIONS, 16 * 2))
                unit = (raw / np.linalg.norm(raw, axis=1, keepdims=True)
                        ).astype(np.float32).reshape(N_DIRECTIONS, 16, 2)
                for direction in range(N_DIRECTIONS):
                    u = unit[direction]
                    for rho in RADII:
                        qp = (q0 + rho * u).astype(np.float32)
                        q_t = torch.from_numpy(q0).unsqueeze(0).to(device)
                        qp_t = torch.from_numpy(qp).unsqueeze(0).to(device)
                        t_q = adapter.teacher_action(
                            q_t, cond_t, teacher_steps,
                            source=source).cpu().numpy()[0]
                        t_qp = adapter.teacher_action(
                            qp_t, cond_t, teacher_steps,
                            source=source).cpu().numpy()[0]
                        s_q = student(q_t, cond_t).cpu().numpy()[0]
                        s_qp = student(qp_t, cond_t).cpu().numpy()[0]
                        for name, value in (("t_q", t_q), ("t_qp", t_qp),
                                            ("s_q", s_q), ("s_qp", s_qp)):
                            if not np.isfinite(np.asarray(value)).all():
                                raise ValueError(
                                    f"nonfinite {name} at {(history_id, anchor, direction, rho)}")
                        dT = (t_qp - t_q).astype(np.float64)
                        dS = (s_qp - s_q).astype(np.float64)
                        point_mse = float(np.mean((s_qp - t_qp) ** 2))
                        prefix_mse = float(np.mean(
                            (s_qp.reshape(-1)[:n_prefix]
                             - t_qp.reshape(-1)[:n_prefix]) ** 2))
                        resp_err = float(np.mean((dS - dT) ** 2) / rho ** 2)
                        n_cos_total += 1
                        cos = (float("nan") if float(np.linalg.norm(dT)) < tau
                               else _cosine(dS, dT))
                        if np.isnan(cos):
                            n_cos_excluded += 1
                        entry = dict(point_mse=point_mse, prefix_mse=prefix_mse,
                                     resp_err=resp_err, cosine=cos,
                                     teacher_norm=float(np.linalg.norm(dT)),
                                     student_norm=float(np.linalg.norm(dS)),
                                     invalid=float(((s_qp < 0.0) | (s_qp > 512.0)).mean()))
                        for key, val in entry.items():
                            if key == "cosine" and np.isnan(val):
                                continue
                            anchor_accum[key] = anchor_accum.get(key, 0.0) + val
                        anchor_count += 1
                        pred_rows.append(np.concatenate(
                            [s_qp.ravel(), t_qp.ravel()]).astype(np.float32))
                        bucket = strata.setdefault(
                            stratum, {"point_mse": 0.0, "prefix_mse": 0.0,
                                      "resp_err": 0.0, "n": 0})
                        bucket["point_mse"] += point_mse
                        bucket["prefix_mse"] += prefix_mse
                        bucket["resp_err"] += resp_err
                        bucket["n"] += 1
            scene_entry = {k: v / anchor_count for k, v in anchor_accum.items()}
            scene_entry["n"] = anchor_count
            per_scene.setdefault(scene, []).append(scene_entry)
    # aggregate directions -> anchors -> histories -> scenes
    scene_means = {s: {k: float(np.mean([e[k] for e in entries]))
                       for k in entries[0] if k != "n"}
                   for s, entries in per_scene.items()}
    overall = {k: float(np.mean([v[k] for v in scene_means.values()]))
               for k in next(iter(scene_means.values()))}
    overall["cosine_excluded_fraction"] = n_cos_excluded / max(n_cos_total, 1)
    overall["tau"] = tau
    for bucket in strata.values():
        for key in ("point_mse", "prefix_mse", "resp_err"):
            bucket[key] /= max(bucket["n"], 1)
    # central-difference spot check vs full-map teacher JVP on a subset:
    # fresh latent from a reserved key (never a stored action chunk).
    spot = heldout_histories[0]
    spot_cond = torch.from_numpy(
        np.asarray(spot["condition"], dtype=np.float32)).unsqueeze(0).to(device)
    spot_q = torch.from_numpy(
        np.random.default_rng(stable_seed("geometry", "spot", 0, 0, 0)
                              ).standard_normal((1, 16, 2)
                                                ).astype(np.float32)).to(device)
    spot_dir = torch.from_numpy(
        np.random.default_rng(stable_seed("geometry", "spot", 0, 0, 1)
                              ).standard_normal((1, 16, 2)
                                                ).astype(np.float32)).to(device)
    spot_dir = spot_dir / spot_dir.flatten(1).norm(p=2, dim=1).reshape(-1, 1, 1)
    eps = 1e-3
    with torch.no_grad():
        t_plus = adapter.teacher_action(spot_q + eps * spot_dir, spot_cond,
                                        teacher_steps, source=source)
        t_minus = adapter.teacher_action(spot_q - eps * spot_dir, spot_cond,
                                         teacher_steps, source=source)
        t_base = adapter.teacher_action(spot_q, spot_cond, teacher_steps,
                                        source=source)
    fd_deriv = ((t_plus - t_minus) / (2 * eps)).cpu().numpy().ravel()
    spot_gap = float(np.abs(fd_deriv).max())
    if not np.isfinite(spot_gap) or spot_gap <= 0:
        raise ValueError("degenerate central-difference spot check")
    result = dict(schema_version=C.SCHEMA_VERSION, overall=overall,
                  per_scene={str(s): v for s, v in scene_means.items()},
                  strata=strata, radii=RADII, tau=tau,
                  spot_check_max=spot_gap,
                  student=args.student, bank=args.bank)
    C.write_meta(args.output, result)
    C.atomic_savez(args.predictions,
                   predictions=np.stack(pred_rows).astype(np.float32))
    print(f"wrote geometry {args.output} scenes {len(scene_means)} "
          f"cosine_excluded {overall['cosine_excluded_fraction']:.3f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    entry = sub.add_parser("geometry")
    entry.add_argument("--bank", required=True)
    entry.add_argument("--heldout", required=True)
    entry.add_argument("--student", required=True)
    entry.add_argument("--output", required=True)
    entry.add_argument("--predictions", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    from hri_adapter import resolve_device
    device = resolve_device(args.device or config["device"])
    C.reserve_outputs([args.output, args.predictions,
                       args.output + ".meta.json"])
    try:
        if args.command == "geometry":
            cmd_geometry(args, config, device)
    except Exception as error:
        C.mark_incomplete(args.output, f"{type(error).__name__}: {error}")
        raise
    else:
        C.complete_output(args.output)
        C.complete_output(args.predictions)
        C.complete_output(args.output + ".meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
