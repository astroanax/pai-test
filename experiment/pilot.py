import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C
from core import (TwoStepStudent, finite_difference_jacobian, metric_scales,
                  objective, pullback_metric_exact, pullback_probes,
                  sample_with_midpoint)
from hri_adapter import (HRIAdapter, load_checkpoint, resolve_device,
                         teacher_half_from, teacher_half_maps, teacher_rollout)


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def sample_source(rng, shape, source):
    if source == "gaussian":
        return rng.standard_normal(shape).astype(np.float32)
    if source == "uniform":
        return (rng.random(shape).astype(np.float32) * 2 - 1)
    raise ValueError("unknown source distribution")


def build_adapter(config, device):
    stats = {key: {inner: np.asarray(val) for inner, val in section.items()}
             for key, section in np.load(config["normalizer"], allow_pickle=True)["stats"].item().items()}
    sys.path.insert(0, os.path.join(config["repository"], "external", "models"))
    vision_encoder, noise_pred_net = load_checkpoint(config["checkpoint"], device)
    return HRIAdapter(vision_encoder, noise_pred_net, stats, device,
                      legacy=config["legacy"], clip_actions=config["clip_actions"])


def load_student(path, config, device):
    """Load a student checkpoint and return (student, payload). The intrinsic
    mode and seed are read from the checkpoint, never from the CLI label."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "student" not in payload:
        raise ValueError("checkpoint lacks a student state dict")
    for key in ("mode", "seed"):
        if key not in payload:
            raise ValueError(f"checkpoint lacks intrinsic {key}")
    student = TwoStepStudent(condition_dim=514, horizon=16, action_dim=2,
                             width=config["width"]).to(device)
    student.load_state_dict(payload["student"])
    for tensor in student.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite student checkpoint tensor")
    return student, payload


def resolve_protocol_id(args, config, device):
    """Recheck the locked protocol before evaluation and analysis."""
    path = getattr(args, "protocol", None) or "runs/protocol_locked.json"
    if not os.path.exists(path):
        return None, None
    protocol = C.require_protocol(path)
    return protocol, protocol.get("protocol_id")


def cmd_normalizer(args, config, device):
    import zarr
    output = args.output
    C.reserve_outputs([output, output + ".meta.json"])
    store = zarr.open(args.dataset, "r")
    action = np.asarray(store["data"]["action"][:], dtype=np.float64).reshape(-1, 2)
    state = np.asarray(store["data"]["state"][:, :2], dtype=np.float64).reshape(-1, 2)
    if not (np.isfinite(action).all() and np.isfinite(state).all()):
        raise ValueError("non-finite values in the demonstration archive")
    stats = {"action": {"min": action.min(0).tolist(), "max": action.max(0).tolist()},
             "agent_pos": {"min": state.min(0).tolist(), "max": state.max(0).tolist()}}
    C.verify_tensors({k: np.array([v["min"], v["max"]]) for k, v in stats.items()},
                     "normalizer")
    C.atomic_savez(output, stats=stats)
    C.write_meta(output + ".meta.json",
                 dict(kind="normalizer", dataset=args.dataset,
                      source_hashes=C.source_hashes(), stats=stats))
    print("wrote normalizer " + output)


def cmd_collect(args, config, device):
    output = args.output
    meta_path = output + ".meta.json"
    C.reserve_outputs([output, meta_path])
    planned = C.scene_range(config, "train")
    adapter = build_adapter(config, device)
    rng = np.random.default_rng(args.seed)
    student = None
    if args.student is not None:
        student, payload = load_student(args.student, config, device)
        student.eval()
    checkpoint_hash = C.sha256_file(args.student) if args.student else None
    rows = []
    start = config["train_scene_start"]
    total = args.episodes if args.episodes is not None else config["train_episodes"]
    if total < 1:
        raise ValueError("--episodes must be >= 1")
    scenes = [start + episode for episode in range(total)]
    unknown = sorted(set(scenes) - planned)
    if unknown:
        raise ValueError(f"collection scenes outside the configured train range: "
                         f"{unknown[:5]}")
    for scene in scenes:
        record = {"scene": scene, "contexts": []}
        env = adapter.new_env(image=True)
        obs, _ = adapter.reset(env, scene)
        history = []
        done = False
        steps = 0
        while not done and steps < config["episode_steps"]:
            cond = adapter.encode_observation(obs["image"], obs["agent_pos"])
            noise = sample_source(rng, (1, 16, 2), config["source"])
            noise_t = torch.from_numpy(noise).to(device)
            keep = (len(history) % 2 == 0
                    and len(record["contexts"]) < config["contexts_per_episode"])
            with torch.no_grad():
                if student is None:
                    # the teacher is the behavior policy here: no supervision is
                    # discarded, its whole rollout IS the executed action chunk
                    chunk = teacher_rollout(adapter.noise_pred_net, noise_t, cond,
                                            config["teacher_steps"])
                    student_mid = None
                else:
                    # behavior only needs the two student steps once; the
                    # midpoint is reused as the stored solver state rather than
                    # recomputed (no teacher work at unsaved decisions)
                    chunk, midpoint = sample_with_midpoint(noise_t, cond, student)
                    student_mid = midpoint.detach().cpu().numpy()[0]
            chunk_n = chunk.detach().cpu().numpy()[0]
            if not np.isfinite(chunk_n).all():
                raise ValueError(f"nonfinite student chunk at scene {scene} step {steps}")
            physical = adapter.decode(chunk_n[:config["execute_steps"]])
            context = None
            if keep:
                if history:
                    check_env = adapter.replay(scene, history)
                    try:
                        signature = [float(v) for v in adapter.signature(check_env).tolist()]
                    finally:
                        check_env.close()
                else:
                    signature = None
                context = {"condition": cond.detach().cpu().numpy()[0], "noise": noise[0],
                           "history": [list(map(float, row)) for row in history],
                           "signature": signature, "scene": scene,
                           "student_mid": None if student_mid is None else student_mid.tolist()}
                record["contexts"].append(context)
            for action in physical:
                obs, _, done, _ = adapter.step(env, action)
                history.append([float(value) for value in np.asarray(action).tolist()])
                steps += 1
                if done or steps >= config["episode_steps"]:
                    break
            if context is not None:
                context["history_end"] = [list(map(float, row)) for row in history]
        env.close()
        rows.append(record)
    scenes_seen = sorted({int(r["scene"]) for r in rows})
    C.validate_scene_sets({"train": scenes_seen,
                           "validation": sorted(C.scene_range(config, "validation")),
                           "development": sorted(C.scene_range(config, "development")),
                           "test": sorted(C.scene_range(config, "test"))})
    payload = dict(records=np.array(rows, dtype=object),
                   meta=json.dumps({"mode": "shared" if student is not None else "teacher",
                                    "student": args.student, "seed": args.seed}),
                   fingerprint=json.dumps(C.cache_fingerprint(config, config["repository"])),
                   lineage=json.dumps(dict(source_hashes=C.source_hashes(),
                                           checkpoint_sha256=checkpoint_hash,
                                           episodes=int(total),
                                           contexts_per_episode=config["contexts_per_episode"])))
    try:
        C.atomic_savez(output, **payload)
    except Exception as error:
        C.mark_incomplete(output, f"collect failed: {error}")
        raise
    C.write_meta(meta_path, json.loads(payload["meta"]) | dict(
        fingerprint=json.loads(payload["fingerprint"]),
        lineage=json.loads(payload["lineage"]), scenes=scenes_seen))
    print("wrote cache " + output + " episodes " + str(len(rows)))


def cmd_metrics(args, config, device):
    output = args.output
    meta_path = output + ".meta.json"
    C.reserve_outputs([output, meta_path])
    adapter = build_adapter(config, device)
    cache = np.load(args.cache, allow_pickle=True)
    info = C.verify_cache(cache, config, config["repository"], expect_mode="shared",
                          expect_scenes=C.scene_range(config, "train"))
    contexts = []
    for record in cache["records"]:
        contexts.extend(record["contexts"])
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        contexts = contexts[:args.limit]
    eps = config["finite_difference_epsilon"]
    steps = config["teacher_steps"]
    mode = config.get("metric_mode", "exact")
    out = []
    for index, context in enumerate(contexts):
        cond = torch.from_numpy(np.asarray(context["condition"], dtype=np.float32)).unsqueeze(0).to(device)
        noise = torch.from_numpy(np.asarray(context["noise"], dtype=np.float32)).unsqueeze(0).to(device)
        student_mid = torch.from_numpy(np.asarray(context["student_mid"], dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            teacher_mid, _ = teacher_half_maps(adapter.noise_pred_net, noise, cond, steps)
            y0 = teacher_mid.detach()
            endpoint = teacher_half_from(y0, adapter.noise_pred_net, cond, steps).detach()
            second = teacher_half_from(student_mid, adapter.noise_pred_net, cond, steps).detach()
        history = [np.asarray(row, dtype=np.float64) for row in context["history"]]
        scene = int(context["scene"])
        if context["signature"] is not None:
            adapter.check_replay(scene, history, np.asarray(context["signature"]))
        prefix_a = endpoint.detach().cpu().numpy()[0][:config["execute_steps"]]
        prefix_b = second.detach().cpu().numpy()[0][:config["execute_steps"]]
        branches = {}
        for tag, prefix in (("start", prefix_a), ("end", prefix_b)):
            jac = finite_difference_jacobian(
                lambda chunk, _scene=scene, _history=history: adapter.execute_from_history(_scene, _history, chunk),
                prefix, epsilon=eps)
            dup = finite_difference_jacobian(
                lambda chunk, _scene=scene, _history=history: adapter.execute_from_history(_scene, _history, chunk),
                prefix, epsilon=eps)
            repeat_gap = float(np.abs(jac - dup).max())
            if not np.isfinite(repeat_gap) or repeat_gap > 1e-6:
                raise ValueError(f"repeated branch disagreement at scene {scene} "
                                 f"({tag}): {repeat_gap}")
            branches[tag] = jac
        jac_start = torch.from_numpy(branches["start"].astype(np.float32)).unsqueeze(0).to(device)

        def suffix(variable):
            return teacher_half_from(variable, adapter.noise_pred_net, cond, steps)

        entry = {"condition": context["condition"], "noise": context["noise"],
                 "midpoint": student_mid.detach().cpu().numpy()[0],
                 "target_mid": y0.detach().cpu().numpy()[0],
                 "target_end": second.detach().cpu().numpy()[0],
                 "teacher_end": endpoint.detach().cpu().numpy()[0],
                 "jacobian_start": branches["start"].astype(np.float32),
                 "jacobian_end": branches["end"].astype(np.float32),
                 "scene": scene, "history": context["history"]}
        if mode == "exact":
            metric = pullback_metric_exact(suffix, y0, jac_start,
                                           config["execute_steps"])
            entry["metric_exact"] = metric.detach().cpu().numpy()[0].astype(np.float32)
        else:
            entry["probes"] = pullback_probes(
                suffix, y0, jac_start, config["execute_steps"],
                num_probes=config["num_probes"]).detach().cpu().numpy()[0].astype(np.float32)
        out.append(entry)
    payload = dict(contexts=np.array(out, dtype=object),
                   meta=json.dumps({"epsilon": eps, "count": len(out),
                                    "metric_mode": mode,
                                    "num_probes": config["num_probes"]}),
                   fingerprint=json.dumps(C.cache_fingerprint(
                       config, config["repository"], extra=C.metric_extra_keys(config))),
                   lineage=json.dumps(dict(source_hashes=C.source_hashes(),
                                           cache=C.sha256_file(args.cache),
                                           cache_meta=info["meta"])))
    try:
        C.atomic_savez(output, **payload)
    except Exception as error:
        C.mark_incomplete(output, f"metrics failed: {error}")
        raise
    C.write_meta(meta_path, json.loads(payload["meta"]) | dict(
        fingerprint=json.loads(payload["fingerprint"]),
        lineage=json.loads(payload["lineage"])))
    print("wrote metrics " + output + " contexts " + str(len(out)))


def batch_from_contexts(contexts, indices, device, metric_mode):
    pick = [contexts[i] for i in indices]
    keys = ["condition", "noise", "midpoint", "target_mid", "target_end",
            "teacher_end", "jacobian_start", "jacobian_end"]
    keys.append("metric_exact" if metric_mode == "exact" else "probes")
    batch = {}
    for key in keys:
        values = np.stack([np.asarray(entry[key]) for entry in pick])
        if not np.isfinite(values).all():
            raise ValueError("nonfinite cached tensor " + key)
        batch[key] = torch.from_numpy(values).to(device)
    return batch


def cmd_train(args, config, device):
    output = args.output
    meta_path = output + ".meta.json"
    C.reserve_outputs([output, meta_path])
    protocol, protocol_id = resolve_protocol_id(args, config, device)
    data = np.load(args.cache, allow_pickle=True)
    info = C.verify_metric_cache(data, config, config["repository"],
                                 expect_scenes=C.scene_range(config, "train"))
    contexts = list(data["contexts"])
    n = len(contexts)
    rows = np.arange(n)
    metric_mode = config.get("metric_mode", "exact")
    metric_key = "metric_exact" if metric_mode == "exact" else "probes"
    cache_tensors = {"jacobian_start": torch.from_numpy(np.stack([c["jacobian_start"] for c in contexts])).float(),
                     "jacobian_end": torch.from_numpy(np.stack([c["jacobian_end"] for c in contexts])).float(),
                     metric_key: torch.from_numpy(np.stack([c[metric_key] for c in contexts])).float()}
    scales, scale_stats = metric_scales(cache_tensors, rows, metric_mode=metric_mode)
    student = TwoStepStudent(condition_dim=514, horizon=16, action_dim=2,
                             width=config["width"]).to(device)
    if args.initial:
        student, initial_payload = load_student(args.initial, config, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(args.seed)
    metric_rows = set(rng.choice(n, size=min(n, config["metric_contexts"]),
                                 replace=False).tolist())
    student.train()
    log = []
    grad_norms = []
    for update in range(args.updates):
        first = rng.integers(0, n, size=config["batch_size"] // 2).tolist()
        marked = rng.choice(sorted(metric_rows), size=config["batch_size"] // 2,
                            replace=True).tolist()
        indices = first + marked
        rng.shuffle(indices)
        batch = batch_from_contexts(contexts, indices, device, metric_mode)
        is_marked = torch.tensor([i in metric_rows for i in indices], device=device)
        order = torch.argsort(~is_marked)
        for key in batch:
            batch[key] = batch[key][order]
        metric_count = int(is_marked.sum().item())
        loss, parts = objective(student, batch, args.mode, config["execute_steps"],
                                {key: torch.tensor(value, device=device) for key, value in scales.items()},
                                metric_weight=config["metric_weight"],
                                anchor_weight=config["anchor_weight"],
                                metric_count=metric_count, metric_mode=metric_mode)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at update {update}")
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0,
                                                   error_if_nonfinite=True)
        optimizer.step()
        grad_norms.append(float(grad_norm))
        if update % max(1, args.updates // 100) == 0 or update == args.updates - 1:
            log.append({"update": update, "loss": float(loss.item()),
                        "base": float(parts["base"].item()),
                        "anchor": float(parts["anchor"].item()),
                        "penalty": float(parts["penalty"].item()),
                        "weighted_penalty": float(parts["weighted_penalty"].item()),
                        "grad_norm": float(grad_norm),
                        "mid_error_rms": float(parts["mid_error_rms"].item()),
                        "end_error_rms": float(parts["end_error_rms"].item()),
                        "endpoint_error_rms": float(parts["endpoint_error_rms"].item())})
    for tensor in student.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite parameter after training; refusing to save")
    payload = dict(student=student.state_dict(), mode=args.mode, seed=args.seed,
                   scales=scales, scale_stats=scale_stats,
                   metric_mode=metric_mode, scales_note="global training median",
                   protocol_id=protocol_id,
                   config_sha256=C.sha256_file(args.config),
                   init_sha256=C.sha256_file(args.initial) if args.initial else None,
                   cache_sha256=C.sha256_file(args.cache),
                   log=log, grad_norm_mean=float(np.mean(grad_norms)),
                   grad_norm_max=float(np.max(grad_norms)))
    C.atomic_save_torch(output, payload)
    C.write_meta(meta_path, dict(kind="student", mode=args.mode, seed=args.seed,
                                 metric_mode=metric_mode, protocol_id=protocol_id,
                                 scales=scales, scale_stats=scale_stats,
                                 grad_norm_mean=float(np.mean(grad_norms)),
                                 updates=args.updates,
                                 source_hashes=C.source_hashes()))
    print("wrote student " + output + " mode " + args.mode +
          " zero_sensitivity=" + str(scale_stats["zero_sensitivity_fraction"]))


def cmd_evaluate(args, config, device):
    output = args.output
    meta_path = output.replace(".jsonl", ".meta.json")
    C.reserve_outputs([output, meta_path])
    protocol, protocol_id = resolve_protocol_id(args, config, device)
    adapter = build_adapter(config, device)
    student = None
    intrinsic_mode = args.name
    intrinsic_seed = args.seed
    checkpoint_hash = None
    if args.student:
        student, payload = load_student(args.student, config, device)
        student.eval()
        intrinsic_mode = str(payload["mode"])
        intrinsic_seed = int(payload["seed"])
        checkpoint_hash = C.sha256_file(args.student)
    source = args.source or config["source"]
    steps = args.steps
    rng = np.random.default_rng(args.seed)
    episodes = args.episodes if args.episodes is not None else (
        config["development_episodes"] if args.development else config["test_episodes"])
    if episodes < 1:
        raise ValueError("--episodes must be >= 1")
    split = "development" if args.development else "test"
    start = config["development_scene_start"] if args.development else config["test_scene_start"]
    scenes = [start + episode for episode in range(episodes)]
    unknown = sorted(set(scenes) - C.scene_range(config, split))
    if unknown:
        raise ValueError(f"evaluation scenes outside the {split} range: {unknown[:5]}")
    C.validate_scene_sets({split: scenes, "train": sorted(C.scene_range(config, "train"))})
    max_decisions = config["episode_steps"] // config["execute_steps"] + 2
    print(f"[eval] device={device} source={source} steps={steps} "
          f"split={split} episodes={len(scenes)} intrinsic_mode={intrinsic_mode}")
    with open(output, "w") as handle:
        for scene in scenes:
            policy_noise = rng.standard_normal((max_decisions, 16, 2)).astype(np.float32)
            env = adapter.new_env(image=True)
            obs, _ = adapter.reset(env, scene)
            done = False
            steps_taken = 0
            best = 0.0
            latencies = []
            head_latencies = []
            oor = 0
            total_cmds = 0
            clipped = 0
            terminated_any = False
            truncated_any = False
            decision = 0
            while not done and steps_taken < config["episode_steps"]:
                cond = adapter.encode_observation(obs["image"], obs["agent_pos"])
                slot = min(decision, max_decisions - 1)
                noise = policy_noise[slot][None, ...]
                noise_t = torch.from_numpy(noise).to(device)
                tick = time.time()
                head_tick = time.time()
                with torch.no_grad():
                    if student is not None:
                        chunk = student.sample(noise_t, cond).cpu().numpy()[0]
                    else:
                        chunk = teacher_rollout(adapter.noise_pred_net, noise_t,
                                                cond, steps).cpu().numpy()[0]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                head_latencies.append((time.time() - head_tick) * 1000.0)
                latencies.append((time.time() - tick) * 1000.0)
                decision += 1
                prefix = chunk[:config["execute_steps"]]
                raw = adapter.decode_raw(prefix)
                raw_oor, raw_total = adapter.count_out_of_range(raw)
                physical = adapter.decode(prefix)
                if adapter.clip_actions:
                    clipped += int(np.count_nonzero(
                        (physical < 0) | (physical > 512)))
                for action in physical:
                    total_cmds += 1
                    oor += int(np.count_nonzero((action < 0) | (action > 512)))
                    obs, reward, terminated, truncated, done = adapter.raw_step(env, action)
                    best = max(best, reward)
                    terminated_any = terminated_any or terminated
                    truncated_any = truncated_any or truncated
                    steps_taken += 1
                    if done or steps_taken >= config["episode_steps"]:
                        break
            env.close()
            capped = bool(steps_taken >= config["episode_steps"] and not terminated_any)
            success = int(best >= 0.95 and terminated_any)
            handle.write(json.dumps({
                "method": intrinsic_mode, "alias": args.name,
                "intrinsic_mode": intrinsic_mode, "intrinsic_seed": intrinsic_seed,
                "training_seed": intrinsic_seed, "scene": scene, "split": split,
                "success": success, "terminated": int(terminated_any),
                "truncated": int(truncated_any), "episode_cap_reached": int(capped),
                "score": best, "steps": steps_taken,
                "source": source, "teacher_steps": steps,
                "latency_median_ms": float(np.median(latencies)),
                "latency_p95_ms": float(np.quantile(latencies, 0.95)),
                "action_head_latency_median_ms": float(np.median(head_latencies)),
                "oor_fraction_raw": raw_oor / max(raw_total, 1),
                "oor_fraction_executed": oor / max(total_cmds, 1),
                "clip_fraction": clipped / max(raw_total, 1),
                "protocol_id": protocol_id,
                "checkpoint_sha256": checkpoint_hash}) + "\n")
    C.write_meta(meta_path, dict(kind="evaluation", alias=args.name,
                                 intrinsic_mode=intrinsic_mode,
                                 intrinsic_seed=intrinsic_seed,
                                 checkpoint_sha256=checkpoint_hash,
                                 protocol_id=protocol_id, split=split,
                                 scenes=scenes, source=source,
                                 teacher_steps=steps, device=str(device),
                                 clip_actions=bool(config["clip_actions"]),
                                 methods=[intrinsic_mode],
                                 source_hashes=C.source_hashes()))
    print("wrote eval " + output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    entry = sub.add_parser("normalizer")
    entry.add_argument("--dataset", required=True)
    entry.add_argument("--output", required=True)
    entry = sub.add_parser("collect")
    entry.add_argument("--output", required=True)
    entry.add_argument("--student", default=None)
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--episodes", type=int, default=None)
    entry = sub.add_parser("metrics")
    entry.add_argument("--cache", required=True)
    entry.add_argument("--output", required=True)
    entry.add_argument("--limit", type=int, default=None)
    entry = sub.add_parser("train")
    entry.add_argument("--cache", required=True)
    entry.add_argument("--output", required=True)
    entry.add_argument("--mode", required=True,
                       choices=["uniform", "prefix", "endpoint", "pullback", "identity", "scalar"])
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--updates", type=int, default=None)
    entry.add_argument("--initial", default=None)
    entry = sub.add_parser("evaluate")
    entry.add_argument("--output", required=True)
    entry.add_argument("--name", required=True)
    entry.add_argument("--student", default=None)
    entry.add_argument("--steps", type=int, default=None)
    entry.add_argument("--source", default=None, choices=["gaussian", "uniform"])
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--episodes", type=int, default=None)
    entry.add_argument("--development", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config["device"])
    if args.command == "train":
        if args.updates is None:
            args.updates = config["updates"]
        if args.updates < 1:
            raise ValueError("--updates must be >= 1")
    if args.command == "evaluate" and args.steps is None:
        args.steps = config["teacher_steps"]
    if args.command == "evaluate" and args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.command == "normalizer":
        cmd_normalizer(args, config, device)
    elif args.command == "collect":
        cmd_collect(args, config, device)
    elif args.command == "metrics":
        cmd_metrics(args, config, device)
    elif args.command == "train":
        cmd_train(args, config, device)
    elif args.command == "evaluate":
        cmd_evaluate(args, config, device)


if __name__ == "__main__":
    main()
