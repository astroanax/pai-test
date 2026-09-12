import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import TwoStepStudent, finite_difference_jacobian, metric_scales, objective, pullback_probes
from hri_adapter import HRIAdapter, load_checkpoint


def load_config(path):
    with open(path) as handle:
        return json.load(handle)


def refuse_overwrite(path):
    if os.path.exists(path):
        raise ValueError("output already exists, refusing silent overwrite")


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


def cmd_normalizer(args, config):
    import zarr
    store = zarr.open(args.dataset, "r")
    action = store["data"]["action"][:]
    state = store["data"]["state"][:, :2]
    stats = {"action": {"min": action.reshape(-1, 2).min(0).tolist(), "max": action.reshape(-1, 2).max(0).tolist()},
             "agent_pos": {"min": state.reshape(-1, 2).min(0).tolist(), "max": state.reshape(-1, 2).max(0).tolist()}}
    refuse_overwrite(args.output)
    np.savez(args.output, stats=stats)
    print("wrote normalizer " + args.output)


def cmd_collect(args, config, device):
    from hri_adapter import teacher_rollout
    adapter = build_adapter(config, device)
    rng = np.random.default_rng(args.seed)
    student = None
    if args.student is not None:
        student = TwoStepStudent(condition_dim=514, horizon=16, action_dim=2, width=config["width"]).to(device)
        student.load_state_dict(torch.load(args.student, map_location=device, weights_only=True)["student"])
        student.eval()
    rows = []
    start = config["train_scene_start"]
    total = args.episodes if args.episodes else config["train_episodes"]
    for episode in range(total):
        scene = start + episode
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
            with torch.no_grad():
                if student is None:
                    chunk = teacher_rollout(adapter.noise_pred_net, noise_t, cond, config["teacher_steps"])
                else:
                    chunk = student.sample(noise_t, cond)
            chunk_n = chunk.detach().cpu().numpy()[0]
            physical = adapter.decode(chunk_n[:config["execute_steps"]])
            context = None
            if len(history) % 2 == 0 and len(record["contexts"]) < config["contexts_per_episode"]:
                if history:
                    check_env = adapter.replay(scene, history)
                    signature = [float(value) for value in adapter.signature(check_env).tolist()]
                    check_env.close()
                else:
                    signature = None
                context = {"condition": cond.detach().cpu().numpy()[0], "noise": noise[0],
                           "history": [list(map(float, row)) for row in history],
                           "signature": signature, "scene": scene}
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
    refuse_overwrite(args.output)
    np.savez(args.output, records=np.array(rows, dtype=object),
             meta=json.dumps({"mode": "shared", "student": args.student, "seed": args.seed}))
    print("wrote cache " + args.output + " episodes " + str(len(rows)))


def cmd_metrics(args, config, device):
    from hri_adapter import teacher_half_maps, teacher_rollout
    adapter = build_adapter(config, device)
    cache = np.load(args.cache, allow_pickle=True)
    records = cache["records"]
    contexts = []
    for record in records:
        contexts.extend(record["contexts"])
    if args.limit:
        contexts = contexts[:args.limit]
    eps = config["finite_difference_epsilon"]
    steps = config["teacher_steps"]
    out = []
    for context in contexts:
        cond = torch.from_numpy(np.asarray(context["condition"], dtype=np.float32)).unsqueeze(0).to(device)
        noise = torch.from_numpy(np.asarray(context["noise"], dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            half_mid, half_end = teacher_half_maps(adapter.noise_pred_net, noise, cond, steps)
            teacher_end = teacher_rollout(adapter.noise_pred_net, noise, cond, steps)
            teacher_midpoint_end = teacher_rollout(adapter.noise_pred_net, half_mid.detach(), cond, steps)
        history = [np.asarray(row, dtype=np.float64) for row in context["history"]]
        scene = int(context["scene"])
        if context["signature"] is not None:
            adapter.check_replay(scene, history, np.asarray(context["signature"]))
        prefix_a = teacher_end.detach().cpu().numpy()[0][:config["execute_steps"]]
        prefix_b = teacher_midpoint_end.detach().cpu().numpy()[0][:config["execute_steps"]]
        branches = {}
        for tag, prefix in (("start", prefix_a), ("end", prefix_b)):
            jac = finite_difference_jacobian(
                lambda chunk, _scene=scene, _history=history: adapter.execute_from_history(_scene, _history, chunk),
                prefix, epsilon=eps)
            dup = finite_difference_jacobian(
                lambda chunk, _scene=scene, _history=history: adapter.execute_from_history(_scene, _history, chunk),
                prefix, epsilon=eps)
            branches[tag] = (jac, float(np.abs(jac - dup).max()))
        suffix_target = half_mid.detach()
        jac_start = torch.from_numpy(branches["start"][0].astype(np.float32)).unsqueeze(0).to(device)

        def suffix(variable):
            return teacher_rollout(adapter.noise_pred_net, variable, cond, steps)

        probes = pullback_probes(suffix, suffix_target, jac_start,
                                 config["execute_steps"], num_probes=config["num_probes"]).detach().cpu().numpy()[0]
        out.append({"condition": context["condition"], "noise": context["noise"],
                    "midpoint": half_mid.detach().cpu().numpy()[0],
                    "target_mid": half_end.detach().cpu().numpy()[0],
                    "target_end": teacher_midpoint_end.detach().cpu().numpy()[0],
                    "teacher_end": teacher_end.detach().cpu().numpy()[0],
                    "jacobian_start": branches["start"][0].astype(np.float32),
                    "jacobian_end": branches["end"][0].astype(np.float32),
                    "probes": probes.astype(np.float32),
                    "scene": scene, "history": context["history"]})
    refuse_overwrite(args.output)
    np.savez(args.output, contexts=np.array(out, dtype=object),
             meta=json.dumps({"epsilon": eps, "count": len(out)}))
    print("wrote metrics " + args.output + " contexts " + str(len(out)))


def batch_from_contexts(contexts, indices, device):
    pick = [contexts[i] for i in indices]
    batch = {}
    for key in ("condition", "noise", "midpoint", "target_mid", "target_end", "teacher_end",
                "jacobian_start", "jacobian_end", "probes"):
        batch[key] = torch.from_numpy(np.stack([np.asarray(entry[key]) for entry in pick])).to(device)
    return batch


def cmd_train(args, config, device):
    data = np.load(args.cache, allow_pickle=True)
    contexts = list(data["contexts"])
    n = len(contexts)
    rows = np.arange(n)
    cache_tensors = {"jacobian_start": torch.from_numpy(np.stack([c["jacobian_start"] for c in contexts])).float(),
                     "jacobian_end": torch.from_numpy(np.stack([c["jacobian_end"] for c in contexts])).float(),
                     "probes": torch.from_numpy(np.stack([c["probes"] for c in contexts])).float()}
    scales = metric_scales(cache_tensors, rows)
    student = TwoStepStudent(condition_dim=514, horizon=16, action_dim=2, width=config["width"]).to(device)
    if args.initial:
        student.load_state_dict(torch.load(args.initial, map_location=device, weights_only=True)["student"])
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rng = np.random.default_rng(args.seed)
    metric_rows = set(rng.choice(n, size=min(n, config["metric_contexts"]), replace=False).tolist())
    student.train()
    for update in range(args.updates):
        first = rng.integers(0, n, size=config["batch_size"] // 2).tolist()
        marked = rng.choice(sorted(metric_rows), size=config["batch_size"] // 2, replace=True).tolist()
        indices = first + marked
        rng.shuffle(indices)
        batch = batch_from_contexts(contexts, indices, device)
        is_marked = torch.tensor([i in metric_rows for i in indices], device=device)
        order = torch.argsort(~is_marked)
        for key in batch:
            batch[key] = batch[key][order]
        metric_count = int(is_marked.sum().item())
        loss, parts = objective(student, batch, args.mode, config["execute_steps"],
                                {key: torch.tensor(value, device=device) for key, value in scales.items()},
                                metric_weight=config["metric_weight"],
                                anchor_weight=config["anchor_weight"], metric_count=metric_count)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
    refuse_overwrite(args.output)
    torch.save({"student": student.state_dict(), "mode": args.mode, "seed": args.seed, "scales": scales}, args.output)
    print("wrote student " + args.output + " mode " + args.mode)


def cmd_evaluate(args, config, device):
    from hri_adapter import teacher_rollout
    adapter = build_adapter(config, device)
    student = None
    if args.student:
        student = TwoStepStudent(condition_dim=514, horizon=16, action_dim=2, width=config["width"]).to(device)
        student.load_state_dict(torch.load(args.student, map_location=device, weights_only=True)["student"])
        student.eval()
    rng = np.random.default_rng(args.seed)
    episodes = args.episodes if args.episodes else config["test_episodes"]
    start = config["development_scene_start"] if args.development else config["test_scene_start"]
    refuse_overwrite(args.output)
    with open(args.output, "w") as handle:
        for episode in range(episodes):
            scene = start + episode
            policy_noise = rng.standard_normal((40, 16, 2)).astype(np.float32)
            env = adapter.new_env(image=True)
            obs, _ = adapter.reset(env, scene)
            done = False
            steps = 0
            best = 0.0
            latencies = []
            oor = 0
            total_cmds = 0
            while not done and steps < config["episode_steps"]:
                cond = adapter.encode_observation(obs["image"], obs["agent_pos"])
                slot = min(steps // config["execute_steps"], 39)
                noise = policy_noise[slot][None, ...]
                noise_t = torch.from_numpy(noise).to(device)
                tick = time.time()
                with torch.no_grad():
                    if student is not None:
                        chunk = student.sample(noise_t, cond).cpu().numpy()[0]
                    else:
                        chunk = teacher_rollout(adapter.noise_pred_net, noise_t, cond, args.steps).cpu().numpy()[0]
                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.time() - tick) * 1000.0)
                physical = adapter.decode(chunk[:config["execute_steps"]])
                for action in physical:
                    total_cmds += 1
                    if bool((action < 0).any() or (action > 512).any()):
                        oor += 1
                    obs, reward, done, _ = adapter.step(env, action)
                    best = max(best, float(reward))
                    steps += 1
                    if done or steps >= config["episode_steps"]:
                        break
            env.close()
            handle.write(json.dumps({"method": args.name, "training_seed": args.seed, "scene": scene,
                                     "success": int(best >= 0.95), "score": best, "steps": steps,
                                     "latency_median_ms": float(np.median(latencies)),
                                     "latency_p95_ms": float(np.quantile(latencies, 0.95)),
                                     "oor_fraction": oor / max(total_cmds, 1)}) + "\n")
    print("wrote eval " + args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
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
    entry.add_argument("--updates", type=int, default=6000)
    entry.add_argument("--initial", default=None)
    entry = sub.add_parser("evaluate")
    entry.add_argument("--output", required=True)
    entry.add_argument("--name", required=True)
    entry.add_argument("--student", default=None)
    entry.add_argument("--steps", type=int, default=16)
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--episodes", type=int, default=None)
    entry.add_argument("--development", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    device = args.device or config["device"]
    if args.command == "normalizer":
        cmd_normalizer(args, config)
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
