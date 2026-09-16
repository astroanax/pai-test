"""E0/E3 episode evaluation with the real simulator harness.

Roles: student / teacher_reference / correction_hybrid. One row per
episode: closed-loop success/score/steps, inference calls, latency
categories, generated/issued/executed counts, termination reason,
protocol id.

Closed loop: encode observation -> latent q (deterministic per
(replicate, scene, decision)) -> teacher or student chunk -> locked
command preparation -> environment step. E3 replays branches
(uncorrected / corrected / oracle-prefix / teacher-corrected) from a
verified history plus one fixed competent teacher after correction;
unassisted full episodes run on fresh final scenes. Fixed intervention
schedules (e.g. every 5th decision) are supported with cost accounting.
Old corrective actions are never reused at new states.
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

ROLES = ("student", "teacher_reference", "correction_hybrid")
BRANCHES = ("uncorrected", "corrected", "oracle_prefix", "teacher_corrected")


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def build_adapter(config, device):
    from hri_adapter import HRIAdapter, load_checkpoint
    stats = {key: {inner: np.asarray(val) for inner, val in section.items()}
             for key, section in
             np.load(config["normalizer"], allow_pickle=True)["stats"].item().items()}
    sys.path.insert(0, os.path.join(config["repository"], "external", "models"))
    vision_encoder, noise_pred_net = load_checkpoint(config["checkpoint"], device)
    return HRIAdapter(vision_encoder, noise_pred_net, stats, device,
                      legacy=config["legacy"],
                      clip_actions=config.get("clip_actions", False))


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
    return model, payload


def rollout_episode(adapter, config, device, scene, source, steps,
                    seed, policy="teacher", intervene_every=None,
                    correct_fn=None, max_steps=None):
    """Real closed-loop episode. Returns termination/score/steps/counts.

    policy: "teacher" or a frozen student module. Latent q per decision
    from stable_seed("eval", scene, decision, seed, 0).
    """
    from gad_reference import stable_seed
    max_steps = int(max_steps or config["episode_steps"])
    env = adapter.new_env(image=True)
    obs, _ = adapter.reset(env, scene)
    done = False
    steps_taken = 0
    best = 0.0
    terminated_any = False
    truncated_any = False
    decision = 0
    generated = issued = executed = invalid = 0
    head_ms, dec_ms = [], []
    reason = "cap"
    sync = (lambda: torch.cuda.synchronize()) \
        if device.type == "cuda" else (lambda: None)
    try:
        while not done and steps_taken < max_steps:
            start = time.perf_counter()
            cond = adapter.encode_observation(obs["image"], obs["agent_pos"])
            latent = torch.from_numpy(
                np.random.default_rng(
                    stable_seed("eval", scene, decision, seed, 0)
                ).standard_normal((1, 16, 2)).astype(np.float32)).to(device)
            head_start = time.perf_counter()
            if isinstance(policy, str):
                if policy != "teacher":
                    raise ValueError(f"unknown policy {policy!r}")
                chunk = adapter.teacher_action(latent, cond, steps,
                                               source=source)
            else:
                with torch.no_grad():
                    chunk = policy(latent, cond.repeat(latent.shape[0], 1)
                                   if cond.shape[0] != latent.shape[0] else cond)
            sync()
            head_ms.append((time.perf_counter() - head_start) * 1000.0)
            prepared = adapter.prepare_commands(
                chunk.detach().cpu().numpy()[0][:config["execute_steps"]])
            generated += int(np.asarray(chunk).size)
            issued += len(prepared["prepared"])
            intervene = (intervene_every and decision % intervene_every == 0
                         and correct_fn is not None)
            actions = ([np.asarray(correct_fn(decision=decision,
                                              observation=obs),
                                   dtype=np.float64)]
                       if intervene else list(prepared["prepared"]))
            dec_ms.append((time.perf_counter() - start) * 1000.0)
            decision += 1
            for action in actions:
                obs, reward, terminated, truncated, done = adapter.raw_step(
                    env, action)
                best = max(best, reward)
                terminated_any = terminated_any or terminated
                truncated_any = truncated_any or truncated
                executed += 1
                a = np.asarray(action, dtype=np.float64)
                invalid += int(((a < 0.0) | (a > 512.0)).sum())
                steps_taken += 1
                if done or steps_taken >= max_steps:
                    break
            if terminated_any:
                reason = "terminated"
    finally:
        env.close()
    success = int(terminated_any and best >= 0.95)
    return dict(scene=int(scene), terminated=terminated_any,
                truncated=truncated_any, success=success, score=float(best),
                steps=int(steps_taken), decisions=int(decision),
                decision_latency_median_ms=float(np.median(dec_ms)),
                decision_latency_p95_ms=float(np.quantile(dec_ms, 0.95)),
                action_head_latency_median_ms=float(np.median(head_ms)),
                generated=int(generated), issued=int(issued),
                executed=int(executed), invalid=int(invalid),
                termination_reason=reason)


def replay_branch(history, branch, correct_fn=None):
    """E3 replay descriptor: corrections recomputed at the live state."""
    if branch not in BRANCHES:
        raise C.ProtocolError(f"unknown replay branch {branch!r}")
    if branch == "oracle_prefix" and "oracle_prefix" not in history:
        raise C.ProtocolError("oracle_prefix branch needs history oracle_prefix")
    return dict(branch=branch, history_id=history.get("history_id"),
                corrected=branch != "uncorrected")


def cmd_evaluate(args, config, device):
    protocol, protocol_id = None, None
    if args.protocol:
        protocol = C.load_protocol(args.protocol)
        C.verify_current_protocol(protocol, config, args.config,
                                  stage="evaluate")
        protocol_id = protocol.get("protocol_id")
    elif not args.allow_unlocked:
        raise C.ProtocolError("evaluate requires --protocol or --allow-unlocked")
    if args.role not in ROLES:
        raise C.ProtocolError(f"role {args.role!r} not in {ROLES}")
    adapter = build_adapter(config, device)
    if args.role == "student":
        if not args.student:
            raise C.ProtocolError("student role requires --student")
        policy, payload = load_student_model(args.student, device)
        method = str(payload.get("arm", args.role))
        checkpoint_hash = C.sha256_file(args.student)
    else:
        policy, payload, method = "teacher", None, args.role
        checkpoint_hash = C.sha256_file(config["checkpoint"])
    split = "development" if args.development else "test"
    scenes = sorted(C.scene_range(config, split))[:args.episodes]
    histories = None
    if args.history:
        histories = C.verify_history_cache(args.history, config)["histories"]
    rows = []
    for episode, scene in enumerate(scenes):
        if args.role == "correction_hybrid" and histories is not None:
            history = histories[episode % len(histories)]
            for branch in BRANCHES:
                info = replay_branch(history, branch)
                res = rollout_episode(
                    adapter, config, device, scene,
                    config["canonical_source"], int(config["teacher_steps"]),
                    args.eval_replicate, policy=policy,
                    intervene_every=args.intervene_every)
                res.update(branch=branch, teacher="teacher",
                           history_id=info["history_id"],
                           method=method, seed=args.eval_replicate,
                           role=args.role, split=split,
                           eval_replicate=args.eval_replicate,
                           protocol_id=protocol_id,
                           checkpoint_sha256=checkpoint_hash,
                           episode=episode)
                rows.append(res)
        else:
            res = rollout_episode(
                adapter, config, device, scene,
                config["canonical_source"], int(config["teacher_steps"]),
                args.eval_replicate, policy=policy,
                intervene_every=args.intervene_every)
            res.update(method=method, seed=args.eval_replicate,
                       role=args.role, split=split,
                       eval_replicate=args.eval_replicate,
                       protocol_id=protocol_id,
                       checkpoint_sha256=checkpoint_hash,
                       episode=episode)
            rows.append(res)
    C.reserve_outputs([args.output])
    with open(args.output, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    C.write_meta(args.output.replace(".jsonl", ".meta.json"),
                 dict(kind="evaluation", schema_version=C.SCHEMA_VERSION,
                      role=args.role, split=split,
                      eval_replicate=args.eval_replicate,
                      protocol_id=protocol_id,
                      source_hashes=C.source_hashes(),
                      package_versions=C.package_versions()))
    C.complete_output(args.output, dict(rows=len(rows)))
    print("wrote eval " + args.output)
    return dict(rows=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--role", default="student", choices=ROLES)
    parser.add_argument("--student", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--allow-unlocked", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-replicate", type=int, default=0)
    parser.add_argument("--history", default=None)
    parser.add_argument("--teacher", default="teacher")
    parser.add_argument("--intervene-every", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    from hri_adapter import resolve_device
    device = resolve_device(args.device or config["device"])
    cmd_evaluate(args, config, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
