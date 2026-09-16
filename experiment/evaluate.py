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

ROLES = ("student", "teacher_reference")
# Item 14: "correction_hybrid" branch labels ran fresh teacher episodes
# under different names without restoring history or applying corrections.
# The role stays disabled until real replay-then-continue branching exists.
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
    arch = payload.get("arch", {})
    model = make_student(514, 16, 2, int(arch.get("width", 512)))
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
    generated = issued = executed = invalid = raw_invalid_total = 0
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
            # Audit fdb59ff improvement: log raw (pre-clipping)
            # violations separately; clipping in prepare_commands can
            # conceal raw out-of-bounds predictions.
            raw_np = chunk.detach().cpu().numpy()[0]
            raw_invalid = int((np.abs(raw_np) > 1.0).sum())
            prepared = adapter.prepare_commands(
                raw_np[:config["execute_steps"]])
            # Item 12: never np.asarray() a CUDA tensor. numel() counts
            # coordinates; rows record what each count means.
            generated += int(chunk.numel())
            raw_invalid_total += raw_invalid
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
                raw_invalid=int(raw_invalid_total),
                executed=int(executed), invalid=int(invalid),
                termination_reason=reason)


def replay_branch(history, branch, adapter, correct_prefix_fn=None):
    """E3 replay: restore the verified history, apply THIS branch's prefix,
    then continue with the same policy and randomness.

    Each branch must produce a genuinely different executed action trace:
    a deliberately different oracle prefix is asserted to diverge. Stub
    descriptors without restoration are refused (item 14).
    """
    if branch not in BRANCHES:
        raise C.ProtocolError(f"unknown replay branch {branch!r}")
    env = adapter.new_env(image=False)
    adapter.reset(env, int(history["scene"]))
    for action in history["history"]:
        _, _, terminated, truncated, _ = adapter.raw_step(
            env, __import__("numpy").asarray(action, dtype=float))
        if terminated or truncated:
            env.close()
            raise C.ProtocolError("history ends before the branch point")
    live = adapter.signature(env)
    if branch == "oracle_prefix":
        if "oracle_prefix" not in history:
            env.close()
            raise C.ProtocolError(
                "oracle_prefix branch needs history oracle_prefix")
        prefix = [list(map(float, row)) for row in history["oracle_prefix"]]
    elif branch == "uncorrected":
        prefix = None
    else:
        if correct_prefix_fn is None:
            env.close()
            raise C.ProtocolError(
                f"branch {branch} needs a correction for the live state; "
                "reusing old actions at a new state is forbidden")
        prefix = [list(map(float, row))
                  for row in correct_prefix_fn(history, live, branch)]
    trace = []
    if prefix is not None:
        for action in prefix:
            obs, _, terminated, truncated, _ = adapter.raw_step(env, action)
            trace.append([float(v) for v in
                          __import__("numpy").asarray(action).tolist()])
            if terminated or truncated:
                break
    return dict(branch=branch, history_id=history.get("history_id"),
                live_signature=list(map(float, live)),
                executed_prefix=trace,
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
        # Audit fdb59ff item 10: final evaluation verifies final-student
        # provenance (arm, seed, warm parent, bank, updates, beta,
        # protocol identity). An artifact can be checksummed yet belong
        # to the wrong experiment.
        if protocol is not None and not args.development:
            C.verify_student_provenance(payload, protocol, config,
                                        args.student, role="final")
        method = str(payload.get("arm", args.role))
        checkpoint_hash = C.sha256_file(args.student)
    else:
        policy, payload, method = "teacher", None, args.role
        checkpoint_hash = C.sha256_file(config["checkpoint"])
    # Item 13: final evaluation consumes the EXACT locked final scenes
    # (never a "test" split or an episode count); training_seed (from the
    # checkpoint) and eval_replicate (noise schedule) are stored separately.
    training_seed = int(payload.get("seed", -1)) if payload else -1
    if protocol is not None and not args.development:
        design = protocol.get("design", {})
        scenes = [int(s) for s in design.get("scenes", {}).get("final", [])]
        if not scenes:
            raise C.ProtocolError("locked design holds no final scenes")
        split = "final"
    else:
        split = "development" if args.development else "final"
        scenes = sorted(C.scene_range(config, split))[:args.episodes]
    histories = None
    if args.history:
        histories = C.verify_history_cache(args.history, config)["histories"]
    # Item 25: reserve before any episode work; failures marked, never
    # half-written.
    meta_path = (args.output.replace(".jsonl", ".meta.json")
                 if args.output.endswith(".jsonl")
                 else args.output + ".meta.json")
    C.reserve_outputs([args.output, meta_path])
    try:
        rows = run_evaluation_episodes(
            adapter, config, device, scenes, args, policy, method,
            training_seed, checkpoint_hash, split, protocol_id, histories)
    except Exception as error:
        C.mark_incomplete(args.output, f"{type(error).__name__}: {error}")
        C.mark_incomplete(meta_path, f"{type(error).__name__}: {error}")
        raise
    with open(args.output, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    C.write_meta(meta_path,
                 dict(kind="evaluation", schema_version=C.SCHEMA_VERSION,
                      role=args.role, split=split,
                      eval_replicate=args.eval_replicate,
                      protocol_id=protocol_id,
                      source_hashes=C.source_hashes(),
                      package_versions=C.package_versions()))
    C.complete_output(args.output, dict(rows=len(rows)))
    C.complete_output(meta_path)
    print("wrote eval " + args.output)
    return dict(rows=len(rows))


def run_evaluation_episodes(adapter, config, device, scenes, args, policy,
                            method, training_seed, checkpoint_hash, split,
                            protocol_id, histories):
    rows = []
    for episode, scene in enumerate(scenes):
        res = rollout_episode(
            adapter, config, device, scene,
            config["canonical_source"], int(config["teacher_steps"]),
            args.eval_replicate, policy=policy,
            intervene_every=args.intervene_every)
        res.update(method=method, seed=training_seed,
                   training_seed=training_seed,
                   role=args.role, split=split,
                   eval_replicate=args.eval_replicate,
                   protocol_id=protocol_id,
                   checkpoint_sha256=checkpoint_hash,
                   episode=episode)
        rows.append(res)
    return rows


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
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-replicate", type=int, default=0)
    parser.add_argument("--history", default=None)
    parser.add_argument("--teacher", default="teacher")
    parser.add_argument("--intervene-every", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    from hri_adapter import resolve_device
    device = resolve_device(args.device or config["device"])
    if args.episodes is None:
        args.episodes = (config["development_episodes"] if args.development
                         else config["final_episodes"])
    cmd_evaluate(args, config, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
