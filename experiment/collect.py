"""History-cache producer: teacher histories + warm-student histories.

Stage 2 (teacher histories): roll out the frozen teacher in closed loop
over train/validation scenes, recording (history_id, scene, split,
decision, physical action history, replay signatures, condition,
teacher endpoint target, collector hash).

Stage 4 (warm histories): same collection driven by a frozen warm
checkpoint instead of the teacher. The warm checkpoint identity is
recorded per context; the pair bank and lock verify the collector is
the locked warm start.

Determinism: scene order fixed; per-(scene, decision) teacher latents
from stable_seed. No test scenes, no outcomes beyond signatures.
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
    from hri_adapter import HRIAdapter, load_checkpoint
    stats = {key: {inner: np.asarray(val) for inner, val in section.items()}
             for key, section in
             np.load(config["normalizer"], allow_pickle=True)["stats"].item().items()}
    sys.path.insert(0, os.path.join(config["repository"], "external", "models"))
    vision_encoder, noise_pred_net = load_checkpoint(config["checkpoint"], device)
    return HRIAdapter(vision_encoder, noise_pred_net, stats, device,
                      legacy=config["legacy"],
                      clip_actions=config.get("clip_actions", False))


def collect_histories(adapter, config, device, scenes, split, driver,
                      decision_limit=None, seed=0):
    """Roll out `driver` (teacher field or student) receding-horizon.

    driver(condition_t) -> normalized action chunk tensor (16, 2).
    Records one context per decision with live signatures and the
    driver's endpoint target for that decision's latent.
    """
    from hri_adapter import normalize_data
    contexts = []
    source = config["canonical_source"]
    steps = int(config["teacher_steps"])
    for scene in scenes:
        env = adapter.new_env(image=True)
        obs, _ = adapter.reset(env, scene)
        done = False
        steps_taken = 0
        history = []
        decision = 0
        sig_initial = adapter.signature(env).tolist()
        while not done and steps_taken < config["episode_steps"]:
            if decision_limit is not None and decision >= decision_limit:
                break
            cond = adapter.encode_observation(obs["image"], obs["agent_pos"])
            sig_live = adapter.signature(env).tolist()
            latent = torch.from_numpy(
                np.random.default_rng(
                    stable_seed("collect", scene, decision, seed, 0)
                ).standard_normal((1, 16, 2)).astype(np.float32)).to(device)
            with torch.no_grad():
                chunk = driver(cond, latent)
            if not torch.isfinite(chunk).all():
                raise ValueError(f"nonfinite driver chunk at scene {scene}")
            endpoint = chunk.detach().cpu().numpy()[0].astype(np.float32)
            prepared = adapter.prepare_commands(
                chunk.detach().cpu().numpy()[0][:config["execute_steps"]])
            contexts.append(dict(
                history_id=f"{split}-{scene}-{decision}",
                scene=int(scene), split=split, decision=int(decision),
                history=[list(map(float, row)) for row in history],
                sig_initial=sig_initial, sig_live=sig_live,
                condition=cond.detach().cpu().numpy()[0].astype(np.float32),
                endpoint=endpoint,
                collector_hash=collect_histories.collector_hash))
            for action in prepared["prepared"]:
                obs, _, terminated, _, done = adapter.raw_step(env, action)
                history.append([float(v) for v in np.asarray(action).tolist()])
                steps_taken += 1
                if done or steps_taken >= config["episode_steps"]:
                    break
            decision += 1
        env.close()
    return contexts


collect_histories.collector_hash = None


def cmd_collect(args, config, device):
    from hri_adapter import resolve_device  # noqa: bound at main
    adapter = build_adapter(config, device)
    if args.student is not None:
        payload = torch.load(args.student, map_location=device,
                             weights_only=True)
        if not isinstance(payload, dict) or "student" not in payload:
            raise ValueError("warm checkpoint lacks a student state dict")
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        from gad_reference import make_student
        student = make_student()
        student.load_state_dict(payload["student"])
        student.to(device).eval()
        for p in student.parameters():
            p.requires_grad_(False)

        def driver(cond, latent, _student=student):
            with torch.no_grad():
                return _student(latent, cond.repeat(latent.shape[0], 1))
        collect_histories.collector_hash = C.sha256_file(args.student)
        mode = "warm"
    else:
        def driver(cond, latent, _adapter=adapter, _steps=int(config["teacher_steps"]),
                   _source=config["canonical_source"]):
            return teacher_endpoint(_adapter.field, latent, cond, _steps,
                                    source=_source)
        collect_histories.collector_hash = C.sha256_file(config["checkpoint"])
        mode = "teacher"
    splits = args.splits or ["train", "validation"]
    contexts = []
    for split in splits:
        if split not in ("train", "validation"):
            raise ValueError("collect supports train and validation splits")
        scenes = sorted(C.scene_range(config, split))
        if args.episodes is not None:
            scenes = scenes[:int(args.episodes)]
        contexts.extend(collect_histories(
            adapter, config, device, scenes, split, driver,
            decision_limit=args.decision_limit, seed=args.seed))
    if not contexts:
        raise ValueError("collection produced no histories")
    C.write_history_cache(args.output, contexts, config)
    print(f"wrote {mode} history cache {args.output} "
          f"contexts {len(contexts)}")
    return dict(contexts=len(contexts), mode=mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--allow-unlocked", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    entry = sub.add_parser("collect")
    entry.add_argument("--output", required=True)
    entry.add_argument("--student", default=None)
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--episodes", type=int, default=None)
    entry.add_argument("--splits", nargs="*", default=None)
    entry.add_argument("--decision-limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.protocol:
        protocol = C.load_protocol(args.protocol)
        C.verify_current_protocol(protocol, config, args.config,
                                  stage="collect")
    elif not args.allow_unlocked:
        raise C.ProtocolError("collect needs --protocol or --allow-unlocked")
    from hri_adapter import resolve_device
    device = resolve_device(args.device or config["device"])
    C.reserve_outputs([args.output])
    try:
        cmd_collect(args, config, device)
    except Exception as error:
        C.mark_incomplete(args.output, f"{type(error).__name__}: {error}")
        raise
    else:
        C.complete_output(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
