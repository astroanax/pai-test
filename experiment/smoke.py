import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C
from core import TwoStepStudent, objective, pullback_metric_exact
from hri_adapter import (check_batch_independence, check_half_map_consistency,
                         resolve_device, teacher_half_from, teacher_rollout)


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def run(config, device, output, modes, updates, seed):
    from pilot import build_adapter
    checks = {}
    assets = {"checkpoint": C.sha256_file(config["checkpoint"]),
              "normalizer": C.sha256_file(config["normalizer"])}

    def record(name, fn):
        start = time.time()
        try:
            value = fn()
            checks[name] = {"ok": True, "detail": value,
                            "seconds": round(time.time() - start, 3)}
        except Exception as error:
            checks[name] = {"ok": False, "error": f"{type(error).__name__}: {error}",
                            "seconds": round(time.time() - start, 3)}
        print(f"[smoke] {name}: {'ok' if checks[name]['ok'] else 'FAIL'}")

    adapter = {}

    def build():
        adapter["a"] = build_adapter(config, device)
        return dict(condition_dim=514, keys=list(config.keys())[:3],
                    clip_actions=config["clip_actions"])
    record("adapter_strict_load", build)
    if "a" not in adapter:
        return finish(checks, assets, config, output)

    a = adapter["a"]

    def envs():
        out = {}
        for image in (True, False):
            env = a.new_env(image=image)
            obs, _ = a.reset(env, config["development_scene_start"])
            action = np.array([256.0, 256.0])
            if image:
                image_obs = a.encode_observation(obs["image"], obs["agent_pos"])
                assert image_obs.shape == (1, 514), "condition shape"
                assert np.isfinite(image_obs.numpy()).all(), "nonfinite condition"
                assert float(np.asarray(obs["image"]).min()) >= 0.0
                assert float(np.asarray(obs["image"]).max()) <= 1.0 + 1e-6
            else:
                assert np.isfinite(a.signature(env)).all(), "nonfinite reset state"
            _, reward, terminated, truncated, done = a.raw_step(env, action)
            assert np.isfinite(reward), "nonfinite reward"
            out["image" if image else "state"] = dict(
                terminated=bool(terminated), truncated=bool(truncated),
                done=bool(done), reward=float(reward))
            env.close()
        return out
    record("env_reset_and_step", envs)

    cond = None

    def half_maps():
        nonlocal cond
        import hri_adapter as H
        env = a.new_env(image=True)
        obs, _ = a.reset(env, config["development_scene_start"])
        cond = a.encode_observation(obs["image"], obs["agent_pos"])
        env.close()
        noise = torch.randn(2, 16, 2, generator=torch.Generator().manual_seed(0)).to(device)
        cond2 = cond.repeat(2, 1)
        gap = H.check_half_map_consistency(a.noise_pred_net, noise, cond2,
                                           config["teacher_steps"])
        full = teacher_rollout(a.noise_pred_net, noise, cond2, config["teacher_steps"])
        assert torch.isfinite(full).all(), "nonfinite teacher solve"
        return dict(max_gap=gap)
    record("teacher_half_map_consistency", half_maps)

    def suffix_vjp():
        assert cond is not None, "condition missing"
        noise = torch.randn(1, 16, 2, generator=torch.Generator().manual_seed(1)).to(device)
        target = teacher_half_from(noise, a.noise_pred_net, cond, config["teacher_steps"]).detach()
        direction = torch.randn_like(target)
        eps = 1e-4

        def suffix(variable):
            return teacher_half_from(variable, a.noise_pred_net, cond,
                                     config["teacher_steps"])
        with torch.enable_grad():
            variable = target.clone().requires_grad_(True)
            completed = suffix(variable)
            cotangent = torch.zeros_like(completed)
            cotangent[:, :config["execute_steps"]] = torch.randn(
                1, config["execute_steps"], 2,
                generator=torch.Generator().manual_seed(2)).to(device)
            gradient, = torch.autograd.grad(completed, variable,
                                            grad_outputs=cotangent)
        analytic = float((gradient * direction).sum())
        with torch.no_grad():
            f_plus = suffix(target + eps * direction)
            f_minus = suffix(target - eps * direction)
        numeric = float(((f_plus - f_minus) / (2 * eps) * cotangent).sum())
        scale = max(abs(analytic), 1e-6)
        rel = abs(analytic - numeric) / scale
        assert rel < 5e-2, f"directional derivative mismatch rel={rel}"
        teacher_params_frozen = all(not p.requires_grad for p in a.noise_pred_net.parameters())
        assert teacher_params_frozen, "teacher parameters are not frozen"
        return dict(analytic=analytic, numeric=numeric, relative_gap=rel)
    record("teacher_suffix_directional_derivative", suffix_vjp)

    def batch_independence():
        return dict(max_gap=check_batch_independence(a.noise_pred_net))
    record("teacher_batch_independence", batch_independence)

    def exact_metric():
        assert cond is not None
        noise = torch.randn(1, 16, 2, generator=torch.Generator().manual_seed(3)).to(device)
        target = teacher_half_from(noise, a.noise_pred_net, cond,
                                   config["teacher_steps"]).detach()
        jac = torch.randn(1, 32, config["execute_steps"] * 2).to(device) * 0.1

        def suffix(variable):
            return teacher_half_from(variable, a.noise_pred_net, cond,
                                     config["teacher_steps"])
        metric = pullback_metric_exact(suffix, target, jac, config["execute_steps"])
        assert metric.shape == (1, 32, config["execute_steps"] * 2), metric.shape
        assert torch.isfinite(metric).all(), "nonfinite exact metric"
        error = torch.randn_like(target) * 0.01
        projected = torch.bmm(metric, error.flatten(1).unsqueeze(-1)).squeeze(-1)
        return dict(shape=list(metric.shape),
                    quadratic=float(projected.square().sum()))
    record("exact_pullback_metric", exact_metric)

    def replay_agreement():
        histories = []
        for scene in (config["development_scene_start"],
                      config["development_scene_start"] + 1):
            env = a.new_env(image=False)
            obs, _ = a.reset(env, scene)
            history = []
            for step in range(24):
                action = np.array([256.0 + 8.0 * np.sin(step / 3.0),
                                   300.0 - 6.0 * step])
                obs, reward, terminated, truncated, done = a.raw_step(env, action)
                history.append(action.tolist())
                if done:
                    break
            sig = a.signature(env)
            env.close()
            gap = a.check_replay(scene, history, sig)
            histories.append(dict(scene=scene, steps=len(history), gap=gap,
                                  block_moved=bool(abs(sig[5]) + abs(sig[6]) > 0)))
        assert len({h["scene"] for h in histories}) == 2, "need two distinct scenes"
        assert any(h["block_moved"] for h in histories), "no motion detected in smoke histories"
        return histories
    record("replay_signature_agreement", replay_agreement)

    def tiny_train_step():
        student = TwoStepStudent(condition_dim=514, horizon=16, action_dim=2,
                                 width=config["width"]).to(device)
        batch = dict(condition=torch.randn(4, 514).to(device),
                     noise=torch.randn(4, 16, 2).to(device),
                     midpoint=torch.randn(4, 16, 2).to(device),
                     target_mid=torch.randn(4, 16, 2).to(device),
                     target_end=torch.randn(4, 16, 2).to(device),
                     teacher_end=torch.randn(4, 16, 2).to(device),
                     jacobian_start=torch.randn(4, 32, 16).to(device) * 0.05,
                     jacobian_end=torch.randn(4, 32, 16).to(device) * 0.05,
                     metric_exact=torch.randn(4, 32, 32).to(device) * 0.05)
        scales = {"endpoint": 1.0, "identity": 1.0, "pullback": 1.0}
        opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
        out = {}
        for mode in modes:
            loss, parts = objective(student, batch, mode, config["execute_steps"],
                                    scales, metric_count=2)
            assert torch.isfinite(loss), f"{mode}: nonfinite loss"
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0,
                                                       error_if_nonfinite=True)
            opt.step()
            assert np.isfinite(float(grad_norm)), f"{mode}: nonfinite grad norm"
            out[mode] = dict(loss=float(loss.item()),
                             penalty=float(parts["penalty"].item()),
                             grad_norm=float(grad_norm))
        frozen = all(not p.requires_grad for p in a.noise_pred_net.parameters())
        assert frozen, "teacher parameters received gradients"
        return out
    record("tiny_train_step_all_modes", tiny_train_step)

    return finish(checks, assets, config, output)


def finish(checks, assets, config, output):
    failed = sorted(name for name, value in checks.items() if not value["ok"])
    report = dict(passed=not failed, failed=failed, checks=checks,
                  source_hashes=C.source_hashes(), asset_hashes=assets,
                  device=str(config.get("device")),
                  teacher_steps=config["teacher_steps"],
                  execute_steps=config["execute_steps"],
                  metric_mode=config.get("metric_mode", "exact"),
                  manifest=C.check_manifest(),
                  packages=C.package_versions(["torch", "torchvision", "numpy",
                                               "gym", "pymunk", "pygame", "imageio_ffmpeg"]),
                  timestamp=time.time())
    C.reserve_outputs([output])
    C.write_meta(output, report)
    print(f"[smoke] {'PASS' if report['passed'] else 'FAIL'} -> {output}")
    if failed:
        print("[smoke] failed checks: " + ", ".join(failed))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--output", default="runs/smoke_report.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--modes", nargs="*",
                        default=["uniform", "prefix", "endpoint", "pullback",
                                 "identity", "scalar"])
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config["device"])
    manifest = C.check_manifest()
    print("[smoke] sources: " + json.dumps(manifest))
    return run(config, device, args.output, args.modes, config["updates"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
