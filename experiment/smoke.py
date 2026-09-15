import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C
from core import (TwoStepStudent, objective, pullback_metric_exact,
                  pullback_probes, pullback_quadratic, sample_source)
from hri_adapter import (check_batch_independence, check_devices,
                         check_half_map_consistency, resolve_device,
                         teacher_half_from, teacher_rollout)


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def run(config, device, output, modes):
    from pilot import build_adapter
    checks = {}
    assets = {"checkpoint": C.sha256_file(config["checkpoint"]),
              "normalizer": C.sha256_file(config["normalizer"])}

    def record(name, fn):
        start = time.perf_counter()
        try:
            value = fn()
            checks[name] = {"ok": True, "detail": value,
                            "seconds": round(time.perf_counter() - start, 3)}
        except Exception as error:
            checks[name] = {"ok": False,
                            "error": f"{type(error).__name__}: {error}",
                            "seconds": round(time.perf_counter() - start, 3)}
        print(f"[smoke] {name}: {'ok' if checks[name]['ok'] else 'FAIL'}", flush=True)

    state = {}

    def build():
        adapter = build_adapter(config, device)
        state["adapter"] = adapter
        return dict(device=str(adapter.device),
                    clip_actions=config["clip_actions"],
                    encoder_device=str(next(adapter.vision_encoder.parameters()).device),
                    field_device=str(next(adapter.noise_pred_net.parameters()).device))
    record("adapter_strict_load", build)
    if "adapter" not in state:
        return finish(checks, assets, config, output)
    adapter = state["adapter"]

    def devices():
        cond = torch.randn(1, 514, device=device)
        actions = torch.randn(1, 16, 2, device=device)
        report = check_devices(adapter.vision_encoder, adapter.noise_pred_net, cond,
                               actions, device)
        mismatch = False
        try:
            check_devices(adapter.vision_encoder, adapter.noise_pred_net,
                          cond.cpu(), actions, device)
        except ValueError:
            mismatch = True
        assert mismatch, "cpu condition was not rejected on a cuda adapter"
        return dict(devices=report, cpu_condition_rejected=True)
    record("device_placement", devices)

    def envs():
        out = {}
        for image in (True, False):
            env = adapter.new_env(image=image)
            _, _ = adapter.reset(env, config["development_scene_start"])
            if image:
                cond = adapter.encode_observation(obs["image"], obs["agent_pos"])
                state["condition"] = cond
                assert cond.shape == (1, 514), cond.shape
                assert torch.isfinite(cond).all(), "nonfinite condition"
                image_values = np.asarray(obs["image"])
                assert image_values.min() >= 0.0 and image_values.max() <= 1.0 + 1e-6
            else:
                assert np.isfinite(adapter.signature(env)).all(), "nonfinite state"
            _, reward, terminated, truncated, done = adapter.raw_step(
                env, np.array([256.0, 256.0]))
            assert np.isfinite(reward), "nonfinite reward"
            out["image" if image else "state"] = dict(
                terminated=bool(terminated), truncated=bool(truncated),
                done=bool(done), reward=float(reward))
            env.close()
        return out
    record("env_reset_and_step", envs)

    def half_maps():
        noise = torch.randn(2, 16, 2, device=device,
                            generator=torch.Generator(device=device).manual_seed(0))
        cond = state["condition"].repeat(2, 1)
        gap = check_half_map_consistency(adapter.noise_pred_net, noise, cond,
                                         config["teacher_steps"])
        full = teacher_rollout(adapter.noise_pred_net, noise, cond,
                               config["teacher_steps"])
        assert torch.isfinite(full).all(), "nonfinite teacher solve"
        return dict(max_gap=gap)
    record("teacher_half_map_consistency", half_maps)

    def suffix_vjp():
        noise = torch.randn(1, 16, 2, device=device,
                            generator=torch.Generator(device=device).manual_seed(1))
        target = teacher_half_from(noise, adapter.noise_pred_net,
                                   state["condition"],
                                   config["teacher_steps"]).detach()
        direction = torch.randn(target.shape, device=device,
                                generator=torch.Generator(device=device).manual_seed(2))
        epsilon = 1e-4

        def suffix(variable):
            return teacher_half_from(variable, adapter.noise_pred_net,
                                     state["condition"], config["teacher_steps"])
        with torch.enable_grad():
            variable = target.clone().requires_grad_(True)
            completed = suffix(variable)
            cotangent = torch.zeros_like(completed)
            cotangent[:, :config["execute_steps"]] = torch.randn(
                completed.shape[0], config["execute_steps"], 2, device=device,
                generator=torch.Generator(device=device).manual_seed(3))
            gradient, = torch.autograd.grad(completed, variable,
                                            grad_outputs=cotangent)
        analytic = float((gradient * direction).sum())
        with torch.no_grad():
            plus = suffix(target + epsilon * direction)
            minus = suffix(target - epsilon * direction)
        numeric = float(((plus - minus) / (2 * epsilon) * cotangent).sum())
        relative = abs(analytic - numeric) / max(abs(analytic), 1e-6)
        assert relative < 5e-2, f"directional derivative mismatch {relative}"
        assert all(not p.requires_grad for p in adapter.noise_pred_net.parameters())
        return dict(analytic=analytic, numeric=numeric, relative_gap=relative)
    record("teacher_suffix_directional_derivative", suffix_vjp)

    def batch_independence():
        return dict(max_gap=check_batch_independence(adapter.noise_pred_net, device))
    record("teacher_batch_independence", batch_independence)

    def exact_metric():
        noise = torch.randn(1, 16, 2, device=device,
                            generator=torch.Generator(device=device).manual_seed(4))
        target = teacher_half_from(noise, adapter.noise_pred_net, state["condition"],
                                   config["teacher_steps"]).detach()
        physical_dim = 4 * config["execute_steps"]
        prefix_dim = config["execute_steps"] * 2
        jacobian = torch.randn(1, physical_dim, prefix_dim, device=device,
                               generator=torch.Generator(device=device).manual_seed(5)) * 0.1

        def suffix(variable):
            return teacher_half_from(variable, adapter.noise_pred_net,
                                     state["condition"], config["teacher_steps"])
        metric = pullback_metric_exact(suffix, target, jacobian,
                                       config["execute_steps"])
        expected = (1, physical_dim, 16 * 2)
        assert tuple(metric.shape) == expected, (metric.shape, expected)
        assert torch.isfinite(metric).all(), "nonfinite exact metric"
        error = torch.randn_like(target) * 0.01
        value = float(pullback_quadratic(metric, error))
        manual = float((torch.bmm(metric, error.flatten(1).unsqueeze(-1)) ** 2).sum())
        assert abs(value - manual) < 1e-4 * max(abs(manual), 1.0)
        probes = pullback_probes(suffix, target, jacobian, config["execute_steps"],
                                 num_probes=config["num_probes"])
        assert probes.shape[1] == config["num_probes"]
        return dict(shape=list(metric.shape), expected_shape=list(expected),
                    quadratic=value, probes=list(probes.shape))
    record("exact_pullback_metric", exact_metric)

    def replay_and_transitions():
        results = []
        for scene in (config["development_scene_start"],
                      config["development_scene_start"] + 1):
            env = adapter.new_env(image=True)
            obs, _ = adapter.reset(env, scene)
            history = []
            before_pose = np.asarray(adapter.signature(env))
            for step in range(24):
                action = np.array([256.0 + 8.0 * np.sin(step / 3.0),
                                   300.0 - 6.0 * step])
                obs, reward, terminated, truncated, done = adapter.raw_step(env, action)
                history.append(action.tolist())
                if done:
                    break
            after_pose = np.asarray(adapter.signature(env))
            live = adapter.signature(env)
            moved = float(np.abs(after_pose[[2, 3, 4, 5, 6, 7]] -
                                 before_pose[[2, 3, 4, 5, 6, 7]]).max())
            env.close()
            gap = adapter.check_replay(scene, history, live)
            equiv = adapter.check_transition_equivalence(scene, history[:-1],
                                                        np.asarray([history[-1]]))
            results.append(dict(scene=scene, steps=len(history), replay_gap=gap,
                                pose_change=moved, image_state_gap=equiv,
                                block_moved=bool(moved > 1e-6)))
        assert all(r["block_moved"] for r in results), \
            "block pose did not change in either smoke history"
        return results
    record("replay_and_image_state_transitions", replay_and_transitions)

    def tiny_train_step():
        student = TwoStepStudent(condition_dim=514, horizon=16, action_dim=2,
                                 width=config["width"]).to(device)
        batch = dict(condition=torch.randn(4, 514, device=device),
                     noise=torch.randn(4, 16, 2, device=device),
                     midpoint=torch.randn(4, 16, 2, device=device),
                     target_mid=torch.randn(4, 16, 2, device=device),
                     target_end=torch.randn(4, 16, 2, device=device),
                     teacher_end=torch.randn(4, 16, 2, device=device))
        if any(m in modes for m in ("endpoint", "pullback", "identity", "scalar")):
            physical_dim = 4 * config["execute_steps"]
            batch["jacobian_start"] = torch.randn(4, physical_dim, 16, device=device) * 0.05
            batch["jacobian_end"] = torch.randn(4, physical_dim, 16, device=device) * 0.05
            batch["metric_exact"] = torch.randn(4, physical_dim, 32, device=device) * 0.05
            batch["probes"] = torch.randn(4, config["num_probes"], 32, device=device) * 0.05
        scales = {"endpoint": 1.0, "identity": 1.0, "pullback": 1.0}
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        out = {}
        for mode in modes:
            needs_metric = mode in ("endpoint", "pullback", "identity", "scalar")
            loss, parts = objective(student, batch, mode, config["execute_steps"],
                                    scales if needs_metric else None, metric_count=2)
            assert torch.isfinite(loss), f"{mode}: nonfinite loss"
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0,
                                                       error_if_nonfinite=True)
            optimizer.step()
            assert np.isfinite(float(grad_norm)), f"{mode}: nonfinite grad norm"
            out[mode] = dict(loss=float(loss.item()),
                             penalty=float(parts["penalty"].item()),
                             grad_norm=float(grad_norm))
        assert all(not p.requires_grad for p in adapter.noise_pred_net.parameters())
        return out
    record("tiny_train_step_all_modes", tiny_train_step)

    def label_only_path():
        """Item 2: the warm-start objective must run with no metric tensors."""
        student = TwoStepStudent(condition_dim=514, horizon=16, action_dim=2,
                                 width=config["width"]).to(device)
        batch = dict(condition=torch.randn(4, 514, device=device),
                     noise=torch.randn(4, 16, 2, device=device),
                     midpoint=torch.randn(4, 16, 2, device=device),
                     target_mid=torch.randn(4, 16, 2, device=device),
                     target_end=torch.randn(4, 16, 2, device=device),
                     teacher_end=torch.randn(4, 16, 2, device=device))
        loss, parts = objective(student, batch, "uniform", config["execute_steps"],
                                None, metric_count=2)
        loss.backward()
        assert torch.isfinite(loss)
        return dict(loss=float(loss.item()),
                    has_metric_keys=sorted(k for k in batch if "jacobian" in k
                                           or "metric" in k or "probes" in k))
    record("uniform_train_without_metrics", label_only_path)

    def source_conventions():
        rng = np.random.default_rng(0)
        gauss = sample_source(rng, (20000,), "gaussian")
        uniform = sample_source(rng, (20000,), "uniform")
        symmetric = sample_source(rng, (20000,), "uniform_symmetric")
        assert abs(float(gauss.mean())) < 0.05 and abs(float(gauss.std()) - 1.0) < 0.05
        assert float(uniform.min()) >= 0.0 and float(uniform.max()) < 1.0
        assert float(symmetric.min()) >= -1.0 and float(symmetric.max()) < 1.0
        try:
            sample_source(rng, (4,), "not_a_source")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown source accepted")
        return dict(gaussian_mean=float(gauss.mean()), gaussian_std=float(gauss.std()),
                    uniform_range=[float(uniform.min()), float(uniform.max())],
                    symmetric_range=[float(symmetric.min()), float(symmetric.max())])
    record("source_conventions", source_conventions)

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
                                               "scipy", "gym", "pymunk", "pygame",
                                               "imageio_ffmpeg"]),
                  schema_version=C.SCHEMA_VERSION,
                  timestamp=time.time())
    C.write_meta(output, report)
    C.complete_output(output, dict(passed=report["passed"]))
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
    C.reserve_outputs([args.output])
    print("[smoke] sources: " + json.dumps(C.check_manifest()))
    print("[smoke] device: " + str(device))
    return run(config, device, args.output, args.modes)


if __name__ == "__main__":
    raise SystemExit(main())
