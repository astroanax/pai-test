import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PARENT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(PARENT, "experiment")):
    sys.path.insert(0, PARENT)
import contract as C


def load_config():
    for candidate in (os.path.join(PARENT, "experiment", "config.json"),
                      os.path.join(HERE, "config.json")):
        if os.path.exists(candidate):
            with open(candidate) as handle:
                return json.load(handle), candidate
    raise FileNotFoundError("config.json not found")


def check_linear_algebra():
    """NumPy-only algebra: finite differences against a known linear map, and the
    pullback quadratic/trace identities re-derived independently."""
    matrix = np.array([[2.0, 0.0, 0.5], [0.0, 0.5, -1.0]])
    offset = np.array([1.0, -2.0])

    def execute(action):
        return matrix @ np.asarray(action) + offset

    action = np.array([0.3, -0.2, 0.1])
    epsilon = 1e-6
    columns = []
    for coordinate in range(action.size):
        step = np.zeros_like(action)
        step[coordinate] = epsilon
        columns.append((execute(action + step) - execute(action - step)) / (2 * epsilon))
    estimated = np.stack(columns, axis=-1)
    gap = float(np.abs(estimated - matrix).max())
    assert gap < 1e-6, f"finite difference gap {gap}"
    C_matrix = np.array([[1.0, 2.0], [0.0, 1.0], [-1.0, 0.5]])
    target = np.array([0.1, -0.2])
    value = float(np.sum((C_matrix @ target) ** 2))
    trace = float(np.sum(C_matrix ** 2))
    assert abs(value - float(np.dot(C_matrix @ target, C_matrix @ target))) < 1e-12
    assert abs(trace - float(np.trace(C_matrix.T @ C_matrix))) < 1e-12
    assert abs(trace - 7.25) < 1e-12, trace
    return dict(finite_difference_gap=gap, quadratic=value, trace=trace)


def check_config_validation():
    config, path = load_config()
    C.validate_config(config, path)
    rejected = []
    cases = [("teacher_steps", 0), ("teacher_steps", 15), ("execute_steps", 0),
             ("execute_steps", 17), ("episode_steps", -1), ("num_probes", 0),
             ("batch_size", 127), ("finite_difference_epsilon", 0.0),
             ("learning_rate", 0.0), ("metric_weight", -1),
             ("validation_metric_contexts", 0), ("native_fast_steps", 0),
             ("source", "gauss"), ("metric_mode", "approx")]
    for key, value in cases:
        broken = dict(config)
        broken[key] = value
        try:
            C.validate_config(broken, "test")
        except ValueError as error:
            rejected.append(f"{key}={value}: {str(error)[:48]}")
        else:
            raise AssertionError(f"validate_config accepted {key}={value}")
    for key in ("validation_scene_start", "test_scene_start"):
        overlap = dict(config)
        overlap[key] = config["train_scene_start"]
        try:
            C.validate_config(overlap, "test")
        except ValueError:
            rejected.append(f"{key} overlap rejected")
        else:
            raise AssertionError(f"{key} overlap accepted")
    return rejected


def synthetic_context(scene, split, rng, with_metric=False, metric_mode="exact"):
    context = {"scene": scene, "split": split, "decision": 0, "step": 0,
               "condition": rng.standard_normal(514).astype(np.float32),
               "noise": rng.standard_normal((16, 2)).astype(np.float32),
               "midpoint": rng.standard_normal((16, 2)).astype(np.float32),
               "target_mid": rng.standard_normal((16, 2)).astype(np.float32),
               "target_end": rng.standard_normal((16, 2)).astype(np.float32),
               "teacher_end": rng.standard_normal((16, 2)).astype(np.float32),
               "history": [], "signature_initial": [0.0],
               "signature_live": [0.0]}
    if with_metric:
        context["jacobian_start"] = rng.standard_normal((32, 16)).astype(np.float32)
        context["jacobian_end"] = rng.standard_normal((32, 16)).astype(np.float32)
        if metric_mode == "exact":
            context["metric_exact"] = rng.standard_normal((32, 32)).astype(np.float32)
        else:
            context["probes"] = rng.standard_normal((4, 32)).astype(np.float32)
    return context


def check_producer_consumer(config, tmpdir):
    """The exact schema regression the audit found: caches written by the
    producer helpers must validate in every consumer."""
    rng = np.random.default_rng(0)
    train_scenes = sorted(C.scene_range(config, "train"))[:4]
    validation_scenes = sorted(C.scene_range(config, "validation"))[:2]
    contexts = ([synthetic_context(s, "train", rng) for s in train_scenes]
                + [synthetic_context(s, "validation", rng) for s in validation_scenes])
    label_path = os.path.join(tmpdir, "labels.npz")
    meta = dict(schema_version=C.SCHEMA_VERSION, mode="shared", source=config["source"],
                seed=0, student="warm.pt", splits={"train": 4, "validation": 2},
                contexts=len(contexts))
    C.write_label_cache(label_path, contexts, config, meta,
                            dict(source_hashes={}, mode="shared"))
    verified = C.verify_label_cache(label_path, config, expect_mode="shared",
                                    allowed_scenes=C.scene_range(config, "train") |
                                    C.scene_range(config, "validation"))
    out = {"label_cache_contexts": verified["n_contexts"],
           "splits": verified["splits"]}
    try:
        C.verify_label_cache(label_path, config, require_metric=True)
    except ValueError:
        out["metric_required_rejected"] = True
    else:
        raise AssertionError("label-only cache accepted where metrics are required")
    train_rows, validation_rows = C.metric_subset(verified, config)
    out["subset"] = dict(train=len(train_rows), validation=len(validation_rows))
    assert all(row["split"] == "train" for row in train_rows)
    assert all(row["split"] == "validation" for row in validation_rows)
    seen_wrong_split = [c for c in contexts if c["split"] != C.split_of_scene(config, c["scene"])]
    assert not seen_wrong_split, "split label disagrees with the configured range"

    label_keys = sorted(set(verified["contexts"][0]) - {"scene", "split", "decision",
                                                      "step", "history",
                                                      "signature_initial",
                                                      "signature_live"})
    assert "metric_exact" not in label_keys and "jacobian_start" not in label_keys
    out["label_context_keys"] = label_keys

    metric_contexts = [synthetic_context(s, "train", rng, with_metric=True)
                       for s in train_scenes]
    metric_path = os.path.join(tmpdir, "metrics.npz")
    metric_meta = dict(schema_version=C.SCHEMA_VERSION,
                       metric_mode=config["metric_mode"], num_probes=config["num_probes"],
                       epsilon=config["finite_difference_epsilon"],
                       counts={"train": len(metric_contexts), "validation": 0,
                               "total": len(metric_contexts)},
                       source=config["source"], splits={"train": len(metric_contexts)})
    C.write_metric_cache(metric_path, metric_contexts, config, metric_meta,
                             dict(cache=label_path))
    metric_verified = C.verify_label_cache(metric_path, config, require_metric=True)
    out["metric_cache_contexts"] = metric_verified["n_contexts"]
    drifted = dict(config)
    drifted["execute_steps"] = 4
    try:
        C.verify_label_cache(metric_path, drifted, require_metric=True)
    except ValueError:
        out["contract_drift_rejected"] = True
    else:
        raise AssertionError("cache accepted a changed execution contract")
    changed_metric = dict(config)
    changed_metric["metric_mode"] = "sketch" if config["metric_mode"] == "exact" else "exact"
    try:
        C.verify_label_cache(metric_path, changed_metric, require_metric=True)
    except ValueError:
        out["metric_mode_change_rejected"] = True
    else:
        raise AssertionError("cache accepted a changed metric mode")
    return out


def check_spearman():
    from scipy.stats import spearmanr
    a = np.array([1.0, 2.0, 2.0, 3.0, 3.0])
    b = np.array([5.0, 4.0, 4.0, 6.0, 6.0])
    double_argsort = np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
    tie_aware = float(spearmanr(a, b).statistic)
    assert abs(tie_aware - double_argsort) > 1e-6, \
        "this fixture must expose the tie handling difference"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        constant = float(spearmanr(np.ones(4), np.arange(4)).statistic)
    assert np.isnan(constant), "constant input must report an undefined rank"
    return dict(tie_aware=tie_aware, double_argsort=float(double_argsort),
                constant_reported_as_nan=bool(np.isnan(constant)))


def check_reserve_and_completion(tmpdir):
    path = os.path.join(tmpdir, "artifact.npz")
    C.reserve_outputs([path])
    assert os.path.exists(path + ".reserved")
    try:
        C.reserve_outputs([path])
    except ValueError:
        pass
    else:
        raise AssertionError("reservation did not block a second claim")
    C.atomic_savez(path, a=np.zeros(3))
    C.complete_output(path, dict(note="synthetic"))
    assert C.is_complete(path), "completion record missing"
    assert not os.path.exists(path + ".reserved")
    C.mark_incomplete(path, "synthetic failure")
    assert not C.is_complete(path), "incomplete artifact still reports complete"
    return dict(reserved=True, completed=True, incomplete_after_failure=True)


def check_protocol_and_analysis(config, config_path, tmpdir):
    protocols = {}
    upstream = C.upstream_hashes(config["repository"])
    protocol = dict(modes=["pullback", "endpoint"], seeds=[0, 1],
                    final_scenes=[10, 11], protocol_id="pid",
                    source_hashes=C.source_hashes(), upstream=upstream,
                    upstream_commit=C.upstream_commit(config["repository"]),
                    packages=C.package_versions(["numpy"]),
                    resolved_config=config, assets=C.asset_hashes(config),
                    checkpoint_sha256=None, normalizer_sha256=None)
    C.verify_current_protocol(protocol, config, config_path)
    protocols["fresh_lock_verified"] = True
    stale = dict(protocol)
    stale["source_hashes"] = {"experiment/core.py": "deadbeef"}
    try:
        C.verify_current_protocol(stale, config, config_path)
    except C.ProtocolError:
        protocols["stale_source_rejected"] = True
    else:
        raise AssertionError("stale source hashes accepted")
    stale = dict(protocol)
    stale["resolved_config"] = dict(config, execute_steps=4)
    try:
        C.verify_current_protocol(stale, config, config_path)
    except C.ProtocolError:
        protocols["config_drift_rejected"] = True
    else:
        raise AssertionError("configuration drift accepted")

    good = [dict(method=m, training_seed=s, scene=c, success=1,
                 protocol_id="pid", split="test", decision_latency_median_ms=1.0,
                 score=0.9, raw_violation_fraction=0.0)
            for m in ("pullback", "endpoint") for s in (0, 1) for c in (10, 11)]
    C.require_complete(good, protocol, "pullback", "endpoint")
    protocols["complete_ok"] = True
    partial = [r for r in good if not (r["training_seed"] == 1 and r["scene"] == 10)]
    try:
        C.require_complete(partial, protocol, "pullback", "endpoint")
    except ValueError:
        protocols["joint_missing_pair_rejected"] = True
    else:
        raise AssertionError("joint missing pair accepted")
    non_binary = [dict(r, success=2) for r in good]
    try:
        C.require_complete(non_binary, protocol, "pullback", "endpoint")
    except ValueError:
        protocols["non_binary_success_rejected"] = True
    else:
        raise AssertionError("non-binary success accepted")

    for mode in ("pullback", "endpoint"):
        for seed in (0, 1):
            jsonl = os.path.join(tmpdir, f"final_{mode}_seed{seed}.jsonl")
            with open(jsonl, "w") as handle:
                for row in [r for r in good if r["method"] == mode
                            and r["training_seed"] == seed]:
                    handle.write(json.dumps(row) + "\n")
            C.write_meta(jsonl.replace(".jsonl", ".meta.json"),
                         dict(kind="evaluation", intrinsic_mode=mode,
                              intrinsic_seed=seed, checkpoint_sha256="sha",
                              protocol_id="pid", split="test", expected_split="test",
                              scenes=[10, 11]))
    inputs = [os.path.join(tmpdir, f"final_{m}_seed{s}.jsonl")
              for m in ("pullback", "endpoint") for s in (0, 1)]
    meta = C.require_eval_metadata(good, protocol, inputs, method="pullback")
    protocols["two_seeds_accepted"] = sorted(meta["by_run"])
    try:
        C.require_eval_metadata(good, protocol, inputs + [inputs[0]],
                                method="pullback")
    except ValueError:
        protocols["duplicate_run_rejected"] = True
    return protocols


def check_manifest():
    manifest_path = os.path.join(HERE, "MANIFEST.json")
    payload = C.write_manifest(manifest_path)
    verified = C.verify_manifest(manifest_path)
    manifest = C.check_manifest()
    assert not manifest["missing"], f"missing sources: {manifest['missing']}"
    return dict(files=payload["count"], verified=verified["verified"],
                delivery_risk=manifest["delivery_risk"][:4])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    config, config_path = load_config()
    results = {"linear_algebra": check_linear_algebra(),
               "config_validation": check_config_validation(),
               "spearman_ties": check_spearman(),
               "manifest": check_manifest()}
    with tempfile.TemporaryDirectory() as tmpdir:
        results["producer_consumer"] = check_producer_consumer(config, tmpdir)
        results["reserve_and_completion"] = check_reserve_and_completion(tmpdir)
        results["protocol_and_analysis"] = check_protocol_and_analysis(
            config, config_path, tmpdir)
    try:
        import torch
        from core import (TwoStepStudent, objective, pullback_metric_exact,
                          pullback_probes, pullback_quadratic, pullback_trace)
        torch.manual_seed(0)
        results["torch_checks"] = check_torch(torch, TwoStepStudent, objective,
                                              pullback_metric_exact, pullback_probes,
                                              pullback_quadratic, pullback_trace)
    except ImportError:
        if args.require_torch:
            raise
        results["torch_checks"] = "skipped (torch unavailable)"
    print("sanity " + json.dumps(results, default=str))
    return 0


def check_torch(torch, TwoStepStudent, objective, pullback_metric_exact,
                pullback_probes, pullback_quadratic, pullback_trace):
    out = {}
    # A KNOWN linear suffix: suffix(y) = M y with M a fixed 32x32 matrix, so
    # J_prefix must equal the first 16 rows of M and C = B0 @ M[:16].
    generator = torch.Generator().manual_seed(0)
    matrix = torch.randn(32, 32, generator=generator)

    def suffix(variable):
        flat = variable.reshape(variable.shape[0], -1)
        return (flat @ matrix.transpose(0, 1)).reshape(variable.shape)

    target = torch.randn(1, 16, 2, generator=torch.Generator().manual_seed(1))
    jacobian = torch.randn(1, 32, 16, generator=torch.Generator().manual_seed(2)) * 0.1
    metric = pullback_metric_exact(suffix, target, jacobian, 8)
    assert tuple(metric.shape) == (1, 32, 32), metric.shape
    expected = jacobian[0] @ matrix[:16, :]
    assert torch.allclose(metric[0], expected, atol=1e-4), \
        "transported metric disagrees with the known linear composition"
    out["linear_composition_max_gap"] = float((metric[0] - expected).abs().max())
    error = torch.randn(1, 16, 2, generator=torch.Generator().manual_seed(3)) * 0.05
    value = float(pullback_quadratic(metric, error))
    manual = float((torch.bmm(metric, error.flatten(1).unsqueeze(-1)) ** 2).sum())
    assert abs(value - manual) < 1e-4 * max(manual, 1.0)
    out["quadratic_matches_manual"] = value
    out["exact_trace_matches"] = abs(
        float(pullback_trace(metric)) - float(expected.square().sum())) < 1e-3
    probes = pullback_probes(suffix, target, jacobian, 8, num_probes=4)
    assert probes.shape == (1, 4, 32), probes.shape
    out["sketch_shape"] = list(probes.shape)
    out["sketch_trace_ratio"] = (float(pullback_trace(probes, sketch=True)) /
                                 max(float(pullback_trace(metric)), 1e-12))
    try:
        pullback_quadratic(metric, error[:, :8, :])
    except ValueError:
        out["prefix_error_rejected"] = True
    else:
        raise AssertionError("a truncated prefix error was accepted as the metric input")

    student = TwoStepStudent(condition_dim=8, horizon=16, action_dim=2, width=32)
    batch = {k: torch.zeros(4, 16, 2) for k in
             ("noise", "midpoint", "target_mid", "target_end", "teacher_end")}
    batch["condition"] = torch.zeros(4, 8)
    clean = objective(student, batch, "prefix", 8, None, metric_count=2)[1]["penalty"]
    poisoned = dict(batch)
    poisoned["target_mid"] = batch["target_mid"].clone()
    poisoned["target_mid"][2:] = 100.0
    poisoned["target_end"] = batch["target_end"].clone()
    poisoned["target_end"][2:] = 100.0
    dirty = objective(student, poisoned, "prefix", 8, None, metric_count=2)[1]["penalty"]
    assert torch.allclose(clean, dirty, atol=1e-6), "prefix penalty leaks the unmarked half"
    whole = objective(student, poisoned, "prefix", 8, None, metric_count=4)[1]["penalty"]
    assert float(whole) > float(clean)
    out["prefix_selection"] = dict(marked_only=float(clean), all_examples=float(whole))

    loss, _ = objective(student, batch, "uniform", 8, None, metric_count=2)
    loss.backward()
    out["uniform_without_metrics_ok"] = float(loss)
    try:
        objective(student, batch, "pullback", 8, None, metric_count=2)
    except (KeyError, ValueError):
        out["pullback_without_metric_rejected"] = True
    else:
        raise AssertionError("metric mode ran without metric tensors")

    guard = torch.nn.Linear(2, 2)
    guard.weight.grad = torch.full_like(guard.weight, float("inf"))
    try:
        torch.nn.utils.clip_grad_norm_(guard.parameters(), 1.0, error_if_nonfinite=True)
    except Exception:
        out["nonfinite_gradient_rejected"] = True
    else:
        raise AssertionError("nonfinite gradient accepted")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
