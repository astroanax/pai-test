import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "experiment" if __import__("os").path.isdir("experiment") else ".")
import contract as C


def check_prefix_selection():
    """Regression: the prefix extra loss must use the marked half, exactly like
    the physical penalties. A batch whose unmarked half carries huge errors must
    not change the prefix penalty."""
    import torch
    from core import objective
    torch.manual_seed(0)
    from core import TwoStepStudent
    student = TwoStepStudent(condition_dim=8, horizon=16, action_dim=2, width=32)
    batch = {k: torch.zeros(4, 16, 2) for k in
             ("noise", "midpoint", "target_mid", "target_end", "teacher_end")}
    batch["condition"] = torch.zeros(4, 8)
    batch["jacobian_start"] = torch.zeros(4, 32, 16)
    batch["jacobian_end"] = torch.zeros(4, 32, 16)
    batch["metric_exact"] = torch.zeros(4, 32, 32)
    scales = {"endpoint": 1.0, "identity": 1.0, "pullback": 1.0}
    clean = objective(student, batch, "prefix", 8, scales, metric_count=2)[1]["penalty"]
    poisoned = dict(batch)
    poisoned["target_mid"] = batch["target_mid"].clone()
    poisoned["target_mid"][2:] = 100.0
    poisoned["target_end"] = batch["target_end"].clone()
    poisoned["target_end"][2:] = 100.0
    dirty = objective(student, poisoned, "prefix", 8, scales, metric_count=2)[1]["penalty"]
    assert torch.allclose(clean, dirty, atol=1e-6), \
        f"prefix penalty leaks the unmarked half: {float(clean)} vs {float(dirty)}"
    whole = objective(student, poisoned, "prefix", 8, scales, metric_count=4)[1]["penalty"]
    assert float(whole) > float(clean), "selection is not restricting the mean"
    return dict(marked_only=float(clean), all_examples=float(whole))


def check_exact_metric_and_sketch():
    """Regression: the exact metric must equal ||C e||^2 and the four-probe
    sketch must leave a direction with large exact consequence unpenalized."""
    import torch
    from core import (pullback_metric_exact, pullback_probes, pullback_quadratic,
                      pullback_trace)
    torch.manual_seed(0)
    matrix = torch.randn(32, 32) * 0.5
    target = torch.randn(1, 16, 2)

    def suffix(variable):
        return variable.reshape(1, -1) @ matrix.T + 0.0 * variable.reshape(1, -1)[:, :0].sum()
    jacobian = torch.randn(1, 32, 16) * 0.1
    exact = pullback_metric_exact(suffix, target, jacobian, 8)
    error = torch.randn_like(target) * 0.05
    value = float(pullback_quadratic(exact, error))
    manual = float((exact @ error.flatten(1).unsqueeze(-1)).square().sum())
    assert abs(value - manual) < 1e-4 * max(abs(manual), 1.0), "exact quadratic mismatch"
    trace = float(pullback_trace(exact))
    assert abs(trace - float(exact.square().sum())) < 1e-4, "exact trace mismatch"
    probes = pullback_probes(suffix, target, jacobian, 8, num_probes=4)
    assert probes.shape[1] == 4, probes.shape
    sketch_trace = float(pullback_trace(probes, sketch=True))
    assert sketch_trace > 0, "sketch trace is zero"
    # unpenalized consequential direction: the null space of the rank-4 sketch
    flat = exact[0]
    projection = probes[0] @ probes[0].transpose(0, 1)
    null_basis = _null_space(probes[0].numpy())
    if null_basis.size:
        worst = None
        for column in null_basis.T:
            direction = torch.from_numpy(column.reshape(1, 16, 2)).float()
            direction = direction / direction.norm()
            exact_cost = float(pullback_quadratic(exact, direction))
            sketch_cost = float(pullback_quadratic(probes, direction, sketch=True))
            ratio = exact_cost / max(sketch_cost, 1e-12)
            if worst is None or ratio > worst["ratio"]:
                worst = dict(ratio=ratio, exact=exact_cost, sketch=sketch_cost)
        assert worst and worst["ratio"] > 1.0, "no consequential direction found"
    else:
        worst = dict(ratio=None, note="no null space at this rank")
    assert float(projection.norm()) > 0
    return dict(exact_quadratic=value, exact_trace=trace,
                sketch_trace=sketch_trace, worst_null_direction=worst)


def _null_space(matrix):
    u, s, vt = np.linalg.svd(matrix.numpy() if hasattr(matrix, "numpy") else matrix,
                             full_matrices=True)
    tol = max(matrix.shape) * (s[0] if s.size else 0) * 1e-12
    rank = int((s > tol).sum())
    return vt[rank:].T


def check_nonfinite_gradient_guard():
    """Regression: clip_grad_norm_ with error_if_nonfinite must raise."""
    import torch
    student = torch.nn.Linear(4, 4)
    student.weight.grad = torch.full_like(student.weight, float("inf"))
    raised = False
    try:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0,
                                       error_if_nonfinite=True)
    except Exception:
        raised = True
    assert raised, "nonfinite gradient was not rejected"
    return dict(raised=raised)


def check_config_validation():
    base = json.load(open("experiment/config.json" if
                          __import__("os").path.isdir("experiment") else "config.json"))
    bad_cases = []
    for key, value in [("teacher_steps", 0), ("teacher_steps", 15),
                       ("execute_steps", 0), ("execute_steps", 17),
                       ("episode_steps", -1), ("num_probes", 0),
                       ("batch_size", 127), ("finite_difference_epsilon", 0.0),
                       ("learning_rate", 0.0), ("metric_weight", -1)]:
        broken = dict(base)
        broken[key] = value
        try:
            C.validate_config(broken, "test")
        except ValueError as error:
            bad_cases.append(f"{key}={value}: {str(error)[:60]}")
        else:
            raise AssertionError(f"validate_config accepted {key}={value}")
    overlap = dict(base)
    overlap["validation_scene_start"] = base["train_scene_start"]
    try:
        C.validate_config(overlap, "test")
    except ValueError:
        bad_cases.append("overlapping scene ranges rejected")
    else:
        raise AssertionError("overlapping scene ranges accepted")
    return bad_cases


def check_cache_and_completeness():
    config = json.load(open("experiment/config.json" if
                            __import__("os").path.isdir("experiment") else "config.json"))
    repository = config["repository"]
    fingerprint = C.cache_fingerprint(config, repository)
    contexts = np.array([{"scene": config["train_scene_start"],
                          "condition": np.zeros(514, np.float32),
                          "noise": np.zeros((16, 2), np.float32),
                          "student_mid": np.zeros((16, 2), np.float32),
                          "history": []}], dtype=object)
    good = dict(contexts=contexts, meta=json.dumps({"mode": "shared"}),
                fingerprint=json.dumps(fingerprint))
    C.verify_cache(good, config, repository, expect_mode="shared")
    drifted = dict(good)
    changed = dict(config)
    changed["execute_steps"] = 4
    try:
        C.verify_cache(drifted, changed, repository, expect_mode="shared")
    except ValueError:
        pass
    else:
        raise AssertionError("cache accepted a changed execution contract")
    frame_module = None
    protocol = dict(modes=["pullback", "endpoint"], seeds=[0, 1],
                    final_scenes=[1, 2], protocol_id="x")
    complete_records = [dict(method=m, training_seed=s, scene=c, success=1)
                        for m in ("pullback", "endpoint")
                        for s in (0, 1) for c in (1, 2)]
    C.require_complete(complete_records, protocol, "pullback", "endpoint")
    completeness = ["complete_ok"]
    dropped_pair = [r for r in complete_records
                    if not (r["training_seed"] == 1 and r["scene"] == 1)]
    for broken, label in [(dropped_pair, "both methods missing seed1/scene1")]:
        try:
            C.require_complete(broken, protocol, "pullback", "endpoint")
        except ValueError:
            completeness.append("dropped_pair_rejected")
        else:
            raise AssertionError("incomplete experiment accepted: " + label)
    only_pullback = [r for r in complete_records if r["method"] == "pullback"]
    try:
        C.require_complete(only_pullback, protocol, "pullback", "endpoint")
    except ValueError:
        completeness.append("missing_baseline_rejected")
    else:
        raise AssertionError("missing baseline accepted")
    try:
        C.require_protocol("runs/definitely_missing_protocol.json")
    except ValueError:
        pass
    else:
        raise AssertionError("missing protocol accepted")
    return dict(checked=["cache_contract", "completeness", "missing_protocol"],
                completeness=completeness)


def check_reserve_and_manifest():
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "artifact.npz")
        C.reserve_outputs([path])
        open(path, "wb").close()
        try:
            C.reserve_outputs([path])
        except ValueError:
            pass
        else:
            raise AssertionError("existing output accepted")
    manifest = C.check_manifest()
    return dict(manifest_incomplete=manifest["untracked"],
                source_files=len(C.SOURCE_FILES))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    results = {"config_validation": check_config_validation(),
               "cache_and_completeness": check_cache_and_completeness(),
               "reserve_and_manifest": check_reserve_and_manifest()}
    try:
        import torch
        results["prefix_selection"] = check_prefix_selection()
        results["exact_metric_and_sketch"] = check_exact_metric_and_sketch()
        results["nonfinite_gradient_guard"] = check_nonfinite_gradient_guard()
    except ImportError:
        if args.require_torch:
            raise
        results["torch_checks"] = "skipped"
    print("sanity " + json.dumps(results, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
