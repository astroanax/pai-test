import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C
from core import (physical_error, pullback_metric_exact, pullback_probes,
                  pullback_quadratic, pullback_trace)
from hri_adapter import resolve_device, teacher_half_from


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def validation_context_rows(contexts, config, per_scene=2, max_contexts=16):
    """Sample validation contexts spread across scenes, never the first rows."""
    want = C.scene_range(config, "validation")
    by_scene = {}
    for index, context in enumerate(contexts):
        scene = int(context["scene"])
        if scene in want:
            by_scene.setdefault(scene, []).append(index)
    if not by_scene:
        raise ValueError("metric cache contains no validation scenes; build the "
                         "validation subset before running the diagnostic")
    rows = []
    for scene in sorted(by_scene)[:max_contexts]:
        rows.extend(by_scene[scene][:per_scene])
    return rows[:max_contexts]


def simulate_outcome(adapter, context, prefix, perturbation=None, scene_arg=None):
    """Realized physical feature discrepancy for a (possibly perturbed) prefix."""
    history = [np.asarray(row, dtype=np.float64) for row in context["history"]]
    scene = int(context["scene"]) if scene_arg is None else int(scene_arg)
    applied = np.asarray(prefix, dtype=np.float64)
    if perturbation is not None:
        applied = applied + np.asarray(perturbation, dtype=np.float64)[
            :applied.shape[0], :applied.shape[1]]
    return adapter.execute_from_history(scene, history, applied)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--student", default=None, help="warm or trained student checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--probe-counts", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--no-simulate", action="store_true",
                        help="skip simulator outcomes (keeps only predictive geometry)")
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config["device"])
    data = np.load(args.cache, allow_pickle=True)
    info = C.verify_metric_cache(data, config, config["repository"])
    cache_hash = C.sha256_file(args.cache)
    protocol = None
    if args.protocol and os.path.exists(args.protocol):
        protocol = C.require_protocol(args.protocol)
    contexts = list(data["contexts"])
    rows = validation_context_rows(contexts, config, max_contexts=16)
    metric_mode = info["metric_mode"]
    radii = [0.005, 0.01, 0.02]
    rng = np.random.default_rng(11)

    student = None
    if args.student:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pilot import load_student
        student, payload = load_student(args.student, config, device)
        student.eval()

    adapter = None
    if not args.no_simulate:
        from pilot import build_adapter
        adapter = build_adapter(config, device)

    report = {"cache": os.path.basename(args.cache), "cache_sha256": cache_hash,
              "metric_mode": metric_mode, "n_contexts": len(contexts),
              "sampled_rows": rows,
              "protocol_id": None if protocol is None else protocol.get("protocol_id"),
              "source_hashes": C.source_hashes(),
              "lineage": json.loads(str(data["lineage"])) if "lineage" in data else None,
              "student": os.path.basename(args.student) if args.student else None,
              "simulated": adapter is not None}
    predictions = {}

    def metric_of(context):
        if metric_mode == "exact":
            return torch.from_numpy(np.asarray(context["metric_exact"], dtype=np.float32)).unsqueeze(0).to(device)
        return torch.from_numpy(np.asarray(context["probes"], dtype=np.float32)).unsqueeze(0).to(device)

    def exact_metric(context, jac_start):
        cond = torch.from_numpy(np.asarray(context["condition"], dtype=np.float32)).unsqueeze(0).to(device)
        target = torch.from_numpy(np.asarray(context["target_mid"], dtype=np.float32)).unsqueeze(0).to(device)
        steps = config["teacher_steps"]

        def suffix(variable):
            return teacher_half_from(variable, adapter.noise_pred_net, cond, steps)

        return pullback_metric_exact(suffix, target, jac_start, config["execute_steps"])

    actual_errors = {}
    if student is not None:
        for index in rows:
            context = contexts[index]
            cond = torch.from_numpy(np.asarray(context["condition"], dtype=np.float32)).unsqueeze(0).to(device)
            noise = torch.from_numpy(np.asarray(context["noise"], dtype=np.float32)).unsqueeze(0).to(device)
            midpoint = torch.from_numpy(np.asarray(context["midpoint"], dtype=np.float32)).unsqueeze(0).to(device)
            target_mid = torch.from_numpy(np.asarray(context["target_mid"], dtype=np.float32)).unsqueeze(0).to(device)
            target_end = torch.from_numpy(np.asarray(context["target_end"], dtype=np.float32)).unsqueeze(0).to(device)
            teacher_end = torch.from_numpy(np.asarray(context["teacher_end"], dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_mid = student(noise, cond, 0.0)
                predicted_end = student(midpoint, cond, 0.5)
                full_end = student(predicted_mid, cond, 0.5)
                e0 = (predicted_mid - target_mid).cpu().numpy()[0]
                e1 = (predicted_end - target_end).cpu().numpy()[0]
                eA = (full_end - teacher_end).cpu().numpy()[0]
            actual_errors[index] = dict(mid=e0.astype(np.float32),
                                        end=e1.astype(np.float32),
                                        endpoint=eA.astype(np.float32))
        stack = {k: np.stack([actual_errors[i][k] for i in rows]).reshape(len(rows), -1)
                 for k in ("mid", "end", "endpoint")}
        report["actual_error_distribution"] = {
            k: dict(rms=float(np.sqrt((v ** 2).mean())),
                    median_norm=float(np.median(np.linalg.norm(v, axis=1))),
                    p95_norm=float(np.quantile(np.linalg.norm(v, axis=1), 0.95)))
            for k, v in stack.items()}

    rank = {}
    for radius in radii:
        rank[str(radius)] = {}
    direction_sets = {}
    for index in rows:
        context = contexts[index]
        flat_dim = config["execute_steps"] * 2
        sets = {"random": []}
        for _ in range(8):
            direction = rng.standard_normal(flat_dim)
            sets["random"].append(direction / np.linalg.norm(direction) * radii[1])
        sets["random_small"] = [d / np.linalg.norm(d) * radii[0]
                                for d in np.stack(sets["random"])]
        sets["random_large"] = [d / np.linalg.norm(d) * radii[2]
                                for d in np.stack(sets["random"])]
        if index in actual_errors:
            sets["actual_mid_error"] = [actual_errors[index]["mid"].reshape(-1)]
        direction_sets[index] = sets

    per_context_records = []
    for index in rows:
        context = contexts[index]
        jac_start = torch.from_numpy(np.asarray(context["jacobian_start"], dtype=np.float32)).unsqueeze(0).to(device)
        jac_end = torch.from_numpy(np.asarray(context["jacobian_end"], dtype=np.float32)).unsqueeze(0).to(device)
        metric = metric_of(context)
        sketch = metric_mode != "exact"
        exact = exact_metric(context, jac_start) if adapter is not None else None
        prefix_a = np.asarray(context["teacher_end"])[:config["execute_steps"]]
        prefix_b = np.asarray(context["target_end"])[:config["execute_steps"]]
        if exact is not None:
            base_outcome = simulate_outcome(adapter, context, prefix_a)
        else:
            base_outcome = None
        # most and least consequential directions under the exact metric
        signif = {}
        if exact is not None:
            gram = exact[0].cpu().numpy()
            _, _, vt = np.linalg.svd(gram, full_matrices=False)
            signif["top_direction"] = (vt[0] * radii[1]).astype(np.float32)
            signif["low_direction"] = (vt[-1] * radii[1]).astype(np.float32)
        for name, directions in list(direction_sets[index].items()) + list(signif.items()):
            for direction in directions:
                error = torch.from_numpy(np.asarray(direction, dtype=np.float32)).reshape(
                    1, config["execute_steps"], 2).to(device)
                row = dict(context=index, scene=int(context["scene"]), set=name,
                           radius=float(np.linalg.norm(direction)))
                if exact is not None:
                    row["exact_pullback"] = float(pullback_quadratic(exact, error).item())
                    row["identity_prefix"] = float(physical_error(
                        jac_start, error, config["execute_steps"]).item())
                if sketch:
                    row["sketch_pullback"] = float(pullback_quadratic(metric, error,
                                                                      sketch=True).item())
                else:
                    row["sketch_pullback"] = None
                row["late_physical"] = float(physical_error(
                    jac_end, error, config["execute_steps"]).item())
                if base_outcome is not None:
                    perturbed = simulate_outcome(adapter, context, prefix_a, direction)
                    row["observed_feature_error"] = float(
                        np.linalg.norm(perturbed - base_outcome))
                per_context_records.append(row)

    # sketch quality: relative error against the exact metric on actual errors
    if metric_mode == "exact" and adapter is not None:
        report["sketch_audit"] = {}
        for index in rows[:8]:
            context = contexts[index]
            cond = torch.from_numpy(np.asarray(context["condition"], dtype=np.float32)).unsqueeze(0).to(device)
            target = torch.from_numpy(np.asarray(context["target_mid"], dtype=np.float32)).unsqueeze(0).to(device)
            jac_start = torch.from_numpy(np.asarray(context["jacobian_start"], dtype=np.float32)).unsqueeze(0).to(device)
            steps = config["teacher_steps"]

            def suffix(variable):
                return teacher_half_from(variable, adapter.noise_pred_net, cond, steps)

            exact = exact_metric(context, jac_start)
            error = torch.from_numpy(np.asarray(
                actual_errors.get(index, {}).get("mid", context["midpoint"]),
                dtype=np.float32)).unsqueeze(0).to(device)
            exact_value = float(pullback_quadratic(exact, error).item())
            exact_trace = float(pullback_trace(exact).item())
            entry = {}
            for count in args.probe_counts:
                probes = pullback_probes(suffix, target, jac_start,
                                         config["execute_steps"], num_probes=count)
                sketch_value = float(pullback_quadratic(probes, error, sketch=True).item())
                sketch_trace = float(pullback_trace(probes, sketch=True).item())
                entry[str(count)] = dict(
                    quadratic_ratio=(sketch_value / exact_value) if exact_value else None,
                    trace_ratio=(sketch_trace / exact_trace) if exact_trace else None)
            report["sketch_audit"][str(index)] = entry

    # rank association between each predictor and the observed outcome
    if per_context_records:
        predictors = [k for k in ("exact_pullback", "identity_prefix", "sketch_pullback",
                                  "late_physical") if any(r.get(k) is not None
                                                          for r in per_context_records)]
        observed_ok = any("observed_feature_error" in r for r in per_context_records)
        for name in sorted({r["set"] for r in per_context_records}):
            subset = [r for r in per_context_records if r["set"] == name]
            if len(subset) < 4:
                continue
            for predictor in predictors:
                values = [r.get(predictor) for r in subset]
                if any(v is None for v in values):
                    continue
                key = f"{name}:{predictor}"
                if observed_ok:
                    truth = [r["observed_feature_error"] for r in subset]
                    rank.setdefault("observed", {})[key] = _spearman(values, truth)
                if predictor != "exact_pullback" and any(
                        r.get("exact_pullback") is not None for r in subset):
                    rank.setdefault("vs_exact", {})[key] = _spearman(
                        values, [r.get("exact_pullback") or 0.0 for r in subset])
    report["rank_association"] = rank
    report["predictor_paths"] = None
    if args.predictions:
        C.reserve_outputs([args.predictions])
        np.savez(args.predictions,
                 records=np.array(per_context_records, dtype=object),
                 rows=np.array(rows),
                 actual_mid=np.array([actual_errors[i]["mid"] for i in rows]) if actual_errors else np.zeros((0,)),
                 actual_end=np.array([actual_errors[i]["end"] for i in rows]) if actual_errors else np.zeros((0,)),
                 actual_endpoint=np.array([actual_errors[i]["endpoint"] for i in rows]) if actual_errors else np.zeros((0,)))
        report["predictor_paths"] = args.predictions
    C.reserve_outputs([args.output])
    C.write_meta(args.output, report)
    print("wrote diagnostic " + args.output)


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
