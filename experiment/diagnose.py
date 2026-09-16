import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

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


DIAGNOSTIC_REQUIRED_KEYS = ("condition", "target_mid", "jacobian_start",
                            "jacobian_end")


def _diagnostic_metric_key(config):
    return "metric_exact" if config.get("metric_mode", "exact") == "exact" \
        else "probes"


def _diagnostic_eligible(context, metric_key):
    if not context.get("has_metric", False):
        return False
    return all(key in context for key in
               DIAGNOSTIC_REQUIRED_KEYS + (metric_key,))


def validation_rows(contexts, config, per_scene=2, max_contexts=16):
    """Diagnostic rows come exclusively from validation scenes, spread across
    them, never the first rows of the cache.

    Only metric-labelled contexts (has_metric plus every tensor the
    diagnostic consumes) are eligible: selecting a validation context
    without Jacobians would fail later in tensors_of with a KeyError.
    Eligible rows spread round-robin (one per scene before any scene
    contributes a second), and the returned values are stable indices
    into ``contexts`` so context IDs are preserved. Insufficient
    validation metric coverage is rejected here, before the teacher is
    loaded.
    """
    wanted = C.scene_range(config, "validation")
    metric_key = _diagnostic_metric_key(config)
    by_scene = {}
    for index, context in enumerate(contexts):
        scene = int(context["scene"])
        if scene in wanted and context.get("split") == "validation" and \
                _diagnostic_eligible(context, metric_key):
            by_scene.setdefault(scene, []).append(index)
    if not by_scene:
        raise ValueError("metric cache contains no validation contexts with "
                         "physical metrics; run the metrics stage for the "
                         "validation split before running the diagnostic")
    eligible_total = sum(len(group) for group in by_scene.values())
    planned = int(config.get("validation_metric_contexts", 0))
    if eligible_total < planned:
        raise ValueError(
            f"insufficient validation metric coverage: {eligible_total} "
            f"metric-labelled validation contexts, below the planned "
            f"{planned}; run the metrics stage for the validation split "
            "before running the diagnostic")
    rows = []
    ordered = sorted(by_scene)
    for round_index in range(per_scene):
        for scene in ordered:
            group = by_scene[scene]
            if round_index < len(group) and len(rows) < max_contexts:
                rows.append(group[round_index])
            if len(rows) >= max_contexts:
                break
        if len(rows) >= max_contexts:
            break
    if len({int(contexts[i]["scene"]) for i in rows}) < 2 and len(ordered) >= 2:
        raise ValueError("diagnostic rows cover fewer than 2 validation "
                         "scenes; collect broader validation metric coverage")
    return rows[:max_contexts]


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return dict(rho=None, n=len(a), note="too few points")
    if np.std(a) == 0 or np.std(b) == 0:
        return dict(rho=None, n=len(a),
                    note=("constant input: one side has zero variance, so rank "
                          "association is undefined and must not be reported as 0"))
    rho, p = spearmanr(a, b)
    return dict(rho=float(rho), p=float(p), n=len(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--student", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--allow-unlocked", action="store_true")
    parser.add_argument("--probe-counts", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--no-simulate", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    C.apply_compute_env(args.device or config["device"])
    device = resolve_device(args.device or config["device"])
    protocol = None
    if args.protocol:
        protocol = C.load_protocol(args.protocol)
        C.verify_current_protocol(protocol, config, args.config, stage="diagnose")
    elif not args.allow_unlocked:
        raise C.ProtocolError("pass --protocol ... or --allow-unlocked for this "
                              "development diagnostic")
    allowed = C.scene_range(config, "train") | C.scene_range(config, "validation")
    cache_path = args.cache
    verified = C.verify_label_cache(cache_path, config, expect_mode="shared",
                                    allowed_scenes=allowed, require_metric=True)
    outputs = [args.output] + ([args.predictions] if args.predictions else [])
    C.reserve_outputs(outputs)
    contexts = verified["contexts"]
    rows = validation_rows(contexts, config)
    metric_mode = verified["meta"]["metric_mode"]
    # Perturbation radii are L2 NORMS of the 32-dimensional chunk direction,
    # not per-coordinate RMS values. For 32 coordinates, L2 0.01 corresponds
    # to RMS ~0.00177 (radius / sqrt(32)).
    radii = [0.005, 0.01, 0.02]
    radius_kind = "L2 norm over the 16x2 chunk"
    rng = np.random.default_rng(11)
    report = {"cache": os.path.basename(cache_path),
              "cache_sha256": C.sha256_file(cache_path),
              "cache_lineage": C.load_cache(cache_path)["lineage"],
              "license_note": "privileged simulator supervision for a diagnostic",
              "metric_mode": metric_mode, "n_contexts": len(contexts),
              "sampled_rows": rows,
              "radii": radii, "radius_kind": radius_kind,
              "radius_rms_equivalent": [r / (32 ** 0.5) for r in radii],
              "sampled_scenes": sorted({int(contexts[i]["scene"]) for i in rows}),
              "protocol_id": None if protocol is None else protocol.get("protocol_id"),
              "source_hashes": C.source_hashes(),
              "student": os.path.basename(args.student) if args.student else None,
              "simulated": not args.no_simulate}
    adapter = None
    if not args.no_simulate:
        from pilot import build_adapter
        adapter = build_adapter(config, device)
    student = None
    if args.student:
        from pilot import load_student
        student, student_payload = load_student(args.student, config, device)
        student.eval()
        if protocol is not None:
            C.verify_student_provenance(student_payload, protocol, config, args.student)

    def tensors_of(context):
        cond = torch.from_numpy(np.asarray(context["condition"], dtype=np.float32)).unsqueeze(0).to(device)
        jac_start = torch.from_numpy(np.asarray(context["jacobian_start"], dtype=np.float32)).unsqueeze(0).to(device)
        jac_end = torch.from_numpy(np.asarray(context["jacobian_end"], dtype=np.float32)).unsqueeze(0).to(device)
        target = torch.from_numpy(np.asarray(context["target_mid"], dtype=np.float32)).unsqueeze(0).to(device)
        return cond, jac_start, jac_end, target

    def suffix_of(context, cond):
        steps = config["teacher_steps"]

        def suffix(variable):
            return teacher_half_from(variable, adapter.noise_pred_net, cond, steps)
        return suffix

    def exact_metric(context):
        cond, jac_start, _, target = tensors_of(context)
        return pullback_metric_exact(suffix_of(context, cond), target, jac_start,
                                     config["execute_steps"])

    actual = {}
    if student is not None:
        for index in rows:
            context = contexts[index]
            cond, _, _, target_mid = tensors_of(context)
            noise = torch.from_numpy(np.asarray(context["noise"], dtype=np.float32)).unsqueeze(0).to(device)
            midpoint = torch.from_numpy(np.asarray(context["midpoint"], dtype=np.float32)).unsqueeze(0).to(device)
            target_end = torch.from_numpy(np.asarray(context["target_end"], dtype=np.float32)).unsqueeze(0).to(device)
            teacher_end = torch.from_numpy(np.asarray(context["teacher_end"], dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_mid = student(noise, cond, 0.0)
                predicted_end = student(midpoint, cond, 0.5)
                full_end = student(predicted_mid, cond, 0.5)
            actual[index] = dict(
                mid=(predicted_mid - target_mid).cpu().numpy()[0],
                end=(predicted_end - target_end).cpu().numpy()[0],
                endpoint=(full_end - teacher_end).cpu().numpy()[0])
        stack = {k: np.stack([actual[i][k] for i in rows]).reshape(len(rows), -1)
                 for k in ("mid", "end", "endpoint")}
        report["actual_error_distribution"] = {
            k: dict(rms=float(np.sqrt((v ** 2).mean())),
                    median_norm=float(np.median(np.linalg.norm(v, axis=1))),
                    p95_norm=float(np.quantile(np.linalg.norm(v, axis=1), 0.95)),
                    n=int(v.shape[0]))
            for k, v in stack.items()}
        report["actual_errors_used_for_physical_tests"] = True
        summary, omitted = {}, []
        for branch in ("mid", "end", "endpoint"):
            vals, scenes = [], []
            for i in rows:
                err = actual[i][branch].reshape(-1)
                norm = float(np.linalg.norm(err))
                if not np.isfinite(norm):
                    omitted.append(dict(context=i, branch=branch, reason="nonfinite"))
                    continue
                vals.append(norm)
                scenes.append(int(contexts[i]["scene"]))
            by_scene = {}
            for value, scene in zip(vals, scenes):
                by_scene.setdefault(str(scene), []).append(value)
            zero_frac = float(sum(1 for v in vals if v == 0.0) / max(len(vals), 1))
            summary[branch] = dict(n_contexts=len(vals), n_omitted=len([o for o in omitted if o["branch"] == branch]),
                                   median_norm=(float(np.median(vals)) if vals else None),
                                   zero_sensitivity_fraction=zero_frac,
                                   by_scene={s: dict(median_norm=float(np.median(v)), n=len(v)) for s, v in sorted(by_scene.items())})
        report["actual_error_summary"] = summary
        report["actual_error_omitted"] = omitted

    records = []
    for index in rows:
        context = contexts[index]
        scene = int(context["scene"])
        history = [np.asarray(row, dtype=np.float64) for row in context["history"]]
        cond, jac_start, jac_end, target = tensors_of(context)
        metric = (torch.from_numpy(np.asarray(context["metric_exact"], dtype=np.float32)).unsqueeze(0).to(device)
                  if metric_mode == "exact"
                  else torch.from_numpy(np.asarray(context["probes"], dtype=np.float32)).unsqueeze(0).to(device))
        sketch = metric_mode != "exact"
        exact = None
        if adapter is not None:
            exact = exact_metric(context)
        elif metric_mode == "exact":
            # --no-simulate reuses the cached exact operator instead of
            # omitting exact predictions: the stored metric IS the exact
            # pullback operator for this context.
            exact = metric.detach().clone()
        sets = {}
        for radius in radii:
            dims = 16 * 2
            directions = []
            for _ in range(8):
                d = rng.standard_normal(dims)
                directions.append((d / np.linalg.norm(d) * radius).astype(np.float32))
            sets["random@%.3f" % radius] = directions
        if index in actual:
            sets["actual_mid_error"] = [actual[index]["mid"].reshape(-1)]
            sets["actual_end_error"] = [actual[index]["end"].reshape(-1)]
            sets["actual_endpoint_error"] = [actual[index]["endpoint"].reshape(-1)]
        if exact is not None:
            _, _, vt = np.linalg.svd(exact[0].cpu().numpy(), full_matrices=False)
            sets["most_consequential"] = [(vt[0] * radii[1]).astype(np.float32)]
            sets["least_consequential"] = [(vt[-1] * radii[1]).astype(np.float32)]
        base_exec = {}
        if adapter is not None:
            base_exec["early"] = None
            base_exec["late"] = adapter.execute_from_history(
                scene, history,
                np.asarray(context["target_end"])[:config["execute_steps"]])
            base_exec["endpoint"] = adapter.execute_from_history(
                scene, history,
                np.asarray(context["teacher_end"])[:config["execute_steps"]])
        target_mid = np.asarray(context["target_mid"], dtype=np.float32)
        # Unperturbed teacher suffix + simulator execution, computed ONCE per
        # context and reused for every direction (was repeated per direction).
        baseline_cache = {}
        if adapter is not None:
            with torch.no_grad():
                baseline = teacher_half_from(
                    torch.from_numpy(target_mid).unsqueeze(0).to(device),
                    adapter.noise_pred_net, cond, config["teacher_steps"])
            baseline_cache["early"] = adapter.execute_from_history(
                scene, history,
                baseline[0, :config["execute_steps"]].cpu().numpy())
        for name, directions in sets.items():
            for direction in directions:
                flat = np.asarray(direction, dtype=np.float32).reshape(-1)
                error = torch.from_numpy(flat).reshape(1, 16, 2).to(device)
                row = dict(context=index, scene=scene, set=name,
                           radius=float(np.linalg.norm(flat)))
                if exact is not None:
                    row["exact_pullback"] = float(pullback_quadratic(exact, error).item())
                    row["identity_prefix"] = float(physical_error(
                        jac_start, error, config["execute_steps"]).item())
                if sketch:
                    row["sketch_pullback"] = float(
                        pullback_quadratic(metric, error, sketch=True).item())
                else:
                    row["sketch_pullback"] = None
                # the late and endpoint outcomes have their own predictors: the
                # late term consumes the end error through jacobian_end and the
                # endpoint term consumes the whole-student error through
                # jacobian_start. Comparing an early predictor against a late
                # outcome would mismatch the trained penalty.
                row["late_predicted"] = float(physical_error(
                    jac_end, error, config["execute_steps"]).item())
                row["endpoint_predicted"] = float(physical_error(
                    jac_start, error, config["execute_steps"]).item())
                if adapter is not None:
                    # EARLY: perturb the full teacher midpoint, run the suffix,
                    # execute its prefix. The direction is never added to the
                    # final physical action array.
                    perturbed = target_mid + flat.reshape(16, 2)
                    with torch.no_grad():
                        completed = teacher_half_from(
                            torch.from_numpy(perturbed).unsqueeze(0).to(device),
                            adapter.noise_pred_net, cond, config["teacher_steps"])
                    early_features = adapter.execute_from_history(
                        scene, history,
                        completed[0, :config["execute_steps"]].cpu().numpy())
                    row["early_physical_squared"] = float(
                        np.sum((early_features - baseline_cache["early"]) ** 2))
                    # LATE: perturb the late-branch input and execute
                    if "actual_end_error" in sets and name == "actual_end_error":
                        late_input = (np.asarray(context["target_end"], dtype=np.float64)
                                      + flat.reshape(16, 2))[:config["execute_steps"]]
                        late_features = adapter.execute_from_history(scene, history,
                                                                     late_input)
                        row["late_physical_squared"] = float(
                            np.sum((late_features - base_exec["late"]) ** 2))
                    # ENDPOINT: perturb the whole-student endpoint and execute
                    if "actual_endpoint_error" in sets and name == "actual_endpoint_error":
                        endpoint_input = (np.asarray(context["teacher_end"], dtype=np.float64)
                                          + flat.reshape(16, 2))[:config["execute_steps"]]
                        endpoint_features = adapter.execute_from_history(
                            scene, history, endpoint_input)
                        row["endpoint_physical_squared"] = float(
                            np.sum((endpoint_features - base_exec["endpoint"]) ** 2))
                records.append(row)

    # predictor quality: each predictor is compared ONLY against its matching
    # outcome, mirroring the trained penalty each one stands for. Association
    # is computed WITHIN each context (directional discrimination: does the
    # predictor rank directions correctly for THIS context?) and then
    # aggregated: median across contexts plus per-scene medians. A pooled
    # correlation across contexts can reflect between-context scale
    # differences rather than useful directional sensitivity, so pooled
    # values are reported separately and never as the headline.
    report["association"] = {}
    report["association_pooled"] = {}
    pairs = [("exact_pullback", "early_physical_squared"),
             ("identity_prefix", "early_physical_squared"),
             ("sketch_pullback", "early_physical_squared"),
             ("late_predicted", "late_physical_squared"),
             ("endpoint_predicted", "endpoint_physical_squared")]
    context_scene = {index: int(contexts[index]["scene"]) for index in rows}
    for predictor, outcome in pairs:
        for name in sorted({r["set"] for r in records}):
            within = []
            by_scene = {}
            for index in rows:
                subset = [r for r in records
                          if r["set"] == name and r["context"] == index]
                preds = [r.get(predictor) for r in subset]
                outs = [r.get(outcome) for r in subset]
                if any(p is None for p in preds) or any(o is None for o in outs):
                    continue
                stat = _spearman(preds, outs)
                if stat.get("rho") is None:
                    continue
                within.append(stat["rho"])
                by_scene.setdefault(context_scene[index], []).append(stat["rho"])
            if within:
                report["association"].setdefault(
                    f"{predictor}->{outcome}", {})[name] = dict(
                        median_rho=float(np.median(within)),
                        n_contexts=len(within),
                        by_scene={str(s): dict(median_rho=float(np.median(v)),
                                               n_contexts=len(v))
                                  for s, v in sorted(by_scene.items())})
            pooled_subset = [r for r in records if r["set"] == name]
            pooled_preds = [r.get(predictor) for r in pooled_subset]
            pooled_outs = [r.get(outcome) for r in pooled_subset]
            if not any(p is None for p in pooled_preds) and \
                    not any(o is None for o in pooled_outs):
                report["association_pooled"].setdefault(
                    f"{predictor}->{outcome}", {})[name] = _spearman(
                        pooled_preds, pooled_outs)
    report["association_note"] = ("within-context directional rank association, "
                                 "median across contexts plus per-scene medians; "
                                 "pooled values live under association_pooled and "
                                 "are never the headline; whole-chunk MSE is "
                                 "constant on a sphere so its association is "
                                 "undefined and is never reported as zero")

    # sketch quality against the exact metric on actual student errors.
    # Without a student there is no actual error: use the cached
    # (midpoint - target_mid) residual as the error direction, never the
    # raw midpoint itself (a position, not an error).
    if metric_mode == "exact" and adapter is not None:
        report["sketch_audit"] = {}
        for index in rows[:8]:
            context = contexts[index]
            cond, jac_start, _, target = tensors_of(context)
            exact = exact_metric(context)
            if index in actual:
                error_flat = actual[index]["mid"].reshape(-1)
            else:
                mid = np.asarray(context["midpoint"], dtype=np.float32)
                tgt = np.asarray(context["target_mid"], dtype=np.float32)
                error_flat = (mid - tgt).reshape(-1)
            error = torch.from_numpy(np.asarray(error_flat, dtype=np.float32)).reshape(1, 16, 2).to(device)
            exact_value = float(pullback_quadratic(exact, error).item())
            exact_trace = float(pullback_trace(exact).item())
            entry = {"exact_quadratic": exact_value, "exact_trace": exact_trace}
            for count in args.probe_counts:
                ratios_q, ratios_t = [], []
                for draw in range(3):
                    gen = torch.Generator(device=device).manual_seed(10_000 + index * 131 + count * 17 + draw)
                    probes = pullback_probes(suffix_of(context, cond), target, jac_start,
                                             config["execute_steps"], num_probes=count,
                                             generator=gen)
                    value = float(pullback_quadratic(probes, error, sketch=True).item())
                    trace = float(pullback_trace(probes, sketch=True).item())
                    if exact_value:
                        ratios_q.append(value / exact_value)
                    if exact_trace:
                        ratios_t.append(trace / exact_trace)
                import statistics as _stats
                entry[str(count)] = dict(
                    quadratic_ratio_mean=(sum(ratios_q) / len(ratios_q)) if ratios_q else None,
                    quadratic_ratio_std=(_stats.pstdev(ratios_q)) if len(ratios_q) > 1 else 0.0,
                    trace_ratio_mean=(sum(ratios_t) / len(ratios_t)) if ratios_t else None,
                    trace_ratio_std=(_stats.pstdev(ratios_t)) if len(ratios_t) > 1 else 0.0,
                    n_draws=3, sketch_seed_base=10_000 + index * 131 + count * 17)
            report["sketch_audit"][str(index)] = entry

    # perturbation-radius stability: eps versus eps/2 on a few contexts
    if adapter is not None:
        stability = {}
        from core import finite_difference_jacobian
        for index in rows[:3]:
            context = contexts[index]
            scene = int(context["scene"])
            history = [np.asarray(row, dtype=np.float64) for row in context["history"]]
            prefix = np.asarray(context["teacher_end"])[:config["execute_steps"]]
            eps = config["finite_difference_epsilon"]

            def jac(radius):
                return finite_difference_jacobian(
                    lambda chunk: adapter.execute_from_history(scene, history, chunk),
                    prefix, epsilon=radius)
            a, b = jac(eps), jac(eps / 2)
            denom = max(float(np.abs(a).max()), 1e-12)
            stability[str(index)] = dict(
                max_abs_change=float(np.abs(a - b).max()),
                relative_change=float(np.abs(a - b).max() / denom),
                epsilon=eps)
        report["epsilon_stability"] = stability

    if args.predictions:
        np.savez(args.predictions,
                 records=np.array(records, dtype=object),
                 rows=np.array(rows),
                 actual_mid=(np.array([actual[i]["mid"] for i in rows]) if actual
                             else np.zeros((0, 16, 2), np.float32)),
                 actual_end=(np.array([actual[i]["end"] for i in rows]) if actual
                             else np.zeros((0, 16, 2), np.float32)),
                 actual_endpoint=(np.array([actual[i]["endpoint"] for i in rows]) if actual
                                  else np.zeros((0, 16, 2), np.float32)))
        report["predictions_path"] = args.predictions
    report["schema_version"] = C.SCHEMA_VERSION
    C.write_meta(args.output, report)
    C.complete_output(args.output, dict(contexts=len(rows)))
    if args.predictions:
        C.complete_output(args.predictions, dict(records=len(records)))
    print("wrote diagnostic " + args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
