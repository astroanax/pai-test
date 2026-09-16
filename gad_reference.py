"""Reference kernels, not a robot training or rollout pipeline."""
import hashlib
import json
import math


def stable_seed(stage, scene, decision, anchor, direction):
    payload = [str(stage), int(scene), int(decision), int(anchor), int(direction)]
    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def require_finite(tensor, name):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite")
    return tensor


def source_from_canonical(latent, source):
    import torch

    if source == "gaussian":
        return latent
    if source == "uniform":
        return 0.5 * (1.0 + torch.erf(latent / math.sqrt(2.0)))
    if source == "uniform_symmetric":
        p = 0.5 * (1.0 + torch.erf(latent / math.sqrt(2.0)))
        return 2.0 * p - 1.0
    raise ValueError(f"unknown source: {source}")


def teacher_endpoint(field, latent, condition, steps, source="gaussian"):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive int")
    require_finite(latent, "latent")
    require_finite(condition, "condition")
    if latent.shape[0] != condition.shape[0]:
        raise ValueError("latent and condition batch sizes must match")
    state = source_from_canonical(latent, source)
    for i in range(steps):
        times = state.new_full((state.shape[0],), float(i) / float(steps))
        update = field(state, times, condition)
        require_finite(update, "field output")
        state = state + update / float(steps)
        require_finite(state, "teacher state")
    return state


def make_student(condition_dim=514, horizon=16, action_dim=2, width=512):
    import torch
    import torch.nn as nn

    for name, value in (
        ("condition_dim", condition_dim),
        ("horizon", horizon),
        ("action_dim", action_dim),
        ("width", width),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int")
    flat = horizon * action_dim

    class TwoStepStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(flat + condition_dim + 1, width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, flat),
            )
            final = self.network[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        def forward(self, latent, condition):
            if latent.dim() != 3:
                raise ValueError("latent must have shape (B, horizon, action_dim)")
            b, h, a = latent.shape
            if h != horizon or a != action_dim:
                raise ValueError(
                    f"latent must have shape (B, {horizon}, {action_dim})"
                )
            if condition.shape != (b, condition_dim):
                raise ValueError(
                    f"condition must have shape (B, {condition_dim})"
                )
            if latent.dtype != torch.float32 or condition.dtype != torch.float32:
                raise ValueError("latent and condition must be FP32")
            if latent.device != condition.device:
                raise ValueError("latent and condition must share a device")
            state = latent
            for k in range(2):
                times = latent.new_full((b, 1), float(k) / 2.0)
                inputs = torch.cat([state.flatten(1), condition, times], dim=1)
                state = state + 0.5 * self.network(inputs).reshape_as(state)
            return state

    student = TwoStepStudent()
    student._condition_dim = condition_dim
    student._horizon = horizon
    student._action_dim = action_dim
    return student


def paired_loss(
    student,
    latent,
    condition,
    direction,
    teacher_anchor,
    teacher_perturbed,
    radius,
    mode="gad",
    beta=0.1,
):
    import torch

    if mode not in ("anchor", "augmented", "gad"):
        raise ValueError("mode must be one of anchor/augmented/gad")
    positive_finite(radius, "radius")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError("beta must be a number")
    if not math.isfinite(float(beta)) or float(beta) < 0:
        raise ValueError("beta must be finite and nonnegative")
    for name, tensor in (
        ("latent", latent),
        ("condition", condition),
        ("direction", direction),
        ("teacher_anchor", teacher_anchor),
        ("teacher_perturbed", teacher_perturbed),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be FP32")
    devices = {t.device for t in (latent, condition, direction, teacher_anchor, teacher_perturbed)}
    if len(devices) != 1:
        raise ValueError("all tensors must share a device")
    for name, tensor in (
        ("latent", latent),
        ("condition", condition),
        ("direction", direction),
        ("teacher_anchor", teacher_anchor),
        ("teacher_perturbed", teacher_perturbed),
    ):
        require_finite(tensor, name)
    if direction.shape != latent.shape:
        raise ValueError("direction must match latent shape")
    if latent.dim() < 2:
        raise ValueError("latent must have a batch dimension")
    b = latent.shape[0]
    if condition.shape[0] != b:
        raise ValueError("condition batch must match latent batch")
    norms = direction.flatten(1).norm(p=2, dim=1)
    if not torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("direction must have unit L2 norm per pair")
    base = latent.detach()
    perturbed = base + float(radius) * direction.detach()
    # Item 6: A0 duplicates the anchor input AND the anchor target, so the
    # anchor loss is plain anchor MSE. Every arm runs ONE 2B forward (2B
    # sample evaluations); the trainer's anchor arm runs two B forwards,
    # which is the same 2B count. Never B + 2B-with-half-discarded (3B).
    second_input = base if mode == "anchor" else perturbed
    target_second = teacher_anchor if mode == "anchor" else teacher_perturbed
    predictions = student(
        torch.cat([base, second_input], dim=0),
        torch.cat([condition.detach(), condition.detach()], dim=0),
    )
    if predictions.shape != (2 * b, *base.shape[1:]):
        raise ValueError("student output shape mismatch")
    pred_anchor, pred_second = predictions.chunk(2, dim=0)
    if pred_anchor.shape != teacher_anchor.shape:
        raise ValueError("teacher_anchor must match student output shape")
    if pred_second.shape != target_second.shape:
        raise ValueError("teacher_perturbed must match student output shape")
    require_finite(pred_anchor, "pred_anchor")
    require_finite(pred_second, "pred_second")
    mse_anchor = torch.mean((pred_anchor - teacher_anchor.detach()) ** 2)
    mse_second = torch.mean((pred_second - target_second.detach()) ** 2)
    value = 0.5 * (mse_anchor + mse_second)
    if mode == "anchor":
        response = torch.zeros((), device=value.device, dtype=value.dtype)
        loss = value
    else:
        response_error = (
            (pred_second - pred_anchor)
            - (teacher_perturbed.detach() - teacher_anchor.detach())
        ) / float(radius)
        response = torch.mean(torch.sum(response_error.flatten(1) ** 2, dim=1))
        loss = value + (float(beta) * response if mode == "gad" else 0.0)
    require_finite(loss, "loss")
    require_finite(value, "value")
    require_finite(response, "response")
    return {
        "loss": loss,
        "value": value.detach(),
        "response": response.detach(),
        "pair_count": b,
        "sample_evals": 2 * b,
    }


def invert_prefix(
    model,
    initial,
    condition,
    target_prefix,
    steps=10,
    learning_rate=0.1,
    trust_rms=0.25,
    gamma=0.01,
):
    import torch

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative int")
    positive_finite(learning_rate, "learning_rate")
    positive_finite(trust_rms, "trust_rms")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
        raise ValueError("gamma must be a number")
    if not math.isfinite(float(gamma)) or float(gamma) < 0:
        raise ValueError("gamma must be finite and nonnegative")
    for name, tensor in (
        ("initial", initial),
        ("condition", condition),
        ("target_prefix", target_prefix),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be FP32")
    devices = {initial.device, condition.device, target_prefix.device}
    if len(devices) != 1:
        raise ValueError("initial, condition, and target_prefix must share a device")
    for name, tensor in (
        ("initial", initial),
        ("condition", condition),
        ("target_prefix", target_prefix),
    ):
        require_finite(tensor, name)
    if initial.dim() < 2:
        raise ValueError("initial must have a batch dimension")
    b = initial.shape[0]
    if condition.shape[0] != b or target_prefix.shape[0] != b:
        raise ValueError("batch sizes must match")
    if (target_prefix.dim() != 3 or target_prefix.shape[2] != initial.shape[2]
            or not 0 < target_prefix.shape[1] <= initial.shape[1]):
        raise ValueError("target prefix shape mismatch")
    params = list(model.parameters()) if hasattr(model, "parameters") else []
    if any(p.requires_grad for p in params):
        raise ValueError("all model parameters must be frozen (requires_grad False)")
    was_training = bool(getattr(model, "training", False))
    model.eval()

    def per_case(action, candidate, origin):
        width = target_prefix.shape[1]
        prefix_mse = torch.mean(
            (action[:, :width] - target_prefix.detach()).flatten(1) ** 2,
            dim=1,
        )
        reg = torch.mean((candidate - origin).flatten(1) ** 2, dim=1)
        return prefix_mse + float(gamma) * reg, prefix_mse

    # Audit fdb59ff item 3: evaluate and select over exactly
    # q0, ..., qK. One forward per iterate (K+1 total); the first K
    # forwards run with grad and their output serves selection AND the
    # backward, the final qK forward is selection-only. The old loop
    # evaluated q0 twice and never evaluated the computed qK, so K=1
    # could not return an improved candidate.
    def _project(point):
        with torch.no_grad():
            delta = point - origin
            flat = delta.flatten(1)
            norms = flat.norm(p=2, dim=1)
            over = norms > trust_l2
            if bool(over.any()):
                scale = torch.ones_like(norms)
                scale[over] = trust_l2 / norms[over]
                return (origin + (flat * scale.unsqueeze(1)
                                  ).reshape_as(delta))
            return point

    def _consider(state, best):
        with torch.no_grad():
            improved = state["objective"] < best["objective"]
            if not bool(improved.any()):
                return best
            out = dict(best)
            for key in ("latent", "action"):
                idx = improved.unsqueeze(1)
                out[key] = torch.where(
                    idx.expand_as(best[key].flatten(1)),
                    state[key].flatten(1),
                    best[key].flatten(1),
                ).reshape_as(best[key])
            # Per-case 1-D objectives: plain elementwise selection.
            for key in ("objective", "prefix"):
                out[key] = torch.where(
                    improved, state[key], best[key])
            return out

    try:
        origin = initial.detach().clone()
        trust_l2 = float(trust_rms) * math.sqrt(initial[0].numel())
        forwards = 0
        backwards = 0
        best = None
        current = origin.clone()
        # Exactly one forward per iterate q0..qK (K+1 total). The first
        # K forwards run with grad and double as the backward source;
        # the qK forward is selection-only.
        for iteration in range(steps + 1):
            last = iteration == steps
            if last:
                with torch.no_grad():
                    action = model(current, condition.detach())
                    forwards += 1
                    objective, prefix = per_case(
                        action, current, origin)
                state = {"latent": current.clone(),
                         "action": action.detach().clone(),
                         "objective": objective.detach().clone(),
                         "prefix": prefix.detach().clone()}
                best = state if best is None else _consider(state, best)
            else:
                with torch.enable_grad():
                    candidate = current.detach().requires_grad_(True)
                    action = model(candidate, condition.detach())
                    forwards += 1
                    objective, prefix = per_case(
                        action, candidate, origin)
                    total = torch.sum(objective)
                state = {"latent": candidate.detach().clone(),
                         "action": action.detach().clone(),
                         "objective": objective.detach().clone(),
                         "prefix": prefix.detach().clone()}
                best = state if best is None else _consider(state, best)
                grads = torch.autograd.grad(total, candidate)[0].detach()
                backwards += 1
                current = _project(
                    candidate.detach() - float(learning_rate) * grads)
        assert forwards == steps + 1, (forwards, steps)
        assert backwards == steps, (backwards, steps)
        return {
            "latent": best["latent"].detach(),
            "action": best["action"].detach(),
            "objective": best["objective"].detach(),
            "prefix_mse": best["prefix"].detach(),
            "forward_calls": forwards,
            "backward_calls": backwards,
            "sample_forward_evaluations": b * forwards,
        }
    finally:
        if was_training:
            model.train()
