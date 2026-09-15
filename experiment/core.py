import numpy as np
import torch
from torch import nn

HORIZON = 16
ACTION_DIM = 2


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


def integrate(field, latent, condition, start, stop, steps):
    if steps < 1 or not 0 <= start < stop <= 1:
        raise ValueError("invalid integration interval")
    delta = (stop - start) / steps
    result = latent
    for index in range(steps):
        times = result.new_full((result.shape[0],), start + index * delta)
        result = result + delta * field(result, times, condition)
    return result


class TwoStepStudent(nn.Module):
    def __init__(self, condition_dim=514, horizon=HORIZON, action_dim=ACTION_DIM,
                 width=512):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        flat_dim = horizon * action_dim
        self.network = nn.Sequential(
            nn.Linear(flat_dim + condition_dim + 1, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, flat_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, latent, condition, start):
        times = latent.new_full((latent.shape[0], 1), start)
        inputs = torch.cat((latent.flatten(1), condition, times), dim=1)
        velocity = self.network(inputs).reshape_as(latent)
        return latent + 0.5 * velocity

    def sample(self, noise, condition):
        midpoint = self(noise, condition, 0.0)
        return self(midpoint, condition, 0.5)


def sample_with_midpoint(noise, condition, student):
    midpoint = student(noise, condition, 0.0)
    return student(midpoint, condition, 0.5), midpoint


def finite_difference_jacobian(execute, action, epsilon=0.01):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite difference epsilon must be positive and finite")
    action = np.asarray(action, dtype=np.float64)
    flat = action.reshape(-1)
    columns = []
    for coordinate in range(flat.size):
        perturbation = np.zeros_like(flat)
        perturbation[coordinate] = epsilon
        positive = execute((flat + perturbation).reshape(action.shape))
        negative = execute((flat - perturbation).reshape(action.shape))
        columns.append((positive - negative) / (2 * epsilon))
    jacobian = np.stack(columns, axis=-1)
    if not np.isfinite(jacobian).all():
        raise ValueError("nonfinite simulator derivative")
    return jacobian


def pullback_probes(suffix, target, physical_jacobian, execute_steps, num_probes=4, generator=None):
    """Sketched metric: L random-probe VJPs transported to the solver input.

    probes[b, l] = J_prefix^T B0^T xi_l, so (probes_l . e)^2 = (xi_l . C e)^2 and
    E_l (probes_l . e)^2 = ||C e||^2 = ||C||_F^2 for the trace. Both are unbiased
    for a FIXED error, but with L fixed the sketch has rank <= L: directions in
    the null space of the sketch stay unpenalized and the student can adapt to
    the particular sketch. Use pullback_metric_exact unless the sketch is the
    declared object of study.
    """
    if num_probes < 1:
        raise ValueError("num_probes must be >= 1")
    with torch.enable_grad():
        variable = target.detach().clone().requires_grad_(True)
        completed = suffix(variable)
        batch, horizon, action_dim = completed.shape
        prefix_dim = execute_steps * action_dim
        if physical_jacobian.shape[-1] != prefix_dim:
            raise ValueError("jacobian uses the wrong action prefix")
        output_dim = physical_jacobian.shape[1]
        probes = []
        for probe_index in range(num_probes):
            signs = torch.randint(
                0, 2, (batch, output_dim), device=target.device,
                generator=generator,
            ).to(target.dtype).mul_(2).sub_(1)
            action_cotangent = torch.bmm(
                physical_jacobian.transpose(1, 2), signs.unsqueeze(-1)
            ).squeeze(-1)
            full_cotangent = torch.zeros_like(completed)
            full_cotangent[:, :execute_steps] = action_cotangent.reshape(
                batch, execute_steps, action_dim
            )
            gradient, = torch.autograd.grad(
                completed, variable, grad_outputs=full_cotangent,
                retain_graph=probe_index + 1 < num_probes,
                create_graph=False,
            )
            probes.append(gradient.detach().flatten(1))
        stacked = torch.stack(probes, dim=1)
        if not torch.isfinite(stacked).all():
            raise ValueError("nonfinite pullback probes")
        return stacked


def pullback_metric_exact(suffix, target, physical_jacobian, execute_steps):
    """Exact transmitted physical metric C = B0 J_prefix.

    J_prefix = d(P_m S(y)) / d y in R^{prefix_dim x flat_dim} is built with one
    VJP per prefix output coordinate (prefix_dim backwards passes, no extra
    simulator branches). Then

        Q0(e) = ||C e||^2 ,     tr M = ||C||_F^2

    is the exact rank-full quadratic form, replacing the rank-<=L sketch. The
    returned tensor is C with shape (batch, output_dim, flat_dim); it must be
    consumed with pullback_quadratic() and pullback_trace(), never by the
    sketch averaging code.
    """
    with torch.enable_grad():
        variable = target.detach().clone().requires_grad_(True)
        completed = suffix(variable)
        batch, horizon, action_dim = completed.shape
        prefix_dim = execute_steps * action_dim
        if physical_jacobian.shape[-1] != prefix_dim:
            raise ValueError("jacobian uses the wrong action prefix")
        output_dim = physical_jacobian.shape[1]
        rows = []
        total = prefix_dim
        for index in range(total):
            cotangent = torch.zeros_like(completed)
            cotangent.view(batch, -1)[:, index] = 1.0
            gradient, = torch.autograd.grad(
                completed, variable, grad_outputs=cotangent,
                retain_graph=index + 1 < total, create_graph=False,
            )
            rows.append(gradient.detach().flatten(1))
        jacobian_prefix = torch.stack(rows, dim=1)
        metric = torch.bmm(physical_jacobian, jacobian_prefix)
        if not torch.isfinite(metric).all():
            raise ValueError("nonfinite exact pullback metric")
        return metric


def pullback_quadratic(metric, error, sketch=False):
    """Q0(e) = ||C e||^2 for exact metrics, or mean_l (r_l . e)^2 for a sketch.

    exact: metric (batch, output_dim, flat_dim), metric = C = B0 J_prefix
    sketch: metric (batch, L, flat_dim), metric = J_prefix^T B0^T xi_l
    """
    flat = error.flatten(1)
    if metric.shape[-1] != flat.shape[1]:
        raise ValueError("metric and error dimensions disagree")
    projected = torch.bmm(metric, flat.unsqueeze(-1)).squeeze(-1)
    if sketch:
        return projected.square().mean(1)
    return projected.square().sum(1)


def pullback_trace(metric, sketch=False):
    if sketch:
        return metric.square().sum(-1).mean(-1)
    return metric.square().sum((1, 2))


def physical_error(jacobian, error, execute_steps):
    prefix = error[:, :execute_steps].flatten(1)
    return torch.bmm(jacobian, prefix.unsqueeze(-1)).squeeze(-1).square().sum(-1)


def objective(student, batch, mode, execute_steps, scales, metric_weight=0.25,
              anchor_weight=0.25, metric_count=None, metric_mode="exact"):
    predicted_mid = student(batch["noise"], batch["condition"], 0.0)
    predicted_end = student(batch["midpoint"], batch["condition"], 0.5)
    mid_error = predicted_mid - batch["target_mid"]
    end_error = predicted_end - batch["target_end"]
    base = 0.5 * (mid_error.square().mean() + end_error.square().mean())
    full_end = student(predicted_mid, batch["condition"], 0.5)
    endpoint_error = full_end - batch["teacher_end"]
    anchor = endpoint_error.square().mean()
    penalty = base.new_zeros(())
    selected = slice(None, metric_count)
    metric = batch["metric_exact"] if metric_mode == "exact" else batch["probes"]
    sketch = metric_mode != "exact"
    if mode == "prefix":
        early = mid_error[selected, :execute_steps]
        late = end_error[selected, :execute_steps]
        penalty = 0.5 * (early.square().mean() + late.square().mean())
    elif mode == "endpoint":
        penalty = physical_error(
            batch["jacobian_start"][selected], endpoint_error[selected], execute_steps
        ).mean() / scales["endpoint"]
    elif mode == "scalar":
        first_trace = pullback_trace(metric[selected], sketch=sketch)
        last_trace = batch["jacobian_end"][selected].square().sum((1, 2))
        early = first_trace * mid_error[selected].square().mean((1, 2))
        late = last_trace * end_error[selected].square().mean((1, 2))
        penalty = 0.5 * (early.mean() + late.mean()) / scales["pullback"]
    elif mode in ("pullback", "identity"):
        if mode == "pullback":
            early = pullback_quadratic(metric[selected], mid_error[selected],
                                       sketch=sketch)
        else:
            early = physical_error(batch["jacobian_start"][selected],
                                   mid_error[selected], execute_steps)
        late = physical_error(batch["jacobian_end"][selected], end_error[selected],
                              execute_steps)
        penalty = 0.5 * (early.mean() + late.mean()) / scales[mode]
    elif mode != "uniform":
        raise ValueError(mode)
    loss = base + anchor_weight * anchor + metric_weight * penalty
    return loss, {"base": base.detach(), "anchor": anchor.detach(),
                  "penalty": penalty.detach(),
                  "weighted_penalty": (metric_weight * penalty).detach(),
                  "mid_error_rms": mid_error.detach().square().mean().sqrt(),
                  "end_error_rms": end_error.detach().square().mean().sqrt(),
                  "endpoint_error_rms": endpoint_error.detach().square().mean().sqrt()}


def metric_scales(cache, rows, metric_mode="exact"):
    first = cache["jacobian_start"][rows].square().sum((1, 2))
    last = cache["jacobian_end"][rows].square().sum((1, 2))
    sketch = metric_mode != "exact"
    key = "probes" if sketch else "metric_exact"
    transported = pullback_trace(cache[key][rows], sketch=sketch)

    def positive_median(values):
        positive = values[values > 1e-12]
        if positive.numel() < 8:
            raise ValueError(
                "too few nonzero sensitivity labels (need >= 8 positive traces); "
                "stop the metric experiment and inspect the cache")
        return float(positive.median().item())

    transported_positive = transported[transported > 1e-12]
    zero_fraction = float((transported <= 1e-12).float().mean().item())
    scales = {"endpoint": positive_median(first),
              "identity": positive_median(0.5 * (first + last)),
              "pullback": positive_median(0.5 * (transported + last))}
    stats = dict(metric_mode=metric_mode,
                 zero_sensitivity_fraction=zero_fraction,
                 n_positive_transported=int(transported_positive.numel()),
                 n_rows=int(len(rows)),
                 endpoint_median=scales["endpoint"],
                 identity_median=scales["identity"],
                 pullback_median=scales["pullback"])
    return scales, stats
