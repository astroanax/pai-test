import numpy as np
import torch
from torch import nn

HORIZON = 16
ACTION_DIM = 2
SOURCE_CONVENTIONS = {
    "gaussian": "standard normal, torch.randn / rng.standard_normal",
    "uniform": "half-open [0, 1), torch.rand / rng.random, upstream native-fast source",
    "uniform_symmetric": "closed [-1, 1), (rand * 2 - 1)",
}


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


def sample_source(rng, shape, source, device=None):
    """One sampler for collection, training targets, and evaluation.

    Conventions are declared in SOURCE_CONVENTIONS and are NOT interchangeable:
    the upstream native-fast example draws `torch.rand` in [0, 1), while its
    training branch draws `torch.randn`. Never substitute one for the other.
    """
    if source not in SOURCE_CONVENTIONS:
        raise ValueError(f"unknown source distribution {source!r}; "
                         f"known: {sorted(SOURCE_CONVENTIONS)}")
    if source == "gaussian":
        values = rng.standard_normal(shape)
    elif source == "uniform":
        values = rng.random(shape)
    else:
        values = rng.random(shape) * 2 - 1
    values = np.asarray(values, dtype=np.float32)
    if device is not None:
        return torch.from_numpy(values).to(device)
    return values


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


def make_student(config, device, seed=None, state_dict=None):
    """Construct the student with an explicit torch seed.

    Fine-tuning arms share one initialization (paired comparison); the warm
    start is built once with its own seed. PyTorch RNG is seeded explicitly so
    NumPy-only seeding cannot leave initialization uncontrolled.
    """
    if seed is not None:
        torch.manual_seed(seed)
    student = TwoStepStudent(condition_dim=514, horizon=HORIZON, action_dim=ACTION_DIM,
                             width=config["width"])
    if state_dict is not None:
        student.load_state_dict(state_dict)
    return student.to(device)


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
    E_l (probes_l . e)^2 = ||C e||^2 for a FIXED error e, while E over probe
    draws of the probe second moment gives the Frobenius norm ||C||_F^2 for
    the trace. The two identities are different quantities: Q(e) = ||Ce||^2
    evaluates one error direction, tr(C^T C) = ||C||_F^2 sums over all of
    them. With L fixed the sketch has rank <= L: directions in the null
    space of the sketch stay unpenalized and the student can adapt to the
    particular sketch. Use pullback_metric_exact unless the sketch is the
    declared object of study.
    """
    if num_probes < 1:
        raise ValueError("num_probes must be >= 1")
    with torch.enable_grad():
        variable = target.detach().clone().requires_grad_(True)
        completed = suffix(variable)
        if completed.shape[1:] != variable.shape[1:]:
            raise ValueError("suffix must return the same horizon and action "
                             "dimension as its input")
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

    J_prefix = d(P_m S(y)) / d y in R^{prefix_dim x flat_dim}, flat_dim =
    horizon * action_dim, is built with one VJP per prefix output coordinate
    (prefix_dim backwards passes, no extra simulator branches). Then

        Q0(e) = ||C e||^2 ,     tr(C^T C) = ||C||_F^2

    Q and tr are different quantities: the quadratic evaluates one fixed
    error, the trace sums over all directions. For C = B J_prefix with m
    executed actions, rank(C) <= 2m (at most 16 for 8 steps on a
    32-dimensional chunk): this is an unsketched pullback operator built
    from finite-difference physical sensitivities, not a full-rank exact
    nonlinear execution-loss metric, and it replaces the rank-<=L sketch. The
    returned tensor is C with shape (batch, output_dim, horizon * action_dim);
    its last axis covers the WHOLE noisy action chunk, not just the executed
    prefix, so it must never be truncated to prefix_dim columns.
    """
    with torch.enable_grad():
        variable = target.detach().clone().requires_grad_(True)
        completed = suffix(variable)
        if completed.shape[1:] != variable.shape[1:]:
            raise ValueError("suffix must return the same horizon and action "
                             "dimension as its input")
        batch, horizon, action_dim = completed.shape
        flat_dim = horizon * action_dim
        prefix_dim = execute_steps * action_dim
        if physical_jacobian.shape[-1] != prefix_dim:
            raise ValueError("jacobian uses the wrong action prefix")
        output_dim = physical_jacobian.shape[1]
        rows = []
        for index in range(prefix_dim):
            cotangent = torch.zeros_like(completed)
            cotangent.view(batch, -1)[:, index] = 1.0
            gradient, = torch.autograd.grad(
                completed, variable, grad_outputs=cotangent,
                retain_graph=index + 1 < prefix_dim, create_graph=False,
            )
            rows.append(gradient.detach().flatten(1))
        jacobian_prefix = torch.stack(rows, dim=1)
        metric = torch.bmm(physical_jacobian, jacobian_prefix)
        if metric.shape != (batch, output_dim, flat_dim):
            raise ValueError(f"metric shape {tuple(metric.shape)} != "
                             f"{(batch, output_dim, flat_dim)}")
        if not torch.isfinite(metric).all():
            raise ValueError("nonfinite exact pullback metric")
        return metric


def pullback_quadratic(metric, error, sketch=False):
    """Q0(e) = ||C e||^2 for exact metrics, or mean_l (r_l . e)^2 for a sketch.

    exact: metric (batch, output_dim, flat_dim), metric = C = B0 J_prefix
    sketch: metric (batch, L, flat_dim), metric = J_prefix^T B0^T xi_l
    `error` is the full generative error with flat dimension horizon * action_dim.
    """
    flat = error.flatten(1)
    if metric.shape[-1] != flat.shape[1]:
        raise ValueError(f"metric last axis {metric.shape[-1]} != error flat "
                         f"dimension {flat.shape[1]}; the error must be the full "
                         f"noisy action chunk, not an executed prefix")
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


METRIC_MODES = ("endpoint", "pullback", "identity", "scalar")


def mode_needs_metric(mode):
    return mode in METRIC_MODES


def objective(student, batch, mode, execute_steps, scales=None, metric_weight=0.25,
              anchor_weight=0.25, metric_count=None, metric_mode="exact"):
    """Shared multi-interval distillation objective plus the mode's extra penalty.

    The metric tensors are only touched inside the branches that consume them, so
    `uniform` and `prefix` training can run on a label-only cache with no
    Jacobians, no transported metric, and no scale estimation.
    """
    predicted_mid = student(batch["noise"], batch["condition"], 0.0)
    midpoint = batch["midpoint"]
    predicted_end = student(midpoint, batch["condition"], 0.5)
    mid_error = predicted_mid - batch["target_mid"]
    end_error = predicted_end - batch["target_end"]
    base = 0.5 * (mid_error.square().mean() + end_error.square().mean())
    full_end = student(predicted_mid, batch["condition"], 0.5)
    endpoint_error = full_end - batch["teacher_end"]
    anchor = endpoint_error.square().mean()
    penalty = base.new_zeros(())
    # metric_count == 0 means this batch carries no designated physical rows:
    # the physical penalty is zero, never an all-batch fallback (an empty mean
    # would be NaN and silently poison the update; a full-batch fallback would
    # charge ordinary rows for sensitivity they were not selected to carry).
    # Batches are built role-first (ordinary rows, then designated physical
    # rows), so the physical slice is the TAIL of the batch, not the head.
    if metric_count:
        selected = slice(len(batch["noise"]) - metric_count, None)
    else:
        selected = None
    if selected is None:
        if mode == "prefix":
            whole = slice(None)
            early = mid_error[whole, :execute_steps]
            late = end_error[whole, :execute_steps]
            penalty = 0.5 * (early.square().mean() + late.square().mean())
        else:
            penalty = base.new_zeros(())
    elif mode == "prefix":
        early = mid_error[selected, :execute_steps]
        late = end_error[selected, :execute_steps]
        penalty = 0.5 * (early.square().mean() + late.square().mean())
    elif mode_needs_metric(mode):
        if scales is None:
            raise ValueError("metric modes require metric scales")
        if metric_mode == "exact":
            metric = batch["metric_exact"]
        else:
            metric = batch["probes"]
        sketch = metric_mode != "exact"
        if mode == "endpoint":
            penalty = physical_error(
                batch["jacobian_start"][selected], endpoint_error[selected],
                execute_steps).mean() / scales["endpoint"]
        elif mode == "scalar":
            first_trace = pullback_trace(metric[selected], sketch=sketch)
            last_trace = batch["jacobian_end"][selected].square().sum((1, 2))
            early = first_trace * mid_error[selected].square().mean((1, 2))
            late = last_trace * end_error[selected].square().mean((1, 2))
            penalty = 0.5 * (early.mean() + late.mean()) / scales["pullback"]
        elif mode == "pullback":
            early = pullback_quadratic(metric[selected], mid_error[selected],
                                       sketch=sketch)
            late = physical_error(batch["jacobian_end"][selected],
                                  end_error[selected], execute_steps)
            penalty = 0.5 * (early.mean() + late.mean()) / scales["pullback"]
        elif mode == "identity":
            early = physical_error(batch["jacobian_start"][selected],
                                   mid_error[selected], execute_steps)
            late = physical_error(batch["jacobian_end"][selected],
                                  end_error[selected], execute_steps)
            penalty = 0.5 * (early.mean() + late.mean()) / scales["identity"]
    elif mode != "uniform":
        raise ValueError(mode)
    loss = base + anchor_weight * anchor + metric_weight * penalty
    return loss, {"base": base.detach(), "anchor": anchor.detach(),
                  "penalty": penalty.detach(),
                  "weighted_penalty": (metric_weight * penalty).detach(),
                  "mid_error_rms": mid_error.detach().square().mean().sqrt(),
                  "end_error_rms": end_error.detach().square().mean().sqrt(),
                  "endpoint_error_rms": endpoint_error.detach().square().mean().sqrt()}


def metric_scales(jacobian_start, jacobian_end, metric, metric_mode="exact"):
    """Global training-only median of positive metric traces.

    Each scale needs at least 3 positive traces to be defined at all; below 8
    the support is thin and the scale is flagged low_support so a reader knows
    the corresponding comparison rests on few sensitive contexts. Aborting on
    thin-but-nondegenerate support would turn legitimate free-space-heavy data
    into a crash; the flag preserves the signal instead.
    """
    first = jacobian_start.square().sum((1, 2))
    last = jacobian_end.square().sum((1, 2))
    sketch = metric_mode != "exact"
    transported = pullback_trace(metric, sketch=sketch)

    def positive_median(values, name):
        positive = values[values > 1e-12]
        count = int(positive.numel())
        if count < 3:
            raise ValueError(
                f"too few nonzero sensitivity labels for the {name} scale "
                f"({count} positive traces); stop the metric experiment and "
                "inspect the cache")
        return float(positive.median().item()), count

    scales = {}
    support = {}
    scales["endpoint"], support["endpoint"] = positive_median(first, "endpoint")
    both = 0.5 * (first + last)
    scales["identity"], support["identity"] = positive_median(both, "identity")
    pulled = 0.5 * (transported + last)
    scales["pullback"], support["pullback"] = positive_median(pulled, "pullback")
    low_support = sorted(name for name, count in support.items() if count < 8)
    stats = dict(metric_mode=metric_mode,
                 zero_sensitivity_fraction=float(
                     (transported <= 1e-12).float().mean().item()),
                 n_positive_transported=int((transported > 1e-12).sum().item()),
                 support=support,
                 low_support=low_support,
                 n_rows=int(len(first)),
                 endpoint_median=scales["endpoint"],
                 identity_median=scales["identity"],
                 pullback_median=scales["pullback"])
    return scales, stats
