import numpy as np
import torch
from torch import nn


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
    def __init__(self, condition_dim=514, horizon=16, action_dim=2, width=512):
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


def finite_difference_jacobian(execute, action, epsilon=0.01):
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
        return torch.stack(probes, dim=1)


def physical_error(jacobian, error, execute_steps):
    prefix = error[:, :execute_steps].flatten(1)
    return torch.bmm(jacobian, prefix.unsqueeze(-1)).squeeze(-1).square().sum(-1)


def objective(student, batch, mode, execute_steps, scales, metric_weight=0.25, anchor_weight=0.25, metric_count=None):
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
    if mode == "prefix":
        penalty = 0.5 * (
            mid_error[:, :execute_steps].square().mean()
            + end_error[:, :execute_steps].square().mean()
        )
    elif mode == "endpoint":
        penalty = physical_error(
            batch["jacobian_start"][selected], endpoint_error[selected], execute_steps
        ).mean() / scales["endpoint"]
    elif mode == "scalar":
        first_trace = batch["probes"][selected].square().sum(-1).mean(-1)
        last_trace = batch["jacobian_end"][selected].square().sum((1, 2))
        early = first_trace * mid_error[selected].square().mean((1, 2))
        late = last_trace * end_error[selected].square().mean((1, 2))
        penalty = 0.5 * (early.mean() + late.mean()) / scales["pullback"]
    elif mode in ("pullback", "identity"):
        if mode == "pullback":
            projected = torch.einsum("bld,bd->bl", batch["probes"][selected], mid_error[selected].flatten(1))
            early = projected.square().mean(1)
        else:
            early = physical_error(batch["jacobian_start"][selected], mid_error[selected], execute_steps)
        late = physical_error(batch["jacobian_end"][selected], end_error[selected], execute_steps)
        penalty = 0.5 * (early.mean() + late.mean()) / scales[mode]
    elif mode != "uniform":
        raise ValueError(mode)
    loss = base + anchor_weight * anchor + metric_weight * penalty
    return loss, {"base": base.detach(), "anchor": anchor.detach(), "penalty": penalty.detach()}


def metric_scales(cache, rows):
    first = cache["jacobian_start"][rows].square().sum((1, 2))
    last = cache["jacobian_end"][rows].square().sum((1, 2))
    transported = cache["probes"][rows].square().sum(-1).mean(-1)

    def positive_median(values):
        positive = values[values > 1e-12]
        if positive.numel() < 8:
            raise ValueError("too few nonzero sensitivity labels")
        return float(positive.median().item())

    return {"endpoint": positive_median(first),
            "identity": positive_median(0.5 * (first + last)),
            "pullback": positive_median(0.5 * (transported + last))}
