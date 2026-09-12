import argparse
import sys

import numpy as np


def check_finite_difference():
    matrix = np.array([[2.0, 0.0], [0.0, 0.5]])
    offset = np.array([1.0, 1.0])

    def execute(action):
        return matrix @ np.asarray(action) + offset

    action = np.array([0.3, -0.2])
    eps = 1e-4
    cols = []
    for coordinate in range(2):
        step = np.zeros(2)
        step[coordinate] = eps
        cols.append((execute(action + step) - execute(action - step)) / (2 * eps))
    estimated = np.stack(cols, axis=-1)
    gap = float(np.abs(estimated - matrix).max())
    if gap > 1e-6:
        raise ValueError("finite difference check failed")
    return gap


def check_trace_scale():
    rng = np.random.default_rng(0)
    probes = rng.standard_normal((8, 4, 32))
    transported = (probes ** 2).sum(-1).mean(-1)
    positive = transported[transported > 1e-12]
    if positive.size < 8:
        raise ValueError("trace check failed")
    return float(np.median(positive))


def check_torch():
    import torch
    sys.path.insert(0, ".")
    from experiment.core import TwoStepStudent, integrate
    torch.manual_seed(0)
    student = TwoStepStudent(condition_dim=8, horizon=4, action_dim=2, width=32)
    noise = torch.randn(2, 4, 2)
    cond = torch.randn(2, 8)
    mid = student(noise, cond, 0.0)
    end = student(mid, cond, 0.5)
    loss = mid.square().mean() + end.square().mean()
    loss.backward()
    grads = [param.grad for param in student.parameters() if param.requires_grad]
    if any(g is None or not np.isfinite(g.detach().numpy()).all() for g in grads):
        raise ValueError("student backward check failed")

    def field(latent, times, condition):
        return times.reshape(-1, 1, 1) * 0.0 + latent * 0.0

    out = integrate(field, noise, cond, 0.0, 1.0, 4)
    if not np.isfinite(out.detach().numpy()).all():
        raise ValueError("integrate check failed")

    target = torch.randn(2, 4, 2)
    variable = target.detach().clone().requires_grad_(True)
    completed = variable * 2.0
    cotangent = torch.ones_like(completed)
    gradient, = torch.autograd.grad(completed, variable, grad_outputs=cotangent)
    if float((gradient - 2.0).abs().max()) > 1e-5:
        raise ValueError("vjp check failed")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    gaps = {"finite_difference": check_finite_difference(), "trace": check_trace_scale()}
    try:
        gaps["torch"] = check_torch()
    except ImportError:
        if args.require_torch:
            raise
        gaps["torch"] = "skipped"
    print("sanity " + str(gaps))


if __name__ == "__main__":
    main()
