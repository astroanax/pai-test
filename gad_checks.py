"""Conformance checks for the reference kernels (numpy + torch)."""
import argparse
import ast
import math
import os
import sys


def numpy_checks():
    import numpy as np

    from gad_reference import paired_loss as _paired_loss  # noqa: F401 (import surface)
    from gad_reference import stable_seed

    rng = np.random.default_rng(194)

    # 1. pair decomposition (Eq.11): ||e0||^2+||e1||^2 = 2||mean||^2 + ||diff||^2/2.
    first = rng.normal(size=(12, 4))
    second = rng.normal(size=(12, 4))
    left = np.sum(first ** 2 + second ** 2, axis=1)
    right = 2 * np.sum(((first + second) / 2) ** 2, axis=1)
    right += 0.5 * np.sum((second - first) ** 2, axis=1)
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)

    # 2. signed-direction covariance: mean over sign patterns is eye(4)/4.
    directions = np.zeros((16, 4))
    for i in range(16):
        for j in range(4):
            directions[i, j] = 1.0 if ((i >> j) & 1) else -1.0
    directions = directions / 2.0
    cov = directions.T @ directions / 16.0
    assert np.allclose(cov, np.eye(4) / 4.0)

    # 3. linear finite differences are exact.
    f = lambda x: 3.0 * x + 1.0  # noqa: E731
    x0 = float(rng.normal())
    rho = 0.01
    assert abs(((f(x0 + rho) - f(x0 - rho)) / (2.0 * rho)) - 3.0) < 1e-9

    # 4. jacobian scaling: mean squared directional response equals sum(M^2)/4.
    matrix = rng.normal(size=(4, 4))
    responses = matrix @ directions.T
    observed = float(np.mean(np.sum(responses**2, axis=0)))
    assert abs(observed - float(np.sum(matrix**2)) / 4.0) < 1e-9

    # 5. L2/RMS convention: norm == radius, rms == radius/2 for D=4.
    radius = 0.4
    vec = np.array([radius / 2.0] * 4)
    assert abs(float(np.linalg.norm(vec)) - radius) < 1e-12
    assert abs(float(np.sqrt(np.mean(vec**2))) - radius / 2.0) < 1e-12

    # 6. key separation: deterministic, stage-sensitive, in [0, 2**63).
    a = stable_seed(" Gad ", 1, 2, 3, 1)
    b = stable_seed(" Gad ", 1, 2, 3, 1)
    c = stable_seed("other", 1, 2, 3, 1)
    assert a == b and a != c
    assert 0 <= a < 2**63

    # 7. NaN radius is rejected by the dependency-free radius
    # validator (audit fdb59ff item 6: numpy_checks must not import
    # torch; the torch-gated paired_loss NaN path is covered by
    # torch_checks below).
    from gad_reference import positive_finite as _positive_finite
    try:
        _positive_finite(float("nan"), "radius")
    except ValueError:
        pass
    else:
        raise AssertionError("NaN radius was not rejected")

    # 8. reference syntax parses.
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "gad_reference.py"), "r", encoding="utf-8") as fh:
        ast.parse(fh.read())

    return {"checks": 8, "status": "passed"}


class _FakeStudent:
    def __call__(self, latent, condition):
        return latent


def torch_checks():
    # NaN radius through the real kernel (torch-gated).
    from gad_reference import paired_loss as _torch_paired_loss
    import torch as _torch_check
    _fake = _FakeStudent()
    _lat = _torch_check.zeros(2, 1, 2, dtype=_torch_check.float32)
    _cond = _torch_check.zeros(2, 3, dtype=_torch_check.float32)
    _dir = _torch_check.zeros_like(_lat)
    _dir[:, 0, 0] = 1.0
    _tgt = _torch_check.zeros(2, 1, 2, dtype=_torch_check.float32)
    try:
        _torch_paired_loss(_fake, _lat, _cond, _dir, _tgt, _tgt,
                           float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("NaN radius was not rejected by paired_loss")
    import math as _math

    import torch
    import torch.nn as nn

    from gad_reference import (
        invert_prefix,
        make_student,
        paired_loss,
        source_from_canonical,
    )

    torch.manual_seed(0)

    class ScaleMap(nn.Module):
        def __init__(self, scale=0.7, offset=0.2):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(scale))
            self.offset = nn.Parameter(torch.tensor(offset))

        def forward(self, latent, condition):
            return self.scale * latent + self.offset

    b, horizon, action_dim, cond_dim = 4, 3, 2, 3
    latent = torch.randn(b, horizon, action_dim, dtype=torch.float32)
    condition = torch.randn(b, cond_dim, dtype=torch.float32)
    direction = torch.randn_like(latent)
    direction = direction / direction.flatten(1).norm(p=2, dim=1).reshape(-1, 1, 1)
    radius = 0.25
    # Teacher must MISMATCH the student scale: identical maps give zero
    # response error and a vacuous gradient test.
    teacher = ScaleMap(scale=2.0, offset=-0.3)
    for p in teacher.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        tgt_anchor = teacher(latent, condition)
        tgt_perturbed = teacher(latent + radius * direction, condition)

    def run(mode, beta):
        model = ScaleMap()
        for p in model.parameters():
            p.requires_grad_(True)
        out = paired_loss(
            model, latent, condition, direction,
            tgt_anchor, tgt_perturbed, radius, mode=mode, beta=beta,
        )
        assert out["pair_count"] == b
        out["loss"].backward()
        grads = [p.grad.detach().clone() for p in model.parameters()]
        model.zero_grad()
        return out, grads

    results = {}
    for mode, beta in [("anchor", 0.1), ("augmented", 0.1), ("gad", 0.0), ("gad", 0.1)]:
        out, grads = run(mode, beta)
        for g in grads:
            assert torch.isfinite(g).all().item()
        results[(mode, beta)] = (float(out["loss"].detach()), grads)
    # augmented(beta=.1) equals gad(beta=0) in loss and grads; gad(beta=.1) differs.
    assert results[("augmented", 0.1)][0] == results[("gad", 0.0)][0]
    for g1, g2 in zip(results[("augmented", 0.1)][1], results[("gad", 0.0)][1]):
        assert torch.equal(g1, g2)
    assert any(
        not torch.equal(g1, g2)
        for g1, g2 in zip(results[("augmented", 0.1)][1], results[("gad", 0.1)][1])
    )

    # identity student test: zero-init final layer maps latent to itself at init.
    student = make_student(condition_dim=5, horizon=2, action_dim=2, width=16)
    lat0 = torch.randn(3, 2, 2, dtype=torch.float32)
    cond0 = torch.randn(3, 5, dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(student(lat0, cond0), lat0)

    # both-steps gradient test.
    lin = nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight[:, :4] = 0.3 * torch.eye(4)
    s = torch.randn(6, 4, dtype=torch.float32, requires_grad=False)
    s0 = s.detach().requires_grad_(False)
    step1 = s0 + 0.5 * lin(torch.cat([s0, torch.zeros(6, 4)], dim=1))
    step2 = step1 + 0.5 * lin(torch.cat([step1, torch.zeros(6, 4)], dim=1))
    step2.sum().backward()
    expected = s0.detach().flatten(1).sum(0) * (1.0 + 0.5 * 0.3)
    for i in range(4):
        assert abs(float(lin.weight.grad[i, i]) - float(expected[i])) < 1e-4

    # inversion checks with a frozen ScaleMap in eval mode.
    inv_model = ScaleMap(scale=0.7, offset=0.2)
    for p in inv_model.parameters():
        p.requires_grad_(False)
    inv_model.eval()
    init = torch.randn(3, 2, 2, dtype=torch.float32)
    cond = torch.randn(3, 4, dtype=torch.float32)

    class _CondScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, lat, c):
            return 0.7 * lat + 0.2

    inv_model2 = _CondScale()
    inv_model2.eval()
    with torch.no_grad():
        target = inv_model2(init + 0.5, cond)
    before = torch.autograd.no_grad()
    with before:
        start_obj = torch.mean((inv_model2(init, cond) - target) ** 2).item()
    r0 = invert_prefix(inv_model2, init, cond, target, steps=0)
    assert torch.equal(r0["latent"], init.detach())
    assert r0["forward_calls"] == 1 and r0["backward_calls"] == 0
    assert r0["sample_forward_evaluations"] == init.shape[0] * 1
    r10 = invert_prefix(
        inv_model2, init, cond, target, steps=10,
        learning_rate=0.5, trust_rms=2.0, gamma=0.0,
    )
    assert float(r10["objective"].mean()) < start_obj
    trust_l2 = 2.0 * _math.sqrt(init[0].numel())
    deltas = (r10["latent"] - init).flatten(1).norm(p=2, dim=1)
    assert bool((deltas <= trust_l2 + 1e-5).all().item())
    assert r10["forward_calls"] == 11 and r10["backward_calls"] == 10
    assert r10["sample_forward_evaluations"] == init.shape[0] * 11
    # batch invariance: joint run matches per-sample runs.
    parts = []
    for i in range(init.shape[0]):
        parts.append(
            invert_prefix(
                inv_model2, init[i : i + 1], cond[i : i + 1], target[i : i + 1],
                steps=10, learning_rate=0.5, trust_rms=2.0, gamma=0.0,
            )["latent"]
        )
    assert torch.allclose(r10["latent"], torch.cat(parts, dim=0), atol=1e-5)
    # unfrozen weights are rejected before any gradient is created.
    bad = ScaleMap()
    bad.train()
    for p in bad.parameters():
        p.requires_grad_(True)
    try:
        invert_prefix(bad, init, cond, target, steps=2)
    except ValueError:
        pass
    else:
        raise AssertionError("unfrozen model was not rejected")
    assert all(p.grad is None for p in bad.parameters())

    # source transform: uniform(0) = 0.5 with slope 1/sqrt(2*pi).
    z = torch.zeros(5, dtype=torch.float32, requires_grad=True)
    u = source_from_canonical(z, "uniform")
    assert torch.allclose(u, torch.full_like(u, 0.5))
    u.sum().backward()
    expected_slope = 1.0 / _math.sqrt(2.0 * _math.pi)
    assert torch.allclose(z.grad, torch.full_like(z.grad, expected_slope), atol=1e-6)

    return {"status": "passed"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="GAD reference kernel checks")
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args(argv)
    scope = "reference kernels only; no robot assets or GPU validation"
    numpy_result = numpy_checks()
    try:
        import torch  # noqa: F401
    except ImportError:
        print(f"torch checks: SKIPPED ({scope})")
        print(f"numpy checks: {numpy_result}")
        return 2 if args.require_torch else 0
    torch_result = torch_checks()
    print(f"numpy checks: {numpy_result}")
    print(f"torch checks: {torch_result}")
    print(f"scope: {scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
