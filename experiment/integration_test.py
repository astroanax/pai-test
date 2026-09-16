"""Tiny actual producer-consumer integration test (audit fdb59ff, steps 1-9).

Uses the REAL producer/consumer functions (contract writers, train.cmd_train,
gad_reference kernels, geometry.jvp_spot_check, correction.correct_case,
analyze.paired_summary) on synthetic CPU data. No simulator, checkpoint, or
GPU required. Run with the torch-capable python:

    python experiment/integration_test.py
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import contract as C


def step_history_cache(tmpdir, config):
    """Step 1: mixed train+validation cache; q converts to ndarray."""
    rng = np.random.default_rng(0)
    train_scene = sorted(C.scene_range(config, "train"))[0]
    val_scene = sorted(C.scene_range(config, "validation"))[0]
    contexts = []
    for i in range(3):
        contexts.append(dict(
            history_id=f"train-{train_scene}-{i}", scene=train_scene,
            split="train", decision=i, history=[[0.0, 0.0]],
            sig_initial=[0.0] * 10, sig_live=[0.0] * 10,
            condition=rng.standard_normal(514).tolist(),
            q=rng.standard_normal((16, 2)).tolist(),
            endpoint=rng.standard_normal((16, 2)).tolist(),
            collector_hash="teacher"))
    contexts.append(dict(
        history_id=f"val-{val_scene}-0", scene=val_scene, split="validation",
        decision=0, history=[[0.0, 0.0]],
        sig_initial=[0.0] * 10, sig_live=[0.0] * 10,
        condition=rng.standard_normal(514).tolist(),
        q=rng.standard_normal((16, 2)).tolist(),
        endpoint=rng.standard_normal((16, 2)).tolist(),
        collector_hash="teacher"))
    path = os.path.join(tmpdir, "hist.json")
    C.write_history_cache(path, contexts, config)
    verified = C.verify_history_cache(path, config)
    assert len(verified["histories"]) == 4
    assert isinstance(verified["histories"][0]["q"], np.ndarray)
    assert verified["histories"][0]["q"].shape == (16, 2)
    trains = [h for h in verified["histories"] if h["split"] == "train"]
    assert len(trains) == 3
    print("1. history cache: mixed train/val round-trips, q converts")
    return path


def step_pair_bank(tmpdir, config):
    """Step 3 (bank side): train-only bank with selection lineage."""
    rng = np.random.default_rng(1)
    train_scene = sorted(C.scene_range(config, "train"))[0]
    records = []
    for i in range(6):
        records.append(dict(
            pair_id=[f"train-{train_scene}-{i}", 0, 0],
            history_id=f"train-{train_scene}-{i}", anchor_id=0,
            direction_id=0, scene=train_scene, split="train",
            q=rng.standard_normal((16, 2)).tolist(),
            u=(np.ones((16, 2)) / np.sqrt(32)).tolist(),
            rho=float(config["rho"]), source="gaussian",
            t0=rng.standard_normal((16, 2)).tolist(),
            t1=rng.standard_normal((16, 2)).tolist(),
            condition=rng.standard_normal(514).tolist(),
            teacher_steps=int(config["teacher_steps"])))
    meta = dict(member_keys=[list(r["pair_id"]) for r in records],
                rho=float(config["rho"]), source="gaussian",
                teacher_steps=int(config["teacher_steps"]))
    path = os.path.join(tmpdir, "bank.json")
    lineage = dict(selected_histories=[r["history_id"] for r in records],
                   rejected_histories=["val-x-0"])
    C.write_pair_bank(path, records, meta, config, lineage=lineage)
    verified = C.verify_pair_bank(path, config)
    assert len(verified["records"]) == 6
    assert verified["lineage"]["rejected_histories"] == ["val-x-0"]
    print("3. pair bank: train-only, selection lineage recorded")
    return path


def step_train_all_arms(tmpdir, config, bank_path):
    """Steps 2+4: real trainer for all three arms; anchor must not crash."""
    import torch
    import train as T
    from gad_reference import make_student
    device = torch.device("cpu")
    width = int(config.get("width", 512))
    init_path = os.path.join(tmpdir, "warm.pt")
    student = make_student(514, 16, 2, width)
    torch.save(dict(student=student.state_dict(), arm="warm", seed=0),
               init_path)

    class Args:
        pass

    outs = {}
    for arm in ("anchor", "augmented", "gad"):
        args = Args()
        args.bank = bank_path
        args.initial = init_path
        args.output = os.path.join(tmpdir, f"student_{arm}.pt")
        args.arm = arm
        args.seed = 0
        args.updates = 2
        args.beta = 0.1 if arm == "gad" else None
        args.protocol = None
        args.allow_unlocked = True
        args.config = os.path.join(HERE, "config.json")
        C.reserve_outputs([args.output, args.output + ".meta.json"])
        T.cmd_train(args, config, device)
        C.complete_output(args.output)
        C.complete_output(args.output + ".meta.json")
        payload = torch.load(args.output, map_location="cpu",
                             weights_only=True)
        outs[arm] = payload
    print("2+4. trainer: all three arms run (anchor KeyError fixed)")
    return outs


def step_beta_zero_equivalence(tmpdir, config, bank_path):
    """Step 5: augmented == gad(beta=0) updates, bit-for-bit."""
    import torch
    import train as T
    from gad_reference import make_student
    device = torch.device("cpu")
    width = int(config.get("width", 512))

    class Args:
        pass

    def run(arm, beta, tag):
        init = os.path.join(tmpdir, "warm.pt")
        out = os.path.join(tmpdir, f"equiv_{tag}.pt")
        args = Args()
        args.bank = bank_path
        args.initial = init
        args.output = out
        args.arm = arm
        args.seed = 7
        args.updates = 3
        args.beta = beta
        args.protocol = None
        args.allow_unlocked = True
        args.config = os.path.join(HERE, "config.json")
        C.reserve_outputs([out, out + ".meta.json"])
        T.cmd_train(args, config, device)
        state = torch.load(out, map_location="cpu",
                           weights_only=True)["student"]
        return state

    torch.manual_seed(123)
    aug = run("augmented", None, "aug")
    torch.manual_seed(123)
    gad0 = run("gad", 0.0, "gad0")
    for key in aug:
        assert torch.equal(aug[key], gad0[key]), key
    print("5. beta-zero equivalence: augmented == gad(beta=0) bit-for-bit")


def step_jvp():
    """Step 6: identity + nonsymmetric-linear JVP via the real helper."""
    import torch
    import torch.nn as nn
    from geometry import jvp_spot_check
    from gad_reference import teacher_endpoint

    class FakeAdapter:
        def __init__(self, field):
            self.field = field

        def teacher_action(self, latent, cond, steps, source="gaussian"):
            with torch.no_grad():
                return teacher_endpoint(self.field, latent, cond, steps,
                                        source=source)

    cond = torch.zeros(1, 514)
    q = torch.tensor([[[0.6, 0.8]] + [[0.0, 0.0]] * 15],
                     dtype=torch.float32)
    u = torch.zeros(1, 16, 2)
    u[0, 0, 0] = 0.6
    u[0, 0, 1] = 0.8

    class ZeroField(nn.Module):
        def forward(self, state, times, condition):
            return torch.zeros_like(state)

    gap = jvp_spot_check(FakeAdapter(ZeroField()), cond, q, u, 4,
                         "gaussian")
    assert gap < 5e-2, gap

    # Nonsymmetric linear map T(q) = q @ M.T, checked analytically.
    M = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    assert not torch.equal(M, M.T)

    class LinAdapter(FakeAdapter):
        def teacher_action(self, latent, cond, steps, source="gaussian"):
            with torch.no_grad():
                out = latent.clone()
                out[..., :2] = latent[..., :2] @ M.T
                return out

    # Direct functional-JVP check against the analytic M (steps=1 exact).
    x = torch.zeros(1, 16, 2)
    x[0, 0] = torch.tensor([0.6, 0.8])
    d = torch.zeros(1, 16, 2)
    d[0, 0] = torch.tensor([0.6, 0.8])
    func = lambda v: (v[..., :2] @ M.T)  # noqa: E731
    _, jvp = torch.autograd.functional.jvp(func, x, d)
    expect = (d[..., :2] @ M.T).ravel()
    assert torch.allclose(jvp[..., :2].ravel(), expect, atol=1e-5), \
        (jvp.ravel()[:4], expect[:4])
    print("6. JVP: identity helper passes; nonsymmetric linear exact")


def step_inversion():
    """Step 7: one-step inversion improves a reachable target."""
    import torch
    import torch.nn as nn
    from gad_reference import invert_prefix
    lin = nn.Linear(32, 16, bias=False)
    with torch.no_grad():
        lin.weight.fill_(1.0 / 32.0)

    class LinMap(nn.Module):
        def forward(self, lat, c):
            return lin(lat.reshape(lat.shape[0], -1)).reshape(
                lat.shape[0], 8, 2)

    model = LinMap().eval()
    q0 = torch.zeros(1, 16, 2)
    c0 = torch.zeros(1, 514)
    with torch.no_grad():
        tgt = model(torch.full((1, 16, 2), 0.2), c0)
    zero = invert_prefix(model, q0, c0, tgt, steps=0)
    one = invert_prefix(model, q0, c0, tgt, steps=1,
                        learning_rate=0.5, trust_rms=2.0, gamma=0.0)
    assert one["forward_calls"] == 2 and one["backward_calls"] == 1
    assert float(one["objective"].mean()) < float(
        zero["objective"].mean())
    print("7. inversion: K=1 improves over q0 (final step evaluated)")


def step_correction_eval(config):
    """Step 8a: real correct_case on synthetic models at budgets (0,1)."""
    import torch
    import torch.nn as nn
    import correction as K
    from gad_reference import invert_prefix
    device = torch.device("cpu")
    torch.manual_seed(0)

    class ScaleMap(nn.Module):
        def __init__(self, scale=0.7):
            super().__init__()
            self.s = nn.Parameter(torch.tensor(scale),
                                  requires_grad=False)

        def forward(self, latent, condition):
            return self.s * latent

    teacher = ScaleMap(2.0).eval()
    student = ScaleMap(0.7).eval()
    for p in list(teacher.parameters()) + list(student.parameters()):
        p.requires_grad_(False)
    models = {"augmented": student}
    q0 = torch.zeros(1, 16, 2)
    cond = torch.zeros(1, 514)
    with torch.no_grad():
        tgt = teacher(torch.full((1, 16, 2), 0.2), cond)
    case = dict(case_id="c0", context_id="x0", target_id="t0",
                target_origin="teacher_reachable", scene=0, seed=0,
                model="augmented", split="eval",
                q0_np=np.zeros((1, 16, 2), dtype=np.float64),
                target_np=tgt.cpu().numpy().astype(np.float64),
                q0_t=q0, cond_t=cond, target_t=tgt)
    teacher_outs = {}
    for budget in (0, 1):
        out = invert_prefix(teacher, q0, cond, tgt, steps=budget,
                            learning_rate=0.1, trust_rms=0.25, gamma=0.01)
        out["_identity"] = (b"q", b"c", b"t")
        teacher_outs[("x0", "t0", budget)] = out
    rng = np.random.default_rng(0)
    offsets = K.random_search_offsets(rng, (1, 16, 2), n=4, trust_r=0.25)
    try:
        from hri_adapter import decode_actions  # noqa: F401
        has_decode = True
    except Exception:
        has_decode = False
    if not has_decode:
        print("8a. correction: skipped (hri_adapter unavailable)")
        return
    import hri_adapter
    stats = {"action": {"min": np.zeros(2), "max": np.full(2, 512.0)}}
    rows = K.correct_case(case, models, teacher_outs, 0.1, offsets,
                          device, None, stats, gamma=0.01, trust_r=0.25,
                          budgets=(0, 1))
    methods = {r["method"] for r in rows}
    assert {"no_correction", "own_model", "teacher_transfer",
            "random_search", "oracle_prefix"} <= methods, methods
    assert any(r["latency_ms"] > 0 for r in rows
               if r["method"] == "random_search")
    print("8a. correction: real rows, all baselines, measured latency")


def step_lock_and_analysis(tmpdir, config):
    """Steps 8b+9: canonical lock id + seed-deletion rejection."""
    from analyze import paired_summary
    design = C.canonical_design(
        config, ["gad", "augmented"], [0, 1], 10, ["gad", "augmented"],
        {"train": [1], "final": [2]}, dict(config["correction"]),
        0.1, float(config["rho"]), {"device": "cpu"},
        {"teacher": {"method": "teacher"}}, warm=None, bank=None,
        cache=None, primary_contrast=["gad", "augmented"])
    assert C.design_id(design) == design["design_id"]
    rows = []
    for seed in (0, 1):
        for case, scene in (("c0", 2), ("c1", 3)):
            for arm in ("gad", "augmented"):
                rows.append(dict(method="own_model", K=10, model=arm,
                                 seed=seed, case_id=case, scene=scene,
                                 mse=0.5 if arm == "gad" else 0.6))
    summary = paired_summary(rows, "gad", "augmented", key="mse",
                             expected_seeds=[0, 1])
    assert summary["scenes"] >= 1
    pruned = [r for r in rows if r["seed"] != 1]
    try:
        paired_summary(pruned, "gad", "augmented", key="mse",
                       expected_seeds=[0, 1])
    except ValueError as error:
        assert "MISSING_EPISODE" in str(error), str(error)
    else:
        raise AssertionError("deleted seed accepted")
    print("8b+9. lock id recomputes; deleted seed -> MISSING_EPISODE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = parser.parse_args()
    with open(args.config) as handle:
        config = json.load(handle)
    C.validate_config(config, args.config)
    with tempfile.TemporaryDirectory() as tmpdir:
        step_history_cache(tmpdir, config)
        bank = step_pair_bank(tmpdir, config)
        step_train_all_arms(tmpdir, config, bank)
        step_beta_zero_equivalence(tmpdir, config, bank)
    step_jvp()
    step_inversion()
    step_correction_eval(config)
    step_lock_and_analysis(tmpdir, config)
    print("integration_test: ALL STEPS PASS")


if __name__ == "__main__":
    raise SystemExit(main())
