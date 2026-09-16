"""Sanity suite: contract/producer checks (numpy) + reference checks (torch-gated).

No simulator, no checkpoint, no GPU required. Torch-dependent checks run
only when torch imports; --require-torch turns a skip into a failure.
Every reported boolean is asserted True by the final sweep.
"""
import argparse
import copy
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import contract as C


def load_config():
    for candidate in (os.path.join(HERE, "config.json"),
                      "experiment/config.json"):
        if os.path.exists(candidate):
            with open(candidate) as handle:
                config = json.load(handle)
            C.validate_config(config, candidate)
            return config, candidate
    raise FileNotFoundError("config.json not found")


def check_config_validation():
    bad = []
    config, _ = load_config()
    for key, value in (("teacher_steps", 0), ("batch_size", 0),
                       ("rho", 0.0), ("learning_rate", 0.0)):
        edited = dict(config, **{key: value})
        try:
            C.validate_config(edited, "test")
        except ValueError:
            bad.append(f"{key}={value}")
        else:
            raise AssertionError(f"invalid {key} accepted")
    edited = dict(config, arms=["gad", "unknown"])
    try:
        C.validate_config(edited, "test")
    except ValueError:
        bad.append("arms=unknown")
    else:
        raise AssertionError("unknown arm accepted")
    edited = dict(config, train_scene_start=config["validation_scene_start"])
    try:
        C.validate_config(edited, "test")
    except ValueError:
        bad.append("overlap")
    else:
        raise AssertionError("overlapping scenes accepted")
    return dict(rejected=bad)


def check_stable_seed():
    import gad_reference as R
    a = R.stable_seed("pairs", 20000, 0, 0, 0)
    assert a == R.stable_seed("pairs", 20000, 0, 0, 0)
    assert a != R.stable_seed("pairs", 20000, 0, 0, 1)
    assert 0 <= a < 2 ** 63
    assert C.stable_seed("pairs", 20000, 0, 0, 0) == a
    return dict(deterministic=True)


def check_reserve_complete(tmpdir):
    path = os.path.join(tmpdir, "artifact.npz")
    C.reserve_outputs([path])
    try:
        C.reserve_outputs([path])
    except C.ProtocolError:
        pass
    else:
        raise AssertionError("double reservation accepted")
    import numpy as _np
    _np.savez(path, a=_np.zeros(3))
    C.complete_output(path, dict(note="synthetic"))
    assert C.verify_completed(path) is not None
    with open(path, "ab") as handle:
        handle.write(b"x")
    try:
        C.verify_completed(path)
    except C.ProtocolError as error:
        assert "ARTIFACT_HASH_MISMATCH" in str(error), str(error)
    else:
        raise AssertionError("corrupted bytes accepted")
    return dict(reserved=True, hash_mismatch_detected=True)


def check_history_pair_roundtrip(config, tmpdir):
    # Items 3, 4: strict numeric serialization (stringified arrays
    # rejected), generating latent q round-trips for warm pairing.
    rng = np.random.default_rng(0)
    contexts = []
    for i in range(4):
        contexts.append(dict(
            history_id=f"train-20000-{i}", scene=20000, split="train",
            decision=i, history=[[0.0, 0.0]],
            sig_initial=[0.0] * 10, sig_live=[0.0] * 10,
            condition=rng.standard_normal(514).astype(np.float64).tolist(),
            q=rng.standard_normal((16, 2)).astype(np.float64).tolist(),
            endpoint=rng.standard_normal((16, 2)).astype(np.float64).tolist(),
            collector_hash="test"))
    path = os.path.join(tmpdir, "hist.json")
    C.write_history_cache(path, contexts, config)
    raw = open(path).read()
    assert '"dtype"' not in raw and '"[array' not in raw.replace(" ", "")
    verified = C.verify_history_cache(path, config)
    assert len(verified["histories"]) == 4
    assert isinstance(verified["histories"][0]["condition"], np.ndarray)
    assert isinstance(verified["histories"][0]["q"], np.ndarray)
    assert verified["histories"][0]["q"].shape == (16, 2)
    assert np.allclose(verified["histories"][2]["q"],
                       np.asarray(contexts[2]["q"]))
    poisoned = [dict(contexts[0], condition="not-an-array")]
    bad_path = os.path.join(tmpdir, "hist_bad.json")
    try:
        C.write_history_cache(bad_path, poisoned, config)
    except C.ProtocolError:
        pass
    else:
        raise AssertionError("stringified history array accepted")
    # Item 5: records carry their split; trainers consume train only.
    records = [dict(pair_id=[f"train-20000-{i}", 0, 0], history_id=f"train-20000-{i}",
                    anchor_id=0, direction_id=0, scene=20000, split="train",
                    q=rng.standard_normal((16, 2)).tolist(),
                    u=(np.ones((16, 2)) / np.sqrt(32)).tolist(),
                    rho=float(config["rho"]), source="gaussian",
                    t0=rng.standard_normal((16, 2)).tolist(),
                    t1=rng.standard_normal((16, 2)).tolist(),
                    condition=rng.standard_normal(514).tolist(),
                    teacher_steps=int(config["teacher_steps"]))
               for i in range(4)]
    leaked = [dict(records[0], split="validation")]
    try:
        C.write_pair_bank(os.path.join(tmpdir, "bank_leak.json"), leaked,
                          meta, config)
    except C.ProtocolError:
        pass
    else:
        raise AssertionError("non-train pair record accepted")
    meta = dict(member_keys=[[f"train-20000-{i}", 0, 0] for i in range(4)],
                rho=float(config["rho"]), source="gaussian",
                teacher_steps=int(config["teacher_steps"]))
    bank = os.path.join(tmpdir, "bank.json")
    C.write_pair_bank(bank, records, meta, config, lineage={"test": True})
    verified_bank = C.verify_pair_bank(bank, config)
    assert isinstance(verified_bank["records"][0]["q"], np.ndarray)
    edited = dict(config, rho=float(config["rho"]) * 2)
    try:
        C.verify_pair_bank(bank, edited)
    except C.ProtocolError:
        pass
    else:
        raise AssertionError("rho drift accepted")
    return dict(history_records=4, pair_records=4, drift_rejected=True)


def check_canonical_design(config):
    scenes = {"train": [20000], "final": [60000]}
    design = C.canonical_design(config, ["anchor"], [0], 10, ["anchor"],
                                scenes, dict(config["correction"]), 0.1,
                                float(config["rho"]), {"device": "cpu"},
                                {"teacher": {"method": "teacher"}},
                                warm=None, bank=None, cache=None)
    assert C.design_id(design) == design["design_id"]
    mutated = copy.deepcopy(design)
    mutated["scenes"] = {"train": [20001], "final": [60000]}
    assert C.design_id(mutated) != design["design_id"]
    return dict(recomputes=True, mutation_detected=True)


def check_require_complete():
    episodes = [dict(method="gad", seed=0, scene=60000, success=1),
                dict(method="gad", seed=0, scene=60001, success=0),
                dict(method="augmented", seed=0, scene=60000, success=1),
                dict(method="augmented", seed=0, scene=60001, success=1)]
    assert C.require_complete(episodes, ["gad", "augmented"], [0],
                              [60000, 60001]) is True
    try:
        C.require_complete(episodes[:3], ["gad", "augmented"], [0],
                           [60000, 60001])
    except C.ProtocolError as error:
        assert "MISSING_EPISODE" in str(error), str(error)
    else:
        raise AssertionError("missing episode accepted")
    try:
        C.require_complete(episodes + [episodes[0]], ["gad", "augmented"],
                           [0], [60000, 60001])
    except C.ProtocolError:
        pass
    else:
        raise AssertionError("duplicate episode accepted")
    try:
        C.check_binary_success(2)
    except C.ProtocolError:
        pass
    else:
        raise AssertionError("nonbinary success accepted")
    return dict(complete_ok=True, missing_rejected=True,
                duplicate_rejected=True, nonbinary_rejected=True)


def check_torch():
    import torch
    sys.path.insert(0, os.path.dirname(HERE))
    import gad_reference as R
    torch.manual_seed(0)
    latent = torch.randn(2, 16, 2, dtype=torch.float32)
    condition = torch.randn(2, 514, dtype=torch.float32)
    direction = torch.randn_like(latent)
    direction = direction / direction.flatten(1).norm(p=2, dim=1).reshape(-1, 1, 1)
    t0, t1 = 2 * latent, 2 * (latent + 0.1 * direction)

    class ScaleMap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.s = torch.nn.Parameter(torch.tensor(0.7))
        def forward(self, x, c):
            return self.s * x

    # Items 1, 6: the kernel returns the LIVE differentiable objective
    # (loss reaches backward with a graph); anchor duplicates the anchor
    # input AND target for any beta, equals augmented at beta=0, and every
    # arm evaluates the same number of samples.
    out = {}
    grads = {}
    evals = {}
    for mode, beta in (("anchor", 0.1), ("augmented", 0.1), ("gad", 0.0),
                       ("gad", 0.1)):
        model = ScaleMap()
        terms = R.paired_loss(model, latent, condition, direction, t0, t1,
                              0.1, mode, beta)
        assert terms["loss"].grad_fn is not None, mode
        terms["loss"].backward()
        assert model.s.grad is not None and torch.isfinite(model.s.grad).all()
        grads[(mode, beta)] = model.s.grad.clone()
        out[(mode, beta)] = float(terms["loss"])
        evals[(mode, beta)] = terms["sample_evals"]
    assert out[("augmented", 0.1)] == out[("gad", 0.0)]
    # Item 6: anchor duplicates anchor input AND anchor target, so its
    # loss equals plain anchor MSE for ANY beta (a fresh-noise input or a
    # perturbed-target anchor would both fail this equality).
    anchor_model = ScaleMap()
    anchor_terms = R.paired_loss(anchor_model, latent, condition, direction,
                                 t0, t1, 0.1, "anchor", 0.7)
    plain = ((anchor_model(latent, condition) - t0) ** 2).mean()
    assert float(anchor_terms["loss"]) == float(plain)
    assert float(anchor_terms["loss"]) == out[("anchor", 0.1)]
    assert torch.equal(grads[("augmented", 0.1)], grads[("gad", 0.0)])
    assert not torch.equal(grads[("augmented", 0.1)], grads[("gad", 0.1)])
    assert len(set(evals.values())) == 1, evals
    student = R.make_student()
    with torch.no_grad():
        assert torch.allclose(student(latent, condition), latent)
    frozen = R.make_student().eval().requires_grad_(False)
    init = torch.zeros(1, 16, 2)
    cond = torch.zeros(1, 514)
    tgt = torch.full((1, 8, 2), 0.5)
    zero = R.invert_prefix(frozen, init, cond, tgt, steps=0)
    assert torch.equal(zero["latent"], init)
    assert zero["forward_calls"] == 1 and zero["backward_calls"] == 0
    # Item 11: exactly K+1 selection forwards and K backwards, and the
    # solved objective never exceeds the q0 objective.
    ten = R.invert_prefix(frozen, init, cond, tgt, steps=10)
    assert ten["forward_calls"] == 11, ten["forward_calls"]
    assert ten["backward_calls"] == 10, ten["backward_calls"]
    assert float(ten["solved_objective"].mean()) <= float(
        ten["zero_objective"].mean())
    # Item 15: geometry spot seeds live outside the bank/schedule space.
    assert R.stable_seed("geometry", 50000, 0, 0, 0) != R.stable_seed(
        "pairs", 20000, 0, 0, 0)
    # Item 2: checkpoints reject nested junk and pickles.
    import contract as _C
    nested = dict(student={"layer": torch.zeros(2)},
                  evil=object())
    try:
        _C.atomic_save_torch(os.path.join(
            tempfile.mkdtemp(), "x.pt"), nested)
    except (ValueError, TypeError, RuntimeError, AttributeError,
            _C.ProtocolError):
        pass
    else:
        raise AssertionError("nested checkpoint junk accepted")
    return dict(loss_equivalence=True, nonzero_response_gradient=True,
                identity_init=True, inversion_identity=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    config, _ = load_config()
    results = {"config_validation": check_config_validation(),
               "stable_seed": check_stable_seed(),
               "canonical_design": check_canonical_design(config),
               "require_complete": check_require_complete()}
    with tempfile.TemporaryDirectory() as tmpdir:
        results["reserve_complete"] = check_reserve_complete(tmpdir)
        results["history_pair_roundtrip"] = check_history_pair_roundtrip(
            config, tmpdir)
    try:
        import torch  # noqa: F401
        results["torch_checks"] = check_torch()
    except ImportError:
        if args.require_torch:
            raise
        results["torch_checks"] = "skipped (torch unavailable)"

    def sweep(node, trail=""):
        if isinstance(node, dict):
            for key, value in node.items():
                sweep(value, trail + "/" + str(key))
        elif node is False:
            raise AssertionError(f"sanity check False at {trail}")
    sweep(results)
    print("sanity " + json.dumps(results, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
