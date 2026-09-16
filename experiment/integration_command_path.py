"""Command-path integration test with a deterministic fake adapter.

Exercises the REAL stage functions end to end on CPU with the external
model/simulator boundary stubbed (fake hri_adapter in sys.modules +
FakeAdapter), without bypassing contract validation:

  teacher-history production (real collect_histories + cache write)
  -> real warm training + checkpoint load
  -> warm-history production + real pair-bank production
  -> real locked student training (both arms, one seed)
  -> base-report production (real closed-loop rollout)
  -> real readiness sweep on the fake adapter (principal mode)
  -> tune-only correction run + saved lr-selection artifact
  -> REAL principal lock creation
  -> REAL locked training initialization (init-role provenance)
  -> final correction run consuming the locked lr selection
  -> locked episode evaluation + manifest-driven analysis
  -> corruption negatives (bytes / marker / failure flag)
  -> whole-case-deletion negative

Run: python experiment/integration_command_path.py (needs torch, CPU).
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


# ---------------------------------------------------------------------------
# Fake external boundary (model + simulator). Contract validation stays real.
# ---------------------------------------------------------------------------
def install_fake_hri_adapter():
    import types
    mod = types.ModuleType("hri_adapter")

    def normalize_data(chunk, stats=None):
        return np.asarray(chunk, dtype=np.float64)

    def decode_actions(normalized, stats):
        a = np.asarray(normalized, dtype=np.float64)
        mn = np.asarray(stats["action"]["min"], dtype=np.float64)
        mx = np.asarray(stats["action"]["max"], dtype=np.float64)
        return (a + 1.0) / 2.0 * (mx - mn) + mn

    def resolve_device(name):
        import torch
        return torch.device("cpu")

    mod.normalize_data = normalize_data
    mod.decode_actions = decode_actions
    mod.resolve_device = resolve_device
    sys.modules["hri_adapter"] = mod
    return mod


class FakeField:
    """Tiny deterministic torch field: contraction, finite everywhere."""

    def __init__(self):
        import torch
        self._lin = torch.nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self._lin.weight.mul_(0.0)
            self._lin.weight.fill_diagonal_(0.05)

    def __call__(self, state, times, condition):
        flat = state.reshape(state.shape[0], -1)
        return (self._lin(flat)
                + 0.01 * condition[:, :32]).reshape_as(state)


class FakeAdapter:
    """Adapter surface used by collect / rollout / pairs / correction."""

    STATS = {"action": {"min": np.zeros(2), "max": np.full(2, 512.0)}}

    def __init__(self, device):
        self.field = FakeField()
        self.stats = copy.deepcopy(self.STATS)
        self.device = device

    class _Env:
        def __init__(self):
            self.steps = 0

        def close(self):
            self.steps = -1

    # -- env boundary --
    def new_env(self, image=True):
        return self._Env()

    def reset(self, env, scene):
        env.steps = 0
        obs = {"image": np.zeros((3, 8, 8)), "agent_pos": np.zeros(2)}
        return obs, {}

    def signature(self, env):
        return np.zeros(10)

    def encode_observation(self, image, agent_pos):
        import torch
        return torch.zeros(1, 514, dtype=torch.float32).to(self.device)

    def prepare_commands(self, chunk):
        phys = np.asarray(chunk, dtype=np.float64) * 0.0 + 256.0
        return {"prepared": [np.asarray(row) for row in phys]}

    def raw_step(self, env, action):
        env.steps += 1
        obs = {"image": np.zeros((3, 8, 8)), "agent_pos": np.zeros(2)}
        return obs, 1.0, True, False, True

    def teacher_action(self, latent, cond, steps, source="gaussian"):
        import torch
        from gad_reference import teacher_endpoint
        with torch.no_grad():
            return teacher_endpoint(
                self.field, latent, cond, int(steps),
                source=source).detach()


class FakeTeacherModule:
    """Correction-side teacher: frozen endpoint map + stats."""

    def __init__(self, adapter, steps, source):
        import torch

        class _Map(torch.nn.Module):
            def forward(self_inner, latent, condition):
                from gad_reference import teacher_endpoint
                return teacher_endpoint(adapter.field, latent, condition,
                                        int(steps), source=source)

        self._map = _Map().eval()
        for p in self._map.parameters():
            p.requires_grad_(False)
        self.adapter = adapter

    def __call__(self, latent, condition):
        return self._map(latent, condition)

    def parameters(self):
        return self._map.parameters()

    def eval(self):
        self._map.eval()
        return self


# ---------------------------------------------------------------------------
# Miniature config
# ---------------------------------------------------------------------------
def mini_config(tmpdir):
    with open(os.path.join(HERE, "config.json")) as handle:
        config = json.load(handle)
    teacher_file = os.path.join(tmpdir, "fake_teacher.bin")
    with open(teacher_file, "wb") as handle:
        handle.write(b"fake-teacher-checkpoint")
    config.update(
        repository=tmpdir, checkpoint=teacher_file,
        normalizer=os.path.join(tmpdir, "fake_norm.npz"),
        device="cpu", width=16, batch_size=4, warm_updates=20,
        updates=6, learning_rate=1e-3, weight_decay=0.0,
        anchors_per_history=1, directions_per_anchor=1,
        execute_steps=2, episode_steps=8,
        development_scene_start=1000, development_episodes=20,
        train_scene_start=2000, train_episodes=3,
        validation_scene_start=3000, validation_episodes=1,
        diagnostic_scene_start=4000, diagnostic_episodes=2,
        final_scene_start=5000, final_episodes=2,
        arms=["anchor", "augmented", "gad"], seeds=[1],
        correction=dict(gamma=0.01, trust_rms=0.25,
                        lrs=[0.03, 0.1], budgets=[0, 1, 10]))
    path = os.path.join(tmpdir, "config.json")
    with open(path, "w") as handle:
        json.dump(config, handle)
    C.validate_config(config, path)
    return config, path


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def reserve_complete(path, fn, *args, **kw):
    C.reserve_outputs([path])
    try:
        out = fn(*args, **kw)
    except Exception as error:
        C.mark_incomplete(path, f"{type(error).__name__}: {error}")
        raise
    C.complete_output(path)
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def run_command_path():
    import torch
    import collect
    import train
    import pairs
    import evaluate
    import readiness
    import correction
    import lock as lock_mod
    import analyze
    from gad_reference import make_student, stable_seed, teacher_endpoint

    install_fake_hri_adapter()
    device = torch.device("cpu")
    tmpdir = tempfile.mkdtemp(prefix="cmdpath_")
    config, config_path = mini_config(tmpdir)
    fake = FakeAdapter(device)

    # 1. Teacher-history production (real collect_histories + cache write).
    collect.collect_histories.collector_hash = C.sha256_file(
        config["checkpoint"])
    teacher_histories = []
    for split in ("train", "validation"):
        scenes = sorted(C.scene_range(config, split))

        def driver(cond, latent, _fake=fake):
            with torch.no_grad():
                return teacher_endpoint(
                    _fake.field, latent, cond,
                    int(config["teacher_steps"]),
                    source=config["canonical_source"])

        teacher_histories.extend(collect.collect_histories(
            fake, config, device, scenes, split, driver, seed=0))
    teacher_cache = os.path.join(tmpdir, "teacher_hist.json")
    C.reserve_outputs([teacher_cache])
    C.write_history_cache(teacher_cache, teacher_histories, config)
    C.complete_output(teacher_cache)
    n_train = sum(1 for h in teacher_histories if h["split"] == "train")
    assert n_train == 3, n_train
    print(f"1. teacher histories: {len(teacher_histories)} contexts "
          f"({n_train} train)")

    # 2. Real warm training + checkpoint load.
    warm_path = os.path.join(tmpdir, "warm.pt")
    C.reserve_outputs([warm_path, warm_path + ".meta.json"])
    wargs = Args(history=teacher_cache, output=warm_path, seed=0,
                 updates=20, config=config_path)
    train.cmd_warm(wargs, config, device)
    C.complete_output(warm_path)
    C.complete_output(warm_path + ".meta.json")
    warm_payload = torch.load(warm_path, map_location="cpu",
                              weights_only=True)
    assert warm_payload["history_sha256"] == C.sha256_file(teacher_cache)
    print("2. warm trained on stored teacher latents; loss "
          f"{warm_payload['first_loss']:.4f} -> "
          f"{warm_payload['final_loss']:.4f}")

    # 3. Warm-history production + real pair-bank production.
    student = make_student(514, 16, 2, 16)
    student.load_state_dict(warm_payload["student"])
    student.to(device).eval()
    for p in student.parameters():
        p.requires_grad_(False)
    collect.collect_histories.collector_hash = C.sha256_file(warm_path)
    warm_histories = []

    def sdriver(cond, latent, _student=student):
        with torch.no_grad():
            return _student(latent, cond.repeat(latent.shape[0], 1)
                            if cond.shape[0] != latent.shape[0]
                            else cond)

    for scene in sorted(C.scene_range(config, "train")):
        warm_histories.extend(collect.collect_histories(
            fake, config, device, [scene], "train", sdriver, seed=0))
    warm_cache = os.path.join(tmpdir, "warm_hist.json")
    C.reserve_outputs([warm_cache])
    C.write_history_cache(warm_cache, warm_histories, config)
    C.complete_output(warm_cache)
    pairs.build_adapter = lambda *a, **k: fake
    bank_path = os.path.join(tmpdir, "bank.json")
    C.reserve_outputs([bank_path, bank_path + ".meta.json"])
    pargs = Args(history=warm_cache, output=bank_path, config=config_path)
    result = pairs.cmd_pairs(pargs, config, device)
    C.write_meta(bank_path + ".meta.json",
                 dict(kind="pair_bank", meta=result["meta"],
                      lineage=result["lineage"]))
    C.complete_output(bank_path)
    C.complete_output(bank_path + ".meta.json")
    assert result["records"] == 3, result
    print(f"3. warm histories + pair bank: {result['records']} records, "
          "train-selected from mixed cache")

    # 4. Real locked-shape student training (both arms, locked seed 1).
    students = {}
    for arm in ("augmented", "gad"):
        out = os.path.join(tmpdir, f"base_{arm}.pt")
        C.reserve_outputs([out, out + ".meta.json"])
        targs = Args(bank=bank_path, initial=warm_path, output=out,
                     arm=arm, seed=1, updates=6,
                     beta=0.1 if arm == "gad" else None,
                     protocol=None, allow_unlocked=True,
                     config=config_path)
        train.cmd_train(targs, config, device)
        C.complete_output(out)
        C.complete_output(out + ".meta.json")
        students[arm] = out
    print("4. base students trained (augmented + gad, seed 1)")

    # 5. Base-report production via real closed-loop rollout.
    evaluate.build_adapter = lambda *a, **k: fake
    base_report = os.path.join(tmpdir, "base_dev.jsonl")
    # NOTE: cmd_evaluate reserves its outputs internally.
    eargs = Args(role="student", student=students["augmented"],
                 output=base_report, protocol=None, allow_unlocked=True,
                 development=True, episodes=20, eval_replicate=0,
                 history=None, intervene_every=None, config=config_path)
    # bind missing attrs expected by cmd_evaluate
    eargs.teacher = "teacher"
    evaluate.cmd_evaluate(eargs, config, device)
    rows = [json.loads(l) for l in open(base_report) if l.strip()]
    assert len(rows) == 20 and all(r["split"] == "development"
                                   for r in rows), len(rows)
    assert all(r["method"] == "augmented" for r in rows)
    assert all(r["checkpoint_sha256"] == C.sha256_file(
        students["augmented"]) for r in rows)
    print(f"5. base report: {len(rows)} dev episodes, "
          f"success {sum(r['success'] for r in rows)}/{len(rows)}")

    # 6. Real readiness sweep (must reach principal on the fake stack).
    evaluate.build_adapter = lambda *a, **k: fake
    read_path = os.path.join(tmpdir, "readiness.json")
    sys.argv = ["readiness.py", "--config", config_path,
                "--output", read_path, "--scenes", "20"]
    assert readiness.main() == 0
    report = json.load(open(read_path))
    assert report["passed"] and report["acceptance_mode"] == "principal", \
        report["gate"]
    print("6. readiness: principal mode, "
          f"success {report['gate']['shared_gate']['success']:.2f}")

    # 7. Tune-only correction run + saved lr-selection artifact.
    correction.build_teacher = lambda cfg, dev: FakeTeacherModule(
        fake, cfg["teacher_steps"], cfg["canonical_source"])
    rng = np.random.default_rng(0)
    tune_scene = sorted(C.scene_range(config, "diagnostic"))[0]
    students_file = os.path.join(tmpdir, "students.json")
    with open(students_file, "w") as handle:
        json.dump(students, handle)
    tune_cases = os.path.join(tmpdir, "tune.jsonl")
    with open(tune_cases, "w") as handle:
        handle.write(json.dumps(dict(
            case_id="tune0", context_id="tctx0", target_id="ttgt0",
            target_origin="teacher_reachable", split="tune",
            scene=tune_scene, seed=1, model="augmented",
            checkpoint=students["augmented"],
            q0=np.zeros((16, 2)).tolist(),
            target=(np.zeros((4, 2)) + 0.1).tolist(),
            condition=np.zeros(514).tolist())) + "\n")
    tune_out = os.path.join(tmpdir, "tune_corr.jsonl")
    sel_path = os.path.join(tmpdir, "lr_selection.json")
    sys.argv = ["correction.py", "--config", config_path,
                "--cases", tune_cases, "--students", students_file,
                "--output", tune_out,
                "--save-lr-selection", sel_path]
    assert correction.main() == 0
    sel = json.load(open(sel_path))
    assert sel["kind"] == "lr_selection" and sel["lr"] in (0.03, 0.1)
    print(f"7. tune-only run: lr selection {sel['lr']} saved + completed")

    # 8. Eval-case manifest + REAL principal lock creation.
    final_scenes = sorted(C.scene_range(config, "final"))
    eval_cases = os.path.join(tmpdir, "eval_cases.jsonl")
    with open(eval_cases, "w") as handle:
        for case_id, scene in (("c0", final_scenes[0]),
                               ("c1", final_scenes[1])):
            for seed in (1,):
                for model in ("gad", "augmented"):
                    handle.write(json.dumps(dict(
                        case_id=case_id, context_id=f"ctx-{case_id}",
                        target_id=f"tgt-{case_id}",
                        target_origin="teacher_reachable", split="eval",
                        scene=scene, seed=seed, model=model,
                        checkpoint=students[model],
                        q0=(np.zeros((16, 2))
                            + 0.01 * seed).tolist(),
                        target=(np.zeros((4, 2)) + 0.05 * (scene % 3)
                                ).tolist(),
                        condition=(np.zeros(514)
                                   + 0.001 * scene).tolist())) + "\n")
    lock_path = os.path.join(tmpdir, "protocol_locked.json")
    sys.argv = ["lock.py", "--confirm-no-final-results",
                "--config", config_path, "--output", lock_path,
                "--readiness-report", read_path,
                "--base-report", base_report,
                "--base-checkpoint", students["augmented"],
                "--warm", warm_path, "--bank", bank_path,
                "--history", warm_cache,
                "--teacher-history", teacher_cache,
                "--eval-cases", eval_cases,
                "--lr-selection", sel_path,
                "--beta-selected", "0.1", "--seeds", "1",
                "--final-prefix", os.path.join(tmpdir, "final_")]
    assert lock_mod.main() == 0
    protocol = C.load_protocol(lock_path)
    design = protocol["design"]
    assert design["warm_seed"] == 0
    assert len(design["eval_cases"]) == 2
    print(f"8. principal lock {protocol['protocol_id'][:12]}: base "
          "checkpoint + teacher/warm histories + eval manifest + "
          "lr selection all bound")

    # 8b. Lock negatives: wrong base checkpoint must refuse.
    bad_lock = os.path.join(tmpdir, "protocol_bad.json")
    argv = ["lock.py", "--confirm-no-final-results",
            "--config", config_path, "--output", bad_lock,
            "--readiness-report", read_path,
            "--base-report", base_report,
            "--base-checkpoint", students["gad"],
            "--warm", warm_path, "--bank", bank_path,
            "--history", warm_cache, "--teacher-history", teacher_cache,
            "--eval-cases", eval_cases, "--lr-selection", sel_path,
            "--beta-selected", "0.1",
            "--final-prefix", os.path.join(tmpdir, "final_")]
    sys.argv = argv
    try:
        lock_mod.main()
    except (ValueError, C.ProtocolError):
        print("8b. lock refuses a teacher/other-student base checkpoint")
    else:
        raise AssertionError("lock accepted wrong base checkpoint")

    # 9. REAL locked training initialization (init-role provenance).
    locked = {}
    for arm in ("gad", "augmented"):
        out = os.path.join(tmpdir, f"final_{arm}.pt")
        C.reserve_outputs([out, out + ".meta.json"])
        targs = Args(bank=bank_path, initial=warm_path, output=out,
                     arm=arm, seed=1, updates=6, beta=None,
                     protocol=lock_path, allow_unlocked=False,
                     config=config_path)
        train.cmd_train(targs, config, device)
        C.complete_output(out)
        C.complete_output(out + ".meta.json")
        locked[arm] = out
    print("9. locked training init: warm provenance (teacher history "
          "+ warm_seed) verified through the real locked path")

    # 10. Final correction consuming the locked selection (no retuning).
    final_students = os.path.join(tmpdir, "students_final.json")
    with open(final_students, "w") as handle:
        json.dump(locked, handle)
    # Repoint eval cases at the locked checkpoints (the manifest
    # fingerprint covers case DATA only, so this changes no identity).
    pointed = []
    for line in open(eval_cases):
        if line.strip():
            row = json.loads(line)
            row["checkpoint"] = locked[row["model"]]
            pointed.append(row)
    with open(eval_cases, "w") as handle:
        for row in pointed:
            handle.write(json.dumps(row) + "\n")
    final_corr = os.path.join(tmpdir, "final_corr.jsonl")
    sys.argv = ["correction.py", "--config", config_path,
                "--protocol", lock_path,
                "--cases", eval_cases, "--students", final_students,
                "--lr-selection", sel_path,
                "--output", final_corr]
    assert correction.main() == 0
    crowns = [json.loads(l) for l in open(final_corr) if l.strip()]
    own = [r for r in crowns if r["method"] == "own_model"
           and r["K"] == 10]
    assert len(own) == 4, len(own)  # 2 cases x 2 arms (seed 1)
    assert all(r["case_data_sha256"] for r in own)
    transfer = [r for r in crowns if r["method"] == "teacher_transfer"
                and r["K"] == 10]
    assert all(r["forwards"] == 1 and r["teacher_edit_forwards"] >= 1
               for r in transfer), transfer[:1]
    print(f"10. final correction: {len(crowns)} rows, locked lr "
          "consumed, transfer/student/teacher costs separate")

    # 11. Locked episode evaluation + manifest-driven analysis.
    eval_files = []
    for arm in ("gad", "augmented"):
        out = os.path.join(tmpdir, f"eval_{arm}.jsonl")
        eargs = Args(role="student", student=locked[arm], output=out,
                     protocol=lock_path, allow_unlocked=False,
                     development=False, episodes=None, eval_replicate=0,
                     history=None, intervene_every=None,
                     config=config_path)
        eargs.teacher = "teacher"
        evaluate.cmd_evaluate(eargs, config, device)
        eval_files.append(out)
    sys.argv = ["analyze.py", "--config", config_path,
                "--inputs", final_corr] + eval_files + [
                    "--output", os.path.join(tmpdir, "analysis.json"),
                    "--protocol", lock_path,
                    "--method", "gad", "--baseline", "augmented"]
    assert analyze.main() == 0
    summary = json.load(open(os.path.join(tmpdir, "analysis.json")))
    assert "difference" in summary
    assert summary["discordant"]["per_seed"], "per-seed discordant missing"
    assert "pooled" in summary["discordant"]["note"]
    print(f"11. analysis: difference {summary['difference']:.4f}, "
          "per-seed discordant + scene-clustered intervals")

    # 12. Corruption negatives on a real artifact.
    import shutil
    victim = os.path.join(tmpdir, "victim.json")
    shutil.copy(sel_path, victim)
    shutil.copy(sel_path + ".complete.json", victim + ".complete.json")
    with open(victim, "a") as handle:
        handle.write(" ")
    try:
        C.verify_completed(victim)
    except C.ProtocolError as error:
        assert "ARTIFACT_HASH_MISMATCH" in str(error), str(error)
    else:
        raise AssertionError("edited bytes accepted")
    os.remove(victim + ".complete.json")
    try:
        C.verify_completed(victim)
    except C.ProtocolError as error:
        assert "MISSING_HASH" in str(error), str(error)
    else:
        raise AssertionError("missing marker accepted")
    shutil.copy(sel_path + ".complete.json", victim + ".complete.json")
    C.mark_incomplete(victim, "synthetic failure")
    try:
        C.verify_completed(victim)
    except C.ProtocolError as error:
        assert "FAILED" in str(error), str(error)
    else:
        raise AssertionError("failure marker accepted")
    print("12. corruption negatives: edited bytes / missing marker / "
          "failure flag all reject with the intended errors")

    # 13. Whole-case deletion on real correction rows.
    from analyze import paired_summary
    kw = dict(expected_seeds=design["seeds"],
              expected_cases=design["eval_cases"])
    pruned = [r for r in crowns
              if not (r.get("method") == "own_model" and r.get("K") == 10
                      and r.get("case_id") == "c1")]
    try:
        paired_summary(pruned, "gad", "augmented", key="mse", **kw)
    except ValueError as error:
        assert "MISSING_EPISODE" in str(error), str(error)
    else:
        raise AssertionError("deleted case accepted on real rows")
    print("13. whole-case deletion across both arms -> MISSING_EPISODE")

    print("command_path: ALL STEPS PASS")


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_command_path()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
