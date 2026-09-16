import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C
from core import (METRIC_MODES, finite_difference_jacobian, make_student,
                  metric_scales, mode_needs_metric, objective,
                  pullback_metric_exact, pullback_probes, sample_source,
                  sample_with_midpoint)
from hri_adapter import (HRIAdapter, load_checkpoint, resolve_device,
                         teacher_half_from, teacher_half_maps, teacher_rollout)


def load_config(path):
    with open(path) as handle:
        config = json.load(handle)
    C.validate_config(config, path)
    return config


def build_adapter(config, device):
    stats = {key: {inner: np.asarray(val) for inner, val in section.items()}
             for key, section in np.load(config["normalizer"], allow_pickle=True)["stats"].item().items()}
    sys.path.insert(0, os.path.join(config["repository"], "external", "models"))
    vision_encoder, noise_pred_net = load_checkpoint(config["checkpoint"], device)
    return HRIAdapter(vision_encoder, noise_pred_net, stats, device,
                      legacy=config["legacy"], clip_actions=config["clip_actions"])


def load_student(path, config, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "student" not in payload:
        raise ValueError("checkpoint lacks a student state dict")
    for key in ("mode", "seed", "updates"):
        if key not in payload:
            raise ValueError(f"checkpoint lacks intrinsic {key}")
    student = make_student(config, device, seed=None,
                           state_dict=payload["student"])
    for tensor in student.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite student checkpoint tensor")
    return student, payload


def protocol_for(args, config):
    """Load the lock and revalidate every fingerprint against the current tree.

    Final evaluation always requires the lock. Training requires it except for
    the warm start (no --initial), which predates the lock by construction.
    Collection, metrics, and diagnostics may run unlocked explicitly.
    """
    path = getattr(args, "protocol", None)
    command = getattr(args, "command", None)
    if command == "evaluate" and not path:
        # Development evaluation (teacher / warm-start checks before locking)
        # is explicitly labeled and can never enter final analysis; final
        # evaluation always requires the lock.
        if getattr(args, "development", False) and \
                getattr(args, "allow_unlocked", False):
            return None, None
        raise C.ProtocolError(
            "evaluate requires --protocol runs/protocol_locked.json; "
            "unlocked test evaluation is not permitted (development "
            "evaluation needs --development --allow-unlocked)")
    if command == "train" and not path:
        if getattr(args, "initial", None) is not None:
            raise C.ProtocolError(
                "fine-tuning (--initial ...) requires --protocol "
                "runs/protocol_locked.json")
        if getattr(args, "allow_unlocked", False):
            return None, None
        raise C.ProtocolError(
            "train: pass --protocol runs/protocol_locked.json, or "
            "--allow-unlocked for the warm start")
    if not path:
        if getattr(args, "allow_unlocked", False):
            return None, None
        raise C.ProtocolError(
            f"{command}: pass --protocol runs/protocol_locked.json, or "
            "--allow-unlocked for a development stage")
    protocol = C.load_protocol(path)
    C.verify_current_protocol(protocol, config, args.config,
                              stage=args.command)
    return protocol, protocol.get("protocol_id")


def cmd_normalizer(args, config, device):
    import zarr

    def produce():
        store = zarr.open(store=args.dataset, mode="r")
        action = np.asarray(store["data"]["action"][:], dtype=np.float64).reshape(-1, 2)
        state = np.asarray(store["data"]["state"][:, :2], dtype=np.float64).reshape(-1, 2)
        if not (np.isfinite(action).all() and np.isfinite(state).all()):
            raise ValueError("non-finite values in the demonstration archive")
        stats = {"action": {"min": action.min(0).tolist(), "max": action.max(0).tolist()},
                 "agent_pos": {"min": state.min(0).tolist(), "max": state.max(0).tolist()}}
        C.verify_tensors({k: np.array([v["min"], v["max"]]) for k, v in stats.items()},
                         "normalizer")
        C.atomic_savez(args.output, stats=stats)
        C.write_meta(args.output + ".meta.json",
                     dict(kind="normalizer", schema_version=C.SCHEMA_VERSION,
                          dataset=args.dataset, stats=stats,
                          source_hashes=C.source_hashes()))
        print("wrote normalizer " + args.output)
        return dict(kind="normalizer", dataset=args.dataset)

    produce()


def _collect_episode(adapter, config, device, scene, student, rng, split,
                     source, decision_limit):
    """One episode with live signatures and saved teacher targets."""
    record = {"scene": scene, "split": split, "contexts": []}
    env = adapter.new_env(image=True)
    obs, _ = adapter.reset(env, scene)
    initial_signature = [float(v) for v in adapter.signature(env).tolist()]
    history = []
    done = False
    steps = 0
    decision = 0
    while not done and steps < config["episode_steps"]:
        cond = adapter.encode_observation(obs["image"], obs["agent_pos"])
        noise_np = sample_source(rng, (1, 16, 2), source)
        noise_t = torch.from_numpy(noise_np).to(device)
        # live signature before executing the next chunk (never a replay)
        live_signature = [float(v) for v in adapter.signature(env).tolist()]
        keep = (decision % 2 == 0
                and len(record["contexts"]) < config["contexts_per_episode"])
        with torch.no_grad():
            if student is None:
                # the teacher IS the behaviour policy; the half map supplies the
                # behaviour chunk and its own targets with no discarded work
                y0, behavior = teacher_half_maps(adapter.noise_pred_net, noise_t,
                                                 cond, config["teacher_steps"])
                student_mid = None
                target_mid = y0
                target_end = behavior
                teacher_end = behavior
            else:
                # behaviour needs only the two student steps; teacher targets are
                # computed solely for decisions that are actually stored
                behavior, midpoint = sample_with_midpoint(noise_t, cond, student)
                student_mid = midpoint.detach()
                y0 = target_mid = target_end = teacher_end = None
                if keep:
                    y0, _ = teacher_half_maps(adapter.noise_pred_net, noise_t, cond,
                                              config["teacher_steps"])
                    teacher_end = teacher_half_from(y0, adapter.noise_pred_net, cond,
                                                    config["teacher_steps"])
                    target_mid = y0
                    # y1 is the teacher answer at the STUDENT's own midpoint
                    target_end = teacher_half_from(student_mid, adapter.noise_pred_net,
                                                   cond, config["teacher_steps"])
        context = None
        if keep:
            context = {"scene": scene, "split": split, "decision": decision,
                       "step": steps,
                       "condition": cond.detach().cpu().numpy()[0],
                       "noise": noise_np[0],
                       "midpoint": (student_mid.detach().cpu().numpy()[0]
                                    if student_mid is not None
                                    else y0.detach().cpu().numpy()[0]),
                       "target_mid": target_mid.detach().cpu().numpy()[0],
                       "target_end": target_end.detach().cpu().numpy()[0],
                       "teacher_end": teacher_end.detach().cpu().numpy()[0],
                       "history": [list(map(float, row)) for row in history],
                       "signature_initial": initial_signature,
                       "signature_live": live_signature}
            record["contexts"].append(context)
        prepared = adapter.prepare_commands(behavior.detach().cpu().numpy()[0][:config["execute_steps"]])
        for action in prepared["prepared"]:
            obs, _, done, _ = adapter.step(env, action)
            history.append([float(value) for value in np.asarray(action).tolist()])
            steps += 1
            if done or steps >= config["episode_steps"]:
                break
        if context is not None:
            context["history_end"] = [list(map(float, row)) for row in history]
        decision += 1
        if decision_limit and decision >= decision_limit:
            break
    env.close()
    return record


def cmd_collect(args, config, device):
    source = args.source or config["source"]
    splits = args.splits or ["train", "validation"]
    for name in splits:
        if name not in ("train", "validation"):
            raise ValueError("collect supports the train and validation splits")
    adapter = build_adapter(config, device)
    rng = np.random.default_rng(args.seed)
    student = None
    checkpoint_hash = None
    if args.student is not None:
        student, payload = load_student(args.student, config, device)
        student.eval()
        checkpoint_hash = C.sha256_file(args.student)
    rows = []
    started = time.time()
    for split in splits:
        planned = sorted(C.scene_range(config, split))
        default_count = (len(planned) if split == "train"
                         else int(config["validation_episodes"]))
        count = args.episodes if args.episodes is not None else default_count
        if count < 1:
            raise ValueError("--episodes must be >= 1")
        scenes = planned[:count]
        for scene in scenes:
            rows.append(_collect_episode(adapter, config, device, scene, student,
                                         rng, split, source, args.decision_limit))
    seen = {split: sorted({int(r["scene"]) for r in rows if r["split"] == split})
            for split in splits}
    C.validate_scene_sets(seen)
    contexts = [context for record in rows for context in record["contexts"]]
    if not contexts:
        raise ValueError("collection produced no contexts")
    meta = dict(schema_version=C.SCHEMA_VERSION,
                mode="shared" if student is not None else "teacher",
                source=source, seed=args.seed, student=args.student,
                splits={k: len(v) for k, v in seen.items()},
                contexts=len(contexts), wall_s=round(time.time() - started, 1))
    lineage = dict(source_hashes=C.source_hashes(),
                   checkpoint_sha256=checkpoint_hash,
                   normalizer_sha256=C.sha256_file(config["normalizer"]),
                   mode=meta["mode"], source=source,
                   split_scenes=seen)
    C.write_label_cache(args.output, contexts, config, meta, lineage)
    print(f"wrote cache {args.output} contexts {len(contexts)} "
          f"splits {meta['splits']}")
    return dict(contexts=len(contexts), splits=meta["splits"])


def _metric_entry(adapter, config, device, context, eps, metric_mode):
    cond = torch.from_numpy(np.asarray(context["condition"], dtype=np.float32)).unsqueeze(0).to(device)
    noise = torch.from_numpy(np.asarray(context["noise"], dtype=np.float32)).unsqueeze(0).to(device)
    steps = config["teacher_steps"]
    with torch.no_grad():
        y0 = torch.from_numpy(np.asarray(context["target_mid"], dtype=np.float32)).unsqueeze(0).to(device)
        teacher_end = torch.from_numpy(np.asarray(context["teacher_end"], dtype=np.float32)).unsqueeze(0).to(device)
        target_end = torch.from_numpy(np.asarray(context["target_end"], dtype=np.float32)).unsqueeze(0).to(device)
    history = [np.asarray(row, dtype=np.float64) for row in context["history"]]
    scene = int(context["scene"])
    adapter.check_replay(scene, history, np.asarray(context["signature_live"]))
    branches = {}
    for tag, prefix in (("start", teacher_end.cpu().numpy()[0][:config["execute_steps"]]),
                        ("end", target_end.cpu().numpy()[0][:config["execute_steps"]])):
        jac = finite_difference_jacobian(
            lambda chunk, _scene=scene, _history=history: adapter.execute_from_history(
                _scene, _history, chunk),
            prefix, epsilon=eps)
        branches[tag] = jac
    jac_start = torch.from_numpy(branches["start"].astype(np.float32)).unsqueeze(0).to(device)

    def suffix(variable):
        return teacher_half_from(variable, adapter.noise_pred_net, cond, steps)

    entry = {key: context[key] for key in
             ("scene", "split", "decision", "step", "condition", "noise",
              "midpoint", "target_mid", "target_end", "teacher_end", "history",
              "signature_initial", "signature_live")}
    entry["jacobian_start"] = branches["start"].astype(np.float32)
    entry["jacobian_end"] = branches["end"].astype(np.float32)
    entry["has_metric"] = True
    if metric_mode == "exact":
        entry["metric_exact"] = pullback_metric_exact(
            suffix, y0, jac_start, config["execute_steps"]).detach().cpu().numpy()[0].astype(np.float32)
    else:
        entry["probes"] = pullback_probes(
            suffix, y0, jac_start, config["execute_steps"],
            num_probes=config["num_probes"]).detach().cpu().numpy()[0].astype(np.float32)
    return entry


def _context_key(context):
    return (int(context["scene"]), str(context["split"]),
            int(context["decision"]), int(context["step"]))


def cmd_metrics(args, config, device):
    adapter = build_adapter(config, device)
    verified = C.verify_label_cache(args.cache, config, expect_mode="shared")
    train_rows, validation_rows = C.metric_subset(verified, config,
                                                  limit=args.limit)
    selected = train_rows + validation_rows
    if not selected:
        raise ValueError("metric subset is empty; collect the shared cache first")
    eps = config["finite_difference_epsilon"]
    metric_mode = config.get("metric_mode", "exact")
    print(f"[metrics] selected {len(train_rows)} train + {len(validation_rows)} "
          f"validation contexts of {verified['n_contexts']} (metric_mode="
          f"{metric_mode}); one repeated-branch determinism check per context")
    out = []
    timings = []
    for index, context in enumerate(selected):
        start = time.perf_counter()
        entry = _metric_entry(adapter, config, device, context, eps, metric_mode)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        out.append(entry)
        if index == 0:
            print(f"[metrics] first context took {elapsed:.2f}s; projected "
                  f"{elapsed * len(selected) / 60:.1f} min for {len(selected)} "
                  f"contexts", flush=True)
        elif (index + 1) % 25 == 0:
            done = sum(timings)
            projected = done / (index + 1) * len(selected)
            print(f"[metrics] {index + 1}/{len(selected)} elapsed {done / 60:.1f} "
                  f"min projected {projected / 60:.1f} min", flush=True)
    # one repeated-branch determinism check instead of duplicating every Jacobian
    probe_context = selected[0]
    prefix = np.asarray(probe_context["teacher_end"], dtype=np.float64)[:config["execute_steps"]]
    history = [np.asarray(row, dtype=np.float64) for row in probe_context["history"]]
    scene = int(probe_context["scene"])
    repeat_a = finite_difference_jacobian(
        lambda chunk: adapter.execute_from_history(scene, history, chunk), prefix,
        epsilon=eps)
    repeat_b = finite_difference_jacobian(
        lambda chunk: adapter.execute_from_history(scene, history, chunk), prefix,
        epsilon=eps)
    repeat_gap = float(np.abs(repeat_a - repeat_b).max())
    if not np.isfinite(repeat_gap) or repeat_gap > 1e-6:
        raise ValueError(f"repeated branch disagreement at scene {scene}: "
                         f"{repeat_gap}")
    # Two linked datasets in one cache: ALL shared contexts are preserved for
    # ordinary distillation and anchoring; the fixed subset additionally
    # carries physical metrics, identified by stable (scene, split, decision,
    # step) keys. Training draws each role from its own dataset, so an
    # ordinary row never becomes physically marked by pool membership.
    by_key = {_context_key(entry): entry for entry in out}
    if len(by_key) != len(out):
        raise ValueError("duplicate metric entries for one context")
    full = []
    for context in verified["contexts"]:
        key = _context_key(context)
        if key in by_key:
            full.append(by_key[key])
        else:
            plain = {k: context[k] for k in
                     ("scene", "split", "decision", "step", "condition", "noise",
                      "midpoint", "target_mid", "target_end", "teacher_end",
                      "history", "signature_initial", "signature_live")}
            plain["has_metric"] = False
            full.append(plain)
    meta = dict(schema_version=C.SCHEMA_VERSION, mode="shared",
                metric_mode=metric_mode,
                num_probes=config["num_probes"], epsilon=eps,
                counts=dict(train=len(train_rows), validation=len(validation_rows),
                            total=len(full), metric_total=len(out)),
                member_keys=[list(_context_key(entry)) for entry in out],
                repeat_branch_gap=repeat_gap,
                per_context_seconds=timings,
                mean_context_seconds=float(np.mean(timings)),
                source=verified["meta"].get("source"),
                splits={row["split"] for row in full})
    meta["splits"] = {split: sum(1 for row in full if row["split"] == split)
                      for split in sorted({row["split"] for row in full})}
    lineage = dict(source_hashes=C.source_hashes(),
                   cache=C.sha256_file(args.cache),
                   cache_meta=verified["meta"],
                   epsilon=eps, metric_mode=metric_mode)
    C.write_metric_cache(args.output, full, config, meta, lineage)
    print(f"wrote metrics {args.output} contexts {len(full)} "
          f"({len(out)} with physical metrics)")
    return dict(contexts=len(full), metric_contexts=len(out), meta=meta)


def batch_from_contexts(contexts, indices, device, metric_mode, need_metric):
    pick = [contexts[i] for i in indices]
    keys = ["condition", "noise", "midpoint", "target_mid", "target_end",
            "teacher_end"]
    if need_metric:
        keys += ["jacobian_start", "jacobian_end"]
        keys.append("metric_exact" if metric_mode == "exact" else "probes")
    batch = {}
    for key in keys:
        values = np.stack([np.asarray(entry[key]) for entry in pick])
        if not np.isfinite(values).all():
            raise ValueError("nonfinite cached tensor " + key)
        batch[key] = torch.from_numpy(values).to(device)
    return batch


def cmd_train(args, config, device):
    protocol, protocol_id = protocol_for(args, config)
    needs_metric = mode_needs_metric(args.mode)
    # the warm start trains from the teacher cache (mode "teacher"); only metric
    # modes require the shared student-collected cache
    verified = C.verify_label_cache(args.cache, config,
                                    expect_mode="shared" if needs_metric else None,
                                    require_metric=needs_metric)
    train_contexts = C.select_split(verified, "train")
    if not train_contexts:
        raise ValueError("cache holds no training-split contexts")
    if not needs_metric:
        print(f"[train] mode {args.mode}: label-only path, no Jacobians, no "
              "transported metric, no scale estimation")
    metric_mode = config.get("metric_mode", "exact")
    scales, scale_stats = None, None
    if needs_metric:
        metric_key = "metric_exact" if metric_mode == "exact" else "probes"
        metric_train = [c for c in train_contexts if c.get("has_metric", False)]
        if not metric_train:
            raise ValueError("metric training needs has_metric rows; "
                             "run the metrics stage first")
        jac_start = np.stack([c["jacobian_start"] for c in metric_train])
        jac_end = np.stack([c["jacobian_end"] for c in metric_train])
        metric = np.stack([c[metric_key] for c in metric_train])
        for name, value in (("jacobian_start", jac_start), ("jacobian_end", jac_end),
                            (metric_key, metric)):
            if not np.isfinite(value).all():
                raise ValueError("nonfinite cached tensor " + name)
        scales, scale_stats = metric_scales(
            torch.from_numpy(jac_start).float(), torch.from_numpy(jac_end).float(),
            torch.from_numpy(metric).float(), metric_mode=metric_mode)
        print("[train] scales " + json.dumps(scales) + " stats " +
              json.dumps(scale_stats))
        if scale_stats.get("low_support"):
            print("[train] WARNING: thin sensitivity support for scales " +
                  json.dumps(scale_stats["low_support"]) + " with support " +
                  json.dumps(scale_stats["support"]) +
                  "; the corresponding comparison rests on few contexts")
    seed_state = torch.load(args.initial, map_location=device,
                            weights_only=True) if args.initial else None
    if seed_state is not None:
        # Initialization is role-checked against the lock BEFORE optimization:
        # only the hash-matched warm start may initialize locked training.
        C.verify_student_provenance(seed_state, protocol, config, args.initial,
                                    role="init")
    if protocol is not None:
        # Validate every locked training argument before spending optimization
        # budget, rather than relying on later checkpoint rejection.
        problems = []
        if args.mode not in protocol.get("modes", []):
            problems.append(f"mode {args.mode} not in locked {protocol['modes']}")
        if int(args.seed) not in [int(s) for s in protocol.get("seeds", [])]:
            problems.append(f"seed {args.seed} not in locked {protocol['seeds']}")
        if int(args.updates) != int(protocol.get("updates", args.updates)):
            problems.append(f"updates {args.updates} != locked {protocol['updates']}")
        if C.sha256_file(args.cache) != protocol.get("cache_sha256"):
            problems.append("training cache differs from the locked cache")
        if problems:
            raise C.ProtocolError("locked training arguments rejected: " +
                                  "; ".join(problems))
    student = make_student(config, device, seed=args.seed,
                           state_dict=seed_state["student"] if seed_state else None)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(args.seed)
    n = len(train_contexts)
    # Role-first batches: the first half is ordinary distillation over ALL
    # shared contexts; the second half is designated physical rows drawn from
    # the fixed metric subset. A role mask travels with the rows through the
    # within-half shuffles, so an ordinary sample never becomes physically
    # marked merely because its context also carries a metric, and every mode
    # sees matched minibatch streams from the same seed.
    half = config["batch_size"] // 2
    metric_pool = [i for i, c in enumerate(train_contexts)
                   if c.get("has_metric", False)]
    if needs_metric and not metric_pool:
        raise ValueError("metric training needs has_metric rows; "
                         "run the metrics stage first")
    student.train()
    log = []
    grad_norms = []
    for update in range(args.updates):
        ordinary = rng.integers(0, n, size=half).tolist()
        rng.shuffle(ordinary)
        if metric_pool:
            physical = rng.choice(metric_pool, size=half,
                                  replace=True).tolist()
        else:
            physical = []
        rng.shuffle(physical)
        ordinary_batch = batch_from_contexts(train_contexts, ordinary, device,
                                             metric_mode, False)
        if physical:
            physical_batch = batch_from_contexts(train_contexts, physical,
                                                 device, metric_mode, needs_metric)
            batch = {k: torch.cat([ordinary_batch[k], physical_batch[k]], dim=0)
                     for k in ordinary_batch}
            # physical-only keys (Jacobians, metric) are NaN-free on the
            # designated rows; ordinary rows carry zeros there and are never
            # selected by the tail slice.
            for k in physical_batch:
                if k not in batch:
                    pad = torch.zeros((len(ordinary),) + physical_batch[k].shape[1:],
                                      dtype=physical_batch[k].dtype,
                                      device=device)
                    batch[k] = torch.cat([pad, physical_batch[k]], dim=0)
        else:
            batch = ordinary_batch
            if needs_metric:
                raise ValueError("empty physical selection for a metric mode")
        metric_count = len(physical)
        loss, parts = objective(student, batch, args.mode, config["execute_steps"],
                                scales, metric_weight=config["metric_weight"],
                                anchor_weight=config["anchor_weight"],
                                metric_count=metric_count, metric_mode=metric_mode)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at update {update}")
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0,
                                                   error_if_nonfinite=True)
        optimizer.step()
        grad_norms.append(float(grad_norm))
        if update % max(1, args.updates // 100) == 0 or update == args.updates - 1:
            log.append({"update": update, "loss": float(loss.item()),
                        "base": float(parts["base"].item()),
                        "anchor": float(parts["anchor"].item()),
                        "penalty": float(parts["penalty"].item()),
                        "weighted_penalty": float(parts["weighted_penalty"].item()),
                        "grad_norm": float(grad_norm),
                        "mid_error_rms": float(parts["mid_error_rms"].item()),
                        "end_error_rms": float(parts["end_error_rms"].item()),
                        "endpoint_error_rms": float(parts["endpoint_error_rms"].item())})
    for tensor in student.state_dict().values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite parameter after training; refusing to save")
    payload = dict(student=student.state_dict(), mode=args.mode, seed=args.seed,
                   updates=args.updates,
                   scales=scales, scale_stats=scale_stats, metric_mode=metric_mode,
                   protocol_id=protocol_id,
                   config_sha256=C.sha256_file(args.config),
                   init_sha256=(C.sha256_file(args.initial) if args.initial else None),
                   cache_sha256=C.sha256_file(args.cache),
                   needs_metric=bool(needs_metric),
                   log=log, grad_norm_mean=float(np.mean(grad_norms)),
                   grad_norm_max=float(np.max(grad_norms)),
                   compute_env=C.apply_compute_env())
    C.atomic_save_torch(args.output, payload)
    C.write_meta(args.output + ".meta.json",
                 dict(kind="student", mode=args.mode, seed=args.seed,
                      updates=args.updates, metric_mode=metric_mode,
                      protocol_id=protocol_id,
                      compute_env=C.apply_compute_env(), scales=scales,
                      scale_stats=scale_stats, needs_metric=bool(needs_metric),
                      grad_norm_mean=float(np.mean(grad_norms)),
                      source_hashes=C.source_hashes()))
    print("wrote student " + args.output + " mode " + args.mode)
    return dict(mode=args.mode, seed=args.seed, updates=args.updates)


def cmd_evaluate(args, config, device):
    protocol, protocol_id = protocol_for(args, config)
    adapter = build_adapter(config, device)
    student = None
    intrinsic_mode = args.name
    intrinsic_seed = args.seed
    checkpoint_hash = None
    if args.student:
        student, payload = load_student(args.student, config, device)
        student.eval()
        if protocol is not None:
            C.verify_student_provenance(payload, protocol, config, args.student,
                                        role="final")
        else:
            # Unlocked runs are development only: the checkpoint must be the
            # warm start evaluated as an explicitly labeled reference.
            C.verify_student_provenance(payload, None, config, args.student,
                                        role="reference")
        intrinsic_mode = str(payload["mode"])
        intrinsic_seed = int(payload["seed"])
        checkpoint_hash = C.sha256_file(args.student)
    source = args.source or config["source"]
    steps = args.steps
    split = "development" if args.development else "test"
    rng = np.random.default_rng(args.seed)
    if protocol is not None:
        if args.development:
            raise C.ProtocolError("locked evaluation is the final test run; "
                                  "development evaluation is unlocked with "
                                  "--development --allow-unlocked")
        # Final evaluation scenes come from the lock, never from CLI/config
        # overrides: a reduced lock must produce a reduced evaluation.
        scenes = list(protocol["final_scenes"])
        if args.episodes is not None and int(args.episodes) != len(scenes):
            raise C.ProtocolError(
                f"--episodes {args.episodes} conflicts with the locked "
                f"{len(scenes)} final scenes; regenerate the lock instead")
    else:
        episodes = args.episodes if args.episodes is not None else (
            config["development_episodes"] if args.development
            else config["test_episodes"])
        if episodes < 1:
            raise ValueError("--episodes must be >= 1")
        start = config["development_scene_start"] if args.development else config["test_scene_start"]
        scenes = [start + episode for episode in range(episodes)]
    unknown = sorted(set(scenes) - C.scene_range(config, split))
    if unknown:
        raise ValueError(f"evaluation scenes outside the {split} range: {unknown[:5]}")
    C.validate_scene_sets({split: scenes, "train": sorted(C.scene_range(config, "train"))})
    max_decisions = config["episode_steps"] // config["execute_steps"] + 2
    warmup = max(0, int(config.get("warmup_episodes", 0)))
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    print(f"[eval] device={device} source={source} steps={steps} split={split} "
          f"episodes={len(scenes)} intrinsic_mode={intrinsic_mode} "
          f"warmup_episodes={warmup} clip={config['clip_actions']}")
    with open(args.output, "w") as handle:
        for episode, scene in enumerate(scenes):
            policy_noise = np.stack([sample_source(rng, (16, 2), source)
                                     for _ in range(max_decisions)])
            env = adapter.new_env(image=True)
            obs, _ = adapter.reset(env, scene)
            done = False
            steps_taken = 0
            best = 0.0
            decision_latencies = []
            head_latencies = []
            raw_violations = 0
            executed_violations = 0
            coordinates = 0
            clipped_coordinates = 0
            terminated_any = False
            truncated_any = False
            decision = 0
            while not done and steps_taken < config["episode_steps"]:
                decision_start = time.perf_counter()
                cond = adapter.encode_observation(obs["image"], obs["agent_pos"])
                sync()
                head_start = time.perf_counter()
                noise = policy_noise[min(decision, max_decisions - 1)][None, ...]
                noise_t = torch.from_numpy(noise).to(device)
                with torch.no_grad():
                    if student is not None:
                        chunk = student.sample(noise_t, cond)
                    else:
                        chunk = teacher_rollout(adapter.noise_pred_net, noise_t,
                                                cond, steps)
                sync()
                head_latencies.append((time.perf_counter() - head_start) * 1000.0)
                prepared = adapter.prepare_commands(
                    chunk.detach().cpu().numpy()[0][:config["execute_steps"]])
                sync()
                decision_latencies.append((time.perf_counter() - decision_start) * 1000.0)
                decision += 1
                raw_violations += prepared["raw_violations"]
                executed_violations += prepared["executed_violations"]
                coordinates += prepared["coordinates"]
                clipped_coordinates += (prepared["raw_violations"]
                                        if prepared["clipped"] else 0)
                for action in prepared["prepared"]:
                    obs, reward, terminated, truncated, done = adapter.raw_step(env, action)
                    best = max(best, reward)
                    terminated_any = terminated_any or terminated
                    truncated_any = truncated_any or truncated
                    steps_taken += 1
                    if done or steps_taken >= config["episode_steps"]:
                        break
            env.close()
            capped = bool(steps_taken >= config["episode_steps"] and not terminated_any)
            success = int(terminated_any and best >= 0.95)
            is_warmup = episode < warmup
            handle.write(json.dumps({
                "method": intrinsic_mode, "alias": args.name,
                "intrinsic_mode": intrinsic_mode, "intrinsic_seed": intrinsic_seed,
                "training_seed": intrinsic_seed, "scene": scene, "split": split,
                "warmup": int(is_warmup),
                "success": success, "terminated": int(terminated_any),
                "truncated": int(truncated_any), "episode_cap_reached": int(capped),
                "score": best, "steps": steps_taken,
                "source": source, "teacher_steps": steps,
                "decision_latency_median_ms": (None if is_warmup else
                                               float(np.median(decision_latencies))),
                "decision_latency_p95_ms": (None if is_warmup else
                                            float(np.quantile(decision_latencies, 0.95))),
                "action_head_latency_median_ms": (None if is_warmup else
                                                  float(np.median(head_latencies))),
                "raw_violation_fraction": raw_violations / max(coordinates, 1),
                "executed_violation_fraction": executed_violations / max(coordinates, 1),
                "clip_fraction": clipped_coordinates / max(coordinates, 1),
                "coordinates": coordinates,
                "protocol_id": protocol_id,
                "checkpoint_sha256": checkpoint_hash}) + "\n")
    C.write_meta(args.output.replace(".jsonl", ".meta.json"),
                 dict(kind="evaluation", alias=args.name,
                      compute_env=C.apply_compute_env(),
                      intrinsic_mode=intrinsic_mode, intrinsic_seed=intrinsic_seed,
                      checkpoint_sha256=checkpoint_hash, protocol_id=protocol_id,
                      split=split, expected_split=split, scenes=scenes,
                      source=source, teacher_steps=steps, device=str(device),
                      clip_actions=bool(config["clip_actions"]),
                      warmup_episodes=warmup,
                      source_hashes=C.source_hashes()))
    print("wrote eval " + args.output)
    return dict(rows=len(scenes), mode=intrinsic_mode, seed=intrinsic_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--allow-unlocked", action="store_true",
                        help="permitted only for development stages")
    sub = parser.add_subparsers(dest="command", required=True)
    entry = sub.add_parser("normalizer")
    entry.add_argument("--dataset", required=True)
    entry.add_argument("--output", required=True)
    entry = sub.add_parser("collect")
    entry.add_argument("--output", required=True)
    entry.add_argument("--student", default=None)
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--episodes", type=int, default=None)
    entry.add_argument("--splits", nargs="*", default=None)
    entry.add_argument("--source", default=None)
    entry.add_argument("--decision-limit", type=int, default=None)
    entry = sub.add_parser("metrics")
    entry.add_argument("--cache", required=True)
    entry.add_argument("--output", required=True)
    entry.add_argument("--limit", type=int, default=None)
    entry = sub.add_parser("train")
    entry.add_argument("--cache", required=True)
    entry.add_argument("--output", required=True)
    entry.add_argument("--mode", required=True,
                       choices=["uniform", "prefix", "endpoint", "pullback",
                                "identity", "scalar"])
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--updates", type=int, default=None)
    entry.add_argument("--initial", default=None)
    entry = sub.add_parser("evaluate")
    entry.add_argument("--output", required=True)
    entry.add_argument("--name", required=True)
    entry.add_argument("--student", default=None)
    entry.add_argument("--steps", type=int, default=None)
    entry.add_argument("--source", default=None,
                       choices=["gaussian", "uniform", "uniform_symmetric"])
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--episodes", type=int, default=None)
    entry.add_argument("--development", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    compute = C.apply_compute_env(args.device or config["device"])
    device = resolve_device(args.device or config["device"])
    if args.command == "train":
        if args.updates is None:
            args.updates = config["updates"]
        if args.updates < 1:
            raise ValueError("--updates must be >= 1")
    if args.command == "evaluate" and args.steps is None:
        args.steps = config["teacher_steps"]
    if args.command == "evaluate" and args.steps < 1:
        raise ValueError("--steps must be >= 1")
    output = getattr(args, "output", None)
    meta_path = None
    if output:
        meta_path = (args.output.replace(".jsonl", ".meta.json")
                     if args.command == "evaluate" else output + ".meta.json")
        C.reserve_outputs([output, meta_path])
    try:
        if args.command == "normalizer":
            cmd_normalizer(args, config, device)
        elif args.command == "collect":
            cmd_collect(args, config, device)
        elif args.command == "metrics":
            cmd_metrics(args, config, device)
        elif args.command == "train":
            cmd_train(args, config, device)
        elif args.command == "evaluate":
            cmd_evaluate(args, config, device)
    except Exception as error:
        if output:
            C.mark_incomplete(output, f"{type(error).__name__}: {error}")
        raise
    else:
        if output:
            C.complete_output(output)
            if meta_path:
                C.complete_output(meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
