"""Artifact contracts for the latent-controllability experiment.

Metric-free: no Jacobians, no pullback probes, no metric weights. This
module governs artifact integrity (hashes, sidecars, reservations),
configuration validation, deterministic seeding, scene splits, cache/bank
lineage, protocol locks, and student provenance.
"""
import hashlib
import json
import os
import subprocess
import tempfile

import numpy as np

SCHEMA_VERSION = 3

SOURCE_FILES = ["contract.py", "hri_adapter.py", "pairs.py", "train.py",
                "geometry.py", "correction.py", "evaluate.py", "readiness.py",
                "lock.py", "analyze.py", "cli.py", "collect.py", "sanity.py",
                "config.json"]
PACKAGES = ["torch", "torchvision", "numpy", "scipy", "pandas", "gym",
            "pygame", "pymunk", "shapely", "cv2", "skimage", "zarr",
            "diffusers", "imageio_ffmpeg"]

SIDECAR_REQUIRED = (".pt", ".pth", ".npz", ".jsonl")
SELF_DESCRIBING = (".json",)

COMPLETION_SUFFIX = ".complete.json"
RESERVED_SUFFIX = ".reserved"
INCOMPLETE_SUFFIX = ".incomplete.json"

SCENE_RANGES = {
    "development": ("development_scene_start", "development_episodes"),
    "train": ("train_scene_start", "train_episodes"),
    "validation": ("validation_scene_start", "validation_episodes"),
    "diagnostic": ("diagnostic_scene_start", "diagnostic_episodes"),
    "final": ("final_scene_start", "final_episodes"),
}

POSITIVE_INTS = ["teacher_steps", "execute_steps", "episode_steps", "width",
                 "batch_size", "warm_updates", "updates",
                 "anchors_per_history", "directions_per_anchor",
                 "development_episodes", "train_episodes",
                 "validation_episodes", "diagnostic_episodes",
                 "final_episodes"]
ALLOWED_ARMS = {"anchor", "augmented", "gad"}

HISTORY_CONTEXT_KEYS = ("history_id", "scene", "split", "decision", "history",
                        "sig_initial", "sig_live", "condition", "endpoint",
                        "collector_hash")
PAIR_RECORD_KEYS = ("pair_id", "history_id", "anchor_id", "direction_id",
                    "q", "u", "rho", "source", "t0", "t1", "condition",
                    "teacher_steps")
PAIR_META_KEYS = ("member_keys", "rho", "source", "teacher_steps")


class ProtocolError(ValueError):
    pass


def experiment_dir():
    return os.path.dirname(os.path.abspath(__file__))


def sha256_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError("missing file " + str(path))
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_many(paths):
    out = {}
    for path in paths:
        try:
            out[path] = sha256_file(path)
        except FileNotFoundError:
            out[path] = "absent"
    return out


def source_hashes():
    here = experiment_dir()
    hashes = _hash_many([os.path.join(here, name) for name in SOURCE_FILES])
    return {os.path.basename(k): v for k, v in hashes.items()}


def package_versions(names=None):
    out = {}
    for name in (names or PACKAGES):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "present")
        except Exception as error:
            out[name] = "missing: " + str(error)
    return out


def asset_hashes(config):
    out = {}
    for key in ("checkpoint", "normalizer"):
        path = config.get(key)
        if path and os.path.exists(path):
            out[key] = sha256_file(path)
        else:
            out[key] = "absent"
    return out


def combined_id(parts):
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def stable_seed(stage, scene, decision, anchor, direction):
    payload = json.dumps([str(stage), int(scene), int(decision), int(anchor),
                          int(direction)],
                         separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def scene_range(config, split):
    if split not in SCENE_RANGES:
        raise ProtocolError(f"unknown split {split!r}; "
                            f"known: {sorted(SCENE_RANGES)}")
    start_key, count_key = SCENE_RANGES[split]
    return set(range(int(config[start_key]),
                     int(config[start_key]) + int(config[count_key])))


def validate_config(config, config_path="config.json"):
    errors = []
    for key in POSITIVE_INTS:
        if key not in config:
            errors.append("missing key " + key)
            continue
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{key} must be a positive integer, got {value!r}")
    rho = config.get("rho")
    if rho is None or not np.isfinite(rho) or rho <= 0:
        errors.append(f"rho must be positive and finite, got {rho!r}")
    betas = config.get("betas")
    if (not isinstance(betas, (list, tuple)) or not betas
            or any(not np.isfinite(b) or b < 0 for b in betas)):
        errors.append(f"betas must be a nonempty list of finite values >= 0, "
                      f"got {betas!r}")
    lr = config.get("learning_rate")
    if lr is None or not np.isfinite(lr) or lr <= 0:
        errors.append("learning_rate must be positive and finite")
    wd = config.get("weight_decay")
    if wd is None or not np.isfinite(wd) or wd < 0:
        errors.append("weight_decay must be finite and >= 0")
    arms = config.get("arms")
    if (not isinstance(arms, (list, tuple)) or not arms
            or any(a not in ALLOWED_ARMS for a in arms)):
        errors.append(f"arms must be a nonempty subset of "
                      f"{sorted(ALLOWED_ARMS)}, got {arms!r}")
    corr = config.get("correction") or {}
    for key in ("lrs", "budgets"):
        if key not in corr:
            errors.append("missing correction key " + key)
    budgets = corr.get("budgets", None)
    if (not isinstance(budgets, (list, tuple)) or
            any(not isinstance(b, int) or isinstance(b, bool) or b < 0
                for b in (budgets or []))):
        errors.append(f"correction.budgets must be nonnegative ints, "
                      f"got {budgets!r}")
    for key in ("checkpoint", "normalizer", "repository"):
        if not config.get(key):
            errors.append("missing path key " + key)
    for key in ("development_scene_start", "train_scene_start",
                "validation_scene_start", "diagnostic_scene_start",
                "final_scene_start"):
        if not isinstance(config.get(key), int) or isinstance(config.get(key), bool):
            errors.append(f"{key} must be an integer")
    if not errors:
        sets = {name: scene_range(config, name) for name in SCENE_RANGES}
        names = sorted(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = sorted(sets[a] & sets[b])
                if overlap:
                    errors.append(f"{a} and {b} scene ranges overlap: "
                                  f"{overlap[:5]}")
    if errors:
        raise ValueError("invalid configuration " + str(config_path) + ": " +
                         "; ".join(errors))
    return True


# ---------------------------------------------------------------------------
# Artifact reservation / completion
# ---------------------------------------------------------------------------

def _needs_sidecar(path):
    return str(path).endswith(SIDECAR_REQUIRED)


def _is_self_describing(path):
    return str(path).endswith(SELF_DESCRIBING)


def reserve_outputs(paths):
    """Exclusively reserve output paths (create sibling .reserved files).

    Refuses when the data file, its .complete.json, or its .reserved file
    already exists, so two processes can never share one artifact.
    """
    for path in paths:
        if os.path.exists(path) or os.path.exists(path + COMPLETION_SUFFIX) \
                or os.path.exists(path + RESERVED_SUFFIX):
            raise ProtocolError(f"output already exists or reserved: {path}")
    for path in paths:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        handle = os.open(path + RESERVED_SUFFIX, flags)
        with os.fdopen(handle, "w") as stream:
            json.dump({"path": path,
                       "schema_version": SCHEMA_VERSION}, stream)


def complete_output(path, role=None, protocol=None):
    """Mark an artifact complete: record sha256+bytes in <path>.complete.json."""
    if not os.path.exists(path):
        raise ProtocolError(f"cannot complete missing artifact: {path}")
    if os.path.exists(path + COMPLETION_SUFFIX):
        raise ProtocolError(f"artifact already completed: {path}")
    sidecar = {"sha256": sha256_file(path),
               "bytes": int(os.path.getsize(path)),
               "schema_version": SCHEMA_VERSION}
    if role is not None:
        sidecar["role"] = role
    if protocol is not None:
        sidecar["protocol"] = protocol
    _atomic_json(path + COMPLETION_SUFFIX, sidecar)
    for suffix in (RESERVED_SUFFIX, INCOMPLETE_SUFFIX):
        try:
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
        except FileNotFoundError:
            pass
    return sidecar


def mark_incomplete(path, reason):
    _atomic_json(path + INCOMPLETE_SUFFIX,
                 {"path": path, "reason": str(reason),
                  "schema_version": SCHEMA_VERSION})


def is_complete(path):
    return os.path.exists(path + COMPLETION_SUFFIX)


def verify_completed(path, role=None, protocol=None):
    """Verify a completed artifact against its sidecar.

    Error codes: MISSING_HASH (no sidecar where required), ARTIFACT_HASH_MISMATCH,
    SIDECAR_MISMATCH (size drift), ROLE_MISMATCH, PROTOCOL_MISMATCH.
    """
    sidecar_path = path + COMPLETION_SUFFIX
    if not os.path.exists(path):
        raise ProtocolError(f"MISSING_HASH: artifact absent: {path}")
    if _is_self_describing(path):
        return {"path": path, "self_describing": True,
                "sha256": sha256_file(path)}
    if not _needs_sidecar(path):
        return {"path": path, "sha256": sha256_file(path)}
    if not os.path.exists(sidecar_path):
        raise ProtocolError(f"MISSING_HASH: no completion sidecar for {path}")
    with open(sidecar_path) as handle:
        sidecar = json.load(handle)
    actual_hash = sha256_file(path)
    if sidecar.get("sha256") != actual_hash:
        raise ProtocolError(f"ARTIFACT_HASH_MISMATCH: {path}")
    if int(sidecar.get("bytes", -1)) != int(os.path.getsize(path)):
        raise ProtocolError(f"SIDECAR_MISMATCH: byte count drift for {path}")
    if role is not None and sidecar.get("role") != role:
        raise ProtocolError(f"ROLE_MISMATCH: expected {role!r}, sidecar has "
                            f"{sidecar.get('role')!r} for {path}")
    if protocol is not None and sidecar.get("protocol") != protocol:
        raise ProtocolError(f"PROTOCOL_MISMATCH: locked protocol differs for "
                            f"{path}")
    return sidecar


def _atomic_json(path, payload):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", dir=directory, delete=False)
    try:
        json.dump(payload, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except Exception:
        try:
            handle.close()
        except Exception:
            pass
        try:
            os.remove(handle.name)
        except Exception:
            pass
        raise


def write_meta(path, payload):
    _atomic_json(path, dict(payload, schema_version=SCHEMA_VERSION))


# ---------------------------------------------------------------------------
# Atomic tensor saves (state dicts only, never pickled modules)
# ---------------------------------------------------------------------------

def atomic_savez(path, **arrays):
    import numpy as _np
    for key, value in arrays.items():
        if not isinstance(value, _np.ndarray):
            raise ProtocolError(f"atomic_savez {path}: {key} is not ndarray")
        if not _np.isfinite(value).all():
            raise ProtocolError(f"atomic_savez {path}: nonfinite {key}")
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(suffix=".npz", dir=directory,
                                         delete=False)
    handle.close()
    try:
        _np.savez(handle.name, **arrays)
        os.replace(handle.name, path)
    except Exception:
        try:
            os.remove(handle.name)
        except Exception:
            pass
        raise


def atomic_save_torch(path, state_dict):
    import torch as _torch
    if not isinstance(state_dict, dict):
        raise ProtocolError("atomic_save_torch requires a state dict, "
                            "never a pickled module")
    for key, value in state_dict.items():
        if not isinstance(value, _torch.Tensor):
            raise ProtocolError(f"atomic_save_torch {path}: {key} is not "
                                "a Tensor; refusing to pickle modules")
        if not _torch.isfinite(value).all():
            raise ProtocolError(f"atomic_save_torch {path}: nonfinite {key}")
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(suffix=".pt", dir=directory,
                                         delete=False)
    handle.close()
    try:
        _torch.save(state_dict, handle.name)
        os.replace(handle.name, path)
    except Exception:
        try:
            os.remove(handle.name)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# History cache and pair bank (with lineage)
# ---------------------------------------------------------------------------

def cache_fingerprint(config, repository=None):
    fingerprint = {"schema_version": SCHEMA_VERSION,
                   "source": config.get("canonical_source"),
                   "teacher_steps": config.get("teacher_steps"),
                   "execute_steps": config.get("execute_steps"),
                   "episode_steps": config.get("episode_steps"),
                   "rho": config.get("rho"),
                   "betas": config.get("betas"),
                   "anchors_per_history": config.get("anchors_per_history"),
                   "directions_per_anchor": config.get("directions_per_anchor"),
                   "legacy": config.get("legacy"),
                   "repository": repository or config.get("repository"),
                   "train_scene_start": config.get("train_scene_start"),
                   "train_episodes": config.get("train_episodes"),
                   "validation_scene_start": config.get("validation_scene_start"),
                   "validation_episodes": config.get("validation_episodes"),
                   "development_scene_start": config.get("development_scene_start"),
                   "development_episodes": config.get("development_episodes"),
                   "diagnostic_scene_start": config.get("diagnostic_scene_start"),
                   "diagnostic_episodes": config.get("diagnostic_episodes"),
                   "final_scene_start": config.get("final_scene_start"),
                   "final_episodes": config.get("final_episodes")}
    fingerprint["sources"] = source_hashes()
    fingerprint["assets"] = asset_hashes(config)
    return fingerprint


def write_history_cache(path, contexts, config, repository=None):
    for context in contexts:
        missing = [k for k in HISTORY_CONTEXT_KEYS if k not in context]
        if missing:
            raise ProtocolError(f"history context missing keys {missing}")
    payload = {"schema_version": SCHEMA_VERSION,
               "fingerprint": cache_fingerprint(config, repository),
               "histories": list(contexts)}
    _atomic_json(path, payload)
    return payload


def _as_float_array(value, name):
    import numpy as _np
    arr = _np.asarray(value, dtype=_np.float32)
    if not _np.isfinite(arr).all():
        raise ProtocolError(f"nonfinite {name} in cache record")
    return arr


def verify_history_cache(path, config, repository=None):
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("history cache schema mismatch")
    expected = cache_fingerprint(config, repository)
    actual = payload.get("fingerprint", {})
    for key, value in expected.items():
        if key in ("sources", "assets"):
            continue
        if actual.get(key) != value:
            raise ProtocolError(f"history cache fingerprint drift at {key}")
    if actual.get("sources") != expected.get("sources"):
        raise ProtocolError("history cache source drift")
    if actual.get("assets") != expected.get("assets"):
        raise ProtocolError("history cache asset drift")
    for context in payload.get("histories", []):
        missing = [k for k in HISTORY_CONTEXT_KEYS if k not in context]
        if missing:
            raise ProtocolError(f"history context missing keys {missing}")
        for key in ("condition", "endpoint"):
            context[key] = _as_float_array(context[key], key)
        context["history"] = [list(map(float, row))
                              for row in context["history"]]
    return payload


def _to_list(value):
    import numpy as _np
    if isinstance(value, _np.ndarray):
        return value.astype(_np.float64).tolist()
    if isinstance(value, (list, tuple)):
        return [ _to_list(v) for v in value ]
    return value


def write_pair_bank(path, records, meta, config, lineage=None):
    serial = []
    for record in records:
        missing = [k for k in PAIR_RECORD_KEYS if k not in record]
        if missing:
            raise ProtocolError(f"pair record missing keys {missing}")
        serial.append({k: _to_list(record[k]) for k in PAIR_RECORD_KEYS})
    missing_meta = [k for k in PAIR_META_KEYS if k not in meta]
    if missing_meta:
        raise ProtocolError(f"pair bank meta missing keys {missing_meta}")
    if float(meta.get("rho", -1)) != float(config.get("rho")):
        raise ProtocolError("pair bank rho does not match config")
    if int(meta.get("teacher_steps", -1)) != int(config.get("teacher_steps")):
        raise ProtocolError("pair bank teacher_steps do not match config")
    payload = {"schema_version": SCHEMA_VERSION, "meta": dict(meta),
               "lineage": dict(lineage or {}),
               "records": serial}
    _atomic_json(path, payload)
    return payload


def verify_pair_bank(path, config):
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("pair bank schema mismatch")
    meta = payload.get("meta", {})
    missing_meta = [k for k in PAIR_META_KEYS if k not in meta]
    if missing_meta:
        raise ProtocolError(f"pair bank meta missing keys {missing_meta}")
    if float(meta.get("rho")) != float(config.get("rho")):
        raise ProtocolError("pair bank rho does not match config")
    if int(meta.get("teacher_steps")) != int(config.get("teacher_steps")):
        raise ProtocolError("pair bank teacher_steps do not match config")
    for record in payload.get("records", []):
        missing = [k for k in PAIR_RECORD_KEYS if k not in record]
        if missing:
            raise ProtocolError(f"pair record missing keys {missing}")
        for key in ("q", "u", "t0", "t1", "condition"):
            record[key] = _as_float_array(record[key], key)
        record["pair_id"] = list(record["pair_id"])
        record["history_id"] = str(record["history_id"])
        record["anchor_id"] = int(record["anchor_id"])
        record["direction_id"] = int(record["direction_id"])
        record["rho"] = float(record["rho"])
        record["teacher_steps"] = int(record["teacher_steps"])
    return payload


# ---------------------------------------------------------------------------
# Protocol lock, canonical design, student provenance
# ---------------------------------------------------------------------------

def upstream_commit(repository):
    try:
        out = subprocess.run(["git", "-C", repository, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def canonical_design(config, modes, seeds, updates, arms, scenes,
                     correction, beta_selected, rho, compute, references,
                     warm=None, bank=None, cache=None):
    design = {"schema_version": SCHEMA_VERSION,
              "config": dict(config),
              "modes": sorted(modes),
              "seeds": sorted(int(s) for s in seeds),
              "updates": int(updates),
              "arms": sorted(arms),
              "scenes": {k: sorted(int(s) for s in v)
                         for k, v in sorted(scenes.items())},
              "correction": correction,
              "beta_selected": float(beta_selected),
              "rho": float(rho),
              "compute": compute,
              "references": references,
              "warm_sha256": (sha256_file(warm)
                              if warm and os.path.exists(warm) else "absent"),
              "bank_sha256": (sha256_file(bank)
                              if bank and os.path.exists(bank) else "absent"),
              "cache_sha256": (sha256_file(cache)
                               if cache and os.path.exists(cache) else "absent")}
    design["design_id"] = combined_id(design)
    return design


def design_id(design):
    body = {k: v for k, v in design.items() if k != "design_id"}
    return combined_id(body)


def load_protocol(path):
    with open(path) as handle:
        return json.load(handle)


def verify_current_protocol(protocol, config, config_path, repository=None):
    repository = repository or config.get("repository")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("protocol schema mismatch")
    if protocol.get("sources") != source_hashes():
        raise ProtocolError("PROTOCOL_MISMATCH: source files changed")
    if protocol.get("config_file") != sha256_file(config_path):
        raise ProtocolError("PROTOCOL_MISMATCH: config file changed")
    if protocol.get("assets") != asset_hashes(config):
        raise ProtocolError("PROTOCOL_MISMATCH: assets changed")
    if protocol.get("packages") != package_versions():
        raise ProtocolError("PROTOCOL_MISMATCH: package set changed")
    recomputed = None
    design = protocol.get("design")
    if design is not None:
        recomputed = design_id(design)
        if recomputed != design.get("design_id"):
            raise ProtocolError("PROTOCOL_MISMATCH: canonical design id "
                                "does not recompute")
    for key in ("artifacts",):
        for entry in (protocol.get(key) or {}).values():
            ref = entry if isinstance(entry, str) else entry.get("path", entry)
            if isinstance(ref, str) and os.path.exists(ref):
                verify_completed(ref)
    return {"design_id": recomputed,
            "upstream_commit": upstream_commit(repository)}


def verify_student_provenance(payload, protocol, config, path,
                              role="final"):
    if role not in ("init", "final"):
        raise ProtocolError(f"unknown student role {role!r}")
    design = protocol.get("design", {}) if protocol else {}
    problems = []
    if protocol is not None:
        if payload.get("protocol_id") is not None and \
                payload.get("protocol_id") != protocol.get("protocol_id"):
            problems.append("checkpoint protocol id differs from the lock")
        if design.get("arms") and payload.get("arm") not in design["arms"] \
                and payload.get("arm") != "warm":
            problems.append(f"checkpoint arm {payload.get('arm')!r} is not locked")
        if design.get("seeds") and int(payload.get("seed", -1)) not in \
                [int(s) for s in design["seeds"]]:
            problems.append("checkpoint seed is not a locked seed")
        if design.get("bank_sha256", "absent") != "absent" and \
                payload.get("bank_sha256") not in (None, design["bank_sha256"]):
            problems.append("checkpoint bank differs from the locked bank")
    if role == "init":
        if protocol is None:
            problems.append("init role requires the lock holding warm_sha256")
        elif sha256_file(path) != design.get("warm_sha256", "absent"):
            problems.append("init file hash != locked warm_sha256")
    if role == "final":
        if payload.get("protocol_id") is None and protocol is not None:
            problems.append("unlocked checkpoint cannot serve as final student")
        if protocol is not None and payload.get("init_sha256") != \
                design.get("warm_sha256", "absent"):
            problems.append("final student init differs from locked warm start")
    if problems:
        raise ProtocolError("checkpoint provenance mismatch for " + str(path)
                            + ": " + "; ".join(problems))
    return True


def compute_env(device="cuda"):
    """Lightweight resolved-compute record (no torch import required)."""
    record = dict(device=str(device), schema_version=SCHEMA_VERSION)
    try:
        import torch as _torch
        record["torch_version"] = str(getattr(_torch, "__version__", "unknown"))
        record["cuda_available"] = bool(_torch.cuda.is_available())
        if _torch.cuda.is_available():
            try:
                record["gpu_name"] = _torch.cuda.get_device_name(device)
            except Exception:
                record["gpu_name"] = "unknown"
        record["deterministic"] = bool(
            _torch.are_deterministic_algorithms_enabled())
    except Exception as error:
        record["torch"] = "missing: " + str(error)
    return record


# ---------------------------------------------------------------------------
# Evaluation gates
# ---------------------------------------------------------------------------

def check_binary_success(value):
    outcome = int(value)
    if outcome not in (0, 1):
        raise ProtocolError(f"binary success must be 0/1, got {value!r}")
    return outcome


def require_complete(episodes, methods, seeds, scenes):
    expected = {(m, int(s), int(c)) for m in methods for s in seeds
                for c in scenes}
    seen = {}
    for episode in episodes:
        key = (episode["method"], int(episode["seed"]), int(episode["scene"]))
        if key in seen:
            raise ProtocolError(f"duplicate episode {key}")
        seen[key] = episode
    missing = sorted(expected - set(seen))
    if missing:
        raise ProtocolError(f"MISSING_EPISODE: {len(missing)} missing, "
                            f"first: {missing[:5]}")
    extra = sorted(set(seen) - expected)
    if extra:
        raise ProtocolError(f"unexpected episodes: {extra[:5]}")
    return True


def require_eval_metadata(metadata):
    for key in ("design_id", "protocol", "seeds", "scenes", "methods"):
        if key not in metadata:
            raise ProtocolError(f"eval metadata missing {key}")
    return True
