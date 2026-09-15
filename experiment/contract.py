import hashlib
import json
import os
import subprocess
import tempfile

import numpy as np

SOURCE_FILES = ["core.py", "hri_adapter.py", "pilot.py", "diagnose.py",
                "analyze.py", "sanity.py", "preflight.py", "lock_protocol.py",
                "contract.py", "smoke.py", "run_comparison.sh", "config.json"]
UPSTREAM_FILES = ["external/models/unet.py", "external/models/resnet.py",
                  "external/models/pusht.py"]
PACKAGES = ["torch", "torchvision", "numpy", "scipy", "pandas", "gym",
            "pygame", "pymunk", "shapely", "cv2", "skimage", "zarr",
            "diffusers", "imageio_ffmpeg"]
CACHE_FINGERPRINT_KEYS = ["source", "teacher_steps", "execute_steps",
                          "episode_steps", "contexts_per_episode",
                          "finite_difference_epsilon", "legacy",
                          "clip_actions", "train_scene_start",
                          "train_episodes", "validation_scene_start",
                          "validation_episodes", "development_scene_start",
                          "development_episodes", "test_scene_start",
                          "test_episodes", "repository"]
SCENE_RANGES = {
    "train": ("train_scene_start", "train_episodes"),
    "validation": ("validation_scene_start", "validation_episodes"),
    "development": ("development_scene_start", "development_episodes"),
    "test": ("test_scene_start", "test_episodes"),
}
POSITIVE_INTS = ["teacher_steps", "execute_steps", "episode_steps",
                 "contexts_per_episode", "batch_size", "width", "num_probes",
                 "metric_contexts", "updates", "train_episodes",
                 "validation_episodes", "development_episodes", "test_episodes"]


def experiment_dir():
    return os.path.dirname(os.path.abspath(__file__))


def sha256_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError("missing file " + path)
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
    root = os.path.dirname(here)
    hashes = _hash_many([os.path.join(here, name) for name in SOURCE_FILES])
    return {os.path.relpath(k, root): v for k, v in hashes.items()}


def upstream_hashes(repository):
    return _hash_many([os.path.join(repository, name) for name in UPSTREAM_FILES])


def upstream_commit(repository):
    try:
        out = subprocess.run(["git", "-C", repository, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def package_versions(names=None):
    out = {}
    for name in (names or PACKAGES):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "present")
        except Exception as error:
            out[name] = "missing: " + str(error)
    return out


def combined_id(parts):
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def protocol_id(config, config_path, cache_path, checkpoint_path,
                normalizer_path, repository):
    parts = dict(config=source_hashes(),
                 config_file=sha256_file(config_path),
                 cache=sha256_file(cache_path),
                 checkpoint=sha256_file(checkpoint_path),
                 normalizer=sha256_file(normalizer_path),
                 upstream=upstream_hashes(repository),
                 upstream_commit=upstream_commit(repository),
                 resolved=dict(config))
    return combined_id(parts), parts


def validate_config(config, config_path="config.json"):
    errors = []
    for key in POSITIVE_INTS:
        if key not in config:
            errors.append("missing key " + key)
            continue
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{key} must be a positive integer, got {value!r}")
    if not errors:
        if config["teacher_steps"] < 2 or config["teacher_steps"] % 2 != 0:
            errors.append("teacher_steps must be an even integer >= 2")
        if not 1 <= config["execute_steps"] <= 16:
            errors.append("execute_steps must satisfy 1 <= execute_steps <= 16")
        if config["contexts_per_episode"] < 1:
            errors.append("contexts_per_episode must be >= 1")
        if config["batch_size"] % 2 != 0:
            errors.append("batch_size must be even to split marked and unmarked halves")
        if config["num_probes"] < 1:
            errors.append("num_probes must be >= 1")
    epsilon = config.get("finite_difference_epsilon")
    if epsilon is None or not np.isfinite(epsilon) or epsilon <= 0:
        errors.append("finite_difference_epsilon must be positive and finite")
    for key in ("metric_weight", "anchor_weight"):
        value = config.get(key)
        if value is None or not np.isfinite(value) or value < 0:
            errors.append(f"{key} must be finite and >= 0")
    lr = config.get("learning_rate")
    if lr is None or not np.isfinite(lr) or lr <= 0:
        errors.append("learning_rate must be positive and finite")
    wd = config.get("weight_decay")
    if wd is None or not np.isfinite(wd) or wd < 0:
        errors.append("weight_decay must be finite and >= 0")
    if config.get("source") not in ("gaussian", "uniform"):
        errors.append("source must be gaussian or uniform")
    if config.get("metric_mode", "exact") not in ("exact", "sketch"):
        errors.append("metric_mode must be exact or sketch")
    for key in ("train_scene_start", "validation_scene_start",
                "development_scene_start", "test_scene_start"):
        if not isinstance(config.get(key), int):
            errors.append(f"{key} must be an integer")
    errors.extend(validate_split_ranges(config)["errors"])
    if errors:
        raise ValueError("invalid configuration " + config_path + ": " +
                         "; ".join(errors))
    return True


def scene_range(config, split):
    start_key, count_key = SCENE_RANGES[split]
    start = int(config[start_key])
    return set(range(start, start + int(config[count_key])))


def validate_split_ranges(config):
    sets = {name: scene_range(config, name) for name in SCENE_RANGES}
    errors = []
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sorted(sets[a] & sets[b])
            if overlap:
                errors.append(f"{a} and {b} scene ranges overlap: {overlap[:5]}")
    return dict(sets=sets, errors=errors)


def validate_scene_sets(scene_records):
    seen = {}
    errors = []
    for split, scenes in scene_records.items():
        for scene in scenes:
            scene = int(scene)
            if scene in seen and seen[scene] != split:
                errors.append(f"scene {scene} appears in {seen[scene]} and {split}")
            seen[scene] = split
    if errors:
        raise ValueError("scene split overlap: " + "; ".join(errors))
    return True


def cache_fingerprint(config, repository, extra=None):
    fingerprint = {key: config.get(key) for key in CACHE_FINGERPRINT_KEYS}
    fingerprint["upstream"] = upstream_hashes(repository)
    fingerprint["upstream_commit"] = upstream_commit(repository)
    fingerprint["feature_version"] = 1
    if extra:
        fingerprint.update(extra)
    return fingerprint


def write_meta(path, payload):
    _atomic_json(path, payload)
    return path


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
        marker = path + ".incomplete"
        if os.path.exists(marker):
            os.remove(marker)
    except Exception:
        handle.close()
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise
    return path


def atomic_savez(path, **arrays):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=directory, delete=False,
                                         suffix=".npz")
    handle.close()
    try:
        np.savez(handle.name, **arrays)
        os.replace(handle.name, path)
        marker = path + ".incomplete"
        if os.path.exists(marker):
            os.remove(marker)
    except Exception:
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise
    return path


def atomic_save_torch(path, payload):
    import torch
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=directory, delete=False,
                                         suffix=".pt")
    handle.close()
    try:
        torch.save(payload, handle.name)
        os.replace(handle.name, path)
        marker = path + ".incomplete"
        if os.path.exists(marker):
            os.remove(marker)
    except Exception:
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise
    return path


def mark_incomplete(path, reason):
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path + ".incomplete", "w") as handle:
        handle.write(reason + "\n")
    return path + ".incomplete"


def reserve_outputs(paths):
    taken = [p for p in paths if p and (os.path.exists(p) or
                                        os.path.exists(p + ".incomplete"))]
    if taken:
        raise ValueError("output path already exists, refusing to overwrite: " +
                         ", ".join(sorted(taken)))
    for path in paths:
        if path:
            parent = os.path.dirname(os.path.abspath(path)) or "."
            os.makedirs(parent, exist_ok=True)
    return True


def check_manifest():
    here = experiment_dir()
    missing = [name for name in SOURCE_FILES
               if not os.path.exists(os.path.join(here, name))]
    tracked, untracked = [], []
    repo = None
    try:
        out = subprocess.run(["git", "-C", os.path.dirname(here),
                              "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            repo = out.stdout.strip()
    except Exception:
        repo = None
    if repo:
        listing = subprocess.run(["git", "-C", repo, "ls-files", here],
                                 capture_output=True, text=True, timeout=30)
        names = {os.path.basename(p) for p in listing.stdout.split()}
        ignored = set()
        for name in SOURCE_FILES:
            (tracked if name in names else untracked).append(name)
        # distinguish "not tracked because ignored" from "not tracked yet"
        check = subprocess.run(["git", "-C", repo, "check-ignore"] +
                               [os.path.join(here, n) for n in untracked],
                               capture_output=True, text=True)
        ignored = {os.path.basename(line) for line in check.stdout.split()}
        return dict(dir=here, source=repo, missing=missing, tracked=sorted(tracked),
                    untracked=sorted(untracked), ignored=sorted(ignored),
                    complete=not missing,
                    delivery_risk=sorted(set(untracked)),
                    note=("commit or archive every file listed in delivery_risk; a "
                          "git-based transfer of tracked files alone is insufficient"))
    return dict(dir=here, source="no_git", missing=missing, tracked=[],
                untracked=list(SOURCE_FILES), ignored=[], complete=not missing,
                delivery_risk=list(SOURCE_FILES),
                note=("no git repository here; ship the explicit file manifest "
                      "and verify it after transfer"))


def write_manifest(path):
    here = experiment_dir()
    entries = {name: sha256_file(os.path.join(here, name)) for name in SOURCE_FILES}
    payload = dict(experiment=os.path.basename(os.path.dirname(here)),
                   dir=here, files=entries, count=len(entries),
                   upstream={k: v for k, v in
                             upstream_hashes(os.path.join(os.path.dirname(here),
                                                          "external", "flow_matching")).items()},
                   required_from_upstream=list(UPSTREAM_FILES),
                   packages=package_versions())
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _atomic_json(path, payload)
    return payload


def verify_manifest(path, experiment_dir_override=None):
    with open(path) as handle:
        payload = json.load(handle)
    here = experiment_dir_override or experiment_dir()
    problems = []
    for name, digest in payload["files"].items():
        target = os.path.join(here, name)
        if not os.path.exists(target):
            problems.append(f"missing {name}")
        elif sha256_file(target) != digest:
            problems.append(f"modified {name}")
    if problems:
        raise ValueError("delivered manifest does not match: " + "; ".join(problems))
    return dict(verified=len(payload["files"]), dir=here)


def verify_tensors(arrays, where):
    for name, value in arrays.items():
        if value is None:
            continue
        if not np.isfinite(np.asarray(value, dtype=np.float64)).all():
            raise ValueError(f"nonfinite values in {where}:{name}")
    return True


def verify_cache(data, config, repository, expect_scenes=None,
                 expect_mode="shared", allow_legacy=False):
    if "contexts" not in data:
        raise ValueError("cache lacks context array")
    fingerprint = json.loads(str(data["fingerprint"])) if "fingerprint" in data else None
    if fingerprint is None:
        if not allow_legacy:
            raise ValueError("cache lacks a contract fingerprint; rebuild it")
    else:
        expected = cache_fingerprint(config, repository)
        mismatch = {k: {"cached": fingerprint.get(k), "expected": v}
                    for k, v in expected.items() if fingerprint.get(k) != v}
        if mismatch:
            raise ValueError("cache contract mismatch: " + json.dumps(mismatch,
                                                                      default=str))
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    if expect_mode and meta.get("mode") != expect_mode:
        raise ValueError(f"cache mode {meta.get('mode')!r} != expected {expect_mode!r}")
    contexts = list(data["contexts"])
    if not contexts:
        raise ValueError("cache is empty")
    scenes = sorted({int(c["scene"]) for c in contexts})
    if expect_scenes is not None:
        unknown = sorted(set(scenes) - set(int(s) for s in expect_scenes))
        if unknown:
            raise ValueError(f"cache contains unplanned scenes: {unknown[:5]}")
    for index, context in enumerate(contexts):
        verify_tensors({k: context.get(k) for k in
                        ("condition", "noise", "student_mid", "history")},
                       f"cache context {index}")
        if context.get("student_mid") is None:
            raise ValueError(
                "cache contexts lack the warm student midpoint; recollect them "
                "from a trained warm student (teacher collection is warm-start only)")
    return dict(scenes=scenes, n_contexts=len(contexts), fingerprint=fingerprint,
                meta=meta)


def verify_metric_cache(data, config, repository, expect_scenes=None):
    if "contexts" not in data:
        raise ValueError("metric cache lacks context array")
    fingerprint = json.loads(str(data["fingerprint"])) if "fingerprint" in data else None
    if fingerprint is None:
        raise ValueError("metric cache lacks a contract fingerprint; rebuild it")
    expected = cache_fingerprint(config, repository,
                                 extra=metric_extra_keys(config))
    mismatch = {k: {"cached": fingerprint.get(k), "expected": v}
                for k, v in expected.items() if fingerprint.get(k) != v}
    if mismatch:
        raise ValueError("metric cache contract mismatch: " + json.dumps(
            mismatch, default=str))
    contexts = list(data["contexts"])
    if not contexts:
        raise ValueError("metric cache is empty")
    scenes = sorted({int(c["scene"]) for c in contexts})
    if expect_scenes is not None:
        unknown = sorted(set(scenes) - set(int(s) for s in expect_scenes))
        if unknown:
            raise ValueError(f"metric cache contains unplanned scenes: {unknown[:5]}")
    mode = config.get("metric_mode", "exact")
    required = ["condition", "noise", "midpoint", "target_mid", "target_end",
                "teacher_end", "jacobian_start", "jacobian_end"]
    required.append("metric_exact" if mode == "exact" else "probes")
    for index, context in enumerate(contexts):
        missing = [k for k in required if k not in context]
        if missing:
            raise ValueError(f"metric context {index} missing {missing}")
        verify_tensors({k: np.asarray(context[k]) for k in required},
                       f"metric context {index}")
    return dict(scenes=scenes, n_contexts=len(contexts), metric_mode=mode,
                fingerprint=fingerprint)


def metric_extra_keys(config):
    return {"metric_mode": config.get("metric_mode", "exact"),
            "num_probes": config.get("num_probes"),
            "execute_steps": config.get("execute_steps"),
            "teacher_steps": config.get("teacher_steps")}


def require_protocol(path, expect_id=None):
    if not os.path.exists(path):
        raise ValueError("protocol lock missing: " + path +
                         " (create it before evaluation or analysis)")
    with open(path) as handle:
        protocol = json.load(handle)
    if expect_id is not None and protocol.get("protocol_id") != expect_id:
        raise ValueError(f"protocol id mismatch: {protocol.get('protocol_id')} "
                         f"!= {expect_id}")
    return protocol


def _records(frame):
    if hasattr(frame, "to_dict"):
        return frame.to_dict("records")
    return list(frame)


def require_complete(frame, protocol, method, baseline, columns=("training_seed", "scene")):
    records = _records(frame)
    expected_methods = list(protocol["modes"])
    expected_seeds = [int(s) for s in protocol["seeds"]]
    expected_scenes = [int(s) for s in protocol["final_scenes"]]
    if method not in expected_methods or baseline not in expected_methods:
        raise ValueError(f"primary contrast {method} vs {baseline} not in the "
                         f"locked modes {expected_methods}")
    expected_pairs = {(s, c) for s in expected_seeds for c in expected_scenes}
    for name in (method, baseline):
        rows = [r for r in records if str(r.get("method")) == name]
        if not rows:
            raise ValueError(f"{name}: no evaluation rows at all")
        present_seeds = sorted({int(r[columns[0]]) for r in rows})
        missing_seeds = sorted(set(expected_seeds) - set(present_seeds))
        if missing_seeds:
            raise ValueError(f"{name}: missing training seeds {missing_seeds}")
        present_scenes = sorted({int(r[columns[1]]) for r in rows})
        missing_scenes = sorted(set(expected_scenes) - set(present_scenes))
        extra_scenes = sorted(set(present_scenes) - set(expected_scenes))
        if missing_scenes:
            raise ValueError(f"{name}: missing scenes {missing_scenes[:5]} "
                             f"({len(missing_scenes)} total)")
        if extra_scenes:
            raise ValueError(f"{name}: unplanned scenes {extra_scenes[:5]}")
        observed = {(int(r[columns[0]]), int(r[columns[1]])) for r in rows}
        missing_pairs = sorted(expected_pairs - observed)
        if missing_pairs:
            raise ValueError(f"{name}: {len(missing_pairs)} planned evaluation "
                             f"episodes missing, e.g. {missing_pairs[:5]}")
        if len(observed) != len(expected_pairs):
            raise ValueError(f"{name}: {len(observed)} rows for "
                             f"{len(expected_pairs)} planned episodes (duplicates?)")
    return True


def require_eval_metadata(frame, protocol, inputs, method=None):
    records = _records(frame)
    present_methods = {str(r.get("method")) for r in records}
    mode_of = {}
    for path in inputs:
        sidecar = path.replace(".jsonl", ".meta.json")
        if not os.path.exists(sidecar):
            raise ValueError("evaluation metadata missing: " + sidecar)
        with open(sidecar) as handle:
            meta = json.load(handle)
        if meta.get("protocol_id") != protocol.get("protocol_id"):
            raise ValueError(f"{sidecar}: protocol id {meta.get('protocol_id')} "
                             f"!= locked {protocol.get('protocol_id')}")
        mode = str(meta.get("intrinsic_mode"))
        if mode in mode_of:
            raise ValueError(f"two evaluation files claim intrinsic mode {mode}")
        if mode not in present_methods:
            raise ValueError(f"{sidecar} claims intrinsic mode {mode} but no rows carry it")
        for key in ("checkpoint_sha256", "intrinsic_seed"):
            if key not in meta:
                raise ValueError(f"metadata for {mode} lacks {key}")
        mode_of[mode] = meta
    if method is not None and method not in mode_of:
        raise ValueError(f"no evaluation metadata with intrinsic mode {method}")
    return mode_of
