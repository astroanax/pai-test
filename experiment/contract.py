import hashlib
import json
import os
import subprocess
import tempfile

import numpy as np

SOURCE_FILES = ["core.py", "hri_adapter.py", "pilot.py", "diagnose.py",
                "analyze.py", "sanity.py", "preflight.py", "lock_protocol.py",
                "contract.py", "smoke.py", "run_comparison.sh", "vertical_test.sh",
                "config.json"]
UPSTREAM_FILES = ["external/models/unet.py", "external/models/resnet.py",
                  "external/models/pusht.py"]
PACKAGES = ["torch", "torchvision", "numpy", "scipy", "pandas", "gym",
            "pygame", "pymunk", "shapely", "cv2", "skimage", "zarr",
            "diffusers", "imageio_ffmpeg"]
SCHEMA_VERSION = 2
CACHE_FINGERPRINT_KEYS = ["source", "teacher_steps", "execute_steps",
                          "episode_steps", "contexts_per_episode",
                          "finite_difference_epsilon", "legacy",
                          "clip_actions", "train_scene_start",
                          "train_episodes", "validation_scene_start",
                          "validation_episodes", "development_scene_start",
                          "development_episodes", "test_scene_start",
                          "test_episodes", "repository", "metric_mode",
                          "num_probes"]
SCENE_RANGES = {
    "train": ("train_scene_start", "train_episodes"),
    "validation": ("validation_scene_start", "validation_episodes"),
    "development": ("development_scene_start", "development_episodes"),
    "test": ("test_scene_start", "test_episodes"),
}
CUDNN_ENV_FLAG = "EXECPB_DISABLE_CUDNN"


def cudnn_healthy(device="cuda", timeout_s=60):
    """Probe whether convolutions actually run on the requested device.

    Importing torch succeeds even when the driver/cuDNN combination cannot
    initialize; this probe catches that before the experiment spends its
    budget. It uses the requested device (not a hard-coded one), synchronizes
    before declaring success, checks the subprocess return code, and
    distinguishes "native cuDNN works" from "the selected fallback works".
    Returns a dict with healthy True/False and a recommendation.
    """
    import subprocess as _sp
    import sys as _sys
    code = (
        "import torch, json; "
        f"device = torch.device({device!r}); "
        "x = torch.randn(1,3,32,32, device=device); "
        "w = torch.randn(8,3,3,3, device=device); "
        "y = torch.nn.functional.conv2d(x, w); "
        "torch.cuda.synchronize() if device.type == 'cuda' else None; "
        "print('conv_ok:' + str(bool(torch.backends.cudnn.is_available() and "
        "torch.backends.cudnn.enabled))); "
        "print('cudnn_enabled=' + str(torch.backends.cudnn.enabled))"
    )
    try:
        out = _sp.run([_sys.executable, "-c", code], capture_output=True,
                      text=True, timeout=timeout_s)
    except Exception as error:
        return dict(healthy=False, error=f"probe failed: {error}",
                    recommendation="rerun on a host with working cuDNN")
    if out.returncode == 0 and "conv_ok:True" in out.stdout:
        return dict(healthy=True, native_cudnn=True)
    if out.returncode == 0 and "conv_ok:False" in out.stdout:
        return dict(healthy=True, native_cudnn=False,
                    note="convolution runs with the selected fallback, not "
                         "native cuDNN; latency comparisons require identical "
                         "settings")
    tail = (out.stderr or out.stdout)[-800:]
    return dict(healthy=False, error=tail,
                recommendation="set EXECPB_DISABLE_CUDNN=1 to run the "
                               "experiment with cuDNN disabled")


def cudnn_disabled():
    import os as _os
    return _os.environ.get(CUDNN_ENV_FLAG, "").strip() in ("1", "true", "yes")


def apply_compute_env(device="cuda"):
    """Apply the compute environment: disable cuDNN process-wide when requested.

    cuDNN is enabled by default; only an explicit EXECPB_DISABLE_CUDNN=1 opts
    out. The full resolved settings (cuDNN availability/enabled/version, TF32,
    deterministic algorithms, device, hardware) are recorded in every smoke
    report and student sidecar so a fallback run is never mistaken for a
    full-cuDNN run, and latency comparisons can require identical settings.
    Returns the resolved settings dict.
    """
    import os as _os
    disabled = cudnn_disabled()
    import torch as _torch
    if disabled:
        _torch.backends.cudnn.enabled = False
        _torch.backends.cudnn.benchmark = False
    try:
        hardware = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "cpu"
    except Exception:
        hardware = "unknown"
    return dict(cudnn_disabled=bool(disabled), flag=CUDNN_ENV_FLAG,
                cudnn_available=bool(_torch.backends.cudnn.is_available()),
                cudnn_enabled=bool(_torch.backends.cudnn.enabled),
                cudnn_version=(_torch.backends.cudnn.version()
                               if _torch.backends.cudnn.is_available() else None),
                tf32_allow=_torch.backends.cuda.matmul.allow_tf32,
                deterministic=_torch.are_deterministic_algorithms_enabled(),
                device=str(device), hardware=hardware,
                # torch 2.6+ exposes __version__ as a TorchVersion object, which
                # weights_only loading rejects: store a plain str instead.
                torch_version=str(getattr(_torch, "__version__", "unknown")))


POSITIVE_INTS = ["teacher_steps", "execute_steps", "episode_steps",
                 "contexts_per_episode", "batch_size", "width", "num_probes",
                 "metric_contexts", "validation_metric_contexts", "updates",
                 "train_episodes", "validation_episodes", "development_episodes",
                 "test_episodes"]
CONTEXT_LABEL_KEYS = ["condition", "noise", "midpoint", "target_mid",
                      "target_end", "teacher_end"]
CONTEXT_METRIC_KEYS = ["jacobian_start", "jacobian_end"]
COMPLETION_SUFFIX = ".complete.json"


class ProtocolError(ValueError):
    pass


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


def asset_hashes(config):
    out = {}
    for key in ("checkpoint", "normalizer"):
        path = config.get(key)
        if path and os.path.exists(path):
            out[key] = sha256_file(path)
        else:
            out[key] = "absent"
    return out


def canonical_choices(config, modes=None, seeds=None, updates=None, warm=None,
                      cache=None, smoke_report=None, final_scenes=None,
                      primary_contrast=None, compute=None):
    """Every experimental choice that must be part of the canonical identifier.

    The final scene list, the primary contrast, and the resolved compute
    settings are included: changing any of them without regenerating the lock
    changes the identifier, so stale locks cannot be reused silently.
    """
    return dict(config_sha256=combined_id(config),
                modes=None if modes is None else sorted(modes),
                seeds=None if seeds is None else sorted(int(s) for s in seeds),
                updates=None if updates is None else int(updates),
                final_scenes=(None if final_scenes is None
                              else sorted(int(s) for s in final_scenes)),
                primary_contrast=(None if primary_contrast is None
                                  else [str(primary_contrast[0]),
                                        str(primary_contrast[1])]),
                compute=compute,
                metric_mode=config.get("metric_mode", "exact"),
                teacher_steps=int(config.get("teacher_steps", 0)),
                source=config.get("source"),
                warm_sha256=(sha256_file(warm) if warm and os.path.exists(warm)
                             else "absent"),
                cache_sha256=(sha256_file(cache) if cache and os.path.exists(cache)
                              else "absent"),
                smoke_report_sha256=(sha256_file(smoke_report)
                                     if smoke_report and os.path.exists(smoke_report)
                                     else "absent"))


def protocol_id(config, config_path, cache_path, warm_path, modes, seeds, updates,
                smoke_report=None, repository=None, final_scenes=None,
                primary_contrast=None, compute=None):
    repository = repository or config["repository"]
    choices = canonical_choices(config, modes, seeds, updates, warm_path,
                                cache_path, smoke_report,
                                final_scenes=final_scenes,
                                primary_contrast=primary_contrast,
                                compute=compute)
    parts = dict(config=source_hashes(),
                 config_file=sha256_file(config_path),
                 cache=sha256_file(cache_path),
                 warm=sha256_file(warm_path) if warm_path and os.path.exists(warm_path) else "absent",
                 smoke_report=(sha256_file(smoke_report)
                               if smoke_report and os.path.exists(smoke_report)
                               else "absent"),
                 assets=asset_hashes(config),
                 upstream=upstream_hashes(repository),
                 upstream_commit=upstream_commit(repository),
                 choices=choices,
                 resolved=config)
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
        if config["batch_size"] % 2 != 0:
            errors.append("batch_size must be even to split marked and unmarked halves")
        if config["native_fast_steps"] < 1:
            errors.append("native_fast_steps must be >= 1")
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
    for key in ("source", "native_fast_source"):
        if config.get(key) not in ("gaussian", "uniform", "uniform_symmetric"):
            errors.append(f"{key} must be gaussian, uniform, or uniform_symmetric")
    if config.get("metric_mode", "exact") not in ("exact", "sketch"):
        errors.append("metric_mode must be exact or sketch")
    for key in ("train_scene_start", "validation_scene_start",
                "development_scene_start", "test_scene_start"):
        if not isinstance(config.get(key), int):
            errors.append(f"{key} must be an integer")
    for key in ("checkpoint", "normalizer", "repository"):
        if not config.get(key):
            errors.append("missing path key " + key)
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


def split_of_scene(config, scene):
    scene = int(scene)
    for name in SCENE_RANGES:
        if scene in scene_range(config, name):
            return name
    raise ValueError(f"scene {scene} is not inside any configured split range")


def cache_fingerprint(config, repository=None, extra=None):
    repository = repository or config["repository"]
    fingerprint = {key: config.get(key) for key in CACHE_FINGERPRINT_KEYS}
    fingerprint["schema_version"] = SCHEMA_VERSION
    fingerprint["upstream"] = upstream_hashes(repository)
    fingerprint["upstream_commit"] = upstream_commit(repository)
    fingerprint["assets"] = asset_hashes(config)
    fingerprint["teacher_checkpoint"] = fingerprint["assets"].get("checkpoint")
    fingerprint["normalizer"] = fingerprint["assets"].get("normalizer")
    fingerprint["feature_version"] = 1
    if extra:
        fingerprint.update(extra)
    return fingerprint


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
        handle.close()
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise
    return path


def write_meta(path, payload):
    return _atomic_json(path, payload)


def atomic_savez(path, **arrays):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("wb", dir=directory, delete=False,
                                         suffix=".npz")
    handle.close()
    try:
        np.savez(handle.name, **arrays)
        os.replace(handle.name, path)
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
    except Exception:
        if os.path.exists(handle.name):
            os.remove(handle.name)
        raise
    return path


def reserve_outputs(paths):
    """Reserve output paths by creating a `.reserved` claim file.

    Existence of the data file is not enough information: a partial artifact or a
    crashed run must also block reuse. The claim is replaced by a completion
    record when the producer finishes, and by an `.incomplete` marker on failure.
    Claims are created exclusively (O_EXCL): two concurrent writers cannot both
    hold the same output.
    """
    paths = [p for p in paths if p]
    taken = []
    for path in paths:
        for marker in (path, path + ".reserved", path + ".incomplete"):
            if os.path.exists(marker):
                taken.append(marker)
    if taken:
        raise ValueError("output already exists or is reserved, refusing to "
                         "overwrite: " + ", ".join(sorted(taken)))
    for path in paths:
        parent = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(parent, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            handle = os.open(path + ".reserved", flags)
        except FileExistsError:
            raise ValueError("output claimed by a concurrent writer: " + path)
        with os.fdopen(handle, "w") as stream:
            stream.write(json.dumps(dict(state="reserved")))
    return paths


def complete_output(path, payload=None):
    detail = dict(payload or {})
    if os.path.exists(path):
        detail.setdefault("sha256", sha256_file(path))
        detail.setdefault("bytes", os.path.getsize(path))
    else:
        detail.setdefault("bytes", 0)
    _atomic_json(path + COMPLETION_SUFFIX,
                 dict(state="complete", artifact=os.path.basename(path),
                      detail=detail))
    for marker in (path + ".reserved", path + ".incomplete"):
        if os.path.exists(marker):
            os.remove(marker)
    return path + COMPLETION_SUFFIX


def mark_incomplete(path, reason):
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    _atomic_json(path + ".incomplete", dict(state="incomplete", reason=str(reason)))
    for marker in (path + ".reserved", path + COMPLETION_SUFFIX):
        if os.path.exists(marker):
            os.remove(marker)
    return path + ".incomplete"


def is_complete(path):
    record = path + COMPLETION_SUFFIX
    if not os.path.exists(record):
        return False
    try:
        with open(record) as handle:
            return json.load(handle).get("state") == "complete"
    except Exception:
        return False


def verify_completed(path, role=None, protocol=None):
    """Verify a skipped artifact before reuse: completion record, existence,
    byte hash, role, and protocol binding.

    Runners must call this instead of trusting marker existence.
    """
    record = path + COMPLETION_SUFFIX
    if not os.path.exists(record):
        raise ValueError(f"{path} has no completion record; rebuild it")
    with open(record) as handle:
        marker = json.load(handle)
    if marker.get("state") != "complete":
        raise ValueError(f"{path} completion state is {marker.get('state')!r}")
    if not os.path.exists(path):
        raise ValueError(f"{path} completed but the artifact is missing")
    recorded = (marker.get("detail") or {}).get("sha256")
    if recorded and sha256_file(path) != recorded:
        raise ValueError(f"{path} bytes differ from its completion record; "
                         "rebuild it")
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
        for name in SOURCE_FILES:
            (tracked if name in names else untracked).append(name)
        check = subprocess.run(["git", "-C", repo, "check-ignore"] +
                               [os.path.join(here, n) for n in untracked],
                               capture_output=True, text=True)
        ignored = {os.path.basename(line) for line in check.stdout.split()}
        return dict(dir=here, source=repo, missing=missing, tracked=sorted(tracked),
                    untracked=sorted(untracked), ignored=sorted(ignored),
                    complete=not missing, delivery_risk=sorted(set(untracked)),
                    note=("commit or archive every file in delivery_risk; a git "
                          "transfer of tracked files alone is insufficient"))
    return dict(dir=here, source="no_git", missing=missing, tracked=[],
                untracked=list(SOURCE_FILES), ignored=[], complete=not missing,
                delivery_risk=list(SOURCE_FILES),
                note=("no git repository here; ship the explicit manifest and "
                      "verify it after transfer"))


def write_manifest(path, experiment_dir_override=None):
    here = experiment_dir_override or experiment_dir()
    entries = {name: sha256_file(os.path.join(here, name)) for name in SOURCE_FILES}
    payload = dict(schema_version=SCHEMA_VERSION,
                   experiment=os.path.basename(os.path.dirname(here)),
                   dir=here, files=entries, count=len(entries),
                   packages=package_versions())
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



def write_label_cache(path, contexts, config, meta, lineage, extra=None):
    """Producer-side schema writer, shared by collection and the schema test."""
    fingerprint = cache_fingerprint(config, config.get("repository"),
                                    extra=dict(source=meta.get("source"),
                                               has_labels=True, has_metrics=False,
                                               **(extra or {})))
    atomic_savez(path, contexts=np.array(contexts, dtype=object),
                 meta=json.dumps(meta), fingerprint=json.dumps(fingerprint),
                 lineage=json.dumps(lineage))
    write_meta(path + ".meta.json", dict(meta, fingerprint=fingerprint,
                                         lineage=lineage))
    return fingerprint


def write_metric_cache(path, contexts, config, meta, lineage):
    fingerprint = cache_fingerprint(
        config, config.get("repository"),
        extra=dict(source=meta.get("source"), has_labels=True, has_metrics=True,
                   metric_mode=meta.get("metric_mode"),
                   num_probes=config.get("num_probes")))
    atomic_savez(path, contexts=np.array(contexts, dtype=object),
                 meta=json.dumps(meta), fingerprint=json.dumps(fingerprint),
                 lineage=json.dumps(lineage))
    write_meta(path + ".meta.json", dict(meta, fingerprint=fingerprint,
                                         lineage=lineage))
    return fingerprint


def verify_tensors(arrays, where):
    for name, value in arrays.items():
        if value is None:
            continue
        if not np.isfinite(np.asarray(value, dtype=np.float64)).all():
            raise ValueError(f"nonfinite values in {where}:{name}")
    return True


def _load_cache(path):
    data = np.load(path, allow_pickle=True)
    if "contexts" not in data:
        raise ValueError(
            f"cache {path} lacks a top-level `contexts` array; schema v{SCHEMA_VERSION} "
            "writes flattened contexts (rebuild the cache with this version)")
    payload = dict(contexts=list(data["contexts"]))
    for key in ("meta", "fingerprint", "lineage"):
        payload[key] = (json.loads(str(data[key])) if key in data else None)
    return payload


def load_cache(path):
    return _load_cache(path)


def _check_fingerprint(stored, expected, kind):
    if stored is None:
        raise ValueError(f"{kind} lacks a contract fingerprint; rebuild it")
    mismatch = {k: {"cached": stored.get(k), "expected": v}
                for k, v in expected.items() if stored.get(k) != v}
    if mismatch:
        raise ValueError(f"{kind} contract mismatch: " +
                         json.dumps(mismatch, default=str))


def verify_label_cache(path_or_data, config, expect_mode=None,
                       allowed_scenes=None, require_metric=False):
    """Validate a flattened cache produced by `pilot.py collect`.

    Label caches carry the two teacher targets and the teacher endpoint, which is
    everything `uniform` and `prefix` training need. Metric caches additionally
    carry the physical Jacobians and the transported metric.
    """
    data = _load_cache(path_or_data) if isinstance(path_or_data, str) else path_or_data
    contexts = data["contexts"]
    if not contexts:
        raise ValueError("cache is empty")
    _check_fingerprint(data["fingerprint"], cache_fingerprint(config),
                       "cache")
    meta = data["meta"] or {}
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"cache schema {meta.get('schema_version')} != "
                         f"{SCHEMA_VERSION}; rebuild the cache")
    lineage = data.get("lineage") or {}
    if lineage.get("source_hashes") != source_hashes():
        raise ValueError("cache was built by different experiment sources "
                         "(collection/normalization/metric implementation "
                         "changed); rebuild it rather than reusing old data")
    if lineage.get("epsilon") != config.get("finite_difference_epsilon") and \
            require_metric:
        raise ValueError("metric cache epsilon differs from the configuration; "
                         "rebuild it")
    if expect_mode and meta.get("mode") != expect_mode:
        raise ValueError(f"cache mode {meta.get('mode')!r} != expected {expect_mode!r}")
    metric_key = "metric_exact" if config.get("metric_mode", "exact") == "exact" \
        else "probes"
    splits = {}
    metric_rows = 0
    for index, context in enumerate(contexts):
        missing = [k for k in CONTEXT_LABEL_KEYS + ["scene", "split", "history"]
                   if k not in context]
        if missing:
            raise ValueError(f"cache context {index} missing {missing}")
        has_metric = context.get("has_metric", metric_key in context)
        if has_metric:
            metric_rows += 1
            missing = [k for k in CONTEXT_METRIC_KEYS + [metric_key]
                       if k not in context]
            if missing:
                raise ValueError(f"metric cache context {index} missing {missing}")
        scene = int(context["scene"])
        expected_split = split_of_scene(config, scene)
        if context["split"] != expected_split:
            raise ValueError(f"cache context {index}: scene {scene} labelled "
                             f"{context['split']!r} but is in {expected_split!r}")
        if allowed_scenes is not None and scene not in allowed_scenes:
            raise ValueError(f"cache contains unplanned scene {scene}")
        splits.setdefault(context["split"], 0)
        splits[context["split"]] += 1
        verify_tensors({k: np.asarray(context[k]) for k in CONTEXT_LABEL_KEYS},
                       f"cache context {index}")
        if has_metric:
            verify_tensors({k: np.asarray(context[k]) for k in
                            CONTEXT_METRIC_KEYS + [metric_key]},
                           f"metric cache context {index}")
        if context.get("history"):
            verify_tensors({"history": np.asarray(context["history"], dtype=np.float64)},
                           f"cache context {index}")
        for key, shape in (("condition", (514,)), ("noise", (16, 2)),
                           ("midpoint", (16, 2)), ("target_mid", (16, 2)),
                           ("target_end", (16, 2)), ("teacher_end", (16, 2))):
            actual_shape = tuple(np.asarray(context[key]).shape)
            if actual_shape != shape:
                raise ValueError(f"cache context {index}: {key} shape "
                                 f"{actual_shape} != {shape}; rebuild the cache")
        if has_metric:
            phys, prefix = 4 * config["execute_steps"], config["execute_steps"] * 2
            for key, shape in (("jacobian_start", (phys, prefix)),
                               ("jacobian_end", (phys, prefix))):
                actual_shape = tuple(np.asarray(context[key]).shape)
                if actual_shape != shape:
                    raise ValueError(f"metric cache context {index}: {key} shape "
                                     f"{actual_shape} != {shape}; rebuild it")
    if require_metric and not metric_rows:
        raise ValueError("metric cache holds no has_metric rows; "
                         "run the metrics stage before metric training")
    return dict(contexts=contexts, meta=meta, splits=splits,
                scenes=sorted({int(c["scene"]) for c in contexts}),
                n_contexts=len(contexts), n_metric=metric_rows)


def select_split(verified, split):
    return [c for c in verified["contexts"] if c["split"] == split]


def metric_rows(verified, config):
    """Rows of a metric cache that carry physical metrics (the linked subset)."""
    metric_key = "metric_exact" if config.get("metric_mode", "exact") == "exact" \
        else "probes"
    return [c for c in verified["contexts"]
            if c.get("has_metric", metric_key in c)]


def metric_subset(verified, config, limit=None, seed=915):
    """Pick the fixed metric subset BEFORE any derivative work.

    Rows are spread round-robin across scenes with a fixed seed, so the subset
    cannot concentrate in the earliest collected episodes. The planned budget is
    `metric_contexts` training rows plus `validation_metric_contexts` validation
    rows. Set `limit` for a small profiling run.
    """
    train = select_split(verified, "train")
    validation = select_split(verified, "validation")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        return train[:limit], []

    def spread(rows, count):
        if not rows or count < 1:
            return []
        by_scene = {}
        for row in rows:
            by_scene.setdefault(int(row["scene"]), []).append(row)
        rng = np.random.default_rng(seed)
        scenes = sorted(by_scene)
        rng.shuffle(scenes)
        picked = []
        round_robin = [list(by_scene[s]) for s in scenes]
        for group in round_robin:
            rng.shuffle(group)
        index = 0
        while len(picked) < min(count, len(rows)):
            for group in round_robin:
                if index < len(group) and len(picked) < min(count, len(rows)):
                    picked.append(group[index])
            index += 1
            if index > max(len(group) for group in round_robin):
                break
        return picked

    return (spread(train, int(config["metric_contexts"])),
            spread(validation, int(config["validation_metric_contexts"])))


def verify_current_protocol(protocol, config, config_path, repository=None,
                            require=True, stage="run"):
    """Recompute every locked fingerprint and compare it with the lock.

    Called by training, final evaluation, and analysis, so a lock cannot go stale
    while work continues.
    """
    if protocol is None:
        if require:
            raise ProtocolError(f"no protocol lock for stage {stage}; "
                                "development stages must pass --allow-unlocked")
        return dict(verified=False, stage=stage, reason="no lock supplied")
    problems = []
    current_sources = source_hashes()
    if protocol.get("source_hashes") != current_sources:
        changed = sorted(set(protocol.get("source_hashes", {}).items()) ^
                         set(current_sources.items()))
        problems.append(f"experiment sources changed ({changed[:3]})")
    repository = repository or config["repository"]
    current_upstream = upstream_hashes(repository)
    if protocol.get("upstream") != current_upstream:
        problems.append("upstream implementation changed")
    if protocol.get("upstream_commit") != upstream_commit(repository):
        problems.append("upstream commit changed")
    if protocol.get("resolved_config") != config:
        problems.append("configuration differs from the lock")
    current_assets = asset_hashes(config)
    if protocol.get("assets") != current_assets:
        problems.append("checkpoint or normalizer bytes changed")
    for key in ("checkpoint", "normalizer"):
        if protocol.get(key + "_sha256") not in (None, current_assets.get(key)):
            problems.append(f"{key} hash differs from the lock")
    current_packages = package_versions(list(protocol.get("packages", {}).keys()) or None)
    if protocol.get("packages") and protocol["packages"] != current_packages:
        differing = {k: (protocol["packages"].get(k), current_packages.get(k))
                     for k in current_packages
                     if protocol["packages"].get(k) != current_packages.get(k)}
        problems.append(f"package versions differ ({differing})")
    for name, key in (("smoke_report", "smoke_report_sha256"),
                      ("cache", "cache_sha256"), ("warm_start", "warm_start_sha256")):
        recorded = protocol.get(key)
        path = protocol.get(name if name != "warm_start" else "warm_start")
        if not recorded or not path:
            problems.append(f"{name} is not recorded in the lock")
        elif not os.path.exists(path):
            problems.append(f"{name} is missing at {path}")
        elif sha256_file(path) != recorded:
            problems.append(f"{name} bytes differ from the lock")
    canonical = protocol.get("canonical_choices")
    if canonical is None:
        problems.append("lock predates canonical-choice binding; regenerate it")
    else:
        rebuilt = combined_id(dict(
            config=protocol.get("config", source_hashes()),
            config_file=(sha256_file(config_path)
                         if os.path.exists(config_path) else "absent"),
            cache=protocol.get("cache_sha256"),
            warm=protocol.get("warm_start_sha256"),
            smoke_report=protocol.get("smoke_report_sha256"),
            assets=protocol.get("assets", asset_hashes(config)),
            upstream=protocol.get("upstream"),
            upstream_commit=protocol.get("upstream_commit"),
            choices=canonical,
            resolved=protocol.get("resolved_config")))
        if rebuilt != protocol.get("protocol_id"):
            problems.append("lock payload was edited without regenerating the "
                            "identifier (scenes, contrast, modes, seeds, "
                            "updates, or compute changed)")
        for field in ("modes", "seeds", "updates", "final_scenes",
                      "primary_contrast"):
            locked = protocol.get(field)
            canon = canonical.get(field)
            norm = (sorted(locked) if isinstance(locked, list)
                    and field in ("modes", "final_scenes") else locked)
            if canon != norm:
                problems.append(f"lock field {field} disagrees with the canonical "
                                "payload; regenerate the lock")
    if problems:
        raise ProtocolError("protocol lock is stale for stage " + stage + ": " +
                            "; ".join(problems))
    return dict(verified=True, stage=stage, protocol_id=protocol.get("protocol_id"))


def load_protocol(path, expect_id=None):
    if not os.path.exists(path):
        raise ProtocolError("protocol lock missing: " + path)
    with open(path) as handle:
        protocol = json.load(handle)
    if expect_id is not None and protocol.get("protocol_id") != expect_id:
        raise ProtocolError(f"protocol id mismatch: {protocol.get('protocol_id')} "
                            f"!= {expect_id}")
    return protocol


def verify_student_provenance(payload, protocol, config, path, role="final"):
    """A checkpoint must belong to the active protocol and its own metadata.

    Roles are validated separately:
    - "init": the file itself must be the locked warm start (hash equality
      with warm_start_sha256). Mode labels are never trusted.
    - "final": full match on protocol id, mode, seed, updates, cache hash,
      and initialization hash.
    - "reference": evaluating a warm-start checkpoint as a behavioral
      reference. The file hash must equal warm_start_sha256 (locked) or the
      protocol must be None with an explicit unlocked development run.

    An unlocked uniform/prefix checkpoint never passes as initialization or
    as a final student merely because its protocol id is None.
    """
    problems = []
    if role == "init":
        if protocol is None:
            problems.append("initialization requires a protocol lock holding "
                            "the warm-start hash")
        elif not path or not os.path.exists(path):
            problems.append("initialization file missing: " + str(path))
        elif sha256_file(path) != protocol.get("warm_start_sha256"):
            problems.append("initialization file hash != locked warm_start_sha256; "
                            "a different unlocked checkpoint cannot initialize "
                            "locked training")
        if payload.get("metric_mode") != config.get("metric_mode", "exact"):
            problems.append("warm-start metric mode differs from the configuration")
        if problems:
            raise ProtocolError("checkpoint provenance mismatch for " + str(path) +
                                ": " + "; ".join(problems))
        return True
    if role == "reference":
        if protocol is not None:
            if not path or not os.path.exists(path):
                problems.append("reference file missing: " + str(path))
            elif sha256_file(path) != protocol.get("warm_start_sha256"):
                problems.append("reference file hash != locked warm_start_sha256")
        elif payload.get("protocol_id") is not None:
            problems.append("reference role with an unlocked protocol requires an "
                            "unlocked warm-start checkpoint")
        if payload.get("mode") not in ("uniform", "prefix"):
            problems.append(f"reference must be a warm-start mode, got "
                            f"{payload.get('mode')!r}")
        if problems:
            raise ProtocolError("reference provenance mismatch for " + str(path) +
                                ": " + "; ".join(problems))
        return True
    if protocol is not None and payload.get("protocol_id") is None:
        problems.append("unlocked checkpoint cannot serve as a final student; "
                        "only the hash-matched warm start may initialize training")
    if protocol is not None:
        if payload.get("protocol_id") != protocol.get("protocol_id"):
            problems.append(f"checkpoint protocol id {payload.get('protocol_id')} "
                            f"!= locked {protocol.get('protocol_id')}")
        if protocol.get("modes") and payload.get("mode") not in protocol["modes"]:
            problems.append(f"checkpoint mode {payload.get('mode')} is not a locked mode")
        if protocol.get("seeds") and payload.get("seed") not in protocol["seeds"]:
            problems.append(f"checkpoint seed {payload.get('seed')} is not a locked seed")
        if protocol.get("cache_sha256") != payload.get("cache_sha256"):
            problems.append("checkpoint was trained from a different cache")
        if protocol.get("warm_start_sha256") != payload.get("init_sha256"):
            problems.append("checkpoint used a different warm start")
        if protocol.get("updates") is not None and \
                int(payload.get("updates", -1)) != int(protocol["updates"]):
            problems.append(f"checkpoint ran {payload.get('updates')} updates, "
                            f"lock says {protocol['updates']}")
    if payload.get("metric_mode") != config.get("metric_mode", "exact"):
        problems.append("checkpoint metric mode differs from the configuration")
    if path and not os.path.exists(path):
        problems.append("checkpoint file missing")
    if problems:
        raise ProtocolError("checkpoint provenance mismatch for " + str(path) +
                            ": " + "; ".join(problems))
    return True


def _records(frame):
    if hasattr(frame, "to_dict"):
        return frame.to_dict("records")
    return list(frame)


def check_binary_success(records):
    bad = sorted({r.get("success") for r in records
                  if r.get("success") not in (0, 1, True, False)})
    if bad:
        raise ValueError(f"success must be binary 0/1, found {bad[:5]}")
    return True


def require_complete(frame, protocol, method, baseline,
                     columns=("training_seed", "scene")):
    records = _records(frame)
    check_binary_success(records)
    expected_methods = list(protocol["modes"])
    expected_seeds = [int(s) for s in protocol["seeds"]]
    expected_scenes = [int(s) for s in protocol["final_scenes"]]
    if method not in expected_methods or baseline not in expected_methods:
        raise ValueError(f"primary contrast {method} vs {baseline} not in the "
                         f"locked modes {expected_methods}")
    expected_pairs = {(s, c) for s in expected_seeds for c in expected_scenes}
    for name in expected_methods:
        rows = [r for r in records if str(r.get("method")) == name]
        if not rows:
            raise ValueError(f"{name}: no evaluation rows at all; every locked "
                             "method must be present")
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
        keys = [(str(r.get("method")), int(r[columns[0]]), int(r[columns[1]]))
                for r in rows if str(r.get("method")) == name]
        if len(keys) != len(set(keys)):
            seen, dupes = set(), set()
            for k in keys:
                if k in seen:
                    dupes.add(k)
                seen.add(k)
            dupes = sorted(dupes)
            raise ValueError(f"{name}: duplicate evaluation episodes {dupes[:5]}; "
                             "never silently deduplicate")
        if len(observed) != len(expected_pairs):
            raise ValueError(f"{name}: {len(observed)} rows for "
                             f"{len(expected_pairs)} planned episodes (duplicates?)")
    return True


def require_eval_metadata(frame, protocol, inputs, method=None, baseline=None):
    """Index metadata by (intrinsic mode, intrinsic seed).

    Two planned fine-tuning seeds legitimately produce two files for one mode;
    only a duplicated RUN is an error.
    """
    records = _records(frame)
    by_run = {}
    for path in inputs:
        sidecar = path.replace(".jsonl", ".meta.json")
        if not os.path.exists(sidecar):
            raise ValueError("evaluation metadata missing: " + sidecar)
        with open(sidecar) as handle:
            meta = json.load(handle)
        if meta.get("protocol_id") != protocol.get("protocol_id"):
            raise ValueError(f"{sidecar}: protocol id {meta.get('protocol_id')} "
                             f"!= locked {protocol.get('protocol_id')}")
        key = (str(meta.get("intrinsic_mode")), int(meta.get("intrinsic_seed", -1)))
        if key in by_run:
            raise ValueError(f"two evaluation files describe the same run {key}")
        for field in ("checkpoint_sha256", "protocol_id", "scenes", "split"):
            if field not in meta:
                raise ValueError(f"metadata for {key} lacks {field}")
        rows = [r for r in records
                if str(r.get("method")) == key[0]
                and int(r.get("training_seed", -1)) == key[1]]
        if not rows:
            raise ValueError(f"{sidecar} describes {key} but no rows carry that "
                             "mode and seed")
        row_scenes = sorted({int(r["scene"]) for r in rows})
        if row_scenes != sorted(int(s) for s in meta["scenes"]):
            raise ValueError(f"{sidecar}: scene set disagrees with its rows")
        if meta.get("expected_split") and any(r.get("split") != meta["expected_split"]
                                              for r in rows):
            raise ValueError(f"{sidecar}: split label disagrees with its rows")
        by_run[key] = meta
    expected_runs = {(str(m), int(s)) for m in protocol["modes"]
                     for s in protocol["seeds"]}
    present = set(by_run)
    missing = sorted(expected_runs - present)
    unexpected = sorted(k for k in present if k[0] in protocol["modes"]
                        and k not in expected_runs)
    if unexpected:
        raise ValueError(f"metadata for unplanned runs: {unexpected}")
    if method is not None:
        key = (method, int(protocol["seeds"][0]))
        if (method, key[1]) not in by_run and not any(k[0] == method for k in by_run):
            raise ValueError(f"no evaluation metadata for the primary method {method}")
    return dict(by_run=by_run, missing_runs=missing)
