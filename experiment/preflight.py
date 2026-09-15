import argparse
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
import contract as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--device", default=None,
                        help="device to require; defaults to the configured device")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="permit a cpu device even when the config requests cuda")
    parser.add_argument("--write-manifest", default=None)
    parser.add_argument("--verify-manifest", default=None)
    args = parser.parse_args()

    if args.write_manifest:
        payload = C.write_manifest(args.write_manifest)
        print("wrote manifest " + args.write_manifest + " files " +
              str(payload["count"]))
    if args.verify_manifest:
        print("manifest verified " + json.dumps(C.verify_manifest(args.verify_manifest)))

    report = {"python": sys.version, "modules": {}, "cuda": None, "assets": {},
              "manifest": C.check_manifest(), "source_hashes": C.source_hashes(),
              "schema_version": C.SCHEMA_VERSION}
    ok = True
    for name in C.PACKAGES:
        try:
            module = importlib.import_module(name)
            report["modules"][name] = getattr(module, "__version__", "present")
        except Exception as error:
            report["modules"][name] = "missing: " + str(error)
            ok = False
    try:
        import torch
        report["cuda"] = {"available": torch.cuda.is_available(),
                          "devices": (torch.cuda.device_count()
                                      if torch.cuda.is_available() else 0),
                          "torch": torch.__version__}
    except Exception as error:
        report["cuda"] = {"error": str(error)}
        ok = False

    config_path = args.config if os.path.exists(args.config) else os.path.join(
        PARENT, args.config)
    if not os.path.exists(config_path):
        report["config"] = "not found: " + args.config
        print(json.dumps(report, indent=2, default=str))
        return 1
    with open(config_path) as handle:
        config = json.load(handle)
    report["config_path"] = config_path
    try:
        C.validate_config(config, config_path)
        report["config_valid"] = True
    except ValueError as error:
        report["config_valid"] = str(error)
        ok = False
    splits = C.validate_split_ranges(config)
    report["split_ranges"] = {k: [min(v), max(v)] for k, v in splits["sets"].items()}
    if splits["errors"]:
        report["split_errors"] = splits["errors"]
        ok = False
    for key in ("checkpoint", "normalizer"):
        path = config[key]
        report["assets"][path] = os.path.exists(path)
        if not os.path.exists(path):
            ok = False
    repository = config["repository"]
    report["assets"][repository] = os.path.isdir(repository)
    if os.path.isdir(repository):
        report["upstream"] = C.upstream_hashes(repository)
        report["upstream_commit"] = C.upstream_commit(repository)
    else:
        ok = False

    requested = args.device or config["device"]
    report["requested_device"] = requested
    if str(requested).startswith("cuda"):
        available = bool(report.get("cuda", {}).get("available"))
        if not available and not args.allow_cpu:
            report["device_error"] = ("cuda was requested and is unavailable; rerun "
                                      "with --allow-cpu for a development check")
            ok = False
        report["cuda_ok"] = available
    for path in ("runs/smoke_report.json", "runs/protocol_locked.json"):
        report.setdefault("artifacts", {})[path] = dict(
            exists=os.path.exists(path), complete=C.is_complete(path))
    if report["manifest"]["missing"]:
        report["manifest"]["error"] = "missing experiment source files"
        ok = False
    print(json.dumps(report, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
