import importlib
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", default=None,
                        help="write a file manifest for transfer verification")
    parser.add_argument("--verify-manifest", default=None,
                        help="verify an existing manifest against this directory")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.write_manifest:
        payload = C.write_manifest(args.write_manifest)
        print("wrote manifest " + args.write_manifest + " files " +
              str(payload["count"]))
    if args.verify_manifest:
        print("manifest verified " + json.dumps(
            C.verify_manifest(args.verify_manifest)))
    report = {"python": sys.version, "modules": {}, "cuda": None, "assets": {},
              "manifest": C.check_manifest(), "source_hashes": C.source_hashes()}
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
                          "devices": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    except Exception as error:
        report["cuda"] = {"error": str(error)}
        ok = False
    config_path = os.environ.get("RH_CONFIG", "experiment/config.json")
    if not os.path.exists(config_path):
        config_path = "config.json"
    config = None
    if os.path.exists(config_path):
        with open(config_path) as handle:
            config = json.load(handle)
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
        if not os.path.isdir(repository):
            ok = False
        else:
            report["upstream"] = C.upstream_hashes(repository)
            report["upstream_commit"] = C.upstream_commit(repository)
    else:
        report["config_valid"] = "config not found"
        ok = False
    if report["manifest"]["missing"]:
        report["manifest"]["error"] = "missing experiment source files"
        ok = False
    for path in ["runs/smoke_report.json", "runs/protocol_locked.json"]:
        report["assets"][path] = os.path.exists(path)
    print(json.dumps(report, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
