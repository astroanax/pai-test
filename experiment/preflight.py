import importlib
import json
import os
import sys


def main():
    report = {"python": sys.version, "modules": {}, "cuda": None, "assets": {}}
    ok = True
    for name in ["torch", "scipy", "pandas", "matplotlib", "gym", "pygame", "pymunk",
                 "shapely", "cv2", "skimage", "zarr", "diffusers", "tqdm", "torchcfm"]:
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
    for path in ["assets/flow_pusht.pth", "assets/normalizer.npz", "external/flow_matching"]:
        report["assets"][path] = os.path.exists(path)
        if not os.path.exists(path):
            ok = False
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
