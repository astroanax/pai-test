import argparse
import hashlib
import json
import os


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-no-final-results", action="store_true", required=True)
    parser.add_argument("--modes", nargs="*", default=["uniform", "prefix", "endpoint", "pullback", "identity", "scalar"])
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--cache", default="runs/shared_metrics.npz")
    parser.add_argument("--output", default="runs/protocol_locked.json")
    args = parser.parse_args()
    for path in [args.config, args.cache, "assets/flow_pusht.pth", "assets/normalizer.npz"]:
        if not os.path.exists(path):
            raise ValueError("missing required asset " + path)
    for name in ["runs/final_pullback_seed0.jsonl", "runs/final_endpoint_seed0.jsonl"]:
        if os.path.exists(name):
            raise ValueError("final outcomes already exist, refusing to lock")
    protocol = {"config": sha256_file(args.config), "cache": sha256_file(args.cache),
                "checkpoint": sha256_file("assets/flow_pusht.pth"),
                "normalizer": sha256_file("assets/normalizer.npz"),
                "modes": args.modes, "updates": args.updates, "seeds": args.seeds,
                "primary_contrast": ["pullback", "endpoint"],
                "final_scenes": list(range(40000, 40100)),
                "no_final_results_examined": True}
    if os.path.exists(args.output):
        raise ValueError("protocol already locked, refusing silent overwrite")
    with open(args.output, "w") as handle:
        json.dump(protocol, handle, indent=2)
    print("locked protocol " + args.output)


if __name__ == "__main__":
    main()
