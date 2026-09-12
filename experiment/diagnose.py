import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = np.load(args.cache, allow_pickle=True)
    contexts = list(data["contexts"])
    if len(contexts) < 8:
        raise ValueError("too few metric contexts for diagnostic")
    radii = [0.005, 0.01, 0.02]
    rng = np.random.default_rng(11)
    report = {"contexts": len(contexts), "radii": {}}
    for radius in radii:
        ranks = []
        for context in contexts[:16]:
            target = np.asarray(context["target_mid"]).reshape(-1)
            jac = np.asarray(context["jacobian_start"])
            probes = np.asarray(context["probes"])
            errors = []
            predicted = []
            for _ in range(8):
                direction = rng.standard_normal(target.shape)
                direction = direction / np.linalg.norm(direction) * radius
                err = direction.reshape(1, 16, 2)
                physical = (jac @ err[0, :8].reshape(-1, 1)).reshape(-1)
                projected = (probes @ direction).mean()
                errors.append(float(np.linalg.norm(physical)))
                predicted.append(float(abs(projected)))
            order_true = np.argsort(errors)
            order_pred = np.argsort(predicted)
            rank = float(np.corrcoef(order_true, order_pred)[0, 1]) if np.std(predicted) > 0 else float("nan")
            ranks.append(rank)
        report["radii"][str(radius)] = {"mean_rank_corr": float(np.nanmean(ranks)), "n": len(ranks)}
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print("wrote diagnostic " + args.output)


if __name__ == "__main__":
    main()
