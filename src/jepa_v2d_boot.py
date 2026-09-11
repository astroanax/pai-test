"""Episode-cluster bootstrap for the DELTA-target JEPA run (remove holdout)."""
import numpy as np, json

def load(tag):
    z = np.load(f"results/pred_{tag}.npz")
    return {k: z[k] for k in z.files}

def cluster_boot(ep, delta, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep)
    by = {e: np.where(ep == e)[0] for e in uniq}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([by[e] for e in pick])
        means.append(delta[sel].mean())
    m = np.array(means)
    return dict(mean=float(delta.mean()),
                lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)),
                p_gt0=float((m > 0).mean()))

D = {t: load(t) for t in ["remove_J0_s0", "remove_J1_s0", "remove_J2_s0"]}
ep = D["remove_J0_s0"]["ep_id"]
print(f"n={len(ep)}, episodes={len(np.unique(ep))}")

out = {}
out["J1-J0_cos"] = cluster_boot(ep, D["remove_J1_s0"]["cos"] - D["remove_J0_s0"]["cos"])
out["J2-J0_cos"] = cluster_boot(ep, D["remove_J2_s0"]["cos"] - D["remove_J0_s0"]["cos"])
out["J2-J1_cos"] = cluster_boot(ep, D["remove_J2_s0"]["cos"] - D["remove_J1_s0"]["cos"])
out["J1_swap-real"] = cluster_boot(ep, D["remove_J1_s0"]["cos_swapped"] - D["remove_J1_s0"]["cos"])
out["J2_swap-real"] = cluster_boot(ep, D["remove_J2_s0"]["cos_swapped"] - D["remove_J2_s0"]["cos"])

for k, v in out.items():
    print(f"{k:16s} mean={v['mean']:+.5f} CI[{v['lo']:+.5f},{v['hi']:+.5f}] P(>0)={v['p_gt0']:.3f}")
json.dump(out, open("results/jepa_v2d_boot.json", "w"), indent=1)
