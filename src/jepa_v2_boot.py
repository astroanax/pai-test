"""Episode-cluster bootstrap analysis for the corrected JEPA rerun.

For each pair (J1-J0, J2-J0, J2-J1) and for real-vs-swapped within J1/J2,
resample EPISODES (clusters) with replacement and report the 95% CI of the
paired mean cosine-error difference. Episode-level clustering is required
because windows inside an episode are not independent.
"""
import numpy as np, json, sys

def load(tag):
    z = np.load(f"results/pred_{tag}.npz")
    return {k: z[k] for k in z.files}

def cluster_boot(ep, delta, n_boot=5000, seed=0):
    """ep: episode ids (N,), delta: per-window difference (N,). Cluster = episode."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(ep)
    idx_by_ep = {e: np.where(ep == e)[0] for e in uniq}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_ep[e] for e in pick])
        means.append(delta[sel].mean())
    means = np.array(means)
    return dict(mean=float(delta.mean()),
                lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)),
                p_gt0=float((means > 0).mean()))

out = {}
D = {t: load(t) for t in ["remove_J0_s0", "remove_J1_s0", "remove_J2_s0"]}
ep = D["remove_J0_s0"]["ep_id"]
n = len(ep)
print(f"n windows={n}, n episodes={len(np.unique(ep))}")

# paired differences in cosine error (positive = first is WORSE)
out["J1_minus_J0_cos"] = cluster_boot(ep, D["remove_J1_s0"]["cos"] - D["remove_J0_s0"]["cos"])
out["J2_minus_J0_cos"] = cluster_boot(ep, D["remove_J2_s0"]["cos"] - D["remove_J0_s0"]["cos"])
out["J2_minus_J1_cos"] = cluster_boot(ep, D["remove_J2_s0"]["cos"] - D["remove_J1_s0"]["cos"])
# real vs swapped (within each force model): positive = swapped WORSE (force relied upon)
out["J1_swap_minus_real"] = cluster_boot(ep, D["remove_J1_s0"]["cos_swapped"] - D["remove_J1_s0"]["cos"])
out["J2_swap_minus_real"] = cluster_boot(ep, D["remove_J2_s0"]["cos_swapped"] - D["remove_J2_s0"]["cos"])

# contact Brier (equal across interfaces here because prevalence ~0.99)
for t in ["remove_J0_s0", "remove_J1_s0", "remove_J2_s0"]:
    Yc, Pc = D[t]["Yc"], D[t]["Pc"]
    b = ((Pc - Yc) ** 2).mean()
    print(f"{t}: contact brier={b:.5f}, mean Pc={Pc.mean():.4f}, Yc rate={Yc[:,1].mean():.4f}")

for k, v in out.items():
    print(f"{k}: mean={v['mean']:+.5f}  95% CI [{v['lo']:+.5f}, {v['hi']:+.5f}]  P(>0)={v['p_gt0']:.3f}")

json.dump(out, open("results/jepa_v2_boot.json", "w"), indent=1)
print("[wrote] results/jepa_v2_boot.json")
