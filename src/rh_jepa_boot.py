"""Cross-hardware bootstrap analysis for RH20T JEPA (J0/J1/J2, 3-way holdout).

Reads results/rh_jepa.json + the per-holdout predrh_*.npz files.
Episode-cluster bootstrap over J1-J0, J2-J0, J2-J1 (delta cos error), plus
real-vs-swapped reliance. Macro-averages across the 3 config holdouts.
"""
import numpy as np, json, os, glob, sys

RES = "results"
OUT = {}

def load_pred(held, iface):
    p = os.path.join(RES, f"predrh_{held}_{iface}_s0.npz")
    if not os.path.exists(p):
        # fall back to remote-copied location
        p = os.path.join("reciprocal-tokens", "results", f"predrh_{held}_{iface}_s0.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
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
                lo=float(np.percentile(m, 2.5)),
                hi=float(np.percentile(m, 97.5)),
                p_gt0=float((m > 0).mean()),
                n_win=int(len(delta)), n_ep=int(len(uniq)))

holds = ["RH20T_cfg1", "RH20T_cfg2", "RH20T_cfg3"]
per_hold = {}
for held in holds:
    D = {i: load_pred(held, i) for i in ["J0", "J1", "J2"]}
    if any(v is None for v in D.values()):
        print(f"missing preds for {held}"); continue
    ep = D["J0"]["ep_id"]
    ph = {}
    ph["J1-J0"] = cluster_boot(ep, D["J1"]["cos"] - D["J0"]["cos"])
    ph["J2-J0"] = cluster_boot(ep, D["J2"]["cos"] - D["J0"]["cos"])
    ph["J2-J1"] = cluster_boot(ep, D["J2"]["cos"] - D["J1"]["cos"])
    ph["J1_swap-real"] = cluster_boot(ep, D["J1"]["cos_swapped"] - D["J1"]["cos"])
    ph["J2_swap-real"] = cluster_boot(ep, D["J2"]["cos_swapped"] - D["J2"]["cos"])
    ph["n_eval"] = len(D["J0"]["cos"])
    ph["mean_cos"] = {i: float(D[i]["cos"].mean()) for i in ["J0", "J1", "J2"]}
    per_hold[held] = ph
    print(f"\n=== {held} (n={ph['n_eval']}) ===")
    print("  mean cos:", {k: round(v, 4) for k, v in ph["mean_cos"].items()})
    for k in ["J1-J0", "J2-J0", "J2-J1", "J1_swap-real", "J2_swap-real"]:
        v = ph[k]
        print(f"  {k:14s} mean={v['mean']:+.5f} CI[{v['lo']:+.5f},{v['hi']:+.5f}] "
              f"P(>0)={v['p_gt0']:.3f}")

# macro-average across holdouts
print("\n=== MACRO (mean over 3 holdouts) ===")
macro = {}
for k in ["J1-J0", "J2-J0", "J2-J1", "J1_swap-real", "J2_swap-real"]:
    means = [per_hold[h][k]["mean"] for h in per_hold]
    macro[k] = dict(mean=float(np.mean(means)), per_hold=means)
    print(f"  {k:14s} mean={np.mean(means):+.5f}  per-hold={[round(x,4) for x in means]}")

OUT = dict(per_hold=per_hold, macro=macro)
json.dump(OUT, open(os.path.join(RES, "rh_jepa_boot.json"), "w"), indent=1)
print("\n[wrote] results/rh_jepa_boot.json")
