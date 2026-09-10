"""Stats lock: paired episode-cluster bootstrap with hierarchical
episodes+seeds resampling, macro-averaged transfers, Holm correction.

Input: per-window prediction npz files from run_factorized.py --save_preds:
  pred_{src}_{tgt}_{iface}_seed{s}.npz with ep_id, win_t, Y, P_real,
  P_zero, P_shuffle, P_same, P_other. Endpoint layout (8,):
  0-2 canonical force vector, 3 |f|, 4 |tau|, 5 onset, 6 offset, 7 cosdir.

Planned comparisons (all on future-|f| MAE unless noted):
  C1: F2 - F0   (canonical force vs no force)
  C2: F2 - F1   (canonical vs raw)
  C3: F3 - F2   (bias/mass/CoM on top of canonical)
  C4: F2 real - F2 shuffled (time-aligned dependence)

Procedure per comparison:
  for each transfer: hierarchical bootstrap with B replicates:
    sample 3 seeds with replacement; sample eval episodes with replacement;
    paired mean error difference over (sampled seeds x sampled episodes).
  macro-average the three transfer-level differences per replicate;
  report mean diff, 95% CI, two-sided p (fraction of replicates crossing 0).
  Holm correction over the 4 comparisons. Relative F2 gains per transfer.
"""
import argparse, glob, json, os
import numpy as np

COL_FMAG = 3
COMPS = [
    ("F2-F0", "F0", "F2", "real", "real"),
    ("F2-F1", "F1", "F2", "real", "real"),
    ("F3-F2", "F2", "F3", "real", "real"),
    ("F2real-F2shuf", "F2", "F2", "shuffle", "real"),
]
SEEDS = [0, 1, 2]


def load_preds(preds_dir):
    """preds[transfer][iface][seed] = dict(ep, Y, P_real, P_shuffle, ...)"""
    preds = {}
    for f in sorted(glob.glob(os.path.join(preds_dir, "pred_*.npz"))):
        base = os.path.basename(f)[5:-4]          # cfg1_cfg2_F2_seed0
        parts = base.split("_seed")
        seed = int(parts[1])
        rest = parts[0].rsplit("_", 1)            # ..._F2
        iface = rest[1]
        transfer = rest[0]                        # cfg1_cfg2
        d = np.load(f)
        preds.setdefault(transfer, {}).setdefault(iface, {})[seed] = dict(
            ep=np.asarray(d["ep_id"]), Y=np.asarray(d["Y"]),
            real=np.asarray(d["P_real"]), shuffle=np.asarray(d["P_shuffle"]))
    return preds


def check_alignment(preds):
    """ev windows must be identical across interfaces/seeds within a transfer."""
    for tr, ifaces in preds.items():
        ref = None
        for iface, seeds in ifaces.items():
            for seed, d in seeds.items():
                key = (d["ep"].tobytes(), d["Y"].tobytes())
                if ref is None:
                    ref = key
                assert key == ref, f"misaligned windows in {tr} {iface} seed{seed}"
    print("alignment OK: ev windows identical across interfaces/seeds per transfer")


def err_fmag(d, cond):
    return np.abs(d[cond][:, COL_FMAG] - d["Y"][:, COL_FMAG])


def boot_comp(preds, transfer, a_iface, b_iface, a_cond, b_cond, B=5000, seed=0):
    """Hierarchical paired bootstrap for (b - a) mean |f| error on one transfer.
    Returns (mean_diff, lo, hi). Positive => b worse than a."""
    rng = np.random.default_rng(seed)
    # union of episodes; per-seed error vectors aligned by window
    d0 = preds[transfer][a_iface][SEEDS[0]]
    ep = d0["ep"]
    ueps = np.unique(ep)
    diffs = []
    for _ in range(B):
        s_seeds = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        s_eps = rng.choice(ueps, size=len(ueps), replace=True)
        per_seed = []
        for s in s_seeds:
            da = preds[transfer][a_iface][s]
            db = preds[transfer][b_iface][s]
            m = np.isin(da["ep"], s_eps)
            if m.sum() == 0:
                continue
            ea = np.abs(da[a_cond][m, COL_FMAG] - da["Y"][m, COL_FMAG])
            eb = np.abs(db[b_cond][m, COL_FMAG] - db["Y"][m, COL_FMAG])
            per_seed.append(eb.mean() - ea.mean())
        if per_seed:
            diffs.append(float(np.mean(per_seed)))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), diffs


def main(preds_dir, out_path, B=5000):
    preds = load_preds(preds_dir)
    check_alignment(preds)
    transfers = sorted(preds)
    print("transfers:", transfers)
    results = {"transfers": transfers, "comparisons": {}}
    raw_ps = []
    for name, a_if, b_if, a_c, b_c in COMPS:
        per_tr = {}
        macro_reps = None
        for tr in transfers:
            mean, lo, hi, reps = boot_comp(preds, tr, a_if, b_if, a_c, b_c, B)
            per_tr[tr] = dict(mean=mean, lo=lo, hi=hi)
            macro_reps = reps if macro_reps is None else macro_reps + reps
        macro_reps = macro_reps / len(transfers)
        m_mean = float(macro_reps.mean())
        m_lo, m_hi = float(np.percentile(macro_reps, 2.5)), float(np.percentile(macro_reps, 97.5))
        # two-sided p: fraction of replicates on the other side of 0
        p = float(min((macro_reps <= 0).mean(), (macro_reps >= 0).mean()) * 2)
        results["comparisons"][name] = dict(per_transfer=per_tr,
                                            macro=dict(mean=m_mean, lo=m_lo, hi=m_hi, p_raw=p))
        raw_ps.append(p)
        print(f"{name}: macro diff={m_mean:+.4f} CI[{m_lo:+.4f},{m_hi:+.4f}] p={p:.4f}")
        for tr in transfers:
            d = per_tr[tr]
            print(f"    {tr}: {d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]")
    # Holm correction
    order = np.argsort(raw_ps)
    holm = np.empty_like(np.array(raw_ps))
    m = len(raw_ps)
    for rank, idx in enumerate(order):
        holm[idx] = min(raw_ps[idx] * (m - rank), 1.0)
    for (name, *_), h in zip(COMPS, holm):
        results["comparisons"][name]["macro"]["p_holm"] = float(h)
        print(f"{name}: p_holm={h:.4f}")
    # relative F2 gains per transfer + macro: (F0 - F2)/F0 on real fmag
    gains = {}
    for tr in transfers:
        f0 = np.mean([np.abs(preds[tr]["F0"][s]["real"][:, COL_FMAG]
                             - preds[tr]["F0"][s]["Y"][:, COL_FMAG]).mean()
                      for s in SEEDS])
        f2 = np.mean([np.abs(preds[tr]["F2"][s]["real"][:, COL_FMAG]
                             - preds[tr]["F2"][s]["Y"][:, COL_FMAG]).mean()
                      for s in SEEDS])
        gains[tr] = dict(F0=float(f0), F2=float(f2), rel=float((f0-f2)/f0))
        print(f"{tr}: F0={f0:.4f} F2={f2:.4f} rel gain={(f0-f2)/f0*100:.1f}%")
    results["relative_F2_gains"] = gains
    json.dump(results, open(out_path, "w"), indent=1)
    print("[wrote]", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds_dir"); ap.add_argument("out")
    ap.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    main(a.preds_dir, a.out, a.B)
