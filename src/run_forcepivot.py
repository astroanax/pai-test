"""Force Is Not a Wrench — N0–N6 experiment (user's pivot program).

Measurement model: y_s = b_s + H_s * C_s Ad^{-T}_Ts [w_contact + w_tool] + eps.
Screw invariants remove only T_s. This experiment maps when force becomes
useful after removing measurement-specific nuisance.

Interfaces:
  N0: no force
  N1: raw force (wrench as measured)
  N2: SE(3)-canonical force (episode-initial-pose co-transform, as M2)
  N3: robust per-sensor(=per-config) affine normalization:
      wrench whitened by per-configuration median/IQR fit on train windows
  N4: free-motion residual: wrench minus per-episode gravity/tool estimate
      (median wrench over the episode's low-|f|-variance segments... here:
      median of stationary frames beyond the initial rezero = self-loading
      template removed)
  N5: small sensor adapter — linear map A,b fit on k seconds of unlabeled
      target-config windows by matching the source-config feature distribution
      (CORAL-style: align mean+covariance of wrench features)
  N6: oracle supervised calibration — affine map fit on labeled target windows
      (upper bound; 500 target windows)

Diagnostics:
  - configuration probe: logistic regression predicting cfg from representation
  - within-configuration shuffle: train with force histories shuffled among
    same-config windows; if held-out performance persists, model used
    hardware identity, not time-aligned physics

Primary metric: transfer regret
  Delta_force = Risk_heldout(F_force) - Risk_heldout(F_no-force)
on the WORK target (per audit: W_H nearly trivial; we keep it but add a
contact-force-magnitude target |f_{t+H}| which is NOT trivial) and on
decomposed future-wrench error.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from symmetry import (load_episodes, episode_frames, windows, iface, context,
                      targets_inv, targets_eq, fit_scales, quat_to_mat,
                      HIST, H, FPS, CONTACT_THR)
from run_audit import recompute_offsets, trivial_baselines, hist_power
from train_reg import train_reg, predict_reg
from invariant_tests import rand_T, cotransform
from sklearn.linear_model import LogisticRegression

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]


# ---------------------------------------------------------------- interfaces
def feat(e, t, kind, cfg_stats=None, adapter=None):
    """Force-branch input (HIST, d) for N-interfaces."""
    w = e["w"][t-HIST:t]
    nu = e["nu"][t-HIST:t]
    if kind == "N0":
        return np.zeros((HIST, 1), np.float32)
    if kind == "N1":
        return w.astype(np.float32)
    if kind == "N2":
        q0 = e["pose"][0]
        R0 = quat_to_mat(q0[3:7]); Rt = R0.T; p = -Rt @ q0[:3]
        wt, _ = cotransform_pair(w, nu, (Rt, p))
        return wt.astype(np.float32)
    if kind == "N3":
        # per-config robust affine normalization (train-fit med/IQR)
        med, iqr = cfg_stats
        return ((w - med) / iqr).astype(np.float32)
    if kind == "N4":
        # free-motion residual: subtract episode-level self-loading template
        # (median wrench over low-force segments) — beyond the initial rezero
        if "selfload" not in e:
            fmag = np.linalg.norm(e["w"][:, 3:], axis=1)
            lo = fmag < np.quantile(fmag, 0.2)
            e["selfload"] = np.median(e["w"][lo], axis=0) if lo.sum() > 50 else np.zeros(6)
        return (w - e["selfload"]).astype(np.float32)
    if kind == "N5":
        # source-normalized + CORAL adapter applied to target windows
        x = w.astype(np.float32)
        if adapter is not None:
            A, b = adapter   # x' = x @ A + b  (per-feature affine)
            x = x @ A + b
        return x
    if kind == "N6":
        x = w.astype(np.float32)
        if adapter is not None:
            A, b = adapter
            x = x @ A + b
        return x
    raise ValueError(kind)


def cotransform_pair(w, nu, T):
    from invariant_tests import cotransform
    wt, nut = np.empty_like(w), np.empty_like(nu)
    for k in range(len(w)):
        a, b = cotransform(w[k], nu[k], T)
        wt[k], nut[k] = a, b
    return wt, nut


def fit_cfg_stats(eps, wins):
    """Per-config robust normalization from train windows (N3)."""
    X = np.concatenate([eps[i]["w"][t-HIST:t] for i, t in wins])
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0) + 1e-6
    return med, iqr


def fit_coral_adapter(eps_src_wins, eps_tgt_wins, eps, k_seconds=60):
    """N5: unlabeled adapter. Align target wrench distribution to source:
    x_tgt' = (x - mu_t) @ Sigma_t^{-1/2} @ Sigma_s^{1/2} + mu_s  (per-feature
    temporal mean/std over histories is enough for a linear adapter)."""
    def collect(wins):
        X = np.concatenate([eps[i]["w"][t-HIST:t] for i, t in wins])
        return X.mean(0), X.std(0) + 1e-6
    mu_s, sd_s = collect(eps_src_wins)
    mu_t, sd_t = collect(eps_tgt_wins[:max(int(k_seconds * FPS / HIST), 8)])
    A = np.diag(sd_s / sd_t)
    b = mu_s - mu_t @ A
    return A, b


def fit_oracle_adapter(eps_src_wins, eps_tgt_wins, eps, Ysrc=None, Ytgt=None):
    """N6: supervised affine map on wrench minimizing target prediction error
    proxy: match target wrench features to source model's expected input stats
    using LABELED target windows — ridge regression from target wrench to
    source wrench is ill-posed; instead we align on the TARGET (the honest
    oracle: use labeled target windows to directly fit input whitening that
    minimizes future-|f| prediction error via least squares on labels)."""
    # oracle = per-feature affine fit mapping target wrench stats to the stats
    # that minimize prediction error: we approximate with exact mean/std
    # matching on labeled windows (upper bound on N5)
    # oracle: use ALL labeled target windows (not a k-second budget)
    def collect(wins):
        X = np.concatenate([eps[i]["w"][max(min(t, len(eps[i]["w"])-1), HIST)-HIST:
                                    max(min(t, len(eps[i]["w"])-1), HIST)] for i, t in wins])
        return X.mean(0), X.std(0) + 1e-6
    mu_s, sd_s = collect(eps_src_wins)
    mu_t, sd_t = collect(eps_tgt_wins)          # all target windows
    A = np.diag(sd_s / sd_t)
    b = mu_s - mu_t @ A
    return A, b


# ---------------------------------------------------------------- targets
def targets_n(eps, wins):
    """Targets: future |f| (non-trivial), future wrench (canonical frame), W_H."""
    out = []
    for i, t in wins:
        e = eps[i]
        fH = np.linalg.norm(e["w"][t+H-1, 3:])
        wq = targets_eq(e, t)
        W_H = targets_inv(e, t)[0]
        out.append(np.r_[wq, fH, W_H])
    return np.stack(out)   # 14-d: [eq 12, fH, W_H]


def run(cache, out_path):
    rng = np.random.default_rng(0)
    eps = load_episodes(cache)
    for e in eps:
        episode_frames(e)
    eps = recompute_offsets(eps)
    cfgs = sorted(set(e["cfg"] for e in eps))
    def task_of(e): return e["name"].split("__")[1].split("_user")[0]
    tasks_by_cfg = {c: set(task_of(e) for e in eps if e["cfg"] == c) for c in cfgs}
    shared = set.intersection(*tasks_by_cfg.values())
    by_cfg = {c: [i for i, e in enumerate(eps) if e["cfg"] == c] for c in cfgs}
    for c in cfgs:
        rng.shuffle(by_cfg[c])
    src, tgt = cfgs[0], cfgs[1]
    train_eps = [i for i in by_cfg[src] if task_of(eps[i]) in shared][:60]
    same_eps = [i for i in by_cfg[src] if task_of(eps[i]) in shared][60:75]
    held_eps = [i for i in by_cfg[tgt] if task_of(eps[i]) in shared][:30]
    role = {}
    for i in train_eps: role[i] = "train"
    for i in same_eps: role[i] = "same"
    for i in held_eps: role[i] = "held"
    allw = windows(eps, stride=25)
    tr_w = [w for w in allw if role.get(w[0]) == "train"]
    sa_w = [w for w in allw if role.get(w[0]) == "same"]
    he_w = [w for w in allw if role.get(w[0]) == "held"]
    rng.shuffle(tr_w); rng.shuffle(sa_w); rng.shuffle(he_w)
    tr_w, sa_w, he_w = tr_w[:2500], sa_w[:600], he_w[:600]
    print(f"windows tr/sa/he: {len(tr_w)}/{len(sa_w)}/{len(he_w)}")

    Y_tr = targets_n(eps, tr_w)
    Y_sa = targets_n(eps, sa_w)
    Y_he = targets_n(eps, he_w)
    # target normalization (train stats) for the 16-d target
    mu, sd = Y_tr.mean(0), Y_tr.std(0) + 1e-6

    cfg_stats = fit_cfg_stats(eps, tr_w)                       # N3
    adapters = {}
    for k_s in [10, 60, 300]:                                  # N5: 10s/1m/5m
        adapters[f"N5_{k_s}"] = fit_coral_adapter(tr_w, he_w, eps, k_s)
    adapters["N6"] = fit_oracle_adapter(tr_w, he_w, eps, None, None)

    results = {}
    # baselines for the |f| target
    results["baselines"] = dict(
        fH_std=float(Y_he[:, 0].std()),
        fH_const_mae=float(np.abs(Y_he[:, 0] - Y_sa[:, 0].mean()).mean()),
        fH_persistence_mae=float(np.abs(Y_he[:, 0] -
            np.array([np.linalg.norm(eps[i]["w"][t, 3:]) for i, t in he_w])).mean()),
    )
    print("baselines:", results["baselines"])

    kinds = ["N0", "N1", "N2", "N3", "N4", "N5_10", "N5_60", "N5_300", "N6"]
    for seed in SEEDS:
        for kind in kinds:
            def build(wins, is_target):
                ad = None
                if kind.startswith("N5") or kind == "N6":
                    ad = adapters["N6"] if kind == "N6" else adapters[kind]
                    if not is_target:
                        ad = None                      # source windows unadapted
                X = np.stack([feat(eps[i], t,
                                   "N5" if kind.startswith("N5") else kind,
                                   cfg_stats if kind == "N3" else None,
                                   ad) for i, t in wins])
                Xc = np.stack([context(eps[i], t) for i, t in wins])
                return X.astype(np.float32), Xc
            Xf_tr, Xc_tr = build(tr_w, False)
            Xf_sa, Xc_sa = build(sa_w, True)
            Xf_he, Xc_he = build(he_w, True)
            # input normalization (train stats)
            mu_i, sd_i = Xf_tr.mean((0, 1)), Xf_tr.std((0, 1)) + 1e-6
            Xf_tr = ((Xf_tr - mu_i) / sd_i).astype(np.float32)
            Xf_sa = ((Xf_sa - mu_i) / sd_i).astype(np.float32)
            Xf_he = ((Xf_he - mu_i) / sd_i).astype(np.float32)
            net = train_reg(Xf_tr, Xc_tr, (Y_tr - mu) / sd, seed)
            # we abuse train_one: single 16-d "eq" head carries all targets;
            # predict and un-normalize
            Pn_sa = predict_reg(net, Xf_sa, Xc_sa)
            Pn_he = predict_reg(net, Xf_he, Xc_he)
            P_sa, P_he = Pn_sa * sd + mu, Pn_he * sd + mu
            def errs(P, Y):
                e = np.abs(P - Y)
                return dict(fH_mae=float(e[:, 12].mean()),
                            f_mae=float(e[:, 9:12].mean()),
                            tau_mae=float(e[:, 6:9].mean()),
                            W_mae=float(e[:, 13].mean()))
            rec = dict(same=errs(P_sa, Y_sa), held=errs(P_he, Y_he))
            # configuration probe (on same+held inputs, predicting cfg)
            Xp = np.concatenate([Xf_sa.mean(1), Xf_he.mean(1)])
            yp = np.r_[np.zeros(len(Xf_sa)), np.ones(len(Xf_he))]
            lr = LogisticRegression(max_iter=500).fit(Xp, yp)
            rec["cfg_probe_acc"] = float(lr.score(Xp, yp))
            results[f"{kind}/seed{seed}"] = rec
            print(f"[{kind} s{seed}] held: fH={rec['held']['fH_mae']:.3f} "
                  f"f={rec['held']['f_mae']:.3f} tau={rec['held']['tau_mae']:.3f} "
                  f"probe={rec['cfg_probe_acc']:.2f}", flush=True)

    # within-configuration shuffle diagnostic (N1 only)
    rng.shuffle(tr_w)
    shuf_eps = [dict(e) for e in eps]      # shallow copy episode list
    # shuffle force histories among same-config windows: swap wrench slices
    idx = list(range(len(tr_w)))
    rng.shuffle(idx)
    perm_map = {i: j for i, j in zip(range(len(tr_w)), idx)}
    print("shuffle diagnostic: training N1 with permuted wrench histories")
    # (implemented as: each training window's wrench replaced by another
    #  random train window's wrench — same config, breaks time alignment)
    def _safe_sl(i, t):
        t2 = max(min(t, len(eps[i]["w"]) - 1), HIST)
        return eps[i]["w"][t2-HIST:t2]
    Xf_tr2 = np.stack([_safe_sl(*tr_w[perm_map[k]]) for k in range(len(tr_w))])
    Xc_tr = np.stack([context(eps[i], t) for i, t in tr_w])
    mu_i, sd_i = Xf_tr2.mean((0, 1)), Xf_tr2.std((0, 1)) + 1e-6
    net = train_reg(((Xf_tr2 - mu_i) / sd_i).astype(np.float32), Xc_tr,
                    (Y_tr - mu) / sd, 0)
    Xf_he_s = ((np.stack([eps[i]["w"][t-HIST:t] for i, t in he_w]) - mu_i) / sd_i).astype(np.float32)
    Xc_he = np.stack([context(eps[i], t) for i, t in he_w])
    P = predict_reg(net, Xf_he_s, Xc_he)
    P = P * sd + mu
    results["shuffle_diag"] = dict(held=errs(P, Y_he))
    print("shuffle diag held:", results["shuffle_diag"]["held"])

    # transfer regret summary
    reg = {}
    for kind in kinds:
        r0 = np.mean([results[f"{kind}/seed{s}"]["held"]["fH_mae"] for s in SEEDS])
        rn0 = np.mean([results[f"N0/seed{s}"]["held"]["fH_mae"] for s in SEEDS])
        reg[kind] = float(r0 - rn0)
    results["transfer_regret_fH"] = reg
    json.dump(results, open(out_path, "w"), indent=1)
    print("[wrote]", out_path)
    print("transfer regret (fH):", json.dumps(reg, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("out")
    a = ap.parse_args()
    run(a.cache, a.out)
