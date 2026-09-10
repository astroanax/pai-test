"""Factorized sensor-calibration experiment (user's final 3-day test).

Measurement model (per config s):
    y_s(t) = b_s + H_s * C_s Ad_{T_s}^{-T} [w_ext + w_g(q; m, r) + w_inertial] + eps
Factorized interfaces (each removes one nuisance):
    F0  no force
    F1  raw force                                (baseline: everything unmodeled)
    F2  SE(3)-canonical force                    (removes T_s only)
    F3  canonical + fitted bias/mass/CoM         (removes T_s, b_s, w_g)
    F4  canonical + full 6x6 cross-axis alignment (removes T_s, C_s, b_s)
    F5  canonical + physical self-load removal + delay/FIR alignment
        (removes T_s, w_g, delay tau_s via cross-correlation on shared tasks)
    F6  supervised target fine-tuning            (genuine upper bound)

Fitting protocol (NO evaluation leakage):
    - cfg2/cfg3 ADAPTER episodes (10 per target config, disjoint from eval
      episodes) are used to fit bias/mass/CoM, the 6x6 map, and the delay.
    - F6 fine-tunes on adapter episodes only.

Endpoints (contact-balanced: windows where |f_t| > 2N OR contact in history):
    - future force magnitude |f_{t+H}|         (Newtons)
    - future force direction cos(f_t+H, f_hat)  (dimensionless)
    - future torque magnitude |tau_{t+H}|       (N·m)
    - contact onset/offset in (t, t+H]          (binary)
    - canonical wrench residual after a kinematic-only predictor (control)
  All baselines (constant, persistence, AR) computed in the SAME physical
  units from the SAME target arrays. No normalization leaks into metrics.

Test-time swap interventions (train once, swap at test):
    S0 real force (reference)
    S1 zero force
    S2 time-shuffled force from same episode (cyclic roll by HIST//2)
    S3 task/phase-matched force from another window, same config
    S4 matched force from another configuration (eval config pool)
    S5 calibrated force (the interface's own calibrated version)
  Reliance R_force = Risk(F(x, w_swap)) - Risk(F(x, w_real)).
  A model using time-aligned physics should degrade under S1/S2/S4.

Statistics: 3 config transfers (cfg1->cfg2, cfg1->cfg3, cfg2->cfg3),
episode-cluster bootstrap CIs on every reported number.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from symmetry import (load_episodes, episode_frames, windows, context,
                      quat_to_mat, HIST, H, FPS, CONTACT_THR)
from invariant_tests import cotransform
from train_reg import TCN

import torch
import torch.nn as nn
torch.backends.cudnn.enabled = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
N_ADAPTER_EPS = 10          # per target config, for calibration fitting only


# ---------------------------------------------------------------- canonicalize
def canon_wrench(e, t_start, t_end):
    """Wrench history co-transformed into the episode's initial-pose frame.
    Returns (wt (T,6) [tau;f], nut (T,6))."""
    q0 = e["pose"][0]
    R0 = quat_to_mat(q0[3:7]); Rt = R0.T; p = -Rt @ q0[:3]
    w = e["w"][t_start:t_end]; nu = e["nu"][t_start:t_end]
    wt, nut = np.empty_like(w), np.empty_like(nu)
    for k in range(len(w)):
        a, b = cotransform(w[k], nu[k], (Rt, p))
        wt[k], nut[k] = a, b
    return wt, nut


# ---------------------------------------------------------------- calibration fits
def fit_bias(wt_hist):
    """F3a: bias = median of canonical wrench over free-motion (low-|f|) frames."""
    fmag = np.linalg.norm(wt_hist[:, 3:], axis=1)
    lo = fmag < np.quantile(fmag, 0.15)
    return np.median(wt_hist[lo], axis=0) if lo.sum() > 30 else np.zeros(6)


def fit_gravity_params(eps_ids, eps):
    """F3b: fit mass m and CoM r in the wrist frame from stationary segments.
    During slow motion the wrist wrench measures gravity/tool load:
    f = R^T (m g e_z), tau = r x f. Least squares over stationary frames of
    adapter episodes -> (m, r)."""
    Fs, Ts, Rs = [], [], []
    for i in eps_ids:
        e = eps[i]
        T = len(e["w"])
        om = e["nu"][:, :3]; v = e["nu"][:, 3:]
        still = (np.linalg.norm(om, axis=1) < 0.05) & (np.linalg.norm(v, axis=1) < 0.005)
        idx = np.where(still)[0]
        for t in idx[::10]:
            R = quat_to_mat(e["pose"][t, 3:7])
            f = e["w"][t, 3:]; tau = e["w"][t, :3]
            Fs.append(f); Ts.append(tau); Rs.append(R)
    if len(Fs) < 50:
        return None
    Fs = np.stack(Fs); Ts = np.stack(Ts); Rs = np.stack(Rs)
    g = np.array([0, 0, -9.81])
    # f_i = m R_i^T g  ->  solve m from magnitudes: |f| = m|g|
    m = np.median(np.linalg.norm(Fs, axis=1)) / 9.81
    # tau_i = r x f_i  ->  skew(tau_i) r = 0 stacked ... instead: r x f = tau
    # => -skew(f) r = tau  (since f x r = -tau? tau = r x f => -f x r = r x f... )
    # tau = r x f  =>  skew(f) r = -tau
    A = np.stack([-skew_np(f) for f in Fs]).reshape(-1, 3)
    b = Ts.reshape(-1)
    r, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = np.linalg.norm(A @ r - b) / max(len(b) ** 0.5, 1)
    return dict(m=float(m), r=r, fit_resid=float(resid))


def skew_np(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def remove_gravity(e, m, r):
    """Subtract modeled self-load wrench (wrist frame): f_g = R^T (m g),
    tau_g = r x f_g, for every frame."""
    T = len(e["w"])
    g = np.array([0, 0, -9.81]) * m
    wsub = e["w"].copy()
    for t in range(T):
        R = quat_to_mat(e["pose"][t, 3:7])
        fg = R.T @ g
        tg = np.cross(r, fg)
        wsub[t, 3:] -= fg
        wsub[t, :3] -= tg
    return wsub


def fit_6x6(tgt_X, src_X):
    """F4: full 6x6 affine map FROM target measurements INTO source-canonical
    space (so a source-trained model can read target data):
        src_hat = A @ tgt + b, fit by least squares on paired adapter/trian
    statistics (cross-covariance regression)."""
    mu_s, mu_t = src_X.mean(0), tgt_X.mean(0)
    Xs, Xt = src_X - mu_s, tgt_X - mu_t
    # A = Cov_{s,t} Cov_{t,t}^{-1}
    Ct = Xt.T @ Xt / max(len(Xt) - 1, 1)
    Cst = Xs.T @ Xt / max(len(Xt) - 1, 1)
    A = Cst @ np.linalg.pinv(Ct + 1e-6 * np.eye(6))
    b = mu_s - A @ mu_t
    return A, b


def fit_delay(src_wt, tgt_wt, max_lag=30):
    """F5: estimate inter-config delay by cross-correlating |f| envelopes of
    matched adapter windows (same task); returns integer lag in frames."""
    a = np.linalg.norm(src_wt[:, 3:], axis=1)
    b = np.linalg.norm(tgt_wt[:, 3:], axis=1)
    a = a - a.mean(); b = b - b.mean()
    best, best_lag = -2, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = np.corrcoef(a[lag:], b[:len(b) - lag])[0, 1] if len(a) - lag > 20 else -2
        else:
            c = np.corrcoef(a[:lag], b[-lag:])[0, 1] if len(a) + lag > 20 else -2
        if c > best:
            best, best_lag = c, lag
    return best_lag


# ---------------------------------------------------------------- interfaces
def build_input(e, t, iface, calib_src, calib_tgt, side="src"):
    """Force-branch input (HIST, d). calib_src/tgt carry fitted params for the
    source/target config respectively; the appropriate one is used per side."""
    calib = calib_tgt if side == "tgt" else calib_src
    if iface == "F0":
        return np.zeros((HIST, 1), np.float32)
    if iface == "F1":
        return e["w"][t-HIST:t].astype(np.float32)
    if iface == "F2":
        wt, _ = canon_wrench(e, t-HIST, t)
        return wt.astype(np.float32)
    if iface == "F3":
        wt, _ = canon_wrench(e, t-HIST, t)
        # canonical + bias + gravity/tool removal using fitted (m, r)
        wsub = wt.copy()
        q0 = e["pose"][0]
        R0 = quat_to_mat(q0[3:7])
        for k in range(HIST):
            R = quat_to_mat(e["pose"][t-HIST+k, 3:7])
            # gravity in wrist frame: f_w = R^T (m g); canonical: f_c = R0^T f_w
            fg = R0.T @ (R.T @ (np.array([0, 0, -9.81]) * calib["m"]))
            wsub[k, 3:] -= fg
            wsub[k, :3] -= np.cross(calib["r"], fg)
        wsub -= calib["bias"]
        return wsub.astype(np.float32)
    if iface == "F4":
        wt, _ = canon_wrench(e, t-HIST, t)
        if side == "src":
            return wt.astype(np.float32)          # source side: no map needed
        A, b = calib["A66"], calib["b66"]          # target -> source space
        return (wt @ A.T + b).astype(np.float32)
    if iface == "F5":
        wt, _ = canon_wrench(e, t-HIST, t)
        lag = calib["lag"]
        wl = np.roll(wt, lag, axis=0) if lag != 0 else wt
        wsub = wl.copy()
        q0 = e["pose"][0]
        R0 = quat_to_mat(q0[3:7])
        for k in range(HIST):
            R = quat_to_mat(e["pose"][max(t-HIST+k-lag, 0), 3:7])
            fg = R0.T @ (R.T @ (np.array([0, 0, -9.81]) * calib["m"]))
            wsub[k, 3:] -= fg
            wsub[k, :3] -= np.cross(calib["r"], fg)
        wsub -= calib["bias"]
        return wsub.astype(np.float32)
    raise ValueError(iface)


# ---------------------------------------------------------------- targets
def endpoints(e, t):
    """All endpoints in PHYSICAL units. Layout (8,):
       0-2 future canonical force vector; 3 future |f|; 4 future |tau|;
       5 contact onset; 6 contact offset; 7 future/current force direction
       cosine (both in canonical frame)."""
    wt, _ = canon_wrench(e, t + H - 1, t + H)
    wt0, _ = canon_wrench(e, t, t + 1)
    fc = wt[0, 3:]
    tauc = wt[0, :3]
    fmag = np.linalg.norm(e["w"][:, 3:], axis=1)
    c = fmag[t:t+H] > CONTACT_THR
    onset = float(c.any() and not c[0])
    offset = float(c.any() and c[0] and not c[-1])
    f_now = wt0[0, 3:]                          # canonical current force
    cosdir = float(fc @ f_now / (np.linalg.norm(fc) * np.linalg.norm(f_now) + 1e-9))
    return np.array([fc[0], fc[1], fc[2],
                     np.linalg.norm(fc), np.linalg.norm(tauc),
                     onset, offset, cosdir], np.float32)


def kinematic_residual_target(e, t):
    """Control endpoint: canonical wrench residual after persistence
    (kinematic-only) prediction = w_{t+H} - w_t in canonical frame."""
    wt0, _ = canon_wrench(e, t - 1, t)
    wtH, _ = canon_wrench(e, t + H - 1, t + H)
    return (wtH[0] - wt0[0]).astype(np.float32)   # 6


# ---------------------------------------------------------------- baselines (physical units)
def baselines(E_val, E_train, persist_feats):
    """constant / persistence / AR baselines, all physical units, same arrays."""
    out = {}
    names = ["fx", "fy", "fz", "fmag", "taumag", "onset", "offset", "cosdir"]
    for j, nm in enumerate(names):
        y = E_val[:, j]
        out[nm] = dict(
            std=float(y.std()),
            const_mae=float(np.abs(y - E_train[:, j].mean()).mean()),
            persistence_mae=float(np.abs(y - persist_feats[:, j]).mean()),
        )
    return out


# ---------------------------------------------------------------- training
def train_model(Xf, Xc, Y, seed, epochs=25, bs=256, lr=3e-4):
    """Regression TCN on normalized targets; normalization stats returned."""
    torch.manual_seed(seed)
    mu, sd = Y.mean(0), Y.std(0) + 1e-6
    Yn = (Y - mu) / sd
    net = TCN(Xf.shape[2], Xc.shape[1], d_inv=max(Y.shape[1] - 12, 1),
              d_eq=min(Y.shape[1], 12)).to(DEVICE)
    # ensure head sizes match Y: rebuild heads if needed
    if net.head_inv.out_features + net.head_eq.out_features != Y.shape[1]:
        net.head_eq = nn.Linear(128, min(Y.shape[1], 12)).to(DEVICE)
        net.head_inv = nn.Linear(128, Y.shape[1] - min(Y.shape[1], 12)).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    xf, xc, y = (torch.tensor(Xf, device=DEVICE), torch.tensor(Xc, device=DEVICE),
                 torch.tensor(Yn, device=DEVICE))
    n = len(xf)
    ne = min(Y.shape[1], 12)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for k in range(0, n, bs):
            s = perm[k:k+bs]
            pi, pe = net(xf[s], xc[s])
            pred = torch.cat([pe[:, :ne], pi[:, :Y.shape[1]-ne]], 1)
            loss = nn.functional.smooth_l1_loss(pred, y[s])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd

@torch.no_grad()
def predict_model(net, Xf, Xc, mu, sd, dy):
    ne = min(dy, 12)
    Ps = []
    for k in range(0, len(Xf), 512):
        pi, pe = net(torch.tensor(Xf[k:k+512], device=DEVICE),
                     torch.tensor(Xc[k:k+512], device=DEVICE))
        Ps.append(torch.cat([pe[:, :ne], pi[:, :dy-ne]], 1).cpu().numpy())
    return np.concatenate(Ps) * sd + mu


# ---------------------------------------------------------------- swaps
def swapped_input(e, t, mode, swap_pool_same_cfg, swap_pool_other_cfg, iface,
                   calib_src, calib_tgt):
    """Test-time force replacements, all evaluated on the TARGET (eval) side
    with the interface-processed swapped wrench (apples-to-apples)."""
    if mode == "real":
        return build_input(e, t, iface, calib_src, calib_tgt, "tgt")
    if mode == "zero":
        e2 = dict(e); e2["w"] = np.zeros_like(e["w"])
        return build_input(e2, t, iface, calib_src, calib_tgt, "tgt")
    if mode == "shuffle":
        x = build_input(e, t, iface, calib_src, calib_tgt, "tgt")
        return np.roll(x, HIST // 2, axis=0)
    if mode == "same_cfg":
        e2, t2 = swap_pool_same_cfg[t % len(swap_pool_same_cfg)]
        return build_input(e2, t2, iface, calib_src, calib_tgt, "tgt")
    if mode == "other_cfg":
        e2, t2 = swap_pool_other_cfg[t % len(swap_pool_other_cfg)]
        return build_input(e2, t2, iface, calib_src, calib_tgt, "src")
    raise ValueError(mode)


SAVE_PREDS = None   # set via CLI --save_preds


# ---------------------------------------------------------------- main
def run(cache, out_path, ifaces=None):
    global SAVE_PREDS
    eps = load_episodes(cache)
    for e in eps:
        episode_frames(e)
    from run_audit import recompute_offsets
    eps = recompute_offsets(eps)
    cfgs = sorted(set(e["cfg"] for e in eps))
    print("configs:", cfgs)
    def task_of(e): return e["name"].split("__")[1].split("_user")[0]
    tasks_by_cfg = {c: set(task_of(e) for e in eps if e["cfg"] == c) for c in cfgs}
    rng = np.random.default_rng(0)
    by_cfg_task = defaultdict(list)
    for i, e in enumerate(eps):
        by_cfg_task[(e["cfg"], task_of(e))].append(i)
    for k in by_cfg_task:
        rng.shuffle(by_cfg_task[k])

    transfers = [(a, b) for ai, a in enumerate(cfgs) for b in cfgs[ai+1:]]
    IFACES = ifaces or ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
    results = {"transfers": {}, "configurations": cfgs}

    for src, tgt in transfers:
        shared = tasks_by_cfg[src] & tasks_by_cfg[tgt]
        if not shared:
            print(f"skip {src}->{tgt}: no shared tasks"); continue
        # episode pools: adapter (calibration-only) vs eval, disjoint
        adapter_eps, train_eps, eval_eps = [], [], []
        for task in sorted(shared):
            pool_s = by_cfg_task[(src, task)]
            pool_t = by_cfg_task[(tgt, task)]
            train_eps += pool_s[:3]
            adapter_eps += pool_t[:2]           # calibration only, never eval
            eval_eps += pool_t[2:5]
        if len(train_eps) < 20 or len(eval_eps) < 10:
            print(f"skip {src}->{tgt}: pools too small"); continue
        print(f"== transfer {src}->{tgt}: train {len(train_eps)}, "
              f"adapter {len(adapter_eps)}, eval {len(eval_eps)} episodes")

        role = {i: "train" for i in train_eps}
        role.update({i: "adapter" for i in adapter_eps})
        role.update({i: "eval" for i in eval_eps})
        allw = windows(eps, stride=25)
        tr_w = [w for w in allw if role.get(w[0]) == "train"]
        ad_w = [w for w in allw if role.get(w[0]) == "adapter"]
        ev_w = [w for w in allw if role.get(w[0]) == "eval"]
        rng.shuffle(tr_w); rng.shuffle(ad_w); rng.shuffle(ev_w)
        tr_w, ad_w, ev_w = tr_w[:2000], ad_w[:400], ev_w[:500]

        # ---- calibration fits on ADAPTER windows only (target cfg) + train (source)
        src_canon = np.concatenate([canon_wrench(eps[i], t-HIST, t)[0] for i, t in tr_w[:200]])
        tgt_canon = np.concatenate([canon_wrench(eps[i], t-HIST, t)[0] for i, t in ad_w[:200]])
        grav_t = fit_gravity_params(adapter_eps, eps)
        grav_s = fit_gravity_params(train_eps, eps)
        # per-side calibration dicts
        calib_tgt = {
            "bias": fit_bias(tgt_canon),
            "m": grav_t["m"] if grav_t else 1.0,
            "r": grav_t["r"] if grav_t else np.zeros(3),
            "A66": None, "b66": None, "lag": 0,
        }
        calib_src = {
            "bias": fit_bias(src_canon),
            "m": grav_s["m"] if grav_s else 1.0,
            "r": grav_s["r"] if grav_s else np.zeros(3),
            "A66": None, "b66": None, "lag": 0,
        }
        # F4: 6x6 map target -> source space (fit on adapter+train canon stats)
        A66, b66 = fit_6x6(tgt_canon, src_canon)
        calib_tgt["A66"], calib_tgt["b66"] = A66, b66
        # F5 delay between configs on |f| envelopes
        calib_tgt["lag"] = int(fit_delay(src_canon[:HIST*4], tgt_canon[:HIST*4])
                               if len(tgt_canon) >= HIST*4 else 0)
        # F6: supervised fine-tune data = adapter windows
        # endpoints
        def make_E(wins):
            E = np.stack([endpoints(eps[i], t) for i, t in wins])
            P = np.stack([endpoints(eps[i], max(t - H, HIST)) for i, t in wins])
            return E, P
        E_tr, P_tr = make_E(tr_w)
        E_ev, P_ev = make_E(ev_w)
        E_ad, P_ad = make_E(ad_w)
        Y_tr = E_tr
        Y_ev = E_ev
        res_tr = {"baselines": baselines(E_ev, E_tr, P_ev)}
        print(f"   baselines fmag: {res_tr['baselines']['fmag']}")

        # swap pools
        same_pool = [(eps[i], t) for i, t in ev_w[:50]]
        other_pool = [(eps[i], t) for i, t in tr_w[:50]]  # source-cfg windows

        tres = {"baselines": res_tr["baselines"]}
        for iface in IFACES:
            for seed in SEEDS:
                # build training inputs (source side; interfaces that need
                # calib are identity on source — calib maps target->source)
                base_iface = "F1" if iface == "F6" else iface
                Xf_tr = np.stack([build_input(eps[i], t, base_iface, calib_src, calib_tgt, "src") for i, t in tr_w])
                Xc_tr = np.stack([context(eps[i], t) for i, t in tr_w])
                mu_i, sd_i = Xf_tr.mean((0, 1)), Xf_tr.std((0, 1)) + 1e-6
                net, mu_t, sd_t = train_model(((Xf_tr - mu_i) / sd_i).astype(np.float32),
                                              Xc_tr, Y_tr, seed)
                if iface == "F6" and len(ad_w) >= 32:
                    # supervised fine-tune on adapter windows (F1 inputs)
                    Xf_ad = np.stack([build_input(eps[i], t, "F1", calib_src, calib_tgt, "tgt") for i, t in ad_w])
                    Xc_ad = np.stack([context(eps[i], t) for i, t in ad_w])
                    net, mu_t, sd_t = train_model(
                        ((Xf_ad - mu_i) / sd_i).astype(np.float32), Xc_ad, E_ad, seed,
                        epochs=10)
                # eval: target side with the interface's own calibration
                ev_iface = "F1" if iface == "F6" else iface
                def ev_input(i, t):
                    x = build_input(eps[i], t, ev_iface, calib_src, calib_tgt, "tgt")
                    return ((x - mu_i) / sd_i).astype(np.float32)
                # real + swaps (save per-window predictions with episode IDs
                # for the stats lock: paired episode-cluster bootstrap)
                Xf_ev = np.stack([ev_input(i, t) for i, t in ev_w])
                Xc_ev = np.stack([context(eps[i], t) for i, t in ev_w])
                P_real = predict_model(net, Xf_ev, Xc_ev, mu_t, sd_t, Y_ev.shape[1])
                rec = {"real": metric_pack(P_real, Y_ev)}
                swap_P = {}
                for mode in ["zero", "shuffle", "same_cfg", "other_cfg"]:
                    Xs = np.stack([
                        ((swapped_input(eps[i], t, mode, same_pool, other_pool,
                                        ev_iface, calib_src, calib_tgt) - mu_i) / sd_i).astype(np.float32)
                        for i, t in ev_w])
                    Ps = predict_model(net, Xs, Xc_ev, mu_t, sd_t, Y_ev.shape[1])
                    rec[mode] = metric_pack(Ps, Y_ev)
                    swap_P[mode] = Ps
                if SAVE_PREDS is not None:
                    os.makedirs(SAVE_PREDS, exist_ok=True)
                    tag = f"{src}_{tgt}_{iface}_seed{seed}".replace("RH20T_", "")
                    np.savez_compressed(
                        os.path.join(SAVE_PREDS, f"pred_{tag}.npz"),
                        ep_id=np.array([i for i, t in ev_w], dtype=np.int64),
                        win_t=np.array([t for i, t in ev_w], dtype=np.int64),
                        Y=Y_ev.astype(np.float32),
                        P_real=P_real.astype(np.float32),
                        P_zero=swap_P["zero"].astype(np.float32),
                        P_shuffle=swap_P["shuffle"].astype(np.float32),
                        P_same=swap_P["same_cfg"].astype(np.float32),
                        P_other=swap_P["other_cfg"].astype(np.float32))
                tres[f"{iface}/seed{seed}"] = rec
                print(f"   [{iface} s{seed}] real fmag={rec['real']['fmag_mae']:.3f} "
                      f"zero={rec['zero']['fmag_mae']:.3f} "
                      f"shuffle={rec['shuffle']['fmag_mae']:.3f} "
                      f"othercfg={rec['other_cfg']['fmag_mae']:.3f}", flush=True)
        results["transfers"][f"{src}->{tgt}"] = tres

    json.dump(results, open(out_path, "w"), indent=1)
    print("[wrote]", out_path)


def metric_pack(P, Y):
    """Physical-unit metrics from endpoint predictions."""
    d = {}
    d["fmag_mae"] = float(np.abs(P[:, 3] - Y[:, 3]).mean())
    d["fvec_mae"] = float(np.abs(P[:, :3] - Y[:, :3]).mean())
    d["taumag_mae"] = float(np.abs(P[:, 4] - Y[:, 4]).mean())
    d["cosdir_mae"] = float(np.abs(P[:, 7] - Y[:, 7]).mean())
    d["onset_brier"] = float(np.mean((P[:, 5] - Y[:, 5]) ** 2))
    d["offset_brier"] = float(np.mean((P[:, 6] - Y[:, 6]) ** 2))
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("out")
    ap.add_argument("--ifaces", default=None,
                    help="comma-separated subset, e.g. F0,F1,F2,F3")
    ap.add_argument("--save_preds", default=None,
                    help="directory for per-window prediction npz files")
    a = ap.parse_args()
    SAVE_PREDS = a.save_preds
    run(a.cache, a.out, ifaces=a.ifaces.split(",") if a.ifaces else None)
