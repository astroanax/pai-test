"""Audit runner (user's 48-hour audit program, executed in full).

Fixes over the original gate run:
  A. Correct wxyz quaternion convention (rebuilt caches).
  B. Trivial-target audit: report target std; constant / persistence / linear-AR
     baselines alongside every model MAE.
  C. Slices: contact-only windows (|f|>2N at t) and high-work windows
     (top-quartile |W_H|) reported separately.
  D. Decomposed equivariant error: force (3), torque (3), omega (3), v (3)
     reported separately, not collapsed.
  E. Per-episode F/T bias recomputation: offset = median of first 0.5s of
     stationary frames (|omega|,|v| below thresholds) subtracted from wrench.
  F. cfg1/cfg2 task matching: train/test restricted to shared task IDs
     (task_XXXX) where possible; unmatched episodes dropped from the
     held-out set, and an unmatched-split variant also reported.
  G. Three seeds.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from symmetry import (load_episodes, episode_frames, windows, iface, context,
                      targets_inv, targets_eq, fit_scales, fit_norm,
                      quat_to_mat, HIST, H, FPS)
from invariant_tests import rand_T, cotransform

import torch
import torch.nn as nn
torch.backends.cudnn.enabled = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPS = ["M0", "M1", "M2", "M3", "M4", "M7"]
SEEDS = [0, 1, 2]


class TCN(nn.Module):
    def __init__(self, dforce, dctx, d_inv=4, d_eq=12, width=128):
        super().__init__()
        self.fin = nn.Conv1d(dforce, width, 5, padding=2)
        self.cin = nn.Linear(dctx, width)
        def block(d):
            return nn.Sequential(nn.Conv1d(width, width, 5, padding=2*d, dilation=d),
                                 nn.GELU(), nn.Dropout(0.1))
        self.b1, self.b2, self.b3 = block(1), block(2), block(4)
        self.head_inv = nn.Linear(width, d_inv)
        self.head_eq = nn.Linear(width, d_eq)
    def forward(self, x, c):
        h = self.fin(x.transpose(1, 2))[:, :, :x.shape[1]]
        h = h + self.cin(c)[:, :, None]
        h = h + self.b1(h); h = h + self.b2(h); h = h + self.b3(h)
        z = h[:, :, -1]
        return self.head_inv(z), self.head_eq(z)


# ---------------------------------------------------------------- audit E
def recompute_offsets(eps):
    """Subtract per-episode F/T bias estimated from initial stationary frames."""
    for e in eps:
        T = len(e["w"])
        n0 = min(int(0.5 * FPS), T // 4)
        seg = e["w"][:n0]
        om = e["nu"][:n0, :3]; v = e["nu"][:n0, 3:]
        still = (np.linalg.norm(om, axis=1) < 0.05) & (np.linalg.norm(v, axis=1) < 0.01)
        ref = seg[still] if still.sum() >= 20 else seg
        bias = np.median(ref, axis=0)
        e["w"] = e["w"] - bias[None, :]
        e.pop("power", None); e.pop("fmag", None); e.pop("contact", None)
        episode_frames(e)
    return eps


# ---------------------------------------------------------------- audit B
def trivial_baselines(Yinv, prev_W):
    """Constant (train-mean), persistence (last observed), and linear-AR baselines."""
    out = {}
    out["target_std"] = float(Yinv[:, 0].std())
    out["const_mae"] = float(np.abs(Yinv[:, 0] - Yinv[:, 0].mean()).mean())
    out["persistence_mae"] = float(np.abs(Yinv[:, 0] - prev_W).mean())
    # linear AR: fit W_{t+H} ~ a * W_hist + b on the fly (closed form)
    A = np.c_[prev_W, np.ones_like(prev_W)]
    coef, *_ = np.linalg.lstsq(A[:len(A)//2], Yinv[:len(A)//2, 0], rcond=None)
    out["linear_ar_mae"] = float(np.abs(A @ coef - Yinv[:, 0]).mean())
    return out


def hist_power(e, t):
    """Mean power over the history window (persistence feature)."""
    return float(e["power"][t-HIST:t].mean())


# ---------------------------------------------------------------- main
def run(cache, out_path, matched_only=True):
    rng = np.random.default_rng(0)
    eps = load_episodes(cache)
    for e in eps:
        episode_frames(e)
    eps = recompute_offsets(eps)                      # audit E
    cfgs = sorted(set(e["cfg"] for e in eps))
    print(f"{len(eps)} episodes; configs {cfgs}")

    # audit F: task matching
    def task_of(e): return e["name"].split("__")[1].split("_user")[0]  # task_XXXX
    tasks_by_cfg = {c: set(task_of(e) for e in eps if e["cfg"] == c) for c in cfgs}
    shared = set.intersection(*tasks_by_cfg.values()) if len(cfgs) > 1 else set()
    print(f"tasks: cfg1={len(tasks_by_cfg[cfgs[0]])}, cfg2={len(tasks_by_cfg[cfgs[1]])}, shared={len(shared)}")

    by_cfg = {c: [i for i, e in enumerate(eps) if e["cfg"] == c] for c in cfgs}
    for c in cfgs:
        rng.shuffle(by_cfg[c])
    held = cfgs[-1]
    if matched_only and shared:
        train_eps = [i for i in by_cfg[cfgs[0]] if task_of(eps[i]) in shared][:60]
        held_eps = [i for i in by_cfg[held] if task_of(eps[i]) in shared][:30]
        same_eps = [i for i in by_cfg[cfgs[0]] if task_of(eps[i]) in shared][60:75]
    else:
        train_eps = by_cfg[cfgs[0]][:60]
        same_eps = by_cfg[cfgs[0]][60:75]
        held_eps = by_cfg[held][:30]
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

    scales = fit_scales(eps, tr_w)
    Yinv_tr = np.stack([targets_inv(eps[i], t) for i, t in tr_w])
    Yinv_sa = np.stack([targets_inv(eps[i], t) for i, t in sa_w])
    Yinv_he = np.stack([targets_inv(eps[i], t) for i, t in he_w])
    Yeq_sa = np.stack([targets_eq(eps[i], t) for i, t in sa_w])
    Yeq_he = np.stack([targets_eq(eps[i], t) for i, t in he_w])
    prevW_sa = np.array([hist_power(eps[i], t) for i, t in sa_w])
    prevW_he = np.array([hist_power(eps[i], t) for i, t in he_w])

    # audit B on the test sets
    audit = {
        "baselines_same": trivial_baselines(Yinv_sa, prevW_sa),
        "baselines_held": trivial_baselines(Yinv_he, prevW_he),
    }
    # audit C slices
    fmag_sa = np.array([np.linalg.norm(eps[i]["w"][t, 3:]) for i, t in sa_w])
    fmag_he = np.array([np.linalg.norm(eps[i]["w"][t, 3:]) for i, t in he_w])
    Wq = np.quantile(np.abs(Yinv_tr[:, 0]), 0.75)
    slices = {
        "contact_same": fmag_sa > 2.0, "highwork_same": np.abs(Yinv_sa[:, 0]) > Wq,
        "contact_held": fmag_he > 2.0, "highwork_held": np.abs(Yinv_he[:, 0]) > Wq,
    }
    audit["slice_sizes"] = {k: int(v.sum()) for k, v in slices.items()}

    results = {"audit": audit}
    for seed in SEEDS:
        for rep in REPS:
            norm = fit_norm(eps, tr_w, rep, scales)
            Xf_tr = np.stack([iface(eps[i], t, rep, scales) for i, t in tr_w])
            Xc_tr = np.stack([context(eps[i], t) for i, t in tr_w])
            Xf_tr = (Xf_tr - norm[0]) / norm[1]
            if rep == "M3":
                rng3 = np.random.default_rng(seed)
                aug = []
                for i, t in tr_w:
                    aug.append(iface(eps[i], t, "M3", scales, rng=rng3, T_aug=rand_T(rng3)))
                Xf_tr = np.concatenate([Xf_tr, (np.stack(aug) - norm[0]) / norm[1]])
                Xc_tr = np.concatenate([Xc_tr, Xc_tr])
                Yinv2, Yeq2 = np.concatenate([Yinv_tr]*2), None
            else:
                Yinv2 = Yinv_tr
            # targets for training: build eq targets for train windows too
            Yeq_tr = np.stack([targets_eq(eps[i], t) for i, t in tr_w])
            if rep == "M3":
                Yeq_tr2 = np.concatenate([Yeq_tr]*2)
            else:
                Yeq_tr2 = Yeq_tr
            net = train_one(Xf_tr.astype(np.float32), Xc_tr, Yinv2, Yeq_tr2, seed)
            # eval
            ev = {}
            for name, wins, Yinv_, Yeq_, prevW_ in [
                    ("same", sa_w, Yinv_sa, Yeq_sa, prevW_sa),
                    ("held", he_w, Yinv_he, Yeq_he, prevW_he)]:
                Xf = np.stack([iface(eps[i], t, rep, scales) for i, t in wins])
                Xc = np.stack([context(eps[i], t) for i, t in wins])
                Xf = ((Xf - norm[0]) / norm[1]).astype(np.float32)
                pi, pe = predict(net, Xf, Xc)
                Werr = np.abs(pi[:, 0] - Yinv_[:, 0])
                # audit D: decomposed equivariant errors
                eqerr = np.abs(pe - Yeq_)
                rec = dict(
                    W_mae=float(Werr.mean()),
                    eq_f=float(eqerr[:, 9:12].mean()),   # layout [om,v,tau,f]
                    eq_tau=float(eqerr[:, 6:9].mean()),
                    eq_om=float(eqerr[:, 0:3].mean()),
                    eq_v=float(eqerr[:, 3:6].mean()),
                )
                # audit C: slices
                sl = {"contact": slices[f"contact_{name}"], "highwork": slices[f"highwork_{name}"]}
                for k, m in sl.items():
                    if m.sum() > 10:
                        rec[f"W_mae_{k}"] = float(Werr[m].mean())
                        rec[f"n_{k}"] = int(m.sum())
                ev[name] = rec
            results[f"{rep}/seed{seed}"] = ev
            print(f"[{rep} s{seed}] same W={ev['same']['W_mae']:.4f} "
                  f"held W={ev['held']['W_mae']:.4f} "
                  f"(f={ev['held']['eq_f']:.4f} tau={ev['held']['eq_tau']:.4f} "
                  f"om={ev['held']['eq_om']:.4f} v={ev['held']['eq_v']:.4f})", flush=True)
    json.dump(results, open(out_path, "w"), indent=1)
    print("[wrote]", out_path)
    print("baselines:", json.dumps(audit["baselines_held"], indent=1))


def train_one(Xf, Xc, Yinv, Yeq, seed, epochs=25, bs=256, lr=3e-4):
    torch.manual_seed(seed)
    net = TCN(Xf.shape[2], Xc.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    xf, xc = torch.tensor(Xf, device=DEVICE), torch.tensor(Xc, device=DEVICE)
    yi, ye = torch.tensor(Yinv, device=DEVICE), torch.tensor(Yeq, device=DEVICE)
    n = len(xf)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for k in range(0, n, bs):
            s = perm[k:k+bs]
            pi, pe = net(xf[s], xc[s])
            li = nn.functional.smooth_l1_loss(pi[:, :2], yi[s][:, :2]) + \
                 nn.functional.binary_cross_entropy_with_logits(pi[:, 2:], yi[s][:, 2:])
            le = nn.functional.smooth_l1_loss(pe, ye[s])
            (li + le).backward()
            opt.step(); opt.zero_grad()
    net.eval()
    return net

@torch.no_grad()
def predict(net, Xf, Xc):
    pi, pe = [], []
    for k in range(0, len(Xf), 512):
        a, b = net(torch.tensor(Xf[k:k+512], device=DEVICE),
                   torch.tensor(Xc[k:k+512], device=DEVICE))
        pi.append(a.cpu().numpy()); pe.append(b.cpu().numpy())
    return np.concatenate(pi), np.concatenate(pe)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("out")
    ap.add_argument("--unmatched", action="store_true")
    a = ap.parse_args()
    run(a.cache, a.out, matched_only=not a.unmatched)
