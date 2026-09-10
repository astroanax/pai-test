"""Kill-experiment runner (main-17 Sec. 11): M0-M4 + M7, one TCN,
invariant (W_H, C_H) and equivariant (future wrench/twist) targets.

Splits: same-frame episode-disjoint; held-out configuration (cfg1 -> cfg2).
Consistency: synthetic gauge co-transforms on test windows.
Run on aiclub A100.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from symmetry import (load_episodes, episode_frames, windows, iface, context,
                      targets_inv, targets_eq, fit_scales, fit_norm,
                      HIST, H, FPS)
from invariant_tests import rand_T, cotransform

import torch
import torch.nn as nn
torch.backends.cudnn.enabled = False   # cuDNN broken on aiclub torch build
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REPS = ["M0", "M1", "M2", "M3", "M4", "M7"]


class TCN(nn.Module):
    """4-layer dilated TCN, width 128, GELU, residual; two heads."""
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
    def forward(self, x, c):                     # x (B,T,d) c (B,dctx)
        h = self.fin(x.transpose(1, 2))[:, :, :x.shape[1]]
        h = h + self.cin(c)[:, :, None]
        h = h + self.b1(h); h = h + self.b2(h); h = h + self.b3(h)
        z = h[:, :, -1]
        return self.head_inv(z), self.head_eq(z)


def build(eps, wins, rep, scales, norm, rng=None):
    Xf = np.stack([iface(eps[i], t, rep, scales, rng=rng) for i, t in wins])
    Xc = np.stack([context(eps[i], t) for i, t in wins])
    mu, sd = norm
    return ((Xf - mu) / sd).astype(np.float32), Xc


def build_targets(eps, wins):
    Yinv = np.stack([targets_inv(eps[i], t) for i, t in wins])
    Yeq = np.stack([targets_eq(eps[i], t) for i, t in wins])
    return Yinv, Yeq


def train(Xf, Xc, Yinv, Yeq, epochs=25, bs=256, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    net = TCN(Xf.shape[2], Xc.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    xf, xc = torch.tensor(Xf, device=DEVICE), torch.tensor(Xc, device=DEVICE)
    yi, ye = (torch.tensor(Yinv, device=DEVICE), torch.tensor(Yeq, device=DEVICE))
    # standardize equivariant target by train stats (per-component)
    n = len(xf)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for k in range(0, n, bs):
            s = perm[k:k+bs]
            pi, pe = net(xf[s], xc[s])
            # invariant: Huber on W/A + BCE on events; equivariant: Huber
            li = nn.functional.smooth_l1_loss(pi[:, :2], yi[s][:, :2]) + \
                 nn.functional.binary_cross_entropy_with_logits(pi[:, 2:], yi[s][:, 2:])
            le = nn.functional.smooth_l1_loss(pe, ye[s])
            loss = li + le
            opt.zero_grad(); loss.backward(); opt.step()
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


def gauge_consistency(net, eps, wins, rep, scales, norm, n=60, n_views=3, seed=0):
    """eps_inv and eps_eq over co-transformed views (Sec. 7.4).
    Equivariant residual: transform prediction analytically by rho(g) and
    compare with prediction from the transformed input."""
    rng = np.random.default_rng(seed)
    di, de = [], []
    for i, t in wins[:n]:
        e = eps[i]
        x0 = iface(e, t, rep, scales)
        c = context(e, t)[None, :]
        pi0, pe0 = predict(net, ((x0 - norm[0]) / norm[1])[None], c)
        for _ in range(n_views - 1):
            T = rand_T(rng)
            # co-transform the physical window
            w = e["w"][t-HIST:t]; nu = e["nu"][t-HIST:t]
            wt, nut = np.empty_like(w), np.empty_like(nu)
            for k in range(HIST):
                a, b = cotransform(w[k], nu[k], T)
                wt[k], nut[k] = a, b
            e2 = dict(w=wt, nu=nut, pose=e["pose"], joint=e["joint"], cfg=e["cfg"])
            episode_frames(e2) if "power" not in e2 else None
            x1 = iface(e2, HIST, rep, scales)
            pi1, pe1 = predict(net, ((x1 - norm[0]) / norm[1])[None], c)
            # invariant residual
            di.append(np.abs(pi1[0] - pi0[0]).mean())
            # equivariant residual: pe0 transported by g should match pe1
            R, p = T
            pe0_w = pe0[0][6:9]; pe0_tau = pe0[0][9:12]  # [nu, w] ordering [om,v,tau,f]
            pe0_om, pe0_v = pe0[0][:3], pe0[0][3:6]
            # rho(g) on (nu, w): Ad_T nu, Ad^{-T} w
            om2 = R @ pe0_om; v2 = R @ pe0_v + np.cross(p, R @ pe0_om)
            f2 = R @ pe0_w; tau2 = R @ pe0_tau + np.cross(p, R @ pe0_w)
            pe0_t = np.r_[om2, v2, tau2, f2]
            de.append(np.abs(pe1[0] - pe0_t).mean())
    return float(np.mean(di)), float(np.mean(de))


def main(cache, out, seed=0, ntrain=2500, ntest=600):
    rng = np.random.default_rng(seed)
    eps = load_episodes(cache)
    for e in eps:
        episode_frames(e)
    cfgs = sorted(set(e["cfg"] for e in eps))
    print(f"{len(eps)} episodes; configs {cfgs}")
    by_cfg = {c: [i for i, e in enumerate(eps) if e["cfg"] == c] for c in cfgs}
    for c in cfgs:
        rng.shuffle(by_cfg[c])
    if len(cfgs) >= 2:
        # held-out config: train on all but the last config, test on it + same-frame
        held = cfgs[-1]
        train_eps = [i for c in cfgs[:-1] for i in by_cfg[c][:40]]
        same_test_eps = [i for c in cfgs[:-1] for i in by_cfg[c][40:45]]
        held_test_eps = by_cfg[held][:15]
    else:
        # single configuration: 80/20 episode split, held-out = same split
        idxs = by_cfg[cfgs[0]]
        k = int(len(idxs) * 0.8)
        train_eps = idxs[:k]
        same_test_eps = idxs[k:int(k + 0.1 * len(idxs))]
        held_test_eps = idxs[int(k + 0.1 * len(idxs)):]
    role = {}
    for i in train_eps: role[i] = "train"
    for i in same_test_eps: role[i] = "same_test"
    for i in held_test_eps: role[i] = "held_test"

    allw = windows(eps, stride=25)
    tr_w = [w for w in allw if role.get(w[0]) == "train"]
    sa_w = [w for w in allw if role.get(w[0]) == "same_test"]
    he_w = [w for w in allw if role.get(w[0]) == "held_test"]
    rng.shuffle(tr_w); rng.shuffle(sa_w); rng.shuffle(he_w)
    tr_w, sa_w, he_w = tr_w[:ntrain], sa_w[:ntest], he_w[:ntest]
    print(f"windows tr/sa/he: {len(tr_w)}/{len(sa_w)}/{len(he_w)}")

    scales = fit_scales(eps, tr_w)
    Yinv_tr, Yeq_tr = build_targets(eps, tr_w)
    Yinv_sa, Yeq_sa = build_targets(eps, sa_w)
    Yinv_he, Yeq_he = build_targets(eps, he_w)
    # target normalization (train)
    mu_i, sd_i = Yinv_tr[:, :2].mean(0), Yinv_tr[:, :2].std(0) + 1e-6
    mu_e, sd_e = Yeq_tr.mean(0), Yeq_tr.std(0) + 1e-6

    results = {}
    for rep in REPS:
        norm = fit_norm(eps, tr_w, rep, scales)
        Xf_tr, Xc_tr = build(eps, tr_w, rep, scales, norm)
        if rep == "M3":
            # augmentation: 2 co-transformed views per window
            Xf_a = []
            for i, t in tr_w:
                Xf_a.append(iface(eps[i], t, "M3", scales, rng=rng, T_aug=rand_T(rng)))
            Xf_tr = np.concatenate([Xf_tr, np.stack(Xf_a).astype(np.float32)])
            Xc_tr = np.concatenate([Xc_tr, Xc_tr])
            Yinv_tr2 = np.concatenate([Yinv_tr, Yinv_tr])
            Yeq_tr2 = np.concatenate([Yeq_tr, Yeq_tr])
        else:
            Yinv_tr2, Yeq_tr2 = Yinv_tr, Yeq_tr
        net = train(Xf_tr, Xc_tr, Yinv_tr2, Yeq_tr2, seed=seed)
        # eval same-frame + held-out
        Xf_sa, Xc_sa = build(eps, sa_w, rep, scales, norm)
        Xf_he, Xc_he = build(eps, he_w, rep, scales, norm)
        pi_sa, pe_sa = predict(net, Xf_sa, Xc_sa)
        pi_he, pe_he = predict(net, Xf_he, Xc_he)
        mae = lambda P, Y: float(np.abs(P - Y).mean())
        # gauge consistency (raw reps see transformed views; M4 sees invariants)
        eps_i, eps_e = gauge_consistency(net, eps, sa_w, rep, scales, norm)
        results[rep] = dict(
            inv_same=dict(W_mae=mae(pi_sa[:, 0], Yinv_sa[:, 0]),
                          A_mae=mae(pi_sa[:, 1], Yinv_sa[:, 1]),
                          C_brier=mae(pi_sa[:, 2:], Yinv_sa[:, 2:])),
            eq_same=dict(eq_mae=mae(pe_sa, Yeq_sa)),
            inv_held=dict(W_mae=mae(pi_he[:, 0], Yinv_he[:, 0]),
                          A_mae=mae(pi_he[:, 1], Yinv_he[:, 1]),
                          C_brier=mae(pi_he[:, 2:], Yinv_he[:, 2:])),
            eq_held=dict(eq_mae=mae(pe_he, Yeq_he)),
            consistency=dict(eps_inv=eps_i, eps_eq=eps_e),
        )
        r = results[rep]
        print(f"[{rep}] same: W={r['inv_same']['W_mae']:.4f} eq={r['eq_same']['eq_mae']:.4f} | "
              f"held: W={r['inv_held']['W_mae']:.4f} eq={r['eq_held']['eq_mae']:.4f} | "
              f"cons: inv={eps_i:.4f} eq={eps_e:.4f}", flush=True)
    json.dump(results, open(out, "w"), indent=1)
    print("[wrote]", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.cache, a.out, a.seed)
