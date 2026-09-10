"""Reciprocal-token study: B0-B5 representations, forecasting, consistency.

Pipeline (prospectus Sec. 5-7):
  windows: 1s causal history (H=100 frames @100Hz), future H in {0.1,0.5,1.0}s
  target: future reciprocal sequence + contact onset/offset + power sign
  models: identical 4-layer TCN (width 128), only force-motion branch changes

Representations:
  B0: no F/T
  B1: raw [w; nu] (12)
  B2: calibrated common-frame [w_b; nu_b] — here: base-frame co-transform using
      episode's initial pose as canonical origin (natural stored-frame shift)
  B3: raw + rigid-frame augmentation (co-transforms, matched optimizer steps)
  B4: power-only [||f||, ||w||, w.nu] (3)
  B5: reciprocal token (6, signed-log, train-fit scales)

Consistency test (7.1): C_H = E ||r_hat(T1) - r_hat(T2)|| on co-transformed views.
Held-out config (7.2): train cfg1 -> eval cfg2 (and rotate).
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from invariant_tests import cotransform, rand_T

FPS = 100.0
HIST = 100          # 1 s
FUT = {0.1: 10, 0.5: 50, 1.0: 100}
CONTACT_THR = 2.0   # N, preregistered

# ---------------------------------------------------------------- invariants
def inv6(w, nu):
    tau, f = w[..., :3], w[..., 3:]
    om, v = nu[..., :3], nu[..., 3:]
    return np.stack([np.sum(f*f,-1), np.sum(om*om,-1), np.sum(f*om,-1),
                     np.sum(f*tau,-1), np.sum(om*v,-1),
                     np.sum(tau*om,-1)+np.sum(f*v,-1)], -1)

def slog(x, s):
    return np.sign(x) * np.log1p(np.abs(x) / s)

def inv_scales(hist_list):
    I = np.concatenate([inv6(w, nu) for w, nu in hist_list])
    return np.maximum(np.abs(np.quantile(I, 0.9, axis=0)), 1e-6)

# ---------------------------------------------------------------- windows
def load_episodes(cache_dir):
    eps = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        d = np.load(f)
        name = os.path.basename(f)[:-4]
        cfg = name.split("__")[0]
        eps.append(dict(name=name, cfg=cfg, w=d["w"], nu=d["nu"],
                        pose=d["pose"], joint=d["joint"]))
    return eps

def make_windows(eps, stride=25):
    """Yield (ep_idx, t) windows with full history+future coverage."""
    W = []
    for i, e in enumerate(eps):
        T = len(e["w"])
        tmax = min(T - HIST - FUT[1.0] - 2, T - 1)
        for t in range(HIST, max(tmax, 0), stride):
            W.append((i, t))
    return W

# ---------------------------------------------------------------- B0-B5 input builders
def build_inputs(eps, win, rep, scales, rng=None, aug_T=None):
    """Return force-motion input sequence for a window under representation rep."""
    i, t = win
    e = eps[i]
    w = e["w"][t-HIST:t]          # (100,6)
    nu = e["nu"][t-HIST:t]
    if rep == "B0":
        return np.zeros((HIST, 1), np.float32)
    if rep == "B1":
        return np.c_[w, nu].astype(np.float32)
    if rep == "B2":
        # canonical base frame: co-transform by inverse of episode initial pose
        p0, q0 = e["pose"][0, :3], e["pose"][0, 3:7]
        # build T^-1 acting on (w,nu): rotate by R0^T, translate by -R0^T p0
        from invariant_tests import skew
        x, y, z, ww = q0
        R0 = np.array([[1-2*(y*y+z*z),2*(x*y-z*ww),2*(x*z+y*ww)],
                       [2*(x*y+z*ww),1-2*(x*x+z*z),2*(y*z-x*ww)],
                       [2*(x*z-y*ww),2*(y*z+x*ww),1-2*(x*x+y*y)]])
        Rt = R0.T
        p = -Rt @ p0
        wt, nut = np.empty_like(w), np.empty_like(nu)
        for k in range(HIST):
            a, b = cotransform(w[k], nu[k], (Rt, p))
            wt[k], nut[k] = a, b
        return np.c_[wt, nut].astype(np.float32)
    if rep == "B3":
        if aug_T is None:
            aug_T = rand_T(rng) if rng else (np.eye(3), np.zeros(3))
        wt, nut = np.empty_like(w), np.empty_like(nu)
        for k in range(HIST):
            a, b = cotransform(w[k], nu[k], aug_T)
            wt[k], nut[k] = a, b
        return np.c_[wt, nut].astype(np.float32)
    if rep == "B4":
        tau, f = w[:, :3], w[:, 3:]
        om, v = nu[:, :3], nu[:, 3:]
        return np.c_[np.linalg.norm(f,axis=1), np.linalg.norm(om,axis=1),
                     np.sum(w*nu,axis=1)].astype(np.float32)
    if rep == "B5":
        I = inv6(w, nu)
        return slog(I, scales).astype(np.float32)
    raise ValueError(rep)

def build_context(eps, win):
    """Shared context branch: joint history stats + gripper (no F/T)."""
    i, t = win
    e = eps[i]
    j = e["joint"][t-HIST:t]
    return np.r_[j.mean(0), j.std(0), j[-1]].astype(np.float32)

def build_target(eps, win, scales):
    """Future reciprocal sequence (downsampled to 10Hz) + contact event + power sign."""
    i, t = win
    e = eps[i]
    out = []
    for hs in [FUT[0.1], FUT[0.5], FUT[1.0]]:
        fut_w = e["w"][t:t+hs]
        fut_nu = e["nu"][t:t+hs]
        I = inv6(fut_w[::10], fut_nu[::10])     # ~10 samples per horizon
        out.append(slog(I, scales).ravel())
    # contact onset/offset in next 0.5s: |f| crosses threshold
    fm = np.linalg.norm(e["w"][t:t+FUT[0.5], 3:], axis=1)
    contact = fm > CONTACT_THR
    onset = 1.0 if (contact.any() and not contact[0]) else 0.0
    offset = 1.0 if (contact.any() and contact[0] and not contact[-1]) else 0.0
    # power sign at +0.25s
    P = np.sum(e["w"][t:t+25] * e["nu"][t:t+25], axis=1).mean()
    psign = 1.0 if P > 0 else 0.0
    return dict(recip=np.concatenate(out).astype(np.float32),
                onset=onset, offset=offset, psign=psign,
                fmag_t=float(fm[0]))

# ---------------------------------------------------------------- model (torch)
import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TCN(nn.Module):
    def __init__(self, din_force, dctx, dout, width=128):
        super().__init__()
        self.force = nn.Conv1d(din_force, width, 5, padding=4, dilation=1)[:-1] if False else None
        self.f1 = nn.Conv1d(din_force, width, 5, padding=4)
        self.c1 = nn.Linear(dctx, width)
        layers = []
        for i, d in enumerate([1, 2, 4]):
            layers += [nn.Conv1d(width, width, 5, padding=4*d, dilation=d),
                       nn.GELU(), nn.Dropout(0.1)]
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(width, dout)
    def forward(self, x, c):          # x (B,T,din), c (B,dctx)
        h = self.f1(x.transpose(1, 2))       # (B,W,T)
        h = h[:, :, :x.shape[1]] if h.shape[-1] > x.shape[1] else h
        h = h + self.c1(c)[:, :, None]
        h = self.tcn(h)
        return self.head(h[:, :, -1])

def train_model(Xf, Xc, Y, rep, epochs=30, bs=256, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    net = TCN(Xf.shape[2], Xc.shape[1], Y.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    xf = torch.tensor(Xf, device=DEVICE); xc = torch.tensor(Xc, device=DEVICE)
    y = torch.tensor(Y, device=DEVICE)
    n = len(xf)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for k in range(0, n, bs):
            s = perm[k:k+bs]
            loss = nn.functional.smooth_l1_loss(net(xf[s], xc[s]), y[s])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net

@torch.no_grad()
def predict(net, Xf, Xc):
    outs = []
    for k in range(0, len(Xf), 512):
        o = net(torch.tensor(Xf[k:k+512], device=DEVICE),
                torch.tensor(Xc[k:k+512], device=DEVICE))
        outs.append(o.cpu().numpy())
    return np.concatenate(outs)

# ---------------------------------------------------------------- eval metrics
def consistency_C(net, eps, wins, rep, scales, n_views=3, seed=0):
    """C_H: mean ||r(T1) - r(T2)|| over co-transformed views of same window."""
    rng = np.random.default_rng(seed)
    ds = []
    for win in wins:
        i, t = win
        e = eps[i]
        w = e["w"][t-HIST:t]; nu = e["nu"][t-HIST:t]
        base = build_inputs(eps, win, "B5", scales)
        views = [base]
        for _ in range(n_views - 1):
            T = rand_T(rng)
            wt, nut = np.empty_like(w), np.empty_like(nu)
            for k in range(HIST):
                a, b = cotransform(w[k], nu[k], T)
                wt[k], nut[k] = a, b
            # build the rep from the transformed raw arrays
            e2 = dict(w=wt, nu=nut, pose=e["pose"], joint=e["joint"])
            eps2 = [e2]
            v = build_inputs(eps2, (0, t), rep, scales)
            views.append(v)
        ctx = build_context(eps, win)[None, :]
        preds = [predict(net, v[None], ctx)[0] for v in views]
        for p in preds[1:]:
            ds.append(np.abs(preds[0] - p).mean())
    return float(np.mean(ds))

def eval_mae(pred, target, scales_len=6):
    return float(np.abs(pred - target).mean())
