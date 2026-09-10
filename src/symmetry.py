"""Target-symmetry benchmark (prospectus main-17, Sec. 7 & 11 kill experiment).

Representations (force-motion interface only; context branch shared):
  M0: no force
  M1: raw screw coordinates [om,v,f,tau] (train-normalized)
  M2: calibrated canonical frame (episode-initial-pose co-transform)
  M3: M1 + rigid-frame co-transform augmentation (matched optimizer steps)
  M4: polynomial pair invariants (Crook-Donelan generators, signed-log)
  M7: minimal equivariant/query-conditioned baseline:
      body-frame twist + wrench expressed in the moving EE frame (gauge-free
      input) plus query-frame transport applied analytically at the output.

Targets:
  invariant:  W_H = int(f.v + tau.om) dt, A_H = int |f.v + tau.om| dt,
              C_H = contact onset/offset events
  equivariant: future wrench zeta_{t+H} and twist xi_{t+H} in query frame q
               (errors computed after mapping predictions into canonical frame)

Consistency residuals (Sec. 7.4):
  eps_inv(F) = E ||F(g.x) - F(x)||        (invariant heads)
  eps_eq(F)  = E ||F(g.x) - rho(g) F(x)|| (equivariant heads)

Splits: same-frame; synthetic gauge (unit test); held-out configuration.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from invariant_tests import cotransform, rand_T, skew, inv

def inv6(w, nu):
    """Vectorized 6 invariants. w: (...,6) [tau;f], nu: (...,6) [om;v]."""
    w = np.atleast_2d(w); nu = np.atleast_2d(nu)
    tau, f = w[:, :3], w[:, 3:]
    om, v = nu[:, :3], nu[:, 3:]
    return np.stack([np.sum(f*f,1), np.sum(om*om,1), np.sum(f*om,1),
                     np.sum(f*tau,1), np.sum(om*v,1),
                     np.sum(tau*om,1)+np.sum(f*v,1)], 1)

FPS = 100.0
HIST = 100            # 1 s @ 100Hz
H = 50                # 0.5 s primary horizon
CONTACT_THR = 2.0     # N

# ------------------------------------------------------------------ helpers
def slog(x, s):
    return np.sign(x) * np.log1p(np.abs(x) / s)

def quat_to_mat(q):
    """q is (w, x, y, z) — RH20T and REASSEMBLE both store w-first."""
    w, x, y, z = q
    n = np.sqrt(x*x + y*y + z*z + w*w + 1e-12)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def twists_from_pose(pose, dt=0.01, smooth=3):
    """Spatial twist (om, v) from (pos, quat) stream with causal smoothing."""
    T = len(pose)
    p = pose[:, :3].copy()
    if smooth > 1:
        k = np.ones(smooth) / smooth
        for i in range(3):
            p[:, i] = np.convolve(p[:, i], k, mode="same")
    R = np.stack([quat_to_mat(q) for q in pose[:, 3:7]])
    v = np.gradient(p, dt, axis=0)
    om = np.zeros((T, 3))
    for t in range(1, T):
        dR = R[t-1].T @ R[t]
        s = (dR - dR.T) / 2
        om[t] = [s[2,1], s[0,2], s[1,0]]
    om = np.gradient(om, dt, axis=0)
    return om, v

# ------------------------------------------------------------------ windows
def load_episodes(cache_dir):
    eps = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        d = np.load(f)
        name = os.path.basename(f)[:-4]
        eps.append(dict(name=name, cfg=name.split("__")[0],
                        w=d["w"], nu=d["nu"], pose=d["pose"], joint=d["joint"]))
    return eps

def episode_frames(e):
    """Cache per-episode derived arrays (power, contact, body-frame wrench)."""
    if "power" in e:
        return e
    e["power"] = np.sum(e["w"] * e["nu"], axis=1)   # tau.om + f.v
    e["fmag"] = np.linalg.norm(e["w"][:, 3:], axis=1)
    e["contact"] = e["fmag"] > CONTACT_THR
    return e

def windows(eps, stride=25):
    W = []
    for i, e in enumerate(eps):
        episode_frames(e)
        T = len(e["w"])
        for t in range(HIST, T - H - 2, stride):
            W.append((i, t))
    return W

# ------------------------------------------------------------------ interfaces
def iface(e, t, rep, scales, rng=None, T_aug=None):
    """Force-motion input sequence (HIST, d)."""
    w = e["w"][t-HIST:t]; nu = e["nu"][t-HIST:t]
    pose = e["pose"][t-HIST:t]
    if rep == "M0":
        return np.zeros((HIST, 1), np.float32)
    if rep in ("M1", "M3"):
        if T_aug is not None:
            wt, nut = np.empty_like(w), np.empty_like(nu)
            for k in range(HIST):
                a, b = cotransform(w[k], nu[k], T_aug)
                wt[k], nut[k] = a, b
            w, nu = wt, nut
        return np.c_[nu, w].astype(np.float32)          # [om,v,tau,f]
    if rep == "M2":
        # canonical frame: co-transform by inverse of episode initial pose
        q0 = e["pose"][0]
        R0 = quat_to_mat(q0[3:7]); Rt = R0.T; p = -Rt @ q0[:3]
        wt, nut = np.empty_like(w), np.empty_like(nu)
        for k in range(HIST):
            a, b = cotransform(w[k], nu[k], (Rt, p))
            wt[k], nut[k] = a, b
        return np.c_[nut, wt].astype(np.float32)
    if rep == "M4":
        I = inv6(w, nu)
        return slog(I, scales).astype(np.float32)
    if rep == "M7":
        # minimal equivariant: express wrench in the moving EE (body) frame.
        # Gauge-free input; equivariance carried analytically at output.
        R = np.stack([quat_to_mat(q) for q in pose[:, 3:7]])
        out = np.empty((HIST, 12), np.float32)
        for k in range(HIST):
            Rk = R[k]
            # body-frame twist: nu_b = [R^T om, R^T v]
            om_b = Rk.T @ nu[k, :3]; v_b = Rk.T @ nu[k, 3:]
            # body-frame wrench (moment rotated at same point, no transport)
            f_b = Rk.T @ w[k, 3:]; tau_b = Rk.T @ w[k, :3]
            out[k] = np.r_[om_b, v_b, tau_b, f_b]
        return out
    raise ValueError(rep)

def context(e, t):
    j = e["joint"][t-HIST:t]
    return np.r_[j.mean(0), j.std(0), j[-1]].astype(np.float32)

# ------------------------------------------------------------------ targets
def targets_inv(e, t):
    """Invariant targets at H: work, abs-work, contact onset/offset."""
    P = e["power"][t:t+H]
    W_H = float(np.sum(P) / FPS)
    A_H = float(np.sum(np.abs(P)) / FPS)
    c = e["contact"][t:t+H]
    onset = float(c.any() and not c[0])
    offset = float(c.any() and c[0] and not c[-1])
    return np.array([W_H, A_H, onset, offset], np.float32)

def targets_eq(e, t, q_R=None, q_p=None):
    """Equivariant targets: future wrench+twist at t+H in query frame.
    Default query frame = episode canonical (initial pose) frame."""
    w = e["w"][t+H-1]; nu = e["nu"][t+H-1]
    if q_R is None:
        q0 = e["pose"][0]
        q_R = quat_to_mat(q0[3:7]).T; q_p = -q_R @ q0[:3]
    wt, nut = cotransform(w, nu, (q_R, q_p))
    return np.r_[nut, wt].astype(np.float32)   # 12

def cotransform_vec(w, nu, T):
    return cotransform(w, nu, T)

# ------------------------------------------------------------------ scales
def fit_scales(eps, wins, n=400):
    rng = np.random.default_rng(0)
    sel = wins if len(wins) <= n else rng.choice(len(wins), n, replace=False).tolist()
    I = np.concatenate([inv6(eps[i]["w"][t-HIST:t], eps[i]["nu"][t-HIST:t])
                        for i, t in [wins[k] for k in sel]])
    return np.maximum(np.abs(np.quantile(I, 0.9, axis=0)), 1e-6)

def fit_norm(eps, wins, rep, scales, n=400):
    """Train-only per-feature mean/std of the interface (for stable training)."""
    rng = np.random.default_rng(1)
    sel = wins if len(wins) <= n else rng.choice(len(wins), n, replace=False).tolist()
    X = np.stack([iface(eps[i], t, rep, scales) for i, t in [wins[k] for k in sel]])
    return X.mean((0, 1)), X.std((0, 1)) + 1e-6
