"""RH20T episode parsing for the reciprocal-token study.

Episode folder: task_XXXX_user_XXXX_scene_XXXX_cfg_XXXX/
  transformed/{joint,force_torque,tcp,...}.npy — aligned low-dim arrays
  metadata.json — calibration timestamp etc.
Config calib: <cfg>/calib/<ts>/{extrinsics,intrinsics,tcp,devices}.npy

We need per-episode, synchronized at ~100Hz (or whatever the arrays give):
  - wrench w (6) with known reference point (sensor/TCP)
  - EE pose (7) -> twist nu via causal smoothing + SE(3)-safe differentiation
  - gripper/joint context (shared context branch)
Outputs one .npz per episode in cache/.
"""
import argparse, glob, json, os
import numpy as np

def quat_to_mat(q):
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

def pose_diff_twist(pose, dt, smooth=3):
    """Body twist from consecutive (pos(3), quat(4)) poses.
    Causal moving-average smoothing of width `smooth` before differentiation."""
    T = len(pose)
    p = pose[:, :3]
    R = np.stack([quat_to_mat(q) for q in pose[:, 3:7]])
    if smooth > 1:
        k = np.ones(smooth) / smooth
        p = np.stack([np.convolve(p[:, i], k, mode="same") for i in range(3)], 1)
        Rf = np.stack([R[max(i-smooth//2,0):i+smooth//2+1].mean(0) for i in range(T)])
        # re-orthonormalize
        R = np.stack([r / np.linalg.norm(r, axis=0, keepdims=True) for r in Rf])
    # linear velocity in world frame
    v = np.gradient(p, dt, axis=0)
    # angular velocity: skew((R_{t-1}^T R_t - I)) / dt approx in body frame
    om = np.zeros((T, 3))
    for t in range(1, T):
        dR = R[t-1].T @ R[t]
        sk = (dR - dR.T) / 2
        om[t] = np.array([sk[2,1], sk[0,2], sk[1,0]])
    om = np.gradient(om, dt, axis=0)
    return om, v, R, p

def parse_episode(ep_dir, out_dir):
    tr = os.path.join(ep_dir, "transformed")
    files = {n: os.path.join(tr, n + ".npy")
             for n in ["force_torque", "tcp", "joint", "ft", "ft_ext", "base"]
             if os.path.exists(os.path.join(tr, n + ".npy"))}
    if "force_torque" not in files or "tcp" not in files:
        return None
    ft = np.load(files["force_torque"])     # (T,6) or (T,>6)
    tcp = np.load(files["tcp"])             # (T,7)
    if ft.ndim != 2 or tcp.ndim != 2 or len(ft) < 60 or len(tcp) < 60:
        return None
    T = min(len(ft), len(tcp))
    ft, tcp = ft[:T], tcp[:T]
    joint = np.load(files["joint"])[:T] if "joint" in files else np.zeros((T, 1))
    dt = 0.01  # 100Hz nominal; verified against timestamps when present
    om, v, R, p = pose_diff_twist(tcp, dt)
    # wrench: RH20T stores [fx fy fz tx ty tz] (force first) -> paper order [tau; f]
    f = ft[:, :3]
    tau = ft[:, 3:6] if ft.shape[1] >= 6 else np.zeros((T, 3))
    w = np.c_[tau, f]                       # [tau; f]
    nu = np.c_[om, v]                       # [om; v]
    name = "_".join(os.path.basename(ep_dir).split("_")[:8])
    cfg = os.path.basename(os.path.dirname(ep_dir))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{cfg}__{name}.npz"),
                        w=w.astype(np.float32), nu=nu.astype(np.float32),
                        pose=tcp.astype(np.float32), joint=joint.astype(np.float32))
    return T

def main(root, out_dir, limit=None):
    eps = sorted(glob.glob(os.path.join(root, "*", "task_*")))
    print(f"{len(eps)} episodes found")
    n_ok = 0
    for i, ep in enumerate(eps):
        if limit and n_ok >= limit:
            break
        if parse_episode(ep, out_dir) is not None:
            n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(eps)} parsed, {n_ok} ok", flush=True)
    print(f"parsed {n_ok}/{len(eps)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.root, a.out, a.limit)
