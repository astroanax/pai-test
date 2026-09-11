"""Corrected screw/twist + wrench-reference conventions (audit fixes).

F1. Twist: SO(3) logarithm / dt for omega (NOT gradient of increments),
     and spatial Plücker linear velocity v = pdot - omega x p.
F1b. Wrench: measurements are in the wrist (sensor) frame; transport to
     base origin: f_b = R f_s, tau_b = R tau_s, tau_O = tau_b + p x f_b.
     This makes (w, nu) a properly co-referenced covector/vector pair so
     power and the screw invariants are physically meaningful.
F2. Canonical frame uses the TIME-VARYING sensor pose:
     T_{0<-s(t)} = T_{b<-0}^{-1} T_{b<-s(t)}.
"""
import numpy as np


def quat_to_mat(q):
    """q is (w, x, y, z)."""
    w, x, y, z = q
    n = np.sqrt(x*x + y*y + z*z + w*w + 1e-12)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def so3_log(R):
    """Rotation vector (axis*angle) from R, |theta| < pi branch."""
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    if theta < 1e-9:
        return np.zeros(3)
    s = (R - R.T) / 2.0
    v = np.array([s[2, 1], s[0, 2], s[1, 0]])
    return v * theta / np.sin(theta)


def twists_from_pose(pose, dt=0.01, smooth=3):
    """Spatial twist (omega, v_Plucker) from (pos, quat-wxyz) stream.

    omega = so3_log(R_{t-1}^T R_t) / dt        [rad/s]
    v     = pdot - omega x p                    [m/s, spatial linear coord]
    Pose smoothing is applied to positions via causal moving average; the
    rotations are NOT re-differentiated.
    """
    T = len(pose)
    p = pose[:, :3].astype(np.float64).copy()
    if smooth > 1:
        k = np.ones(smooth) / smooth
        for i in range(3):
            p[:, i] = np.convolve(p[:, i], k, mode="same")
    R = np.stack([quat_to_mat(q) for q in pose[:, 3:7]])
    om = np.zeros((T, 3))
    for t in range(1, T):
        dR = R[t-1].T @ R[t]
        om[t] = so3_log(dR) / dt
    pdot = np.gradient(p, dt, axis=0)
    v = pdot - np.cross(om, p)
    return om, v


def wrench_base_origin(w_s, pose):
    """w_s = [tau_s; f_s] in wrist/sensor frame -> base-origin co-referenced
    w = [tau_O; f_b]."""
    f_s = w_s[3:]; tau_s = w_s[:3]
    R = quat_to_mat(pose[3:7])
    p = pose[:3]
    f_b = R @ f_s
    tau_b = R @ tau_s
    tau_O = tau_b + np.cross(p, f_b)
    return np.r_[tau_O, f_b]


def cotransform(w, nu, T):
    """Dual-adjoint on wrench, adjoint on twist, for (R, p)."""
    R, p = T
    tau, f = w[:3], w[3:]
    om, v = nu[:3], nu[3:]
    f2 = R @ f
    tau2 = R @ tau + np.cross(p, R @ f)
    om2 = R @ om
    v2 = R @ v + np.cross(p, R @ om)
    return np.r_[tau2, f2], np.r_[om2, v2]


def canonical_wrench_series(w_base_origin_hist, pose_hist, pose0):
    """Map base-origin wrench history into the INITIAL sensor frame.

    The time-varying sensor pose is already absorbed by the base-origin
    transport (wrench_base_origin); the remaining change of frame is the
    CONSTANT initial-pose transform T_{b<-0}:
        w_0(t) = Ad_{T_{0<-b}}^{-T} w_O(t),  T_{0<-b} = (R0^T, -R0^T p0)
    """
    R0 = quat_to_mat(pose0[3:7]); p0 = pose0[:3]
    T_0b = (R0.T, -R0.T @ p0)          # base -> initial-sensor frame
    out = np.empty_like(w_base_origin_hist)
    for k in range(len(w_base_origin_hist)):
        wt, _ = cotransform(w_base_origin_hist[k], np.zeros(6), T_0b)
        out[k] = wt
    return out


def canonical_wrench_single(w_base_origin, pose_t, pose0):
    R0 = quat_to_mat(pose0[3:7]); p0 = pose0[:3]
    T_0b = (R0.T, -R0.T @ p0)
    wt, _ = cotransform(w_base_origin, np.zeros(6), T_0b)
    return wt
