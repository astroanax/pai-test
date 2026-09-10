"""Unit tests for the reciprocal token (prospectus Props 1-2).

I(w, nu) = [f.f, om.om, f.om, f.tau, om.v, tau.om + f.v], w=[tau;f], nu=[om;v]
Cotransform: f'=Rf, tau'=R tau + p x R f, om'=R om, v'=R v + p x R om.
"""
import numpy as np

def rand_T(rng, scale=0.5):
    A = rng.normal(size=(3, 3)); Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    p = rng.normal(size=3) * scale
    return Q, p

def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

def inv(w, nu):
    tau, f = w[:3], w[3:]
    om, v = nu[:3], nu[3:]
    return np.array([f @ f, om @ om, f @ om, f @ tau, om @ v,
                     tau @ om + f @ v])

def cotransform(w, nu, T):
    Q, p = T
    tau, f = w[:3], w[3:]
    om, v = nu[:3], nu[3:]
    f2 = Q @ f
    tau2 = Q @ tau + np.cross(p, Q @ f)
    om2 = Q @ om
    v2 = Q @ v + np.cross(p, Q @ om)
    return np.r_[tau2, f2], np.r_[om2, v2]

def power(w, nu):
    return w @ nu

def test_invariance(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    err = []
    for _ in range(n):
        w = rng.normal(size=6) * np.array([.05, .05, .05, 10, 10, 10])
        nu = rng.normal(size=6) * np.array([.5, .5, .5, .2, .2, .2])
        T = rand_T(rng)
        w2, nu2 = cotransform(w, nu, T)
        err.append(np.max(np.abs(inv(w, nu) - inv(w2, nu2))))
    return max(err)

def test_power(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    err = []
    for _ in range(n):
        w = rng.normal(size=6); nu = rng.normal(size=6)
        T = rand_T(rng)
        w2, nu2 = cotransform(w, nu, T)
        err.append(abs(power(w, nu) - power(w2, nu2)))
    return max(err)

def find_transform(w, nu, w2, nu2):
    """Given two pairs with equal invariants (non-degenerate), find (Q,p)."""
    f, tau = w[3:], w[:3]; om, v = nu[:3], nu[3:]
    f2, tau2 = w2[3:], w2[:3]; om2, v2 = nu2[:3], nu2[3:]
    # build right-handed frames from the non-collinear pair
    def frame(a, b):
        e1 = a / np.linalg.norm(a)
        e2 = b - (b @ e1) * e1
        e2 = e2 / np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        return np.stack([e1, e2, e3])          # rows
    F, F2 = frame(f, om), frame(f2, om2)
    Q = F2.T @ F                                # maps F-rows to F2-rows
    if not np.allclose(Q @ f, f2, atol=1e-5):
        return None
    if not np.allclose(Q @ om, om2, atol=1e-5):
        return None
    # tau2 - Q tau = p x (Q f);  v2 - Q v = p x (Q om)  =>  -skew(Qf) p = dtau
    A = np.vstack([-skew(Q @ f), -skew(Q @ om)])    # 6x3, solves A p = b
    b = np.r_[tau2 - Q @ tau, v2 - Q @ v]
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.max(np.abs(A @ p - b)) > 1e-5:
        return None
    return Q, p

def test_completeness(n=500, seed=2):
    """Round trip: transform a random non-degenerate pair, check the
    invariants are equal and the transform is recoverable (Prop 2 'if')."""
    rng = np.random.default_rng(seed)
    ok, equal, degenerate = 0, 0, 0
    for _ in range(n):
        w = rng.normal(size=6) * np.array([.05, .05, .05, 10, 10, 10])
        nu = rng.normal(size=6) * np.array([.5, .5, .5, .2, .2, .2])
        f, om = w[3:], nu[:3]
        if np.linalg.norm(np.cross(f, om)) < 1e-3:
            degenerate += 1
            continue
        T = rand_T(rng)
        w2, nu2 = cotransform(w, nu, T)
        if np.allclose(inv(w, nu), inv(w2, nu2), atol=1e-6):
            equal += 1
        Trec = find_transform(w, nu, w2, nu2)
        if Trec is not None:
            ok += 1
    return equal, ok, degenerate, n

if __name__ == "__main__":
    print("Prop1 invariance max err (1000 transforms):", test_invariance())
    print("Power invariance max err:", test_power())
    eq, rec, deg, n = test_completeness()
    print(f"Prop2 round-trip: invariants equal {eq}/{n-deg}, "
          f"transform recovered {rec}/{n-deg}, degenerate {deg}")
