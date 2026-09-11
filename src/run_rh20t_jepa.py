"""RH20T cross-hardware JEPA: J0/J1/J2 with 3-way config holdout, delta target.

Train on two configurations, hold out the third, rotating all 3 ways. For each
holdout we run J0 (no force), J1 (raw sensor wrench), J2 (canonical wrench).
Reuses train_jepa / predict_full from run_jepa_v2 (same A-dict contract).
Windows at the (video=force) frame rate; HIST=30 (~3s), H_FUT=5 (~0.5s),
RGB_OFFS=[9,6,3,0] for 4 current frames, future target at +5.
"""
import numpy as np, os, sys, pickle, glob, json, argparse
sys.path.insert(0, os.path.expanduser("~/rehan/reciprocal-tokens/src"))
from screw_fixed import (quat_to_mat, so3_log, wrench_base_origin,
                         canonical_wrench_series, canonical_wrench_single,
                         twists_from_pose)
from symmetry import CONTACT_THR
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from run_jepa_v2 import (train_jepa, predict_full, swap_matched,
                         last_frame_baseline, _delta_target)

ROOT = os.path.expanduser("~/rehan/reciprocal-tokens")
CACHE = os.path.join(ROOT, "cache_rh20t_v")
HIST = 30          # 3s force history at 10fps
H_FUT = 5          # 0.5s future
RGB_OFFS = [9, 6, 3, 0]

def epkey(cfg, name):
    return f"{cfg}__{name}"

def load_ep(cfg, name):
    z = np.load(os.path.join(CACHE, epkey(cfg, name) + ".npz"))
    return {k: z[k] for k in z.files}

def all_windows_rh():
    """All (cfg, name, t) windows across cached episodes."""
    recs = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        base = os.path.basename(f)[:-4]
        cfg = base.split("__")[0]
        name = "__".join(base.split("__")[1:])
        e = load_ep(cfg, name)
        n = len(e["w"])
        if n < HIST + H_FUT + 2:
            continue
        fmag = np.linalg.norm(e["w"][:, 3:], axis=1)
        contact = fmag > CONTACT_THR
        for t in range(HIST, n - H_FUT - 1):
            hist_c = bool(contact[t-HIST:t].any())
            fut = contact[t:t+H_FUT]
            onset = bool(fut.any() and not fut[0])
            anyf = bool(fut.any())
            recs.append(dict(cfg=cfg, name=name, t=int(t),
                             hist_contact=hist_c, fut_onset=onset, fut_any=anyf))
    return recs

def assemble_rh(recs, feats):
    """Build the A-dict used by train_jepa. w_s raw (sensor), canonical via
    base-origin transport + init frame. Quat reorder (x,y,z,w)->(w,x,y,z).
    meta = (episode_id, t) with episode_id a stable int per (cfg,name)."""
    Xv, Xp, Xw_raw, Xw_can, Xph = [], [], [], [], []
    Yv_idx, Yc, Yw_raw, Yw_can, meta = [], [], [], [], []
    ep_id = {}
    nxt = 0
    for w in recs:
        cfg, name, t = w["cfg"], w["name"], w["t"]
        if (cfg, name) not in ep_id:
            ep_id[(cfg, name)] = nxt; nxt += 1
        keys = [(cfg, name, t - o) for o in RGB_OFFS] + [(cfg, name, t + H_FUT)]
        if any(k not in feats for k in keys):
            continue
        e = load_ep(cfg, name)
        if t + H_FUT >= len(e["w"]):
            continue
        # pose reorder: tcp (x,y,z,qx,qy,qz,qw) -> (x,y,z,w,x,y,z)
        pose = e["pose"]
        pose_w = np.c_[pose[:, :3], pose[:, 6], pose[:, 3:6]]
        w_s = e["w"]
        # raw sensor-frame wrench history (J1)
        w_raw = w_s[t-HIST:t].astype(np.float32)
        # canonical: base-origin transport then init-frame (J2)
        w_O = np.stack([wrench_base_origin(w_s[k], pose_w[k])
                        for k in range(t-HIST, t)])
        w_can = canonical_wrench_series(w_O, pose_w[t-HIST:t], pose_w[0])
        # future canonical target
        w_O_f = wrench_base_origin(w_s[t+H_FUT], pose_w[t+H_FUT])
        w_can_f = canonical_wrench_single(w_O_f, pose_w[t+H_FUT], pose_w[0])
        # proprio: mean/std of pose over history + future pose
        prop = np.r_[pose_w[t-HIST:t].mean(0), pose_w[t-HIST:t].std(0),
                     pose_w[t+H_FUT]].astype(np.float32)
        fmag = np.linalg.norm(w_s[t-HIST:t, 3:], axis=1)
        phase = np.array([fmag.mean(), fmag.max(),
                          float((fmag > CONTACT_THR).mean())], np.float32)
        Xv.append(np.stack([feats[k] for k in keys[:4]]))
        Xp.append(prop); Xw_raw.append(w_raw); Xw_can.append(w_can)
        Xph.append(phase)
        Yv_idx.append(keys[4])
        Yc.append([float(w["fut_onset"]), float(w["fut_any"])])
        Yw_raw.append(w_s[t+H_FUT]); Yw_can.append(w_can_f)
        meta.append((ep_id[(cfg, name)], t))
    if not Xv:
        return None
    return dict(vis=np.stack(Xv), prop=np.stack(Xp),
                w_raw=np.stack(Xw_raw).astype(np.float32),
                w_can=np.stack(Xw_can).astype(np.float32),
                phase=np.stack(Xph),
                Yv=np.stack([feats[k] for k in Yv_idx]),
                Yc=np.array(Yc, np.float32),
                Yw_raw=np.stack(Yw_raw).astype(np.float32),
                Yw_can=np.stack(Yw_can).astype(np.float32),
                meta=np.array(meta, dtype=np.int64))

def swap_matched_rh(A, recs, rng):
    """Matched cross-window swap within (cfg, hist_contact) buckets.
    Returns (swapped_w_can_array, idx)."""
    buckets = {}
    for i, w in enumerate(recs):
        key = (w["cfg"], w["hist_contact"])
        buckets.setdefault(key, []).append(i)
    idx = np.arange(len(recs))
    for key, members in buckets.items():
        m = np.array(members)
        if len(m) < 2:
            continue
        target = np.roll(np.arange(len(m)), 1)   # derangement
        idx[m] = m[target]
    return A["w_can"].copy()[idx], idx

def split_holdout(recs, held_cfg, seed):
    """Train = episodes from the other 2 configs; eval = held-out config.
    Episode = (cfg, name). Returns (train_recs, eval_recs)."""
    rng = np.random.default_rng(seed)
    eps = sorted(set((r["cfg"], r["name"]) for r in recs))
    tr_eps = [e for e in eps if e[0] != held_cfg]
    ev_eps = [e for e in eps if e[0] == held_cfg]
    rng.shuffle(tr_eps)
    # cap training episodes for balance
    tr_eps = tr_eps[:80]
    trset = set(tr_eps); evset = set(ev_eps)
    tr = [r for r in recs if (r["cfg"], r["name"]) in trset]
    ev = [r for r in recs if (r["cfg"], r["name"]) in evset]
    return tr, ev

def run_rh(seed=0):
    feats = pickle.load(open(os.path.join(CACHE, "visfeat.pkl"), "rb"))
    recs = all_windows_rh()
    print(f"total windows: {len(recs)}", flush=True)
    cfgset = ["RH20T_cfg1", "RH20T_cfg2", "RH20T_cfg3"]
    out = {}
    for held in cfgset:
        tr, ev = split_holdout(recs, held, seed)
        A_tr = assemble_rh(tr, feats)
        A_ev = assemble_rh(ev, feats)
        if A_tr is None or A_ev is None:
            print(f"  HOLD {held}: no data", flush=True); continue
        # assert raw != canonical
        d1 = np.abs(A_tr["w_raw"] - A_tr["w_can"]).mean()
        assert d1 > 1e-3, "J1/J2 identical"
        prev = float(A_ev["Yc"][:, 1].mean())
        Yd = A_ev["Yv"] - A_ev["vis"][:, -1]
        res = {"n_train": len(A_tr["vis"]), "n_eval": len(A_ev["vis"]),
               "prevalence": prev,
               "baselines": {"last_frame_cos": last_frame_baseline(A_ev),
                             "mean_delta_cos": float(
                                 (1 - (Yd @ Yd.mean(0)) /
                                  (np.linalg.norm(Yd, axis=1) *
                                   np.linalg.norm(Yd.mean(0)) + 1e-9)).mean())}}
        print(f"\n=== HOLD {held}: train {len(A_tr['vis'])}, eval {len(A_ev['vis'])}, "
              f"prev {prev:.2f} ===", flush=True)
        print(f"  baselines: last_frame_cos={res['baselines']['last_frame_cos']:.4f} "
              f"mean_delta_cos={res['baselines']['mean_delta_cos']:.4f}", flush=True)
        for iface in ["J0", "J1", "J2"]:
            wcol = "w_can" if iface == "J2" else ("w_raw" if iface == "J1" else "w_can")
            net, stats = train_jepa(A_tr, iface, seed=seed, w_col=wcol, delta=True,
                                    epochs=50, bs=256)
            Pv, Pc, Pw = predict_full(net, stats, A_ev, iface, wcol)
            Y = A_ev["Yv"] - A_ev["vis"][:, -1]
            cos = 1 - (Pv * Y).sum(1) / (
                np.linalg.norm(Pv, axis=1) * np.linalg.norm(Y, axis=1) + 1e-9)
            # matched swap on canonical
            Wsw, _ = swap_matched_rh(A_ev, ev, np.random.default_rng(seed))
            Asw = dict(A_ev); Asw[wcol] = Wsw
            Pv_s, _, _ = predict_full(net, stats, Asw, iface, wcol)
            cos_sw = 1 - (Pv_s * Y).sum(1) / (
                np.linalg.norm(Pv_s, axis=1) * np.linalg.norm(Y, axis=1) + 1e-9)
            rec = dict(cos_err=float(cos.mean()),
                       cos_swapped=float(cos_sw.mean()),
                       contact_brier=float(((Pc - A_ev["Yc"]) ** 2).mean()),
                       fmag_mae=float(np.abs(np.linalg.norm(Pw[:, 3:], axis=1) -
                                             np.linalg.norm(A_ev["Yw_can"][:, 3:],
                                                            axis=1)).mean()))
            np.savez_compressed(os.path.join(CACHE,
                                f"predrh_{held}_{iface}_s{seed}.npz"),
                                ep_id=A_ev["meta"][:, 0],
                                win_t=A_ev["meta"][:, 1],
                                Yv=A_ev["Yv"].astype(np.float32), Pv=Pv.astype(np.float32),
                                Yc=A_ev["Yc"].astype(np.float32), Pc=Pc.astype(np.float32),
                                cos=cos.astype(np.float32), cos_swapped=cos_sw.astype(np.float32))
            res[iface] = rec
            print(f"  [{iface}] cos={rec['cos_err']:.4f} (sw {rec['cos_swapped']:.4f}) "
                  f"brier={rec['contact_brier']:.5f}", flush=True)
        out[held] = res
    json.dump(out, open(os.path.join(ROOT, "results/rh_jepa.json"), "w"), indent=1)
    print("\n[wrote] results/rh_jepa.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run_rh(a.seed)
