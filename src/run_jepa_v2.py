"""JEPA minimal valid rerun (audit-driven, run_jepa_v2).

Fixes over run_jepa.py:
  F1  correct SO(3)/SE(3) twist (so3_log/dt, spatial Plucker v)
  F1b wrench transported to base origin (tau_O = R tau_s + p x R f_s)
  F2  SEPARATE w_raw (sensor frame) and w_canonical (initial sensor frame,
      via T_{0<-s(t)} = T_{b<-0}^{-1} T_{b<-s(t)}); J1 uses raw, J2 canonical;
      numeric assertion that they differ
  F3  contact sampling on FUTURE transitions; eval prevalence reported
  F4  matched cross-window force swap (same group+phase bucket) instead of
      a fixed temporal permutation
  F5  per-window targets AND predictions saved (visual, wrench, contact)
  F6  overfit sanity (32-64 windows), last-current-frame baseline,
      video sync montage
Only J0 (no force), J1 (raw), J2 (canonical) x one holdout x one seed.
"""
import argparse, json, os, sys, pickle
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from screw_fixed import (quat_to_mat, twists_from_pose, wrench_base_origin,
                         canonical_wrench_series, canonical_wrench_single,
                         cotransform)
from jepa_data import task_group, load_task_names, pick_camera, FPS, HIST, H_FUT, RGB_OFFS
from jepa_model import JEPA, DEVICE

GROUPS = ["insert", "remove", "place_pick"]
CONTACT_THR = 2.0


# ---------------------------------------------------------------- data
_EP_CACHE, _SHARD_CACHE = {}, {}

def _shards(data_dir):
    if data_dir not in _SHARD_CACHE:
        import glob as gb
        _SHARD_CACHE[data_dir] = [pd.read_parquet(f) for f in
                                  sorted(gb.glob(os.path.join(data_dir, "data/chunk-000/*.parquet")))]
    return _SHARD_CACHE[data_dir]

def get_episode(data_dir, ep):
    key = (data_dir, ep)
    if key not in _EP_CACHE:
        out = [df[df.episode_index == ep] for df in _shards(data_dir)]
        out = [o for o in out if len(o)]
        _EP_CACHE[key] = pd.concat(out).sort_values("frame_index").reset_index(drop=True) if out else None
    return _EP_CACHE[key]


def all_windows(data_dir, feat_cache, stride=1):
    """Windows whose every required frame (4 current + future) has a cached
    frozen visual feature. Candidates come FROM the feature cache, so no
    window is silently dropped at assemble time."""
    feats = pickle.load(open(feat_cache, "rb"))
    by_ep = {}
    for (e, t) in feats:
        by_ep.setdefault(e, set()).add(int(t))
    names = load_task_names(data_dir)
    eps = pd.read_parquet(os.path.join(data_dir, "meta/episodes/chunk-000/file-000.parquet"))
    recs = []
    for _, row in eps.iterrows():
        ep = int(row["episode_index"])
        S = by_ep.get(ep)
        if not S:
            continue
        dfe = get_episode(data_dir, ep)
        if dfe is None or len(dfe) < HIST + H_FUT + 5:
            continue
        force = np.stack(dfe["observation.force"].values)
        contact = np.linalg.norm(force, axis=1) > CONTACT_THR
        tids = dfe["task_index"].values
        tmax = len(dfe) - H_FUT - 2
        for t in sorted(S):
            if t - max(RGB_OFFS) < 0 or t + H_FUT > tmax:
                continue
            if any((t - o) not in S for o in RGB_OFFS) or (t + H_FUT) not in S:
                continue
            fut = contact[t:t+H_FUT]
            recs.append(dict(ep=ep, t=int(t),
                             group=task_group(names[int(tids[t])]),
                             hist_contact=bool(contact[t-HIST:t].any()),
                             fut_onset=bool(fut.any() and not fut[0]),
                             fut_any=bool(fut.any())))
    return recs


def split_episodes(wins, seed=0):
    rng = np.random.default_rng(seed)
    eps = sorted(set(w["ep"] for w in wins))
    rng.shuffle(eps)
    k = int(len(eps) * 0.7)
    return set(eps[:k]), set(eps[k:])


def select_windows(wins, train_pool, held_group, max_per_cell=700):
    """Balance train/adapter on FUTURE transitions (fut_any);
    keep eval at natural prevalence."""
    rng = np.random.default_rng(1)
    others = [g for g in GROUPS if g != held_group]
    train_w, adapter_w, eval_w = [], [], []
    for g in others:
        cell = [w for w in wins if w["ep"] in train_pool and w["group"] == g]
        for c in [True, False]:          # balance on future contact
            sub = [w for w in cell if w["fut_any"] == c]
            rng.shuffle(sub)
            train_w += sub[:max_per_cell]
    cell = [w for w in wins if w["ep"] in train_pool and w["group"] == held_group]
    for c in [True, False]:
        sub = [w for w in cell if w["fut_any"] == c]
        rng.shuffle(sub)
        adapter_w += sub[:150]
    cell = [w for w in wins if w["ep"] not in train_pool and w["group"] == held_group]
    # EVAL: natural prevalence (no balancing), keep everything available
    rng.shuffle(cell)
    eval_w = cell
    rng.shuffle(train_w)
    return train_w, adapter_w, eval_w


# ---------------------------------------------------------------- assemble (fixed)
def assemble(data_dir, wins, feat_cache):
    """Per-window arrays. Force histories: w_raw (sensor frame), w_canon
    (initial sensor frame via base-origin transport + constant init frame).
    Saves per-window TARGETS AND PREDICTIONS incl. future embeddings."""
    feats = pickle.load(open(feat_cache, "rb"))
    names = load_task_names(data_dir)
    Xv, Xp, Xw_raw, Xw_can, Xph = [], [], [], [], []
    Yv_idx, Yc, Yw_raw, Yw_can, meta = [], [], [], [], []
    for w in wins:
        ep, t = w["ep"], w["t"]
        keys = [(ep, t - o) for o in RGB_OFFS] + [(ep, t + H_FUT)]
        if any(k not in feats for k in keys):
            continue
        dfe = get_episode(data_dir, ep)
        if dfe is None or t + H_FUT >= len(dfe):
            continue
        ee = np.stack(dfe["observation.state.ee_pose"].values)
        grip = np.stack(dfe["observation.state.gripper_position"].values)
        act = np.stack(dfe["action"].values)
        # ---- wrench (sensor frame, as measured)
        force = np.stack(dfe["observation.force"].values)
        torque = np.stack(dfe["observation.torque"].values)
        w_s = np.c_[torque, force].astype(np.float32)          # [tau_s; f_s]
        # ---- twist (fixed): spatial Plucker, base frame
        om_f, v_f = twists_from_pose(ee[max(t-HIST, 0):t+H_FUT], 1.0/FPS, smooth=3)
        off = HIST - min(HIST, t)
        nu_hist = np.c_[om_f[off:off+HIST], v_f[off:off+HIST]].astype(np.float32)
        # ---- raw: sensor-frame wrench history (J1)
        w_raw = w_s[t-HIST:t]
        # ---- canonical: base-origin transport + initial-frame (J2)
        w_O = np.stack([wrench_base_origin(w_s[k], ee[k]) for k in range(t-HIST, t)])
        w_can = canonical_wrench_series(w_O, ee[t-HIST:t], ee[0])
        # ---- future canonical wrench target
        w_O_f = wrench_base_origin(w_s[t+H_FUT], ee[t+H_FUT])
        w_can_f = canonical_wrench_single(w_O_f, ee[t+H_FUT], ee[0])
        # ---- proprio
        prop = np.r_[ee[t-HIST:t].mean(0), ee[t-HIST:t].std(0), ee[t+H_FUT-1],
                     grip[t-HIST:t].mean(0), grip[t-1],
                     act[t-HIST:t].mean(0), act[t+H_FUT-1]].astype(np.float32)
        fmag = np.linalg.norm(force[t-HIST:t], axis=1)
        phase = np.array([fmag.mean(), fmag.max(), float((fmag > CONTACT_THR).mean())], np.float32)
        Xv.append(np.stack([feats[k] for k in keys[:4]]))
        Xp.append(prop)
        Xw_raw.append(w_raw); Xw_can.append(w_can); Xph.append(phase)
        Yv_idx.append(keys[4])
        Yc.append([float(w["fut_onset"]), float(w["fut_any"])])
        Yw_raw.append(w_s[t+H_FUT]); Yw_can.append(w_can_f)
        meta.append((ep, t))
    return dict(vis=np.stack(Xv), prop=np.stack(Xp),
                w_raw=np.stack(Xw_raw).astype(np.float32),
                w_can=np.stack(Xw_can).astype(np.float32),
                phase=np.stack(Xph),
                Yv=np.stack([feats[k] for k in Yv_idx]),
                Yc=np.array(Yc, np.float32),
                Yw_raw=np.stack(Yw_raw).astype(np.float32),
                Yw_can=np.stack(Yw_can).astype(np.float32),
                meta=np.array(meta, dtype=np.int64))


# ---------------------------------------------------------------- training
def _delta_target(A):
    """Predict the CHANGE in visual embedding: Yv_delta = future - last-current.
    Removes the persistence shortcut (copy last frame) that dominates absolute
    future prediction. Returns (Yv_delta, last) unit-normalized components."""
    last = A["vis"][:, -1]
    d = A["Yv"] - last
    return d, last


def train_jepa(A_tr, iface, seed=0, epochs=150, bs=64, lr=1e-3, w_col="w_can",
               delta=True):
    torch.manual_seed(seed)
    d_prop = A_tr["prop"].shape[1]
    net = JEPA(force_kind=iface, d_prop=d_prop).to(DEVICE)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    W = A_tr[w_col] if w_col in A_tr else A_tr["w_can"]
    mu_w, sd_w = W.mean((0, 1)), W.std((0, 1)) + 1e-6
    mu_p, sd_p = A_tr["prop"].mean(0), A_tr["prop"].std(0) + 1e-6
    mu_yw, sd_yw = A_tr["Yw_can"].mean(0), A_tr["Yw_can"].std(0) + 1e-6
    X = {"vis": torch.tensor(A_tr["vis"], device=DEVICE),
         "prop": torch.tensor((A_tr["prop"] - mu_p) / sd_p, device=DEVICE)}
    if delta:
        Yv = torch.tensor(_delta_target(A_tr)[0], device=DEVICE)  # raw delta
    else:
        Yv = torch.tensor(A_tr["Yv"], device=DEVICE)
    Yc = torch.tensor(A_tr["Yc"], device=DEVICE)
    Yw = torch.tensor((A_tr["Yw_can"] - mu_yw) / sd_yw, device=DEVICE)
    n = len(X["vis"])
    Wn = (W - mu_w) / sd_w
    # Precompute GPU tensors ONCE (per-batch CPU<->GPU round-trips were the
    # bottleneck: 99% CPU / low GPU util on 36k windows x 100 epochs).
    Wn_t = torch.tensor(Wn, device=DEVICE)
    Phase_t = torch.tensor(A_tr["phase"], device=DEVICE)
    for epn in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for k in range(0, n, bs):
            s = perm[k:k+bs]
            h, pv, pc, pw = net(X["vis"][s], X["prop"][s], Wn_t[s], Phase_t[s])
            l_vis = (1 - torch.nn.functional.cosine_similarity(pv, Yv[s])).mean()
            l_c = torch.nn.functional.binary_cross_entropy_with_logits(pc, Yc[s])
            l_w = torch.nn.functional.smooth_l1_loss(pw, Yw[s])
            (l_vis + l_c + 0.5 * l_w).backward()
            opt.step(); opt.zero_grad()
    net.eval()
    stats = dict(mu_w=mu_w, sd_w=sd_w, mu_p=mu_p, sd_p=sd_p,
                 mu_yw=mu_yw, sd_yw=sd_yw, delta=delta)
    return net, stats


@torch.no_grad()
def predict_full(net, stats, A, iface, w_col="w_can"):
    W = A[w_col].copy()
    Wn = (W - stats["mu_w"]) / stats["sd_w"]
    Pv, Pc, Pw = [], [], []
    for k in range(0, len(W), 256):
        sl = slice(k, k+256)
        h, pv, pc, pw = net(
            torch.tensor(A["vis"][sl], device=DEVICE),
            torch.tensor((A["prop"][sl] - stats["mu_p"]) / stats["sd_p"], device=DEVICE),
            torch.tensor(Wn[sl], device=DEVICE),
            torch.tensor(A["phase"][sl], device=DEVICE))
        Pv.append(pv.cpu().numpy())
        Pc.append(torch.sigmoid(pc).cpu().numpy())
        Pw.append(pw.cpu().numpy() * stats["sd_yw"] + stats["mu_yw"])
    return (np.concatenate(Pv), np.concatenate(Pc), np.concatenate(Pw))


def swap_matched(A, wins, group_map, phase_key, rng):
    """Matched cross-window force swap: each window's wrench history is
    replaced by one from a DIFFERENT window in the same (group, hist_contact)
    bucket (rotation = derangement, so no window keeps its own force)."""
    buckets = {}
    for i, w in enumerate(wins):
        key = (w["group"], w["hist_contact"])
        buckets.setdefault(key, []).append(i)
    idx = np.arange(len(wins))
    for key, members in buckets.items():
        m = np.array(members)
        if len(m) < 2:
            continue
        target = np.roll(np.arange(len(m)), 1)   # derangement
        idx[m] = m[target]
    W = A["w_can"].copy()
    return W[idx], idx


def last_frame_baseline(A):
    """Cosine distance between LAST current-frame embedding and future
    embedding (no-model baseline)."""
    last = A["vis"][:, -1]
    fut = A["Yv"]
    c = (last * fut).sum(1) / (np.linalg.norm(last, axis=1) * np.linalg.norm(fut, axis=1) + 1e-9)
    return float((1 - c).mean())


def make_montage(data_dir, wins, feat_cache, out_path, n=3):
    """Render current+future frame montage for sync verification."""
    import cv2, imageio_ffmpeg
    import torchvision.transforms as T
    sys.path.insert(0, os.path.dirname(__file__))
    from jepa_data import episode_rows
    feats = pickle.load(open(feat_cache, "rb"))
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    # decode needed frames for first n windows
    grid = []
    for w in wins[:n]:
        ep, t = w["ep"], w["t"]
        cam, (vpath, fts, tts) = pick_camera(data_dir, ep)
        if vpath is None: continue
        dfe = get_episode(data_dir, ep)
        ts = dfe["timestamp"].values
        idxs = [t - o for o in RGB_OFFS] + [t + H_FUT]
        want = {round(float(ts[i] - ts[0]) * 30): i for i in idxs}
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        frames = {}
        for vf, fr in enumerate(decode_frames(exe, vpath)):
            if vf in want:
                frames[vf] = cv2.resize(fr, (224, 224))
        if len(frames) < len(want): continue
        row = np.concatenate([frames[vf] for vf in sorted(frames)], axis=1)
        grid.append(row)
    if grid:
        montage = np.concatenate(grid, axis=0)
        cv2.imwrite(out_path, montage[:, :, ::-1])
        print(f"[montage] wrote {out_path} ({len(grid)} windows)")
    else:
        print("[montage] no frames available")


def decode_frames(exe, vpath):
    import subprocess
    import cv2
    cap = cv2.VideoCapture(vpath)
    W, H = int(cap.get(3)), int(cap.get(4))
    cap.release()
    p = subprocess.Popen([exe, "-v", "error", "-i", vpath, "-pix_fmt", "rgb24",
                          "-f", "rawvideo", "pipe:1"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = W * H * 3
    buf = b""
    while True:
        while len(buf) < n:
            chunk = p.stdout.read(n - len(buf))
            if not chunk: break
            buf += chunk
        if len(buf) < n: break
        yield np.frombuffer(buf[:n], np.uint8).reshape(H, W, 3)
        buf = buf[n:]
    p.wait()


# ---------------------------------------------------------------- run
def run(cache_dir, data_dir, out_path, held_group="remove", seed=0):
    rng = np.random.default_rng(seed)
    feat_cache = os.path.join(cache_dir, "visfeat.pkl")
    wins = all_windows(data_dir, feat_cache)
    train_pool, eval_pool = split_episodes(wins, seed)
    tr_w, ad_w, ev_w = select_windows(wins, train_pool, held_group)
    print(f"holdout {held_group}: train {len(tr_w)}, adapter {len(ad_w)}, "
          f"eval {len(ev_w)}", flush=True)
    A_tr = assemble(data_dir, tr_w, os.path.join(cache_dir, "visfeat.pkl"))
    A_ad = assemble(data_dir, ad_w, os.path.join(cache_dir, "visfeat.pkl"))
    A_ev = assemble(data_dir, ev_w, os.path.join(cache_dir, "visfeat.pkl"))
    # prevalence of future contact in eval
    prev = float(A_ev["Yc"][:, 1].mean())
    print(f"eval future-contact prevalence: {prev:.3f} (n={len(A_ev['vis'])})", flush=True)

    # F2 assertion: raw vs canonical differ
    d1 = np.abs(A_tr["w_raw"] - A_tr["w_can"]).mean()
    print(f"assert: mean |w_raw - w_can| = {d1:.4f} (must be > 1e-3)", flush=True)
    assert d1 > 1e-3, "J1/J2 inputs identical — bug"

    res = {"prevalence": prev, "n_eval": len(A_ev["vis"]),
           "baselines": {"last_frame_cos": last_frame_baseline(A_ev),
                         "contact_rate": prev}}
    # DELTA mode: the model predicts the visual CHANGE (future - last current),
    # and the no-model baseline is the mean delta (persistence is removed).
    Yd = A_ev["Yv"] - A_ev["vis"][:, -1]          # true delta target
    mean_delta = Yd.mean(0)
    md = np.linalg.norm(Yd) * np.linalg.norm(mean_delta) + 1e-9
    res["baselines"]["mean_delta_cos"] = float(
        (1 - (Yd @ mean_delta) / md).mean())
    res["baselines"]["delta_norm_mean"] = float(np.linalg.norm(Yd, axis=1).mean())
    res["baselines"]["abs_future_norm"] = float(np.linalg.norm(A_ev["Yv"], axis=1).mean())
    print(f"delta target: |Yd| mean={res['baselines']['delta_norm_mean']:.3f} "
          f"vs |Yv| {res['baselines']['abs_future_norm']:.3f}; "
          f"mean-delta baseline cos={res['baselines']['mean_delta_cos']:.4f}", flush=True)
    for iface in ["J0", "J1", "J2"]:
        wcol = "w_can" if iface in ("J2",) else ("w_raw" if iface == "J1" else "w_can")
        net, stats = train_jepa(A_tr, iface, seed=seed, w_col=wcol, delta=True)
        Pv, Pc, Pw = predict_full(net, stats, A_ev, iface, wcol)
        # per-window metrics against the DELTA target
        Y = Yd
        cos = 1 - (Pv * Y).sum(1) / (
            np.linalg.norm(Pv, axis=1) * np.linalg.norm(Y, axis=1) + 1e-9)
        # matched swap
        Wsw, idx = swap_matched(A_ev, ev_w, None, None, rng)
        Asw = dict(A_ev); Asw[wcol] = Wsw
        Pv_s, _, _ = predict_full(net, stats, Asw, iface, wcol)
        cos_sw = 1 - (Pv_s * Y).sum(1) / (
            np.linalg.norm(Pv_s, axis=1) * np.linalg.norm(Y, axis=1) + 1e-9)
        rec = dict(cos_err=float(cos.mean()),
                   cos_swapped=float(cos_sw.mean()),
                   contact_brier=float(((Pc - A_ev["Yc"]) ** 2).mean()),
                   fmag_mae=float(np.abs(np.linalg.norm(Pw[:, 3:], axis=1) -
                                         np.linalg.norm(A_ev["Yw_can"][:, 3:], axis=1)).mean()))
        # save per-window predictions (F5)
        np.savez_compressed(os.path.join(cache_dir, f"pred_{held_group}_{iface}_s{seed}.npz"),
                            ep_id=A_ev["meta"][:, 0], win_t=A_ev["meta"][:, 1],
                            Yv=A_ev["Yv"].astype(np.float32),
                            Yd=Y.astype(np.float32),
                            Pv=Pv.astype(np.float32),
                            Yc=A_ev["Yc"].astype(np.float32), Pc=Pc.astype(np.float32),
                            Yw=A_ev["Yw_can"].astype(np.float32), Pw=Pw.astype(np.float32),
                            cos=cos.astype(np.float32), cos_swapped=cos_sw.astype(np.float32))
        res[iface] = rec
        print(f"  [{iface}] cos={rec['cos_err']:.4f} (swapped {rec['cos_swapped']:.4f}) "
              f"brier={rec['contact_brier']:.5f} fH={rec['fmag_mae']:.3f}", flush=True)

    # F6: overfit sanity on 48 windows (delta mode)
    A_ov = {k: v[:48] for k, v in A_tr.items()}
    net, stats = train_jepa(A_ov, "J2", seed=0, epochs=60, delta=True)
    Pv, Pc, Pw = predict_full(net, stats, A_ov, "J2")
    Yd_ov = A_ov["Yv"] - A_ov["vis"][:, -1]
    ov_cos = float((1 - (Pv * Yd_ov).sum(1) / (
        np.linalg.norm(Pv, axis=1) * np.linalg.norm(Yd_ov, axis=1) + 1e-9)).mean())
    res["overfit"] = dict(n=48, train_cos=ov_cos)
    print(f"  [overfit J2] 48-window train cos = {ov_cos:.4f}", flush=True)

    json.dump(res, open(out_path, "w"), indent=1)
    print("[wrote]", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="cache_jepa")
    ap.add_argument("--data_dir", default="data/reassemble")
    ap.add_argument("--out", default="results/jepa_v2.json")
    ap.add_argument("--held_group", default="remove")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.cache_dir, a.data_dir, a.out, a.held_group, a.seed)
