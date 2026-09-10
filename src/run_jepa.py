"""JEPA runner: J0-J4 x 3 task-group holdouts x 3 seeds, test-time swaps.

Holdout design (REASSEMBLE, single embodiment):
  groups = {insert, remove, place_pick} from per-frame task_index.
  episodes split into train-pool (70%) / eval-pool (30%) FIRST (disjoint).
  For holdout g: train windows = train-pool episodes x {other two groups};
  adapter episodes = train-pool episodes (held-out group windows, disjoint
  from eval-pool); eval windows = eval-pool episodes x group g.
  Contact-balanced sampling per (split, group).

Feature cache (built once): frozen ResNet18 embeddings for the unique
  (episode, video_frame) set needed by selected windows; canonical wrench
  histories + proprio + targets cached alongside.

Metrics: future-latent cosine error, contact Brier/AUPRC, canonical |f|
  MAE (aux), real-vs-shuffled cosine error. All predictions saved with
  (episode, window t) for episode-cluster bootstrap.
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from jepa_data import (task_group, load_task_names, episode_rows, pick_camera,
                       FPS, HIST, H_FUT, RGB_OFFS)
from symmetry import quat_to_mat, cotransform

GROUPS = ["insert", "remove", "place_pick"]
SEEDS = [0, 1, 2]
IFACES = ["J0", "J1", "J2", "J3", "J4"]


# ---------------------------------------------------------------- windows
_EP_CACHE = {}   # (data_dir, ep) -> dataframe (loaded once; assemble/all_windows
                 # were re-reading all 10 parquet shards per window -> hang)
_SHARD_CACHE = {}  # data_dir -> list of shard dataframes (loaded once)

def _shards(data_dir):
    sh = _SHARD_CACHE.get(data_dir)
    if sh is None:
        import glob as gb
        import pandas as pd
        sh = []
        for f in sorted(gb.glob(os.path.join(data_dir, "data/chunk-000/*.parquet"))):
            sh.append(pd.read_parquet(f))
        _SHARD_CACHE[data_dir] = sh
    return sh

def get_episode(data_dir, ep):
    key = (data_dir, ep)
    d = _EP_CACHE.get(key)
    if d is None:
        import pandas as pd
        out = []
        for df in _shards(data_dir):
            dfe = df[df.episode_index == ep]
            if len(dfe):
                out.append(dfe)
        d = pd.concat(out).sort_values("frame_index").reset_index(drop=True) if out else None
        _EP_CACHE[key] = d
    return d


def all_windows(data_dir, stride=15):
    names = load_task_names(data_dir)
    eps = pd.read_parquet(os.path.join(data_dir, "meta/episodes/chunk-000/file-000.parquet"))
    recs = []
    for _, row in eps.iterrows():
        ep = int(row["episode_index"])
        dfe = get_episode(data_dir, ep)
        if dfe is None or len(dfe) < HIST + H_FUT + 5:
            continue
        force = np.stack(dfe["observation.force"].values)
        fmag = np.linalg.norm(force, axis=1)
        contact = fmag > 2.0
        tids = dfe["task_index"].values
        for t in range(HIST, len(dfe) - H_FUT - 1, stride):
            fut = contact[t:t+H_FUT]
            recs.append(dict(ep=ep, t=int(t),
                             group=task_group(names[int(tids[t])]),
                             hist_contact=bool(contact[t-HIST:t].any()),
                             onset=bool(fut.any() and not fut[0]),
                             offset=bool(fut.any() and fut[0] and not fut[-1])))
    return recs


def split_episodes(wins, seed=0):
    rng = np.random.default_rng(seed)
    eps = sorted(set(w["ep"] for w in wins))
    rng.shuffle(eps)
    k = int(len(eps) * 0.7)
    return set(eps[:k]), set(eps[k:])


def select_windows(wins, train_pool, held_group, max_per_cell=900):
    """Contact-balanced selection. Returns (train_w, adapter_w, eval_w)."""
    rng = np.random.default_rng(1)
    others = [g for g in GROUPS if g != held_group]
    train_w, adapter_w, eval_w = [], [], []
    for g in others:
        cell = [w for w in wins if w["ep"] in train_pool and w["group"] == g]
        for c in [True, False]:
            sub = [w for w in cell if w["hist_contact"] == c]
            rng.shuffle(sub)
            train_w += sub[:max_per_cell]
    cell = [w for w in wins if w["ep"] in train_pool and w["group"] == held_group]
    for c in [True, False]:
        sub = [w for w in cell if w["hist_contact"] == c]
        rng.shuffle(sub)
        adapter_w += sub[:200]
    cell = [w for w in wins if w["ep"] not in train_pool and w["group"] == held_group]
    for c in [True, False]:
        sub = [w for w in cell if w["hist_contact"] == c]
        rng.shuffle(sub)
        eval_w += sub[:300]
    rng.shuffle(train_w); rng.shuffle(eval_w)
    return train_w, adapter_w, eval_w


# ---------------------------------------------------------------- features
@np.errstate(all="ignore")
def decode_video_frames(exe, vpath):
    """Yield RGB frames (H,W,3 uint8) using bundled ffmpeg (AV1-capable)."""
    import subprocess
    import numpy as np
    p = subprocess.Popen([exe, "-v", "error", "-i", vpath, "-pix_fmt", "rgb24",
                          "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    # probe dims first
    import cv2
    cap = cv2.VideoCapture(vpath)
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    buf = b""
    n = W * H * 3
    while True:
        while len(buf) < n:
            chunk = p.stdout.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        if len(buf) < n:
            break
        fr = np.frombuffer(buf[:n], np.uint8).reshape(H, W, 3)
        buf = buf[n:]
        yield fr
    p.wait()
    return fps


def encode_frames_offline(data_dir, wins, cache_path, img_size=224, batch=256):
    """Decode needed video frames once; encode with frozen ResNet18; cache."""
    import cv2
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from jepa_model import get_frozen_encoder, DEVICE
    import torchvision.transforms as T
    # unique (ep, video_frame_idx) needed
    need = {}   # ep -> set of parquet-frame indices (video frame == parquet frame check below)
    for w in wins:
        ep, t = w["ep"], w["t"]
        idxs = [t - o for o in RGB_OFFS] + [t + H_FUT]
        need.setdefault(ep, set()).update(idxs)
    tfm = T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    enc = get_frozen_encoder()
    feats = {}
    import pickle
    eps_meta = pd.read_parquet(os.path.join(
        data_dir, "meta/episodes/chunk-000/file-000.parquet"))
    # group episodes by video file (files may concatenate episodes)
    file_eps = {}   # (cam, vpath) -> [(ep, fts, tts, n_parquet)]
    for ep in sorted(need):
        got_cam = pick_camera(data_dir, ep)
        if got_cam[0] is None:
            continue
        cam, (vpath, fts, tts) = got_cam
        dfe = get_episode(data_dir, ep)
        if dfe is None:
            continue
        file_eps.setdefault((cam, vpath), []).append((ep, fts, tts, len(dfe)))
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    for (cam, vpath), eps_in_file in sorted(file_eps.items()):
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(vpath)
        fps = _cap.get(_cv2.CAP_PROP_FPS)
        _cap.release()
        if not fps or fps <= 0:
            fps = FPS
        total = None
        exp = [(ep, round((tts - fts) * fps), npar) for ep, fts, tts, npar in eps_in_file]
        if not all(abs(e - npar) <= max(5, 0.02 * npar) for _, e, npar in exp):
            print(f"  SKIP {vpath}: slice/parquet mismatch", flush=True)
            continue
        order = sorted(range(len(eps_in_file)), key=lambda k: eps_in_file[k][1])
        offs, acc = {}, 0
        for k in order:
            offs[eps_in_file[k][0]] = acc
            acc += exp[k][1]
        # probe decoded frame count once, then decode for real
        n_probe = sum(1 for _ in decode_video_frames(exe, vpath))
        if abs(acc - n_probe) > max(10, 0.03 * acc):
            print(f"  SKIP {vpath}: decoded {n_probe} vs expected {acc}", flush=True)
            continue
        want = {}  # absolute video frame -> (ep, parquet_idx)
        for ep, fts, tts, npar in eps_in_file:
            if ep not in need:
                continue
            dfe = get_episode(data_dir, ep)
            ts = dfe["timestamp"].values
            for i in sorted(need[ep]):
                if 0 <= i < len(dfe):
                    want[offs[ep] + round(float(ts[i] - ts[0]) * fps)] = (ep, i)
        if not want:
            continue
        imgs = {}
        for vf, fr in enumerate(decode_video_frames(exe, vpath)):
            if vf in want:
                imgs[vf] = tfm(fr)
        if len(imgs) < 0.9 * len(want):
            print(f"  SKIP {vpath}: decoded {len(imgs)}/{len(want)}", flush=True)
            continue
        with torch.no_grad():
            keys = sorted(imgs)
            X = torch.stack([imgs[vf] for vf in keys]).to(DEVICE)
            Z = np.concatenate([enc(X[k:k+batch]).cpu().numpy()
                                for k in range(0, len(X), batch)])
        for vf, z in zip(keys, Z):
            feats[want[vf]] = z
        print(f"  {os.path.basename(vpath)}: {len(keys)} frames "
              f"({len(feats)} total)", flush=True)
    with open(cache_path, "wb") as f:
        pickle.dump(feats, f)
    print(f"[wrote] {cache_path}: {len(feats)} frame features")


def assemble(data_dir, wins, feat_cache):
    """Build model-ready arrays for a window list. Drops windows with
    missing video features."""
    import pickle
    feats = pickle.load(open(feat_cache, "rb"))
    names = load_task_names(data_dir)
    Xv, Xp, Xw, Xph, Yv_idx, Yc, Yw, meta = [], [], [], [], [], [], [], []
    for w in wins:
        ep, t = w["ep"], w["t"]
        keys = [(ep, t - o) for o in RGB_OFFS] + [(ep, t + H_FUT)]
        if any(k not in feats for k in keys):
            continue
        dfe = get_episode(data_dir, ep)
        if dfe is None or t + H_FUT >= len(dfe):
            continue
        # proprio: EE pose + gripper + action history stats
        ee = np.stack(dfe["observation.state.ee_pose"].values)[t-HIST:t]
        grip = np.stack(dfe["observation.state.gripper_position"].values)[t-HIST:t]
        act = np.stack(dfe["action"].values)[t-HIST:t]
        prop = np.r_[ee.mean(0), ee.std(0), ee[-1], grip.mean(0), grip[-1],
                     act.mean(0), act[-1]].astype(np.float32)
        # force: raw + canonical wrench history; twist over history+future
        # so the future twist index is valid
        force = np.stack(dfe["observation.force"].values)
        torque = np.stack(dfe["observation.torque"].values)
        wraw_full = np.c_[torque, force].astype(np.float32)   # [tau; f]
        ee_full = np.stack(dfe["observation.state.ee_pose"].values)
        om_f, v_f = twists_from_pose_local(
            ee_full[max(t-HIST, 0):t+H_FUT])
        off = HIST - min(HIST, t)  # start offset inside om_f (==0 for t>=HIST)
        nu_hist = np.c_[om_f[off:off+HIST], v_f[off:off+HIST]].astype(np.float32)
        nu_fut = np.r_[om_f[off+HIST+H_FUT-1], v_f[off+HIST+H_FUT-1]].astype(np.float32)
        wraw = wraw_full[t-HIST:t]
        R0 = quat_to_mat(ee_full[0, 3:7]); Rt = R0.T; p0 = -Rt @ ee_full[0, :3]
        from symmetry import cotransform as _ct
        wt = np.empty_like(wraw); nut = np.empty_like(nu_hist)
        for k in range(HIST):
            a, b = _ct(wraw[k], nu_hist[k], (Rt, p0))
            wt[k], nut[k] = a, b
        fmag = np.linalg.norm(force[t-HIST:t], axis=1)
        phase = np.array([fmag.mean(), fmag.max(),
                          float((fmag > 2.0).mean())], np.float32)
        Xv.append(np.stack([feats[k] for k in keys[:4]]))
        Xp.append(prop)
        Xw.append(wt); Xph.append(phase)
        Yv_idx.append(keys[4])
        Yc.append([float(w["onset"]), float(w["offset"])])
        # future canonical wrench at t+H_FUT
        a, b = _ct(wraw_full[t+H_FUT], nu_fut, (Rt, p0))
        Yw.append(a)
        meta.append((ep, t))
    Yv = np.stack([feats[k] for k in Yv_idx])
    return (dict(vis=np.stack(Xv), prop=np.stack(Xp), w=np.stack(Xw),
                 phase=np.stack(Xph), Yv=Yv,
                 Yc=np.array(Yc, np.float32),
                 Yw=np.stack(Yw).astype(np.float32),
                 meta=np.array(meta, dtype=np.int64)))


def twists_from_pose_local(ee):
    from symmetry import twists_from_pose
    return twists_from_pose(ee, 1.0 / 30.0, smooth=3)


# ---------------------------------------------------------------- training
def train_jepa(A_tr, Yv_tr, iface, seed=0, epochs=15, bs=128, lr=3e-4):
    import torch
    import torch.nn as nn
    from jepa_model import JEPA, DEVICE
    torch.manual_seed(seed)
    d_prop = A_tr["prop"].shape[1]
    net = JEPA(force_kind=iface, d_prop=d_prop).to(DEVICE)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    # input norm (train stats) for force + proprio
    mu_w, sd_w = A_tr["w"].mean((0, 1)), A_tr["w"].std((0, 1)) + 1e-6
    mu_p, sd_p = A_tr["prop"].mean(0), A_tr["prop"].std(0) + 1e-6
    mu_yw, sd_yw = A_tr["Yw"].mean(0), A_tr["Yw"].std(0) + 1e-6
    def prep_w(W):
        return (W - mu_w) / sd_w
    n = len(A_tr["vis"])
    X = {k: torch.tensor(A_tr[k], device=DEVICE) for k in ("vis", "prop")}
    # NOTE: normalize proprio in numpy BEFORE tensor conversion — a CUDA
    # tensor minus a numpy array dispatches to numpy and crashes (__array__).
    X["prop"] = torch.tensor((A_tr["prop"] - mu_p) / sd_p, device=DEVICE)
    Yv = torch.tensor(Yv_tr, device=DEVICE)
    Yc = torch.tensor(A_tr["Yc"], device=DEVICE)
    Yw = torch.tensor((A_tr["Yw"] - mu_yw) / sd_yw, device=DEVICE)
    for epn in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for k in range(0, n, bs):
            s = perm[k:k+bs]
            w = torch.tensor(prep_w(A_tr["w"][s.cpu().numpy()]), device=DEVICE)
            h, pv, pc, pw = net(X["vis"][s], X["prop"][s], w,
                                torch.tensor(A_tr["phase"][s.cpu().numpy()], device=DEVICE))
            l_vis = (1 - nn.functional.cosine_similarity(pv, Yv[s])).mean()
            l_c = nn.functional.binary_cross_entropy_with_logits(pc, Yc[s])
            l_w = nn.functional.smooth_l1_loss(pw, Yw[s])
            (l_vis + l_c + 0.5 * l_w).backward()
            opt.step(); opt.zero_grad()
    net.eval()
    stats = dict(mu_w=mu_w, sd_w=sd_w, mu_p=mu_p, sd_p=sd_p, mu_yw=mu_yw, sd_yw=sd_yw)
    return net, stats


@torch.no_grad()
def eval_jepa(net, stats, A, Yv, iface, swap=None, swap_pool=None):
    import torch
    import torch.nn as nn
    from jepa_model import DEVICE
    mu_w, sd_w = stats["mu_w"], stats["sd_w"]
    W = A["w"].copy()
    if swap == "shuffle":
        # fixed permutation: destroys time alignment identically for all windows
        W = W[:, np.random.default_rng(0).permutation(HIST)]
    elif swap == "zero":
        W = np.zeros_like(W)
    W = (W - mu_w) / sd_w
    outs = []
    for k in range(0, len(W), 256):
        sl = slice(k, k + 256)
        h, pv, pc, pw = net(
            torch.tensor(A["vis"][sl], device=DEVICE),
            torch.tensor((A["prop"][sl] - stats["mu_p"]) / stats["sd_p"], device=DEVICE),
            torch.tensor(W[sl], device=DEVICE),
            torch.tensor(A["phase"][sl], device=DEVICE))
        outs.append((pv.cpu().numpy(), torch.sigmoid(pc).cpu().numpy(),
                     pw.cpu().numpy() * stats["sd_yw"] + stats["mu_yw"]))
    pv, pc, pw = (np.concatenate(x) for x in zip(*outs))
    cos_err = float((1 - (pv * Yv).sum(1) /
                     (np.linalg.norm(pv, axis=1) * np.linalg.norm(Yv, axis=1) + 1e-9)).mean())
    return dict(cos_err=cos_err, pc=pc, pw=pw)


def run(cache_dir, data_dir, out_path, pred_dir=None):
    import torch
    import torch.nn as nn
    from jepa_model import JEPA, DEVICE
    wins = all_windows(data_dir)
    train_pool, eval_pool = split_episodes(wins)
    print(f"{len(wins)} windows total")
    results = {"holdouts": {}}
    for held in GROUPS:
        tr_w, ad_w, ev_w = select_windows(wins, train_pool, held)
        print(f"== holdout {held}: train {len(tr_w)}, adapter {len(ad_w)}, eval {len(ev_w)}",
              flush=True)
        A_tr = assemble(data_dir, tr_w, os.path.join(cache_dir, "visfeat.pkl"))
        A_ad = assemble(data_dir, ad_w, os.path.join(cache_dir, "visfeat.pkl"))
        A_ev = assemble(data_dir, ev_w, os.path.join(cache_dir, "visfeat.pkl"))
        yc = A_ev["Yc"]
        base = dict(contact_rate=float(yc.mean()),
                    brier_const=float(((yc.mean(0)[None, :] - yc) ** 2).mean()))
        hres = {"baselines": base, "n_train": len(A_tr["vis"]),
                "n_eval": len(A_ev["vis"])}
        for iface in IFACES:
            for seed in SEEDS:
                adapter = None
                if iface == "J4" and len(A_ad["w"]) >= 32:
                    Wa = A_ad["w"].reshape(-1, 6)
                    Wt = A_tr["w"].reshape(-1, 6)
                    mu_a, mu_t = Wa.mean(0), Wt.mean(0)
                    Ca = np.cov(Wa.T) + 1e-4 * np.eye(6)
                    Ct = np.cov(Wt.T) + 1e-4 * np.eye(6)
                    ea, Ua = np.linalg.eigh(Ca); et, Ut = np.linalg.eigh(Ct)
                    Aad = (Ut * np.sqrt(np.maximum(et, 1e-8))) @ (Ua * (1.0 / np.sqrt(np.maximum(ea, 1e-8)))).T
                    bad = mu_t - Aad @ mu_a
                    adapter = (Aad.astype(np.float32), bad.astype(np.float32))
                net, stats = train_jepa(A_tr, A_tr["Yv"], iface, seed=seed)
                # J4: adapter (fit on adapter episodes) maps held-out-group
                # wrench into train distribution — applied to EVAL inputs only
                A_ev_use = A_ev
                if iface == "J4" and adapter is not None:
                    Aad, bad = adapter
                    A_ev_use = dict(A_ev)
                    A_ev_use["w"] = (A_ev["w"] @ Aad.T + bad).astype(np.float32)
                Yv_ev = A_ev_use["Yv"]
                P_real = eval_jepa(net, stats, A_ev_use, Yv_ev, iface)
                P_zero = eval_jepa(net, stats, A_ev_use, Yv_ev, iface, swap="zero")
                P_shuf = eval_jepa(net, stats, A_ev_use, Yv_ev, iface, swap="shuffle")
                rec = {}
                for nm, P in [("real", P_real), ("zero", P_zero), ("shuffle", P_shuf)]:
                    rec[nm] = dict(
                        cos_err=P["cos_err"],
                        contact_brier=float(((P["pc"] - A_ev["Yc"]) ** 2).mean()),
                        fmag_mae=float(np.abs(np.linalg.norm(P["pw"][:, 3:], axis=1)
                                              - np.linalg.norm(A_ev["Yw"][:, 3:], axis=1)).mean()))
                try:
                    from sklearn.metrics import average_precision_score
                    rec["real"]["contact_auprc"] = float(average_precision_score(
                        (A_ev["Yc"].sum(1) > 0).astype(int), P_real["pc"].max(1)))
                except Exception:
                    rec["real"]["contact_auprc"] = float("nan")
                hres[f"{iface}/seed{seed}"] = rec
                if pred_dir is not None:
                    os.makedirs(pred_dir, exist_ok=True)
                    np.savez_compressed(
                        os.path.join(pred_dir, f"jepa_{held}_{iface}_seed{seed}.npz"),
                        ep_id=A_ev["meta"][:, 0], win_t=A_ev["meta"][:, 1],
                        Yc=A_ev["Yc"].astype(np.float32),
                        Pc_real=P_real["pc"].astype(np.float32),
                        cos_real=np.array([P_real["cos_err"]]),
                        cos_shuf=np.array([P_shuf["cos_err"]]))
                print(f"   [{iface} s{seed}] cos={rec['real']['cos_err']:.4f} "
                      f"(zero {rec['zero']['cos_err']:.4f}, shuf {rec['shuffle']['cos_err']:.4f}) "
                      f"brier={rec['real']['contact_brier']:.4f}", flush=True)
        results["holdouts"][f"holdout_{held}"] = hres
    json.dump(results, open(out_path, "w"), indent=1)
    print("[wrote]", out_path)


def selected_union(wins):
    """Mirror of run()'s selection for all 3 holdouts: union of windows
    that will ever need video features."""
    train_pool, eval_pool = split_episodes(wins)
    union = {}
    for held in GROUPS:
        tr_w, ad_w, ev_w = select_windows(wins, train_pool, held)
        for w in tr_w + ad_w + ev_w:
            union[(w["ep"], w["t"])] = w
    return list(union.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_windows", default=None)
    ap.add_argument("--data_dir", default="data/reassemble")
    ap.add_argument("--build_cache", default=None,
                    help="build frozen visual feature cache at this path")
    ap.add_argument("--cache_dir", default="cache_jepa")
    ap.add_argument("--out", default="results/jepa.json")
    ap.add_argument("--pred_dir", default=None)
    ap.add_argument("--wins_json", default=None,
                    help="window manifest json (else rebuilt)")
    args = ap.parse_args()
    if args.build_windows:
        from jepa_data import build_windows
        build_windows(args.data_dir, args.build_windows)
    elif args.build_cache:
        wins = json.load(open(args.wins_json))
        sel = selected_union(wins)
        print(f"selected {len(sel)} windows for feature caching")
        encode_frames_offline(args.data_dir, sel, args.build_cache)
    else:
        run(args.cache_dir, args.data_dir, args.out, pred_dir=args.pred_dir)
