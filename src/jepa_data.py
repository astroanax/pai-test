"""JEPA data layer: REASSEMBLE windows + frozen visual features.

Per-window (30fps):
  history: 30 frames (1 s) of force/torque, EE pose, gripper, action
  current RGB: 4 frames at t-27, t-18, t-9, t  (hama1, fallback hama2/hand)
  future target: visual embedding at t+15 (0.5 s ahead, stop-gradient at train)
  aux targets: contact onset/offset in (t, t+15], canonical wrench at t+15
Task group per window from task_index -> {insert, remove, place_pick}.
Contact-balanced sampling per (pool, group).
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from symmetry import quat_to_mat, twists_from_pose, CONTACT_THR as F_THR

FPS = 30.0
HIST = 30
H_FUT = 15
RGB_OFFS = [27, 18, 9, 0]
CAMERAS = ["hama1", "hama2", "hand"]


def task_group(name):
    if name.startswith("Insert"):
        return "insert"
    if name.startswith("Remove"):
        return "remove"
    return "place_pick"


def load_task_names(data_dir):
    tasks = pd.read_parquet(os.path.join(data_dir, "meta/tasks.parquet"))
    return tasks.index.tolist()


def episode_rows(data_dir, ep):
    """All parquet rows for one episode (across shards)."""
    import glob as gb
    out = []
    for f in sorted(gb.glob(os.path.join(data_dir, "data/chunk-000/*.parquet"))):
        df = pd.read_parquet(f)
        dfe = df[df.episode_index == ep]
        if len(dfe):
            out.append(dfe)
    if not out:
        return None
    return pd.concat(out).sort_values("frame_index").reset_index(drop=True)


def video_slice_info(data_dir, ep, camera):
    """(video_path, frame_offset, fps) for an episode+camera, or None."""
    eps = pd.read_parquet(os.path.join(data_dir, "meta/episodes/chunk-000/file-000.parquet"))
    row = eps[eps.episode_index == ep]
    if not len(row):
        return None
    row = row.iloc[0]
    key = f"videos/observation.images.{camera}"
    try:
        ci = int(row[key + "/chunk_index"]); fi = int(row[key + "/file_index"])
        fts = float(row[key + "/from_timestamp"]); tts = float(row[key + "/to_timestamp"])
    except Exception:
        return None
    v = os.path.join(data_dir, key, f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")
    if not os.path.exists(v):
        return None
    return v, fts, tts


def pick_camera(data_dir, ep):
    for cam in CAMERAS:
        info = video_slice_info(data_dir, ep, cam)
        if info is not None:
            return cam, info
    return None, None


def build_windows(data_dir, out_path, stride=15):
    """Window manifest (no video): one row per window with all metadata."""
    names = load_task_names(data_dir)
    eps = pd.read_parquet(os.path.join(data_dir, "meta/episodes/chunk-000/file-000.parquet"))
    recs = []
    for _, row in eps.iterrows():
        ep = int(row["episode_index"])
        dfe = episode_rows(data_dir, ep)
        if dfe is None or len(dfe) < HIST + H_FUT + 5:
            continue
        force = np.stack(dfe["observation.force"].values)
        fmag = np.linalg.norm(force, axis=1)
        contact = fmag > F_THR
        tids = dfe["task_index"].values
        ts = dfe["timestamp"].values
        for t in range(HIST, len(dfe) - H_FUT - 1, stride):
            grp = task_group(names[int(tids[t])])
            hist_contact = bool(contact[t-HIST:t].any())
            fut = contact[t:t+H_FUT]
            onset = bool(fut.any() and not fut[0])
            offset = bool(fut.any() and fut[0] and not fut[-1])
            recs.append(dict(ep=ep, t=int(t), ts=float(ts[t]),
                             group=grp, hist_contact=hist_contact,
                             fut_onset=onset, fut_offset=offset,
                             fmag_t=float(fmag[t])))
    import json as js
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        js.dump(recs, f)
    print(f"[wrote] {out_path}: {len(recs)} windows")
    return recs
