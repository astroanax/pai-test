"""RH20T cross-hardware JEPA data builder.

For each of cfg1/cfg2/cfg3 and each shared-task episode in rgb_plan.json:
  - color.mp4  : video frames (index i)
  - timestamps.npy['color'] : epoch-ms per frame (sample-synced with base streams)
  - force_torque_base.npy[cam] : per-sample {'timestamp','zeroed','raw'}  (100Hz-ish)
  - tcp_base.npy[cam]          : per-sample {'timestamp','tcp','robot_ft'}
  - joint not present -> derive gripper-free; use robot_ft absent -> zeros
Alignment: within one camera serial, video frame i == force sample i == tcp sample i
           (verified: same length, timestamps match <1ms).

Output: cache_rh20t_v/<cfg>__<ep>.npz with aligned (w, nu, pose, joint) arrays at
the video frame rate, plus cache_rh20t_v/visfeat.pkl of ResNet18 embeddings.
Fixed camera serial used across all configs.
"""
import numpy as np, os, sys, pickle, json, glob
sys.path.insert(0, os.path.expanduser("~/rehan/reciprocal-tokens/src"))
from screw_fixed import quat_to_mat, so3_log, wrench_base_origin
from jepa_model import get_frozen_encoder, DEVICE

ROOT = os.path.expanduser("~/rehan/reciprocal-tokens")
# Per-config fixed camera (hardware differs across configs). Chosen as the
# most frequently present camera serial in each config's extracted episodes.
CAM = {
    "RH20T_cfg1": "038522063145",
    "RH20T_cfg2": "036422060215",
    "RH20T_cfg3": "104122062295",
}
OUT = os.path.join(ROOT, "cache_rh20t_v")
os.makedirs(OUT, exist_ok=True)
RGB = os.path.join(ROOT, "rgb")
EX = os.path.join(ROOT, "extracted")
FPS = 10.0                     # measured median dt ~97ms for color
CONTACT_THR = 2.0

def cfg_low(cfg, name):
    base = "ali.abouzeid/" if cfg == "RH20T_cfg1" else ""
    return os.path.join(EX, base + cfg, name)

def cfg_rgb(cfg, name):
    base = "ali.abouzeid/" if cfg == "RH20T_cfg1" else ""
    return os.path.join(RGB, base + cfg, name)

def load_fttcp(cfg, name, cam):
    """(w_s[tau;f], pose[xyzw], ts_ms) aligned arrays for one camera."""
    low = cfg_low(cfg, name)
    ftb = np.load(os.path.join(low, "transformed/force_torque_base.npy"),
                  allow_pickle=True).item()
    tcpb = np.load(os.path.join(low, "transformed/tcp_base.npy"),
                   allow_pickle=True).item()
    ft = ftb[cam]              # list of {'timestamp','zeroed','raw'}
    tc = tcpb[cam]             # list of {'timestamp','tcp','robot_ft'}
    n = min(len(ft), len(tc))
    ts = np.array([ft[i]["timestamp"] for i in range(n)], dtype=float)
    # zeroed FT: [fx fy fz tx ty tz] (force first) -> paper [tau; f]
    raw = np.stack([ft[i]["zeroed"] for i in range(n)])[:, :6].astype(np.float64)
    f = raw[:, :3]; tau = raw[:, 3:6]
    w_s = np.c_[tau, f]        # [tau; f]
    pose = np.stack([tc[i]["tcp"] for i in range(n)]).astype(np.float64)
    return w_s, pose, ts

def build_episode(cfg, name):
    """Build aligned low-dim cache + return needed frame indices."""
    cam = CAM[cfg]
    r = cfg_rgb(cfg, name)
    camd = os.path.join(r, "cam_" + cam)
    if not os.path.isdir(camd):
        return None
    w_s, pose, ts = load_fttcp(cfg, name, cam)
    td = np.load(os.path.join(camd, "timestamps.npy"), allow_pickle=True).item()
    cts = np.array(td["color"], dtype=float)
    n = min(len(w_s), len(cts))
    if n < 100:
        return None
    # twist (fixed): spatial Plucker from pose (x,y,z,qx,qy,qz,qw -> wxyz)
    pose_w = np.c_[pose[:n, :3], pose[:n, 6], pose[:n, 3:6]]
    om, v = twists_from_pose_fast(pose_w[:n], 1.0 / FPS)
    nu = np.c_[om, v].astype(np.float32)
    np.savez_compressed(os.path.join(OUT, f"{cfg}__{name}.npz"),
                        w=w_s[:n].astype(np.float32), nu=nu,
                        pose=pose[:n].astype(np.float32),
                        joint=np.zeros((n, 1), np.float32),
                        ts=ts[:n].astype(np.float64))
    return dict(n=n, video=os.path.join(camd, "color.mp4"), name=name, cfg=cfg)

def twists_from_pose_fast(pose, dt):
    """Spatial Plucker twist (om, v= pdot - om x p) from (pos, quat-wxyz)."""
    T = len(pose)
    p = pose[:, :3]
    R = np.stack([quat_to_mat(pose[t, 3:7]) for t in range(T)])
    om = np.zeros((T, 3))
    for t in range(1, T):
        om[t] = so3_log(R[t-1].T @ R[t]) / dt
    v = np.gradient(p, dt, axis=0) - np.cross(om, p)
    return om, v

def main():
    plan = json.load(open(os.path.join(ROOT, "rgb_plan.json")))
    eps_meta = []
    need_video = {}
    for cfg, tasks in plan["per_cfg_episodes"].items():
        for tid, names in tasks.items():
            for name in names:
                got = build_episode(cfg, name)
                if got:
                    eps_meta.append(got)
                    # map (epkey)->(video, n) for feature pass
                    key = (cfg, name)
                    need_video[key] = got
    print(f"built {len(eps_meta)} aligned episode caches", flush=True)
    # feature extraction on color.mp4
    import torch, torchvision.transforms as T, cv2, subprocess, imageio_ffmpeg
    enc = get_frozen_encoder()
    tfm = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    feats = {}
    for (cfg, name), info in sorted(need_video.items()):
        vpath = info["video"]
        # decode all frames, encode in batches
        cap = cv2.VideoCapture(vpath)
        W, H = int(cap.get(3)), int(cap.get(4))
        cap.release()
        proc = subprocess.Popen([exe, "-v", "error", "-i", vpath, "-pix_fmt", "rgb24",
                                 "-f", "rawvideo", "pipe:1"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        nbytes = W * H * 3
        buf = b""; i = 0; Z = []
        while True:
            while len(buf) < nbytes:
                chunk = proc.stdout.read(nbytes - len(buf))
                if not chunk: break
                buf += chunk
            if len(buf) < nbytes: break
            fr = np.frombuffer(buf[:nbytes], np.uint8).reshape(H, W, 3)
            buf = buf[nbytes:]
            Z.append((i, tfm(fr))); i += 1
        proc.wait()
        key = (cfg, name)
        if len(Z) < 0.9 * info["n"]:
            print(f"  SKIP {name}: decoded {len(Z)}/{info['n']}", flush=True)
            continue
        with torch.no_grad():
            for k in range(0, len(Z), 256):
                batch = Z[k:k+256]
                idxs = [b[0] for b in batch]
                X = torch.stack([b[1] for b in batch]).to(DEVICE)
                out = enc(X).cpu().numpy()
                for fi, z in zip(idxs, out):
                    feats[(cfg, name, fi)] = z
        print(f"  {name}: {len(Z)} frames -> {len(out)} feat", flush=True)
    with open(os.path.join(OUT, "visfeat.pkl"), "wb") as f:
        pickle.dump(feats, f)
    print(f"[wrote] {len(feats)} frame features", flush=True)

if __name__ == "__main__":
    main()
