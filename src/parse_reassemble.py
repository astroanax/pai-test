"""REASSEMBLE adapter for the symmetry benchmark: build episode npz cache
from the LeRobot v3 parquet shards (EE pose -> twist, wrist F/T wrench)."""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from symmetry import twists_from_pose

DATA = "data/reassemble"
OUT = "cache_reassemble"


def main(limit=None):
    eps_meta = pd.read_parquet(f"{DATA}/meta/episodes/chunk-000/file-000.parquet")
    tasks = pd.read_parquet(f"{DATA}/meta/tasks.parquet")
    task_names = tasks.index.tolist()
    shards = {}
    for _, row in eps_meta.iterrows():
        shards.setdefault(int(row["data/file_index"]), []).append(int(row["episode_index"]))
    dfs = {fi: pd.read_parquet(f"{DATA}/data/chunk-000/file-{fi:03d}.parquet")
           for fi in sorted(shards)}
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for _, row in eps_meta.iterrows():
        ep = int(row["episode_index"])
        df = dfs[int(row["data/file_index"])]
        dfe = df[df.episode_index == ep].sort_values("frame_index").reset_index(drop=True)
        if len(dfe) < 250:
            continue
        ee = np.stack(dfe["observation.state.ee_pose"].values)      # (T,7)
        force = np.stack(dfe["observation.force"].values)           # wrist frame
        torque = np.stack(dfe["observation.torque"].values)
        grip = np.stack(dfe["observation.state.gripper_position"].values)
        dt = 1.0 / 30.0
        om, v = twists_from_pose(ee, dt, smooth=3)
        w = np.c_[torque, force]        # [tau; f]
        nu = np.c_[om, v]               # [om; v]
        task = task_names[int(dfe["task_index"].iloc[0])]
        seg_ok = np.stack(dfe["segment.success"].values).ravel().astype(bool)
        np.savez_compressed(f"{OUT}/cfgRA__ep{ep:04d}_{task[:20].replace(' ','_')}.npz",
                            w=w.astype(np.float32), nu=nu.astype(np.float32),
                            pose=ee.astype(np.float32),
                            joint=np.c_[grip, grip].astype(np.float32),
                            success=seg_ok.astype(np.float32))
        n += 1
        if limit and n >= limit:
            break
    print(f"cached {n} REASSEMBLE episodes -> {OUT}")


if __name__ == "__main__":
    main()
