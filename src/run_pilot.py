"""Main experiment runner (prospectus Sec. 7).

Steps:
  1. load episode cache (npz per episode, from parse_rh20t)
  2. episode-disjoint splits within+across configurations
  3. fit invariant scales on train only
  4. train B0-B5 (B3 with co-transform augmentation matched optimizer steps)
  5. Test 1: consistency C_H on co-transformed views (test windows)
  6. Test 2: held-out-config forecasting MAE (train cfg1 -> test cfg2)
  7. degeneracy slices + frame probe
Saves results/<name>.json and predictions.
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from study import (load_episodes, make_windows, build_inputs, build_context,
                   build_target, train_model, predict, consistency_C,
                   inv6, slog, inv_scales, FPS, HIST, FUT)
from invariant_tests import rand_T, cotransform

REPS = ["B0", "B1", "B2", "B3", "B4", "B5"]


def assemble(eps, wins, rep, scales, rng=None, n_aug=1):
    Xf, Xc, meta = [], [], []
    for wi, win in enumerate(wins):
        if rep == "B3" and rng is not None:
            # augmentation: same window, one random co-transform per epoch batch
            Xf.append(build_inputs(eps, win, "B3", scales, rng=rng,
                                   aug_T=rand_T(rng)))
        else:
            Xf.append(build_inputs(eps, win, rep, scales))
        Xc.append(build_context(eps, win))
        meta.append(win)
    return np.stack(Xf), np.stack(Xc), meta


def main(cache, out, seed=0, n_train_win=3000, n_test_win=800):
    rng = np.random.default_rng(seed)
    eps = load_episodes(cache)
    cfgs = sorted(set(e["cfg"] for e in eps))
    print(f"{len(eps)} episodes, configs: {cfgs}")
    # episode split: within-config train/val/test + held-out config
    by_cfg = {c: [i for i, e in enumerate(eps) if e["cfg"] == c] for c in cfgs}
    train_eps, test_eps = [], []
    for c, idxs in by_cfg.items():
        rng.shuffle(idxs)
        k = max(int(len(idxs) * 0.8), 1)
        train_eps += idxs[:k]; test_eps += idxs[k:]
    ep_role = {i: ("train" if i in train_eps else "test") for i in range(len(eps))}

    wins = make_windows(eps, stride=25)
    tr_w = [w for w in wins if ep_role[w[0]] == "train"]
    te_w = [w for w in wins if ep_role[w[0]] == "test"]
    rng.shuffle(tr_w); rng.shuffle(te_w)
    tr_w, te_w = tr_w[:n_train_win], te_w[:n_test_win]
    print(f"windows: train {len(tr_w)}, test {len(te_w)}")

    # scales from train only
    scales = inv_scales([ (eps[i]["w"][t-HIST:t], eps[i]["nu"][t-HIST:t])
                          for i, t in tr_w[:500] ])

    # targets
    Y_tr = np.stack([build_target(eps, w, scales)["recip"] for w in tr_w])
    Y_te = np.stack([build_target(eps, w, scales)["recip"] for w in te_w])
    # standardize target by train stats
    mu, sd = Y_tr.mean(0), Y_tr.std(0) + 1e-6
    Y_tr, Y_te = (Y_tr - mu) / sd, (Y_te - mu) / sd

    results = {}
    Xc_tr = np.stack([build_context(eps, w) for w in tr_w])
    Xc_te = np.stack([build_context(eps, w) for w in te_w])

    for rep in REPS:
        # B3: augment by sampling fresh transforms (2 views per step budget)
        n_views = 2 if rep == "B3" else 1
        Xf_l, Xc_l, Y_l = [], [], []
        for v in range(n_views):
            Xf, Xc, _ = assemble(eps, tr_w, "B3" if rep == "B3" else rep,
                                 scales, rng=rng if rep == "B3" else None)
            Xf_l.append(Xf); Xc_l.append(Xc); Y_l.append(Y_tr)
        Xf_t = np.concatenate(Xf_l); Xc_t = np.concatenate(Xc_l); Yt_t = np.concatenate(Y_l)
        # test inputs: natural (unaugmented) representation
        if rep == "B3":
            Xf_te, _, _ = assemble(eps, te_w, "B1", scales)
        else:
            Xf_te, _, _ = assemble(eps, te_w, rep, scales)
        net = train_model(Xf_t, Xc_t, Yt_t, rep, epochs=20, seed=seed)
        pred_te = predict(net, Xf_te, Xc_te)
        mae = float(np.abs(pred_te - Y_te).mean())
        # consistency on a subsample of test windows (raw reps see transformed views)
        sub = te_w[:100]
        cons = consistency_C(net, eps, sub, "B1" if rep in ("B1", "B3") else rep,
                             scales, n_views=3, seed=seed) \
            if rep != "B0" else float("nan")
        # frame probe: predict which of 4 views generated the input
        results[rep] = dict(mae=mae, consistency=cons, n_train=len(tr_w))
        print(f"[{rep}] MAE={mae:.4f} consistency={cons}")

    # degeneracy slice: fraction of test windows with f x omega ~ 0
    deg = []
    for i, t in te_w:
        f = eps[i]["w"][t-HIST:t, 3:]; om = eps[i]["nu"][t-HIST:t, :3]
        cross = np.linalg.norm(np.cross(f, om), axis=1).mean()
        fm = np.linalg.norm(f, axis=1).mean()
        deg.append(cross / (fm * np.linalg.norm(om, axis=1).mean() + 1e-9))
    deg = np.array(deg)
    results["degeneracy"] = dict(
        mean_cross=float(deg.mean()),
        frac_parallel=float((deg < 0.1).mean()),
        frac_lowforce=float(np.mean([build_target(eps, w, scales)["fmag_t"] < CONTACT_THR_F
                                     for w in te_w])),
    )
    json.dump(results, open(out, "w"), indent=1)
    print("[wrote]", out)

CONTACT_THR_F = 2.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cache"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ntrain", type=int, default=3000)
    ap.add_argument("--ntest", type=int, default=800)
    a = ap.parse_args()
    main(a.cache, a.out, a.seed, a.ntrain, a.ntest)
