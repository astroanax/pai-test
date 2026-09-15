# Execution-Pullback Pilot — Audit Bundle

Date: 2026-09-16. Machine: local copy of work run on `aiclub` (A100 80GB) plus a local CPU vertical.

## What this is

Few-step flow-policy distillation pilot on visual Push-T: does an execution-pullback
penalty (physical sensitivity transported through the frozen teacher) beat an
endpoint-linearized baseline, matched on architecture, data, and update count?
Primary contrast: **pullback vs endpoint**, 6 modes x 2 seeds, 100 final scenes.

## Pipeline status (all on aiclub, `runs-full/`, protocol `65ab600e`)

Complete end to end with no crashes: preflight, sanity (all torch checks green),
smoke 11/11, teacher cache (2048 contexts), warm student (6000 updates, label-only),
shared cache, metric cache (288 contexts, scales healthy: 95/96 positives, no
low-support flags), geometry diagnostic, protocol lock, all 12 students trained,
all 12 student evals + reference-teacher + native-fast evals, paired analysis.

## Headline result

Principal contrast **pullback vs endpoint: difference 0.0, CI [0.0, 0.0], 0 wins /
0 losses on both seeds** — but the comparison is VOID, not negative:

- Every student: 0.0 success (scores ~0.20-0.24).
- 16-step Gaussian reference teacher: 0.0 success (score 0.25).
- 1-step uniform native-fast baseline: 0.15 success (score 0.52).

The students faithfully learned an incompetent teacher. Suspected cause:
source-distribution mismatch (Gaussian eval vs the checkpoint's native uniform
1-step convention) and/or eval-harness mismatch. Next step per the paper's own
gate: reproduce a competent teacher (1-step uniform, 16-step uniform, 16-step
Gaussian on 20 dev scenes) before any further training. If none reaches score
0.65 / success 0.50, the honest deliverable is the negative/readiness diagnosis.

Geometry diagnostic (supporting, not behavioral): exact-pullback rank
association with observed physical error rho 0.82-1.0 on actual student errors.

## Contents

- `experiment/`: full pipeline code (see its README.md for run order).
- `remote-runs/runs-full/`: the complete A100 run — logs, per-episode JSONL,
  `paired_pullback_endpoint.json`, `geometry.json`, protocol lock, smoke report.
- `remote-runs/runs-vertical2/`: tiny vertical test that passed end to end
  (uniform vs pullback, 2 seeds, 2 scenes), incl. negative control.
- `remote-runs/runs-vertical/`, `remote-runs/runs/`: earlier partial runs.
- `remote-runs/notes-remote-logs/`: remote console logs (vertical-*.log).
- `runs-local/`, `vertical-local*.log`: local CPU run (same pipeline shape).
- `assets/`: NOT bundled (335MB checkpoint + 31MB dataset). Checkpoint: HRI
  Flow-Matching Push-T (Drive id `19A0sdo-OygRE8WnZG8OYkB0nEvsIPY1R`); dataset:
  `pusht_cchi_v7_replay.zarr`; upstream commit `516e8e1`.

## Excluded from this zip

`*.pt` / `*.pth` checkpoints, `*.npz` caches, rendered images, `__pycache__`,
venvs, and `assets/`/`external/`. Checkpoints can be retrained from the locked
protocol + caches; hashes for every artifact are in the `.meta.json` sidecars.
