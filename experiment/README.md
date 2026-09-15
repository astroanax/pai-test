execution pullback pilot, project-i workspace

Files, all under execution-pullback/experiment/
- contract.py      config validation, split checks, hashing, cache schema v2
                   writers/verifiers, protocol id and revalidation, atomic writes,
                   reservation plus completion records, file manifest
- core.py          student, integration, one shared source sampler, finite
                   differences, VJP probes, exact transmitted metric, six losses,
                   scale matching
- hri_adapter.py   device-aware strict checkpoint load, complete-history replay,
                   flag separation, single command-preparation pathway
- pilot.py         normalizer, collect (train plus validation, with labels),
                   metrics (pre-selected budget), train (label-only or metric),
                   evaluate
- diagnose.py      held-out geometry diagnostic on actual student errors
- analyze.py       protocol-bound paired analysis over the full locked run set
- smoke.py         target-stack smoke report, required before locking
- sanity.py        algebra, schema, protocol, and estimator regressions
- preflight.py     imports, cuda requirement, assets, manifest, exit status
- lock_protocol.py protocol lock with source, upstream, package, and choice
                   provenance
- run_comparison.sh stage-2 runner: verifies locked artifacts, reads every
                   setting from the lock, never creates the lock
- MANIFEST.json    transfer manifest

Run order (schema v2 artifacts are incompatible with v1; rebuild them)
1  python experiment/preflight.py --config experiment/config.json \
       --write-manifest experiment/MANIFEST.json
2  python experiment/pilot.py --config experiment/config.json \
       normalizer --dataset <zarr path> --output assets/normalizer.npz
   (global flags such as --config, --device, --protocol, and --allow-unlocked
   must precede the subcommand; argparse rejects them after it)
3  python experiment/sanity.py --require-torch
4  python experiment/smoke.py --config experiment/config.json \
       --output runs/smoke_report.json
5  teacher cache:     pilot.py collect --output runs/teacher_cache.npz
   warm student:      pilot.py --config experiment/config.json --allow-unlocked \
                          train --cache runs/teacher_cache.npz \
                          --mode uniform --output runs/warm.pt
6  shared cache:      pilot.py collect --student runs/warm.pt \
                          --output runs/shared_cache.npz
7  metric profile:    pilot.py metrics --cache runs/shared_cache.npz --limit 8 \
                          --output runs/profile_metrics.npz
   metric cache:      pilot.py metrics --cache runs/shared_cache.npz \
                          --output runs/shared_metrics.npz
8  geometry:          diagnose.py --cache runs/shared_metrics.npz \
                          --student runs/warm.pt --output runs/geometry.json \
                          --predictions runs/geometry_predictions.npz
9  lock:              lock_protocol.py --confirm-no-final-results
10 bash experiment/run_comparison.sh

Notes on conventions
- source distributions are not interchangeable: `gaussian` is randn, `uniform`
  is the upstream native-fast [0, 1) draw, `uniform_symmetric` is [-1, 1). The
  same sampler serves collection and evaluation.
- do not add ImageNet normalization; the HRI image environment already
  normalizes the rendered frames.
- teacher_steps must be even so the half map is well defined.
- `uniform` and `prefix` train on labels only; `endpoint`, `pullback`,
  `identity`, and `scalar` require the metric cache. The warm start never needs
  simulator derivatives.

Fixes applied against the third audit
1  teacher modules are moved to the selected device and the batch-independence
   test builds its tensors on the model device; smoke asserts placements
2  warm start restored: collection stores the teacher targets, `uniform` and
   `prefix` train from labels alone, and metric scales are computed only for
   metric modes
3  one versioned schema: collection and metrics both write a flattened top-level
   `contexts` array through contract.py, and every consumer validates that
   normalized form
4  collection covers train and validation with split labels; training filters the
   train split; the diagnostic uses validation scenes only
5  the sanity suffix returns its input shape and the transported metric is
   compared against an independently known linear composition
6  smoke no longer calls numpy on a cuda tensor and derives the exact-metric
   shape (batch, physical dim, horizon x action dim) = (1, 32, 32)
7  the diagnostic uses 16x2 generative errors, 32-dimensional directions,
   singular vectors wrapped as single-element lists, and converts to the
   8-action prefix only at the execution boundary
8  the runner verifies existing artifacts, refuses to create the lock, and reads
   modes, seeds, updates, cache, warm start, reference, and native-fast settings
   from the lock
9  evaluation metadata is indexed by (intrinsic mode, intrinsic seed), verified
   against each file's own rows, scene set, and split
10 evaluation honours the requested source, and one shared sampler implements
   the declared conventions
11 the diagnostic now perturbs the teacher midpoint and runs the suffix for the
   early term, and tests late and endpoint terms on their own inputs with squared
   feature differences, plus an epsilon versus epsilon/2 stability check
12 live signatures are recorded from the image environment before each chunk and
   the initial signature is stored; smoke compares image and state transitions
13 one prepare_commands pathway decodes, records raw violations, clips once, and
   returns exactly what is executed; counters are per coordinate
14 training, evaluation, and analysis call verify_current_protocol, which
   recomputes sources, upstream, packages, configuration, and asset hashes; the
   canonical id includes warm start, modes, seeds, updates, and the smoke report
15 the metric subset is chosen before any derivative work, validation contexts
   are included, one repeated-branch check replaces duplicate Jacobians, labels
   are kept for the full cache, and per-context timing is logged
16 decision timing spans encoding through host-side command preparation, the
   action head is timed separately, cuda is synchronized at explicit boundaries,
   perf_counter is used, and warm-up episodes are flagged and excluded
plus tie-aware Spearman with an explicit constant-input note, preflight honouring
   --config and failing on an unavailable cuda request, real output reservation
   with completion records, descriptive tables that require every locked run,
   binary success validation, torch seeding tied to the training seed, and a
   transfer manifest

Vertical test (run this first on the GPU machine, before the A100 budget)
- bash experiment/vertical_test.sh [run_dir]: two training scenes plus two
  validation scenes through every real producer/consumer command, including the
  lock, brief training, two-seed evaluation, analysis, and a negative control
  that must reject a deliberately incomplete experiment.

Fixes applied against the fourth audit
- smoke envs() keeps the reset observation instead of discarding it; the
  exact-metric shape is derived from the live tensors, never hard-coded
- device comparison normalizes cuda indices, so "cuda" and "cuda:0" match
- warm training accepts the teacher cache; only metric modes require mode
  "shared", which the metric writer now records
- metric_count 0 falls back to the full batch instead of an empty NaN mean
- the unlocked uniform warm start passes provenance; unlocked metric
  checkpoints are still rejected
- the metric subset is spread round-robin across scenes with a fixed seed
- the marked pool is drawn identically for every mode, so all arms share the
  same minibatch streams
- the diagnostic pairs each predictor only with its matching outcome, with
  dedicated late and endpoint predictors
- global flags must precede the subcommand; the README warm-start line is fixed
- missing locked artifacts now fail verification, and final evaluation always
  requires the lock (training needs it except for the unlocked warm start)
- student collection computes teacher targets only for stored decisions
- lock_protocol accepts --min-metric-fraction, --final-prefix, and
  --test-episodes so the vertical test can lock a tiny predeclared run

Not run here: no GPU, no torch, no simulator, and no upstream checkout are
present in this workspace, so nothing is executed end to end. Run the vertical
test on the GPU machine before locking the main protocol.
