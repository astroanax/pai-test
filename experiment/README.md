execution pullback pilot, project-i workspace

layout, all under execution-pullback/experiment/
- contract.py      config validation, split checks, source/upstream/package hashing,
                   cache-contract fingerprint, protocol id, atomic writes, output
                   reservation, completeness/metadata requirements, file manifest
- core.py          student, integration, finite differences, VJP probes, exact
                   transmitted metric, six losses, scale matching
- hri_adapter.py   strict checkpoint load, complete-history replay, flag separation
- pilot.py         normalizer, collect, metrics, train, evaluate
- diagnose.py      held-out geometry diagnostic on actual student errors
- analyze.py       protocol-bound paired analysis
- smoke.py         target-stack smoke report (required before locking)
- sanity.py        algebra, contract, and estimator regression tests
- preflight.py     imports, cuda, assets, manifest, config, exit status
- lock_protocol.py protocol lock with source, upstream, and package provenance
- run_comparison.sh locked matched comparison including reference and native-fast
- MANIFEST.json    file manifest for transfer verification

conventions fixed from upstream
- training source is gaussian noise; the upstream test example uses uniform noise
  with one euler step, so native-fast is evaluated explicitly (native_fast_* keys)
- pred horizon 16, action dim 2, execution horizon 8, condition dim 514,
  agent action range 0 to 512, success is block coverage above 0.95 within 300 steps
- do not add ImageNet normalization: the HRI image environment already normalizes
- teacher_steps must be even so the half map is well defined

method notes
- execution_pullback exact metric: C = B0 J_prefix, Q0(e) = ||C e||^2,
  tr M = ||C||_F^2, built with one VJP per prefix output (16 backwards passes, no
  extra simulator branches). This replaces the rank-4 probe sketch, which leaves
  consequential directions unpenalized; metric_mode="sketch" keeps the probe path
  with an audit against the exact metric and a probe-count sweep in diagnose.py
- switch modes never mix the two metric representations; the cache records which
  one it holds and contract.py rejects a mismatch

run order
- python experiment/preflight.py --write-manifest experiment/MANIFEST.json
- python experiment/sanity.py --require-torch
- python experiment/smoke.py            (must pass before locking)
- python experiment/pilot.py normalizer --dataset <zarr> --output assets/normalizer.npz
- python experiment/pilot.py collect --output runs/teacher_cache.npz
- python experiment/pilot.py train --cache runs/teacher_cache.npz --mode uniform --output runs/warm.pt
- python experiment/pilot.py collect --student runs/warm.pt --output runs/shared_cache.npz
- python experiment/pilot.py metrics --cache runs/shared_cache.npz --limit 8 --output runs/profile_metrics.npz
- python experiment/pilot.py metrics --cache runs/shared_cache.npz --output runs/shared_metrics.npz
- python experiment/diagnose.py --cache runs/shared_metrics.npz --student runs/warm.pt \
    --output runs/geometry_diagnostic.json --predictions runs/geometry_predictions.npz
- python experiment/pilot.py train --cache runs/shared_metrics.npz --mode pullback \
    --updates 250 --initial runs/warm.pt --output runs/profile_student.pt
- python experiment/lock_protocol.py --confirm-no-final-results
- bash experiment/run_comparison.sh

audit fixes, all against the second friend audit
1  analysis loads the protocol lock, requires the full method x seed x scene product
   for both arms, rejects missing runs even when the remainder pairs, requires the
   declared primary contrast, and refuses a degenerate bootstrap frame
2  the lock hashes the experiment sources, the upstream model files and commit, and
   package versions; the protocol id is written into every checkpoint, evaluation
   row, and sidecar, and rechecked before evaluation and analysis
3  cache contract fingerprint covers source distribution, solver, execution horizon,
   action decoding, physical features, upstream implementation, and collection
   episode cap; diagnose.py verifies the cache and records its lineage
4  configured scene ranges must be pairwise disjoint; caches, evaluations, and the
   lock are checked against the declared splits
5  the prefix extra loss uses the marked metric half, matching the physical penalties
6  exact transmitted metric replaces the four-probe sketch by default
7  the diagnostic accepts a student checkpoint, measures actual mid/end/endpoint
   errors, tests consequences along those errors, includes the identity comparator,
   covers the late and endpoint terms, samples validation contexts across scenes,
   and saves every prediction
8  the reference teacher is read from the locked config, and native-fast is a
   separate configured evaluation with its own source distribution
9  smoke.py instantiates the adapter, loads the checkpoint strictly, resets both
   environments, checks ranges and finiteness, compares half maps, checks a suffix
   directional derivative against finite differences, checks replay agreement,
   runs one training step per mode, and writes a hash-bound pass/fail report
10 gradients use error_if_nonfinite=True; the gradient norm is logged and parameters
   are checked before saving
11 student collection reuses one forward for the midpoint instead of recomputing it,
   and computes no teacher supervision at unsaved decisions
12 outputs are reserved at command startup, written atomically, and marked
   .incomplete on failure
13 configuration validation rejects non-positive counts, odd or single-step
   teachers, out-of-range execution horizons, bad epsilon, and bad optimizer
   settings; CLI defaults use is None; development defaults use development_episodes;
   warmup_updates was removed rather than left unused
14 success, termination, truncation, and episode-cap exhaustion are recorded
   separately and success requires the environment's own success signal
15 checkpoints, evaluation rows, and sidecars carry the intrinsic mode and seed;
   analysis verifies checkpoint and protocol metadata, not just the label string
16 training logs base, anchor, penalty, weighted penalty, gradient norm, mid/end/
   endpoint error magnitudes, and the zero-sensitivity fraction
17 normalizers, cached tensors, actions, signatures, and replay comparisons are
   finiteness-checked before any tolerance test
18 the device is resolved once, cuda synchronization is guarded, and a cuda request
   on a cpu-only host fails loudly
19 out-of-range commands are counted on raw decoded actions, with executed and
   clipped counts reported separately
20 preflight writes and verifies a file manifest, and reports which sources are not
   tracked or are ignored by version control

not run in this workspace
- no gpu, no torch, no simulator, and no upstream checkout are present, so nothing
  here is executed end to end; run preflight, sanity --require-torch, and smoke.py
  on the GPU machine before locking the protocol
