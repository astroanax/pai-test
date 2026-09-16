set -euo pipefail
# Two-scene vertical integration test: exercises the ACTUAL producer/consumer
# commands end to end on a tiny budget before committing the A100 budget.
# Usage: bash experiment/vertical_test.sh [run_dir]
# Requires: the teacher checkpoint plus assets/normalizer.npz already built
# (e.g. `python experiment/pilot.py normalizer --dataset ... --output
# assets/normalizer.npz`). This script builds nothing from RH_DATASET.

RUN="${1:-runs-vertical}"
CONFIG="${CONFIG:-experiment/config.json}"
mkdir -p "${RUN}"
echo "[vertical] run dir: ${RUN}"

fail() { echo "[vertical] FAIL: $1" >&2; exit 1; }

python experiment/preflight.py --config "${CONFIG}" --allow-cpu > "${RUN}/preflight.json" \
  || fail "preflight"
python experiment/sanity.py || fail "sanity"
if [ ! -f "${RUN}/smoke_report.json.complete.json" ]; then
  rm -f "${RUN}/smoke_report.json" "${RUN}/smoke_report.json.reserved" "${RUN}/smoke_report.json.incomplete"
  python experiment/smoke.py --config "${CONFIG}" --output "${RUN}/smoke_report.json" \
    || fail "smoke (fix the target stack before anything else)"
fi

# 12 training scenes + 12 validation scenes, 4 decisions each (2 kept per
# episode): the sensitivity gate needs >= 8 positive traces, and free-space
# contexts are legitimately zero-sensitivity, so 4 train contexts cannot pass
python experiment/pilot.py --config "${CONFIG}" --allow-unlocked \
  collect --output "${RUN}/teacher_cache.npz" --episodes 12 \
  --splits train validation --decision-limit 4 || fail "teacher collect"
python experiment/pilot.py --config "${CONFIG}" --allow-unlocked \
  train --cache "${RUN}/teacher_cache.npz" --mode uniform \
  --output "${RUN}/warm.pt" --updates 5 || fail "warm train (label-only path)"
python experiment/pilot.py --config "${CONFIG}" --allow-unlocked \
  collect --student "${RUN}/warm.pt" --output "${RUN}/shared_cache.npz" \
  --episodes 12 --decision-limit 4 || fail "shared collect"
python experiment/pilot.py --config "${CONFIG}" --allow-unlocked \
  metrics --cache "${RUN}/shared_cache.npz" --limit 4 \
  --output "${RUN}/profile_metrics.npz" || fail "metric profile"
python experiment/pilot.py --config "${CONFIG}" --allow-unlocked \
  metrics --cache "${RUN}/shared_cache.npz" --output "${RUN}/shared_metrics.npz" \
  || fail "metric cache"
python experiment/diagnose.py --config "${CONFIG}" --allow-unlocked \
  --cache "${RUN}/shared_metrics.npz" --student "${RUN}/warm.pt" \
  --output "${RUN}/geometry.json" --predictions "${RUN}/geometry_predictions.npz" \
  || fail "diagnostic"

python experiment/lock_protocol.py --config "${CONFIG}" --confirm-no-final-results \
  --kind integration_test \
  --modes uniform pullback --seeds 0 1 --updates 5 --test-episodes 2 \
  --min-metric-fraction 0 --min-support 0 --final-prefix "${RUN}/final_" \
  --cache "${RUN}/shared_metrics.npz" --warm "${RUN}/warm.pt" \
  --smoke-report "${RUN}/smoke_report.json" --output "${RUN}/protocol_locked.json" \
  || fail "protocol lock"

for seed in 0 1; do
  for mode in uniform pullback; do
    python experiment/pilot.py --config "${CONFIG}" --protocol "${RUN}/protocol_locked.json" \
      train --cache "${RUN}/shared_metrics.npz" \
      --output "${RUN}/student_${mode}_seed${seed}.pt" --mode "${mode}" \
      --seed "${seed}" --updates 5 --initial "${RUN}/warm.pt" || fail "train ${mode}/${seed}"
    python experiment/pilot.py --config "${CONFIG}" --protocol "${RUN}/protocol_locked.json" \
      evaluate --student "${RUN}/student_${mode}_seed${seed}.pt" --name "${mode}" \
      --seed "${seed}" --output "${RUN}/final_${mode}_seed${seed}.jsonl" \
      || fail "evaluate ${mode}/${seed}"
  done
done

python experiment/analyze.py --config "${CONFIG}" --protocol "${RUN}/protocol_locked.json" \
  --inputs "${RUN}"/final_*.jsonl --output "${RUN}/paired.json" \
  || fail "analysis of the complete tiny result"

# negative control, step 1: an INTACT copy (evaluations + sidecars + completion
# records) must still pass analysis
mkdir -p "${RUN}/negative"
cp "${RUN}"/final_*.jsonl "${RUN}/negative/"
cp "${RUN}"/final_*.meta.json "${RUN}/negative/"
cp "${RUN}"/final_*.complete.json "${RUN}/negative/"
python experiment/analyze.py --config "${CONFIG}" --protocol "${RUN}/protocol_locked.json" \
  --inputs "${RUN}"/negative/final_*.jsonl --output "${RUN}/negative/paired_intact.json" \
  || fail "analysis rejected an intact copy (negative-control fixture broken)"
echo "[vertical] negative control step 1 passed: intact copy accepted"
# step 2a: byte corruption must trip the hash layer first (ARTIFACT_HASH_MISMATCH),
# independently of any completeness reasoning
head -n -1 "${RUN}/negative/final_pullback_seed1.jsonl" > "${RUN}/negative/partial.jsonl" \
  && mv "${RUN}/negative/partial.jsonl" "${RUN}/negative/final_pullback_seed1.jsonl"
if python experiment/analyze.py --config "${CONFIG}" --protocol "${RUN}/protocol_locked.json" \
    --inputs "${RUN}"/negative/final_*.jsonl --output "${RUN}/negative/paired.json" \
    > "${RUN}/negative/analysis_err_hash.txt" 2>&1; then
  fail "analysis accepted a byte-corrupted evaluation"
else
  grep -q "ARTIFACT_HASH_MISMATCH" "${RUN}/negative/analysis_err_hash.txt" \
    || fail "analysis rejected the corrupted fixture for the wrong reason: $(cat "${RUN}/negative/analysis_err_hash.txt")"
  echo "[vertical] negative control step 2a passed: hash corruption rejected with ARTIFACT_HASH_MISMATCH"
fi
# step 2b: a truncated-but-recompleted fixture (fresh identity record over the
# truncated bytes, consistent sidecar) must fail on completeness with
# MISSING_EPISODE -- never silently accepted, never confused with corruption
python - "${RUN}/negative/final_pullback_seed1.meta.json" <<'PY'
import json, sys
path = sys.argv[1]
meta = json.load(open(path))
meta["scenes"] = meta["scenes"][:-1]
json.dump(meta, open(path, "w"), indent=2)
PY
python - "${RUN}/negative/final_pullback_seed1.jsonl" "${RUN}/negative/final_pullback_seed1.meta.json" <<'PY'
import sys
sys.path.insert(0, "experiment")
import contract as C
C.complete_output(sys.argv[1])
C.complete_output(sys.argv[2])
PY
if python experiment/analyze.py --config "${CONFIG}" --protocol "${RUN}/protocol_locked.json" \
    --inputs "${RUN}"/negative/final_*.jsonl --output "${RUN}/negative/paired.json" \
    > "${RUN}/negative/analysis_err.txt" 2>&1; then
  fail "analysis accepted a deliberately incomplete experiment"
else
  grep -q "MISSING_EPISODE" "${RUN}/negative/analysis_err.txt" \
    || fail "analysis rejected the partial fixture for the wrong reason: $(cat "${RUN}/negative/analysis_err.txt")"
  echo "[vertical] negative control step 2b passed: missing episode rejected with MISSING_EPISODE"
fi

echo "[vertical] PASS: all producer/consumer stages ran end to end in ${RUN}"
