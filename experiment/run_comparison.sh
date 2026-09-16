set -euo pipefail

CONFIG="${CONFIG:-experiment/config.json}"
PROTOCOL="${PROTOCOL:-runs/protocol_locked.json}"
RUN="${RUN:-runs}"

# Stage 2 of the two-stage workflow: this runner VERIFIES artifacts the user has
# already produced deliberately (smoke report, warm start, metric cache,
# protocol lock). It never creates the lock and never injects the acknowledgement
# flag on the user's behalf. Locked paths are read from the lock FIRST, then
# those exact artifacts are validated (hash, completion record, protocol).
if [ ! -f "${PROTOCOL}" ]; then
  echo "no protocol lock at ${PROTOCOL}; run experiment/lock_protocol.py --confirm-no-final-results yourself" >&2
  exit 1
fi
LOCK_SMOKE="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['smoke_report'])" "${PROTOCOL}")"
LOCK_CACHE="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['cache'])" "${PROTOCOL}")"
LOCK_WARM="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['warm_start'])" "${PROTOCOL}")"
python - "${LOCK_SMOKE}" "${LOCK_CACHE}" "${LOCK_WARM}" <<'PY'
import sys
sys.path.insert(0, "experiment")
import contract as C
for path in sys.argv[1:]:
    C.verify_completed(path)
print("locked artifacts verified:", sys.argv[1:])
PY

if ! python - "${PROTOCOL}" <<'PY'
import json, sys
protocol = json.load(open(sys.argv[1]))
for key in ("modes", "updates", "seeds", "teacher_steps", "source_distribution",
            "native_fast", "cache", "warm_start", "protocol_id"):
    assert key in protocol, f"lock lacks {key}"
PY
then
  echo "protocol lock is incomplete" >&2
  exit 1
fi
MODES="$(python -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]))['modes']))" "${PROTOCOL}")"
UPDATES="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['updates'])" "${PROTOCOL}")"
SEEDS="$(python -c "import json,sys;print(' '.join(str(s) for s in json.load(open(sys.argv[1]))['seeds']))" "${PROTOCOL}")"
REF_STEPS="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['teacher_steps'])" "${PROTOCOL}")"
REF_SOURCE="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['source_distribution'])" "${PROTOCOL}")"
FAST_STEPS="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['native_fast']['steps'])" "${PROTOCOL}")"
FAST_SOURCE="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['native_fast']['source'])" "${PROTOCOL}")"
CACHE="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['cache'])" "${PROTOCOL}")"
WARM="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['warm_start'])" "${PROTOCOL}")"

echo "locked: modes=[${MODES}] updates=${UPDATES} seeds=[${SEEDS}]"
echo "reference teacher: ${REF_STEPS} steps, source=${REF_SOURCE}"

for seed in ${SEEDS}; do
  for mode in ${MODES}; do
    out="${RUN}/student_${mode}_seed${seed}.pt"
    if [ -f "${out}.complete.json" ]; then
      echo "skip existing ${out}"
      continue
    fi
    if [ -f "${out}" ]; then
      echo "${out} exists without a completion record; refusing to reuse" >&2
      exit 1
    fi
    python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
      train --cache "${CACHE}" --output "${out}" --mode "${mode}" \
      --seed "${seed}" --updates "${UPDATES}" --initial "${WARM}"
  done
done

for seed in ${SEEDS}; do
  for mode in ${MODES}; do
    out="${RUN}/final_${mode}_seed${seed}.jsonl"
    if [ -f "${out}.complete.json" ]; then
      echo "skip existing ${out}"
      continue
    fi
    python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
      evaluate --student "${RUN}/student_${mode}_seed${seed}.pt" \
      --name "${mode}" --seed "${seed}" --output "${out}"
  done
done

ref="${RUN}/final_teacher.jsonl"
if [ ! -f "${ref}.complete.json" ]; then
  python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
    evaluate --name "teacher" \
    --steps "${REF_STEPS}" --source "${REF_SOURCE}" --output "${ref}"
fi

fast="${RUN}/final_native_fast.jsonl"
if [ ! -f "${fast}.complete.json" ]; then
  python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
    evaluate --name "native_fast" \
    --steps "${FAST_STEPS}" --source "${FAST_SOURCE}" --output "${fast}"
fi

python experiment/analyze.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
  --inputs "${RUN}"/final_*.jsonl --output "${RUN}/paired_pullback_endpoint.json"
echo "comparison complete"
