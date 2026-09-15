set -euo pipefail

CONFIG="${CONFIG:-experiment/config.json}"
PROTOCOL="${PROTOCOL:-runs/protocol_locked.json}"
RUN="${RUN:-runs}"

# Stage 2 of the two-stage workflow: this runner VERIFIES artifacts the user has
# already produced deliberately (smoke report, warm start, metric cache,
# protocol lock). It never creates the lock and never injects the acknowledgement
# flag on the user's behalf.
if [ ! -f "${PROTOCOL}" ]; then
  echo "no protocol lock at ${PROTOCOL}; run experiment/lock_protocol.py --confirm-no-final-results yourself" >&2
  exit 1
fi
for artifact in "${RUN}/smoke_report.json" "${RUN}/warm.pt" "${RUN}/shared_metrics.npz"; do
  if [ ! -f "${artifact}" ]; then
    echo "missing required artifact ${artifact}" >&2
    exit 1
  fi
  if [ ! -f "${artifact}.complete.json" ]; then
    echo "${artifact} has no completion record; it is partial or predates the current schema" >&2
    exit 1
  fi
done

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

ref="${RUN}/final_reference_teacher${REF_STEPS}_${REF_SOURCE}.jsonl"
if [ ! -f "${ref}.complete.json" ]; then
  python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
    evaluate --name "reference_teacher${REF_STEPS}_${REF_SOURCE}" \
    --steps "${REF_STEPS}" --source "${REF_SOURCE}" --output "${ref}"
fi

fast="${RUN}/final_nativefast${FAST_STEPS}_${FAST_SOURCE}.jsonl"
if [ ! -f "${fast}.complete.json" ]; then
  python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
    evaluate --name "nativefast${FAST_STEPS}_${FAST_SOURCE}" \
    --steps "${FAST_STEPS}" --source "${FAST_SOURCE}" --output "${fast}"
fi

python experiment/analyze.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
  --inputs "${RUN}"/final_*.jsonl --output "${RUN}/paired_pullback_endpoint.json"
echo "comparison complete"
