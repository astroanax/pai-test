set -euo pipefail
CONFIG="${CONFIG:-experiment/config.json}"
MODES="${MODES:-uniform prefix endpoint pullback identity scalar}"
PROTOCOL="${PROTOCOL:-runs/protocol_locked.json}"
UPDATES="${UPDATES:-6000}"
TEACHER_STEPS="$(python -c "import json;print(json.load(open('${CONFIG}'))['teacher_steps'])")"
TEACHER_SOURCE="$(python -c "import json;print(json.load(open('${CONFIG}'))['source'])")"
FAST_STEPS="$(python -c "import json;print(json.load(open('${CONFIG}'))['native_fast_steps'])")"
FAST_SOURCE="$(python -c "import json;print(json.load(open('${CONFIG}'))['native_fast_source'])")"
RUN="${RUN:-runs}"
mkdir -p "${RUN}"

python experiment/smoke.py --config "${CONFIG}" --output "${RUN}/smoke_report.json" || {
  echo "smoke report failed; do not start the comparison" >&2
  exit 1
}

python experiment/lock_protocol.py --config "${CONFIG}" --modes ${MODES} \
  --updates "${UPDATES}" --cache "${RUN}/shared_metrics.npz" \
  --warm "${RUN}/warm.pt" --smoke-report "${RUN}/smoke_report.json" \
  --output "${PROTOCOL}"

for seed in 0 1; do
  for mode in ${MODES}; do
    out="${RUN}/student_${mode}_seed${seed}.pt"
    if [ ! -f "${out}" ]; then
      python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
        train --cache "${RUN}/shared_metrics.npz" --output "${out}" \
        --mode "${mode}" --seed "${seed}" --updates "${UPDATES}" \
        --initial "${RUN}/warm.pt"
    fi
  done
done

for seed in 0 1; do
  for mode in ${MODES}; do
    out="${RUN}/final_${mode}_seed${seed}.jsonl"
    if [ ! -f "${out}" ]; then
      python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
        evaluate --student "${RUN}/student_${mode}_seed${seed}.pt" \
        --name "${mode}" --seed "${seed}" --output "${out}"
    fi
  done
done

ref="${RUN}/final_teacher${TEACHER_STEPS}_${TEACHER_SOURCE}.jsonl"
if [ ! -f "${ref}" ]; then
  python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
    evaluate --name "teacher${TEACHER_STEPS}_${TEACHER_SOURCE}" \
    --steps "${TEACHER_STEPS}" --source "${TEACHER_SOURCE}" --output "${ref}"
fi

fast="${RUN}/final_nativefast${FAST_STEPS}_${FAST_SOURCE}.jsonl"
if [ ! -f "${fast}" ]; then
  python experiment/pilot.py --config "${CONFIG}" --protocol "${PROTOCOL}" \
    evaluate --name "nativefast${FAST_STEPS}_${FAST_SOURCE}" \
    --steps "${FAST_STEPS}" --source "${FAST_SOURCE}" --output "${fast}"
fi

python experiment/analyze.py --inputs "${RUN}"/final_*.jsonl \
  --protocol "${PROTOCOL}" --output "${RUN}/paired_pullback_endpoint.json"
echo "comparison complete"
