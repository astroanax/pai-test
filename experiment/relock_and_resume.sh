set -euo pipefail
RUN=runs-vertical
CONFIG=experiment/config.json
fail() { echo "[relock] FAIL: $1"; exit 1; }
if [ "${RELOCK_ACK:-}" != "1" ]; then
  echo '[relock] refusing: set RELOCK_ACK=1 to acknowledge you are deliberately creating a NEW protocol (final outcomes unseen)' >&2
  exit 1
fi
python3 experiment/lock_protocol.py --config "$CONFIG" --confirm-no-final-results \
  --modes uniform pullback --seeds 0 1 --updates 5 --test-episodes 2 \
  --kind integration_test \
  --min-metric-fraction 0 --min-support 0 --final-prefix "$RUN/final_" \
  --cache "$RUN/shared_metrics.npz" --warm "$RUN/warm.pt" \
  --smoke-report "$RUN/smoke_report.json" --output "$RUN/protocol_locked.json" \
  || fail "lock"
bash experiment/resume_vertical.sh || fail "resume"
echo "[relock] DONE"
