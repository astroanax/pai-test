set -euo pipefail
cd ~/rehan/execution-pullback-test
export EXECPB_DISABLE_CUDNN=1
RUN=runs-full
log() { echo "[finish] $1" >> "$RUN/finish.log"; }

# 1. wait for the reference-teacher eval started earlier
log "waiting for teacher eval"
for i in $(seq 1 120); do
  if [ -f "$RUN/final_reference_teacher16.jsonl.complete.json" ]; then break; fi
  if grep -q FAIL "$RUN/eval-teacher.log" 2>/dev/null; then log "teacher eval FAILED"; exit 1; fi
  sleep 60
done
[ -f "$RUN/final_reference_teacher16.jsonl.complete.json" ] || { log "teacher eval timed out"; exit 1; }
log "teacher eval done"

# 2. native-fast baseline (1 step, uniform source)
if [ ! -f "$RUN/final_nativefast1_uniform.jsonl.complete.json" ]; then
  python3 -u experiment/pilot.py --config experiment/config.json --protocol "$RUN/protocol_locked.json" \
    evaluate --name nativefast1_uniform --steps 1 --source uniform \
    --output "$RUN/final_nativefast1_uniform.jsonl" >> "$RUN/eval-nativefast.log" 2>&1 \
    || { log "nativefast eval FAILED"; exit 1; }
fi
log "nativefast done"

# 3. paired analysis (primary: pullback vs endpoint)
python3 experiment/analyze.py --config experiment/config.json --protocol "$RUN/protocol_locked.json" \
  --inputs "$RUN"/final_*.jsonl --output "$RUN/paired_pullback_endpoint.json" \
  >> "$RUN/finish.log" 2>&1 || { log "analysis FAILED"; exit 1; }
log "analysis done"

echo DONE >> "$RUN/finish.log"
log "pipeline complete"
