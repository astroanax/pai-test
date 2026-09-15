set -euo pipefail
RUN=runs-vertical
CONFIG=experiment/config.json
fail() { echo "[resume] FAIL: $1"; exit 1; }
for seed in 0 1; do
  for mode in uniform pullback; do
    if [ -f "$RUN/student_${mode}_seed${seed}.pt.complete.json" ]; then
      echo "[resume] skip train $mode/$seed"
    else
      python3 experiment/pilot.py --config "$CONFIG" --protocol "$RUN/protocol_locked.json" \
        train --cache "$RUN/shared_metrics.npz" \
        --output "$RUN/student_${mode}_seed${seed}.pt" --mode "$mode" \
        --seed "$seed" --updates 5 --initial "$RUN/warm.pt" || fail "train $mode/$seed"
    fi
    if [ -f "$RUN/final_${mode}_seed${seed}.jsonl.complete.json" ]; then
      echo "[resume] skip eval $mode/$seed"
    else
      python3 experiment/pilot.py --config "$CONFIG" --protocol "$RUN/protocol_locked.json" \
        evaluate --student "$RUN/student_${mode}_seed${seed}.pt" --name "$mode" \
        --seed "$seed" --episodes 2 --output "$RUN/final_${mode}_seed${seed}.jsonl" \
        || fail "evaluate $mode/$seed"
    fi
  done
done
python3 experiment/analyze.py --config "$CONFIG" --protocol "$RUN/protocol_locked.json" \
  --inputs "$RUN"/final_*.jsonl --output "$RUN/paired.json" || fail "analysis"
mkdir -p "$RUN/negative"
cp "$RUN"/final_*.jsonl "$RUN/negative/"
head -n -1 "$RUN/negative/final_pullback_seed1.jsonl" > "$RUN/negative/partial.jsonl" \
  && mv "$RUN/negative/partial.jsonl" "$RUN/negative/final_pullback_seed1.jsonl"
cp "$RUN"/final_*.meta.json "$RUN/negative/" 2>/dev/null || true
if python3 experiment/analyze.py --config "$CONFIG" --protocol "$RUN/protocol_locked.json" \
    --inputs "$RUN"/negative/final_*.jsonl --output "$RUN/negative/paired.json" 2>/dev/null; then
  fail "negative control accepted incomplete"
else
  echo "[resume] negative control passed"
fi
echo "[resume] DONE"
