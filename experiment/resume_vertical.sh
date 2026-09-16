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
        --seed "$seed" --output "$RUN/final_${mode}_seed${seed}.jsonl" \
        || fail "evaluate $mode/$seed"
    fi
  done
done
python3 experiment/analyze.py --config "$CONFIG" --protocol "$RUN/protocol_locked.json" \
  --inputs "$RUN"/final_*.jsonl --output "$RUN/paired.json" || fail "analysis"
mkdir -p "$RUN/negative"
cp "$RUN"/final_*.jsonl "$RUN/negative/"
cp "$RUN"/final_*.meta.json "$RUN/negative/"
cp "$RUN"/final_*.complete.json "$RUN/negative/"
python3 experiment/analyze.py --config "$CONFIG" --protocol "$RUN/protocol_locked.json" \
  --inputs "$RUN"/negative/final_*.jsonl --output "$RUN/negative/paired_intact.json" \
  || fail "analysis rejected an intact copy"
head -n -1 "$RUN/negative/final_pullback_seed1.jsonl" > "$RUN/negative/partial.jsonl" \
  && mv "$RUN/negative/partial.jsonl" "$RUN/negative/final_pullback_seed1.jsonl"
python3 - "$RUN/negative/final_pullback_seed1.meta.json" <<'PY'
import json, sys
path = sys.argv[1]
meta = json.load(open(path))
meta["scenes"] = meta["scenes"][:-1]
json.dump(meta, open(path, "w"), indent=2)
PY
if python3 experiment/analyze.py --config "$CONFIG" --protocol "$RUN/protocol_locked.json" \
    --inputs "$RUN"/negative/final_*.jsonl --output "$RUN/negative/paired.json" \
    > "$RUN/negative/analysis_err.txt" 2>&1; then
  fail "negative control accepted incomplete"
elif ! grep -q -E "missing|incomplete|no evaluation rows" "$RUN/negative/analysis_err.txt"; then
  fail "negative control rejected for the wrong reason: $(cat "$RUN/negative/analysis_err.txt")"
else
  echo "[resume] negative control passed"
fi
echo "[resume] DONE"
