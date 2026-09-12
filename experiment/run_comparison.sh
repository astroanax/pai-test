set -euo pipefail
modes="${MODES:-uniform prefix endpoint pullback identity scalar}"
for seed in 0 1; do
  for mode in ${modes}; do
    python experiment/pilot.py train --cache runs/shared_metrics.npz --output runs/student_${mode}_seed${seed}.pt --mode "${mode}" --seed "${seed}" --updates 6000 --initial runs/warm.pt
  done
done
for seed in 0 1; do
  for mode in ${modes}; do
    python experiment/pilot.py evaluate --student runs/student_${mode}_seed${seed}.pt --name "${mode}" --seed "${seed}" --episodes 100 --output runs/final_${mode}_seed${seed}.jsonl
  done
done
python experiment/pilot.py evaluate --name teacher16 --steps 16 --episodes 100 --output runs/final_teacher16.jsonl
python experiment/analyze.py --inputs runs/final_*.jsonl --output runs/paired_pullback_endpoint.json --method pullback --baseline endpoint
echo "comparison complete"
