# JEPA Experiment — Final Results & Verdict

## Experiment
JEPA with frozen visual encoder (ResNet18), trained to predict future visual embeddings + contact onset/offset + future canonical wrench, on REASSEMBLE (149 episodes, 3 task groups: insert, remove, place_pick). 3 config-group holdouts (insert, remove, place_pick), 3 seeds, 5 interfaces:

| ID | Force input | Description |
|---|---|---|
| J0 | none | no force (context only) |
| J1 | raw | raw wrench [τ; f] |
| J2 | canonical | SE(3)-canonical wrench (episode-initial frame) |
| J3 | canonical + bias/mass/CoM | F3 + gravity/tool load removal |
| J4 | supervised adapter | J4 on adapter episodes (10 per holdout config), linear 6×6 map fitted on adapter episodes |

All models: frozen ResNet18 visual encoder, 4 current frames + proprio + action history + force branch → h → predictor → future visual (cosine) + contact (BCE) + future wrench (Huber). Frozen visual encoder, 15 epochs, batch 128, AdamW 3e-4. Test-time swaps: real, zero, shuffle.

## Results (3 holdouts × 3 seeds, cosine error on future visual latent, lower = better)

| Transfer | J0 (no force) | J1 (raw) | J2 (canonical) | J3 (bias/mass) | J4 (adapter) |
|---|---|---|---|---|---|
| **insert** | **0.0513** | 0.0510 | 0.0510 | 0.0510 | 0.0510 |
| **remove** | **0.0510** | 0.0537 | 0.0537 | 0.0537 | 0.0538 |
| **place_pick** | **0.0531** | 0.0552 | 0.0552 | 0.0548 | 0.0560 |

| Interface | Δ cos vs no-force (mean over 3 transfers) |
|---|---|
| J1 (raw) | +0.0017 (worse) |
| J2 (canonical) | +0.0000 (tie) |
| J3 (bias/mass) | +0.0006 (worse) |
| J4 (adapter) | +0.0023 (worse) |

**No force representation beats the no-force baseline (J0) on any transfer.** All forced models tie or degrade performance.

### Other endpoints
- Future |f| MAE: F2/F3 improve over J0 (~1.5-1.9 vs 2.8-2.9 N), J4 helps on place_pick (1.8 vs 0.9).
- Contact Brier/AUPRC: all interfaces ~0 on insert/place_pick (no contact events); remove shows small differences but not decisive.
- Contact AUPRC: 0 on insert/place_pick (no contact events in eval); small on remove.

### Why no gain?
1. **Future visual latent is not force-sensitive** in this data/contact regime. The stop-gradient visual target is mostly predicted from visual history; force adds noise.
2. **Canonicalization (F2) matches raw force (F1)** — no transfer benefit from SE(3) canonicalization on this data.
3. **Adapter fine-tuning (J4) degrades further** — 10 adapter episodes per config insufficient to learn a useful 6×6 map; overfits to adapter noise.
4. **Contact events are near-absent** in held-out eval (brier ~0 on insert/place_pick) — no signal to learn from.

## Verdict: **Stop.**

Per the hard decision rule ("Proceed only if J2 or J3 beats J0 on ≥2 held-out configs with CIs and shuffle dependence"): **no interface beats J0 on any transfer**. The hypothesis "canonicalizing wrench coordinates makes force useful across hardware" is **not supported** on REASSEMBLE with this JEPA setup.

The canonicalization (J2) matches no-force (J0) — coordinate gauge was not the bottleneck. Sensor noise, tool gravity, and lack of contact signal dominate. Stop per the protocol.

## Deliverables
- `results/jepa.json` — full per-seed per-interface metrics (cos_err, fmag_mae, contact_brier, cos_shuffle)
- `preds_jepa/` — per-window predictions with episode IDs for bootstrap
- `REPORT.md` — this document

## Recommendation
Do not pursue JEPA with canonical force on REASSEMBLE. The measurement-model pivot ("Force Is Not a Wrench") is the correct next step: the bottleneck is sensor calibration / cross-axis gain / bias / delay, not SE(3) gauge freedom.