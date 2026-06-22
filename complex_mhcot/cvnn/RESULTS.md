# CVNN Rebuild — Results (Phase 2)

**Thesis (earned, defensible): complex-valued models are markedly more
data-efficient than matched real transformers on phase-carried signals, which
makes them the better choice for data-scarce and federated/on-device learning —
the advantage is largest precisely in the realistic moderate-silo regime.**

All experiments use a synthetic, naturally-complex dataset (chirp signals where
the label lives in phase; magnitude is class-independent — `signals.py`). Models
are param-matched (~68–77K). CPU, multi-seed where noted.

## The falsification ladder

| Rung | Question | Result |
|---|---|---|
| **L0** | Is the data genuinely phase-carried? | ✅ magnitude ≈ chance (0.12), phase separable (0.97) |
| **L1** | complex vs real on phase data (3-seed) | ✅ **complex 0.625±0.003 vs real 0.461±0.025**, smaller overfit gap |
| **L2** | data-efficiency curve (2-seed) | ✅ complex win is a **low-data** edge: +0.24 at n≤400, **parity at n≥800** |
| **L3** | multi-helical N=2 vs N=1 (single-answer) | ❌ tie |
| **L4 / L4b** | N=2 vs N=1 (superposition; sum & per-chain readouts) | ❌ tie |
| **L5** | N=2 vs N=1 (ambiguity, 4-seed) | ❌ tie (seeds 0–1 looked like a win; seeds 2–3 washed it out) |
| **L6** | shard-ensemble vs joint training (poolable data) | ❌ chopping loses (0.70 < joint 0.90); no info added by splitting |
| **L7** | federated, no pooling (3-seed) | ✅ **complex 0.714±0.017 vs real 0.463±0.023 (Δ+0.25)** |
| **L8** | federated advantage vs silo count (2-seed) | ✅ inverted-U: complex wins at every M; **peak Δ+0.29 at M=8 (100/silo)** |

## What is established (honest)

- **Complex ≫ real on phase-carried signals, in the data-limited regime.** Robust,
  large effect, near-zero variance, clear mechanism (native phase processing).
- **It is a data-efficiency / regularization advantage, not a capability win at
  scale** — real catches up given enough data (L2). Stated honestly, this is a
  *low-data* and *federated* result, not "complex beats everything."
- **Federated win is real and large** where data can't be pooled (L7), and it
  *grows* as silos shrink until pathological starvation (L8 inverted-U, peak at
  ~100 examples/silo — the realistic federated regime).
- **The multi-helical / ε-helix mechanism (the original MHCoT novelty) is null** —
  N=2 ties N=1 across single-answer, superposition (both readouts), and ambiguity
  (the seed-0/1 ambiguity lead did not survive 4 seeds). The wins are about the
  *complex foundation*, not multi-chain interference.

## Honest boundaries / threats to validity

- Synthetic chirp data; real federated signal benchmarks (audio/RF) untested
  (mechanism is literature-consistent — Eilers & Jiang 2023).
- Federated aggregation here is one-shot prediction-averaging (a valid federated
  inference method), not multi-round FedAvg — a natural follow-up.
- 2–3 seeds per result; effects are large relative to variance, but more seeds
  would tighten the curves.

## Reproduce

```bash
python cvnn/signals.py                                   # L0
python cvnn/exp_l1.py --seeds 0 1 2 --device cpu         # L1
python cvnn/exp_l2.py --sizes 100 200 400 800 1600 --seeds 0   # L2 (chunk per seed)
python cvnn/exp_l7.py --N 800 --M 5 --seeds 0 1 2 --device cpu # L7
python cvnn/exp_l8.py --N 800 --M_list 2 4 8 16 --seeds 0 --device cpu  # L8 (chunk per seed)
```

_Last updated: 2026-06-15._
