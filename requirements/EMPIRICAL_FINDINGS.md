# MHCoT — Empirical Findings

Living record of what we have **actually measured** (not designed). Updated as
results land. This is the source of truth for results — keep it in git so a
sync/revert can't lose it.

---

## Setup

- **Data:** 605 GSM8K problems processed through Graph-of-Thoughts (DeepSeek-
  R1-Distill-Qwen-1.5B as the local LLM). Each problem → GoT candidates +
  one distilled `final_node`. 596 usable final-node sequences.
- **Model (MHCoTEncoder, ~1M params):** frozen backbone → real down-projection
  1536→128 → ComplexLift → N=2 phase-separated chains → 2 semantic + 2 soliton-
  coupling complex layers → answer-region (last-16 tokens) pooled |Ψ| → scorer.
- **Task:** predict whether a reasoning trace is correct (a SCORER, not a solver).
- **Eval:** val = 120 held-out (seed 0, stratified). ⚠️ single split, single seed.

---

## Finding 0 — Pooling matters (methodological)

Mean-pooling over all ~250 tokens washed out the answer signal (gate AUC 0.59,
candidates 93% cosine-similar). **Answer-region pooling (last-16 tokens)**
recovered it (AUC 0.69, same-vs-different-answer separation 10× larger). All
results below use answer-region pooling.

---

## Finding 1 — Accuracy: complex ≈ real (NO win)

| Model | val AUC |
|---|---|
| complex MHCoT (N=2) | ~0.78–0.80 (peak 0.797) |
| real twin (matched ~1M params) | ~0.78–0.79 (stable 0.784) |
| untrained interference probe | 0.689 |

On **accuracy**, complex and real tie. The real twin is actually *more stable*
and trains 10× faster. **Complex does not beat real on accuracy.** Both overfit
(task loss → 0) on the small data.

---

## Finding 2 — Calibration: complex wins, and it is the MULTI-HELICAL INTERFERENCE ★

The central result. ECE = Expected Calibration Error (lower = better; does an
"80% confident" prediction get it right 80% of the time?).

| Model | ECE | What it rules out |
|---|---|---|
| real (raw readout) | 0.240 | — (badly over-confident) |
| real (`|features|` readout) | 0.232 | **not** the magnitude readout |
| complex **N=1** (single chain) | 0.225 | **not** general complex processing |
| complex N=2, **task-only** (no ε/wave) | 0.070 | **not** the auxiliary losses |
| complex N=2, full losses | 0.072 | — |

**Three confounds eliminated, one cause left standing: the interference between
two phase-separated chains** (the superposition Ψ = ψ⁰ + ψ¹). A real model can't
do it, a magnitude readout can't do it, a single complex chain can't do it.
Only the N=2 multi-helical superposition calibrates.

**Mechanism:** where the two phase-separated chains agree → constructive
interference (large |Ψ|); where they disagree → destructive (small |Ψ|). That
per-dimension interference is an honest confidence signal that exists only with
≥2 chains. **Calibration is literally the interference.**

This is uniquely MHCoT (not DCN-adjacent): a single complex chain (the DCN-style
baseline) does NOT calibrate.

---

## Finding 3 — Option 3 (cross-candidate): real aggregation wins; delineates the niche

Interference between K=3 *genuinely different* GoT candidates (trained, on
answer-region pooled candidate vectors), predicting consensus correctness:

| Model | AUC | ECE |
|---|---|---|
| complex (interference) | 0.850 | 0.086 |
| real (aggregation) | 0.881 | **0.057** |
| discrete agreement (Self-Consistency) | 0.883 | — |

When genuine diversity already exists (3 candidates), **simple real aggregation
already calibrates** (ensemble effect), and complex interference adds nothing /
slightly hurts (phase cancellation discards information the real sum keeps).

**Interpretation:** interference helps when *manufacturing* diversity from ONE
trace (Finding 2 / Option 1); it does not help when *aggregating* diversity that
already exists (Option 3).

---

## ⚠️ REPLICATION VERDICT (supersedes Findings 2 & 4): the calibration finding was NOISE

Findings 2 and 4 below were measured on a SINGLE 120-sample in-distribution
val split (seed 0). Rigorous follow-up shows they do **not** hold:

**Multi-seed reproduction (exact original conditions, best-AUC ckpt):**
```
            seed0   seed1   seed2     mean ± std
real        0.240   0.038   0.192   0.157 ± 0.086
N=1         0.079   0.112   0.182   0.124 ± 0.043
N=2 full    0.075   0.170   0.184   0.143 ± 0.049
```
ECE swings wildly; seed 0 (the original) was a fluke; seed 1 reverses it
(real 0.038 < N=2 0.170). Across seeds N=2 (0.143) ≈ real (0.157), std dwarfs
the gap. **The 0.070-vs-0.240 headline was single-seed noise.**

**Full differentiated model on held-out (train 361 → eval 596):**
```
complex_N2_full: ECE 0.188  AUC 0.700  div 1.312 (chains DID differentiate)
vs real 0.208 | N=1 0.205 | N=2-taskonly 0.197   (all ~0.19-0.21, tied)
```
Even the genuinely multi-helical model (chains differentiated to 1.31) shows
no calibration advantage on held-out data.

**CONCLUSION: MHCoT does not robustly beat a matched real transformer on
GSM8K trace scoring — not in accuracy (always a tie) and not in calibration
(the apparent advantage was single-seed noise). The mechanism works but does
not help. This is an honest NEGATIVE result.** Findings 2 and 4 below are
retained only as a record of what the small-sample run showed; they are
SUPERSEDED by this verdict.

---

## Finding 4 — Difficulty-awareness (confirms calibration from another angle)

Does the model know which problems are hard? Difficulty = # calculation steps
in the GSM8K gold. Spearman(confidence, difficulty), negative = aware:

| Signal | Spearman rho | p-value |
|---|---|---|
| complex confidence | **−0.338** | **<0.001** (significant) |
| real confidence | −0.145 | 0.115 (NOT significant) |
| complex interference | −0.226 | 0.013 (significant) |

(Sanity: accuracy easy 0.85 / med 0.88 / hard 0.65 — proxy valid.)

**MHCoT's confidence drops significantly on harder problems; the real model's
does not (statistically indistinguishable from no relationship).** MHCoT is
difficulty-aware; the over-confident real model is difficulty-blind. NOTE: this
is a *consequence* of calibration (a calibrated model should track difficulty),
so it confirms Finding 2 from another angle rather than being independent.

---

## The contribution these findings support

> **MHCoT provides calibrated reasoning assessment from a SINGLE trace, via
> phase-separated complex interference — a property absent in real transformers
> and single complex chains.**

| Setup | ECE | Inference cost |
|---|---|---|
| 1 trace, real (N=1) | 0.225 | 1× |
| **1 trace, MHCoT** | **0.070** | **1×** |
| 3 traces, real aggregation (≈ Self-Consistency) | 0.057 | 3× |

MHCoT approaches Self-Consistency calibration **from one trace at 1/3 the cost.**
It does **not** beat SC; it matches it cheaply. Contribution = mechanism
(interference→calibration, rigorously isolated) + efficiency (single-trace).
NOT accuracy.

---

## Honest threats to validity (must address before publishing)

1. **Replication.** 120 val, **one seed, one split, one task (GSM8K).** The ECE
   gaps are large (0.070 vs 0.225–0.240 — unlikely pure noise), but we have NOT
   shown the ordering holds across seeds or more data. **This is the #1 gate.**
2. **Scale.** ~1M-param models on 476 training samples; both overfit. More data
   needed for trustworthy numbers and to confirm Finding 2 robustly.
3. **Single task.** Calibration win shown only on GSM8K correctness. Generalization untested.
4. **Accuracy is a tie** — the contribution is explicitly calibration + efficiency.

---

## Capability ladder (what MHCoT can offer)

| # | Capability | Status |
|---|---|---|
| 1 | Calibration (knows when uncertain) | ✅ complex wins (multi-helical), Finding 2 |
| — | Cross-candidate (Option 3) | ✅ tested — real wins; delineates niche |
| 3 | Difficulty / OOD detection | ✅ complex wins (rho −0.34 vs −0.15), Finding 4 |
| 2 | Distractor robustness | ⬜ needs distractor-augmented data (generation phase) |
| 4 | Ambiguity handling | ⬜ needs ambiguous data (generation phase) |
| 5 | Multi-path planning ("higher-dimensional") | ⬜ needs SOLVER expansion (paper-2 scale) |

**Status:** all scorer-compatible, existing-data tests are DONE (#1, #3,
Option 3). Remaining capabilities (#2 distractor, #4 ambiguity) need NEW data;
#5 needs a solver. → the data-generation phase serves replication + #2 + #4.

**Scorer vs solver:** current MHCoT is a *scorer*. Capabilities 2–3 are testable
now. Capability 5 (the "higher-dimensional" frontier — where N chains ARE the
competing solution paths and the ε-helix prevents premature commitment) requires
turning MHCoT into a *solver* — a larger, paper-2-scale build.

---

## Next steps (priority order)

1. **Replicate Finding 2 on more data + multiple seeds** (the #1 rigor gate) —
   scale to ~2,500 GoT problems on Colab A100, re-run real / N=1 / N=2 ECE.
2. **Difficulty/OOD detection** — does interference-confidence track problem hardness?
3. **Distractor robustness** — does interference flag misleading context?
4. (Paper-2) **Solver expansion** — MHCoT as a multi-path reasoner.
