# CVNN — Complex-Valued Rebuild (Phase 2)

Rebuilding on the lesson from Eilers & Jiang 2023 ("Building Blocks for a
Complex-Valued Transformer Architecture") + our own 6 negatives: we had been
testing complex/multi-helical nets on the **wrong data** (real reasoning text,
no intrinsic complex structure) and for the **wrong benefit** (raw accuracy,
where everything ties). Their result — and the whole CVNN literature — says
complex nets win on **naturally complex-valued data** and on **generalization /
overfitting-robustness**, not accuracy.

So this rebuild inverts every miss:

1. **Right data** — naturally complex-valued signals where *phase carries the
   label* (magnitude alone is non-discriminative). `signals.py`.
2. **Right metric** — generalization gap / overfitting-robustness as PRIMARY
   (train-test gap, robustness in the small-data regime), accuracy secondary.
3. **Right foundation** — a clean complex transformer (their-style complex SDPA
   + complex layernorm) vs a matched real baseline. Reproduce their finding first
   (complex ≈ real on accuracy, better on overfitting) before adding novelty.
4. **THEN the multi-helical layer** — does N=2 co-evolution add robustness *over*
   N=1, on complex data, in the overfitting-prone regime? The experiment we never
   ran in the conditions that could favor it.

## Falsification ladder (cheap gates first)

- **L0 — Data is genuinely phase-carried.** Magnitude-only ≈ chance; phase-aware
  separable. (`signals.py` self-test.) ← we are here
- **L1 — Complex transformer ≈ real on accuracy, BETTER on overfitting-robustness**
  (reproduce the paper on our data). If this fails, the foundation is wrong.
- **L2 — N=1 vs real generalization gap** in the small-data regime (where the
  paper's complex win appeared).
- **L3 — N=2 (multi-helical) vs N=1** on the same data/metric. The real question.

Pre-committed bars at each rung. Each is multi-seed. Fail cheap, learn fast.
The complex-transformer layer (L1/L2) has literature support; the multi-helical
layer (L3) is the open question — now tested on its home field for the first time.
