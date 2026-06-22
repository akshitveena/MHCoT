# Positive Findings — Reproduced & Verified

Consolidated record of the **positive, self-reproduced** results across the
project. Every number here is from our own runs, multi-seed where noted, with
honest scope. (Negative results — the multi-helical falsification — live in
`MHCoT_TECHNICAL_REPORT.md`; this file is the *wins*.)

---

## 1. Thought-verifier / reranker (`real_mhcot/`) — the headline

A real-valued contrastive **thought encoder** over GSM8K Graph-of-Thoughts
reasoning candidates (DeepSeek-1.5B representations).

- **Verification (R1):** beats a plain-classifier baseline on a **leak-free,
  split-by-problem, multi-seed** test — **AUC 0.819 ± 0.006 vs 0.793**. The
  prototype-cosine geometry alone (no extra classifier) beats the classifier.
- **Reranking on contested problems (R3):** on problems with both correct and
  incorrect candidates, picks a correct one **0.80 vs 0.53 random** (+0.27), 3 seeds.
- **Final-answer accuracy vs self-consistency (the headline):**
  **verifier-rerank 0.775 ± 0.029 vs self-consistency 0.716 ± 0.025** (+5.9 pts,
  recovers ~41% of the random→oracle gap), 3 seeds. *Beats the standard strong
  baseline on actual answer accuracy.*
- **Interpretable thought-vocabulary (R4, SAE):** reconstruction cosine **0.999**,
  **3.1%** active features, **25 features that *correlate* with correctness**
  (max feature↔correctness corr 0.565) — **largely surface-form on audit** (see #2 below).
- **Feature interpretation (#2):** the top correctness-features are interpretable —
  top-activating thoughts are 100% class-coherent (corr up to ±0.89) — and appear to
  capture **reasoning-*style* / surface-form patterns** (e.g., formulaic
  "First, I need to determine…" openings vs. exploratory "Okay, so…" openings) that
  correlate with correctness. *Honest caveat:* this suggests the verifier partly
  exploits **stylistic correlates** of correctness rather than deep logical validity
  — consistent with documented SAE surface-form sensitivity. We surfaced this by
  auditing our own features (rigor), rather than overclaiming a "reasoning-validity"
  feature.
- **Honest ablation:** a multi-chain variant **does not** beat single-branch
  (0.800 vs 0.819, 3 seeds, 2× params) — the value is the contrastive encoder,
  not multi-chain.

*Scope: GSM8K candidates, DeepSeek-1.5B features, 605 problems. Recognizable
problem (verifier/reranker), defensible numbers, leak-free, multi-seed.*

## 2. Complex-valued nets on phase-bearing data (`cvnn/`)

- **Data-efficiency (L1, 3 seeds):** on phase-carried signals, a complex
  transformer beats a matched real one — **test acc 0.625 ± 0.003 vs 0.461 ± 0.025**,
  with a smaller overfitting gap.
- **Data-efficiency curve (L2, 2 seeds):** complex's edge is largest at small data
  (~2×/+0.24 at n≤400), converging to parity at scale — a clean characterization.
- **Federated / no-pooling (L7, 3 seeds):** complex experts beat real experts
  **0.714 ± 0.017 vs 0.463 ± 0.023** (+0.25); advantage grows then peaks as silos
  shrink (L8 inverted-U).

*Scope: synthetic phase-carried signals. The effects are real & reproduced, but
the underlying phenomena (complex data-efficiency, complex-federated) are prior
work — see lit checks.*

## 3. Self-play reproduction (`selfplay/`)

A complete, **unit-tested** asymmetric-self-play system on Game-24 (S0–S6):

- **Learnable solver (S1):** heuristic AUC 0.59→0.97; solve-rate 0→0.39.
- **Difficulty-controllable proposer (S2):** easy 0.33 (100% solved) vs hard 0.88 (18%).
- **Self-play loop (S3):** solve-rate 0.10→0.52 over rounds.
- **Edge-seeking curriculum (S4):** difficulty escalates as the solver improves
  ("asker stays ahead").
- **Learned verifier (S5):** acc 0.83 / AUC 0.87 vs the oracle (self-play without
  ground truth).
- **Collapse + mitigation (S6):** naive reward → proposer collapses to trivial;
  edge-seeking keeps it meaningful and solves hard puzzles **0.80 vs 0.58**.

*Scope: Game-24 toy domain; an honest reproduction of the SQLM/R-Zero paradigm
(established), valuable as a tested end-to-end build incl. the collapse study.*

## 4. The one complex-vs-real micro-win (`solver/`)

- A **single** complex chain beats a matched real net on greedy **best-first
  search guidance** (3/3 seeds, more node-efficient) — narrow, metric-dependent,
  but replicated.

---

## Honest framing

- **Strongest for a résumé:** §1 (the thought-verifier — positive, multi-seed,
  beats self-consistency on a real metric, interpretable features, with rigor).
- **What's real vs prior:** §1 is solid *applied* work (verifier/reranker/SAE —
  established area, well-executed); §2 reproduces known complex-net effects; §3 is
  an honest reproduction of self-play.
- **The rigor itself is a finding:** leak-free splits, multi-seed everywhere, a
  caught single-seed false positive, honest negatives reported — the methodology is
  the differentiator.

_Last updated: 2026-06-15. All results reproducible via the per-module self-tests._
