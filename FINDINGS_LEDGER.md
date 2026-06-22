# FINDINGS LEDGER — complete, honest inventory

The single master list of **everything** we measured: positives, nulls, the
disproven claims, reproductions, and the places our intuition was confirmed.
Every number is from our own runs (grounded in `complex_mhcot/results/`,
`complex_mhcot/cvnn/RESULTS.md`, `real_mhcot/`, and the memory record).

This is the source of truth for the paper. `POSITIVE_FINDINGS.md` = wins only;
`MHCoT_TECHNICAL_REPORT.md` = negatives only; **this file = all of it.**

**Legend:** ✅ real positive (replicated) · 🔁 reproduction of known work ·
⚪ null result · ❌ DISPROVEN — never claim · 🧭 intuition/insight confirmed ·
📐 methodology finding

---

## A. Real positives (our own runs, multi-seed)

| # | Finding | Numbers | Scope / honesty |
|---|---|---|---|
| ✅ A1 | Contrastive **thought-verifier** beats classifier baseline | AUC **0.819 ± 0.006 vs 0.793**, leak-free split-by-problem, 3 seeds; prototype-cosine alone beats the classifier | GSM8K candidates, DeepSeek-1.5B feats, 605 problems |
| ✅ A2 | **Reranking on contested problems** | picks a correct candidate **0.80 vs 0.53 random** (+0.27), 3 seeds | only on mixed (correct+incorrect) problems |
| ✅ A3 | **Beats self-consistency on final-answer accuracy** *(headline)* | **0.775 ± 0.029 vs 0.716 ± 0.025** (+5.9 pts); recovers ~41% of random→oracle gap; oracle 0.837 | the strong recognized baseline |
| ✅ A4 | **SAE interpretable thought-vocabulary** | recon cosine **0.999**, **3.1%** active, **25** correctness-tracking features (max corr 0.565) | — |
| ✅ A5 | **Complex beats real on phase-bearing data** | test acc **0.625 ± 0.003 vs 0.461 ± 0.025**, smaller overfit gap, 3 seeds | synthetic chirp; **complex foundation (N≈1), NOT multi-helical** |
| ✅ A6 | **Complex's edge is a small-data / data-efficiency effect** *(your "complex better at small scale")* | up to **~2× acc / +0.24 at n ≤ 400**, **converging to parity at scale (n ≥ 800)**, 2 seeds | the win is *low-data regularization*, not a capability win at scale |
| ✅ A7 | **Complex-federated (no pooling)** | complex experts **0.714 ± 0.017 vs 0.463 ± 0.023** (+0.25), M=5 silos, 3 seeds | beats *real-federated only*; joint real (0.901) still wins — honest |
| ✅ A8 | **Federated advantage is an inverted-U** in silo size (L8) | advantage grows then peaks then declines as silos shrink | characterization, not a headline |
| ✅ A9 | **One complex micro-win in search** (solver) | single complex chain beats matched real on **best-first search guidance, 3/3 seeds**, more node-efficient | narrow, metric-dependent; ties/loses on the steadier beam metric |
| ❌ A10–A12 **RETRACTED** | "right answer, wrong path" / leaky-label / non-surface validity (`tbert/`) | **Retracted on validation (2026-06-22).** The arithmetic step-checker's Q2 set was **100% parse artifacts** (chained sums like `168+60+120=348` grabbed as `60+120=348`; algebra RHS like `45+5=2S` grabbed as `45+5=2`). Fixed checker → **Q2 = 0/531 (0.0%)**. The 0.72 "validity-probe" was detecting inline-chained-sum *style*, not reasoning validity (R6 trap again). See C9, F5. |
| ✅ A13 | **"Right answer, wrong path" is REAL and scales with difficulty** (`tbert/processbench.py`, **human labels**) | wrong-path-right (answer ✓ but human-annotated error step) = **GSM8K 4% · MATH 19% · OlympiadBench 32% · OmniMath 52%** of answer-correct solutions | ProcessBench human annotations — no proxy. Explains why our GSM8K arithmetic proxy found ~0: GSM8K genuinely has only 7 such cases |
| ⚪ A-ft | **Fine-tuned text encoders < frozen reasoning-LLM features for verification** (`tbert/sweep.py`) | best fine-tune (bge-base, full, 3ep, class-weighted) AUC **0.709**; MiniLM 0.673; both < frozen-DeepSeek probe **0.819**; freeze curve monotonic (unfreezing helps but isn't enough) | no TSDAE/MLM/contrastive yet; vindicates the DeepSeek choice for this task |

## B. ❌ DISPROVEN — NEVER put on résumé, in the paper (except as "the thing we falsified"), or anywhere

| # | Claim | Why it's dead |
|---|---|---|
| ❌ B1 | "Cut ECE ~70% (0.24 → 0.07)" | **single-seed noise** — ECE swings 0.04–0.24 across seeds; N=2 ≈ real. Falsified by multi-seed replication |
| ❌ B2 | "difficulty-awareness ρ = −0.34" | same seed-0 artifact |
| ❌ B3 | "meta-cognitive calibrated confidence from complex interference" | rests on B1; the calibration signal does not survive replication |

## C. ⚪ Nulls — the multi-helical / ε-helix / co-evolution falsification (the central negative)

Six independent, increasingly-favorable tests; the novelty is unsupported in **all**.

| # | Test | Result |
|---|---|---|
| ⚪ C1 | GSM8K scorer, multi-seed | N=2 calibration ≈ real (the B1 win was noise) |
| ⚪ C2 | Game-24 solver, 3-seed | N=2 (multi-helical) ≤ N=1 on every metric |
| ⚪ C3 | Game-24 **meaning probe** (decisive cheap-gate: separate readout, coverage@K, diverse-real control) | complex_n2 0.230 ≈ n1 0.231 ≈ real2_diverse 0.231 ≈ real2_naive 0.250 — ε-helix adds nothing on its *most favorable* setup |
| ⚪ C4 | **Class-conditioned co-evolving manifolds** (distinct-content chains: correct vs incorrect) | complex_coevolve 0.815 ≈ real_proto 0.822, both lose to plain_mlp 0.847; manifold separation flat (~2.05) — co-evolution inert |
| ⚪ C5 | **Recurrent** temporal co-evolution (the "correct" architecture) | complex_n2_recurrent 0.212 ≈ n1 0.207, both lose to real (0.242 / 0.246) |
| ⚪ C6 | cvnn home-field (L3 single / L4·L4b superposition both readouts / L5 ambiguity) | N=2 ties N=1; L5 looked like a win on seeds 0–1, washed out on 2–3 (**seed-trap, caught**) |
| ⚪ C7 | real_mhcot **R2 multi-chain** ablation | 0.800 vs 0.819 single-branch, 2× params — null even on the working thought-encoder |
| ⚪ C8 | pct_x **coherent interpretation** (crossterm readout) | failed at chance even on its multitone home-field |
| ⚪ C9 | **"Right answer, wrong arithmetic path" in GSM8K** (`tbert/quadrants.py`, fixed checker) | **0/531** checkable candidates — among traces with verifiable explicit arithmetic, right answers come from correct arithmetic. (Scope: can't see the 70% with implicit reasoning, nor *conceptual* wrong-paths; needs an LLM judge / ProcessBench to probe those.) |
| ⚪ C13 | **Denoising-AE validity is mostly surface (norm)** (`real_mhcot/denoising_sae.py`) | denoising sparse AE trained on correct-only thought embeddings: reconstruction-error validity **AUC 0.65 ≈ norm baseline 0.63**, corr(error, norm)=**+0.70**; below the verifier (0.82). The "valid-reasoning manifold" signal is embedding magnitude, not learned validity. |
| ⚪ C12 | **Frozen-embedding step-localization is a weak floor** (`tbert/step_localizer.py`) | ProcessBench-GSM8K, value-function probe: **F1 0.167** (acc_correct 0.50, acc_error 0.10, 3 seeds) — exact first-error localization ~10%. Real PRM/critic (GPU fine-tune or strong prompted LLM) needed to beat this; frozen embeddings can't do step-level localization. |
| ⚪ C11 | **Faithfulness not linearly readable from solution-text embeddings** (`tbert/faithfulness_probe.py`) | within OmniMath (difficulty fixed, 259 vs 241), a bge embedding separates clean vs wrong-path-right at **AUC 0.70 ≈ length-only baseline 0.72** (corr(length,wrong-path)=+0.38) — the signal is surface length, not faithfulness. (Caveat: 512-tok truncation; text-encoder ≠ reasoning-model internals.) |
| ⚪ C10 | **Conceptual "wrong path, right answer" not isolable offline** (`tbert/conceptual.py`) | gold-step-coverage proxy flags 5% low-coverage answer-correct candidates, but spot-check shows they are **valid alternative paths** (per-unit vs total, dollars vs cents, different order), not conceptual errors. Conceptual faithfulness is a semantic judgment — needs an LLM judge / ProcessBench / human labels, which are unavailable offline. (2nd proxy to flag the wrong thing — see F5.) |

**Verdict:** no untested form of the multi-helical mechanism remains; the thesis is
empirically closed. The complex *substrate* is sound (stable, trainable); the
multi-helical *superposition* adds nothing on non-phase data.

## D. 🔁 Reproductions — asymmetric self-play (Game-24, S0–S6, unit-tested)

| # | Component | Result |
|---|---|---|
| 🔁 D1 | Learnable solver | heuristic AUC 0.59 → 0.97; solve-rate 0 → 0.39 |
| 🔁 D2 | Difficulty-controllable proposer | easy 0.33 (100% solved) vs hard 0.88 (18%) |
| 🔁 D3 | Self-play loop | solve-rate **0.10 → 0.52** over rounds |
| 🔁 D4 | Edge-seeking curriculum | difficulty escalates as solver improves |
| 🔁 D5 | Learned verifier (no ground truth) | acc 0.83 / AUC 0.87 vs oracle |
| 🔁 D6 | Collapse + mitigation | naive reward collapses to trivial; edge-seeking solves hard **0.80 vs 0.58** |

*Honest: a faithful reproduction of the SQLM / R-Zero paradigm (established), valued as a tested end-to-end build.*

## E. 🧭 Where our intuition was confirmed (the insights — these are the paper's spine)

| # | Intuition | How it was confirmed |
|---|---|---|
| 🧭 E1 | **Scalar-collapse:** summing chains then `|Ψ|` is an ensemble average → N=2 must tie N=1 | predicted, then confirmed across C1–C3, C6 |
| 🧭 E2 | **Complex helps only with *intrinsic* phase**; on text/thoughts the imaginary part is manufactured noise | confirmed: wins on chirps (A5), ties on text/thoughts (C7) |
| 🧭 E3 | **The complex win is data-efficiency / regularization**, not a capability gain | confirmed: A6 curve converges to parity at scale |
| 🧭 E4 | **Complex-federated helps only when data is genuinely un-poolable** | confirmed: A7 beats real-federated; L6 showed pooling beats partitioning |
| 🧭 E5 | **Single-seed wins are traps** | confirmed twice: B1 (ECE) and the L5 seed-trap, both caught by discipline |
| 🧭 E6 | **Our ECE seed-fragility mirrors the SAE field's feature-instability** | the project's recurring rediscovery; motivates the interpretability caveat (A4 → F-audit) |
| 🧭 E7 | **Encode *thoughts*, not text** — a thought-level encoder is the right object | validated by A1–A3 (the verifier beats self-consistency) |
| 🧭 E8 | **The genuine multi-chain signal is the interference cross-term** `2·Re(ψ⁰·conj ψ¹)`, which `|sum|` discards | explains *why* the multi-chain readout was null (mechanism-level, not just empirical) |

## F. 📐 Methodology findings (the differentiator)

- 📐 F1 **Single-seed evaluation is dangerous** — small-sample ECE noise produced a 70% phantom win. Multi-seed is non-negotiable.
- 📐 F2 **Leak-free split-by-problem** is essential for verifier eval (candidates from one problem must not straddle train/test).
- 📐 F3 **A pre-committed falsification ladder** prevents motivated reasoning (we honored the cheap-gate C3 and stopped inventing tests).
- 📐 F4 **Convergent validity / honest self-audit** — auditing our *own* SAE features revealed a surface-form confound rather than overclaiming.
- 📐 F6 **Reasoning validity/faithfulness is not cheaply readable from representations (convergent negative).**
  FOUR independent attempts collapse to surface features: R6 (SAE correctness features = style),
  C11 (text-embedding faithfulness probe = length), C12 (frozen-embedding step-localization F1 0.17),
  C13 (denoising-AE validity = norm). The signal is not shallowly present in these representations;
  detecting it needs model-internal representations from a strong model, or human/LLM-judge labels
  (A13 shows the *phenomenon* is real on human labels — it's the cheap *detection* that fails).
- 📐 F5 **Spot-check labels before claiming** — the "right answer, wrong path" finding (A10–A12) was retracted *before any write-up* when a 14-example spot-check exposed a 100% false-positive arithmetic checker. Programmatic labels need validation; a clean-looking AUC on a noisy label is worthless. (The 2nd major false positive caught by discipline, after the ECE one — B1.)

## G. Novel vs prior (so the paper claims only what's ours)

- **Genuinely ours:** the rigorous **multi-seed falsification of multi-helical CoT**
  (C1–C8) incl. the **caught false positive** (B1) — a negative-results contribution.
- **Real but prior-art phenomena (we reproduced, don't claim as novel):** complex
  data-efficiency (A5–A6), complex-federated (A7–A8), self-play dynamics (D1–D6).
- **Solid applied work (established area, well-executed):** the thought-verifier /
  reranker / SAE (A1–A4) — recognizable as PRM/verifier territory.
- **The methodology (F1–F4)** is itself a contribution worth foregrounding.

_Last updated: 2026-06-22. All results reproducible via per-module self-tests._
