# Multi-Helical Chain-of-Thought (MHCoT): A Falsification Study

**A complex-valued, multi-chain reasoning architecture — designed, implemented,
and rigorously tested. This report documents the full investigation, including a
self-caught false positive, four independent negative results for the central
hypothesis, one narrow replicated positive, and the methodology that holds it all
together.**

Author: Akshit Veena · Date: 2026-06-14 · Status: **honest negative result (core
hypothesis), with reusable artifacts and methodology**

---

## 0. Abstract

MHCoT proposes that reasoning is best represented not as real-valued vectors but
as **high-dimensional complex signals**, carried by **multiple co-evolving chains**
that hold competing hypotheses without collapsing prematurely (an "ε-helix"
anti-collapse dynamic), ultimately to be trained inside a reinforcement-learning
environment. We implemented the full complex-valued architecture (complex
transformer primitives, cross-chain soliton coupling, ε-helix phase loss) and
tested its central claim — that the **multi-helical superposition** outperforms a
matched real-valued network — across four increasingly favorable settings:
GSM8K trace scoring (calibration & accuracy), cross-candidate aggregation,
Game-24 search-heuristic guidance, and a Game-24 multi-answer coverage probe
explicitly designed to remove all known confounds.

**Result: the multi-helical mechanism is not supported.** An initially striking
calibration result (≈70% ECE reduction) was shown by multi-seed replication to be
single-seed noise. Across all subsequent tests the complex multi-chain model
**ties** matched real baselines. We do report one genuine, replicated, but narrow
positive: a *single* complex chain improves greedy best-first search guidance over
a real baseline (3/3 seeds). The complex substrate itself is sound (a verified
stable, trainable, signal-rich dynamical system). The lasting contributions are
(a) a complete, reusable complex-architecture stack on Apple Silicon, (b) a
falsification methodology that caught its own false positive, and (c) a precise
conceptual framework for "thoughts as signals."

---

## 1. Motivation and vision

The guiding question: **what is the degree of a thought — how many dimensions does
a thought have?** A thought is not a scalar. If a thought is high-dimensional, it
can be viewed as a **signal**, with signal properties: amplitude, phase, frequency,
and interference. From this:

- **Reasoning** = a *chain* of such signals unfolding in sequence.
- **Multiple chains** = a superposition of competing reasoning trajectories.
- **Intelligence** emerges when chains **co-evolve**, influencing one another as
  they generate.
- The system comes **alive inside a reinforcement-learning environment (RLE)** —
  shaped by consequence, not labels.

MHCoT is the attempt to operationalize this. The novel claim under test is
specifically the **multi-helical** part: that ≥2 phase-separated, co-evolving
complex chains (held apart by an ε-helix dynamic) produce something a single chain
or a real network cannot — via the **interference** between them.

---

## 2. Architecture

All modules are **real-decomposed** (complex math expressed as real operations) so
they run on CPU / CUDA / Apple-Silicon MPS, and each has a unit test.

### 2.1 Complex primitives (`main/complex_ops.py`)
- **ComplexLinear** — DCN-style real-parametrized complex linear layer
  (`y_re = W_re x_re − W_im x_im`, `y_im = W_re x_im + W_im x_re`).
- **ComplexLift** — bridge ℝ→ℂ. Init `W_real = I` (Re(z) = input preserves the
  source), `W_imag` = small random (non-trivial phase from step 0).
- **ComplexPositionalEncoding** — multiplicative (rotation) positional encoding;
  amplitude preserved, phase carries position.
- **modReLU** — complex activation gating magnitude, preserving phase.
- **MagnitudeLN** — layer norm on |z|, phase preserved.
- **ComplexAttention** — multi-head attention with magnitude-softmax
  (`attn = softmax(|Q·Kᴴ|/√d)`, V stays complex).

### 2.2 Multi-helical dynamics (`main/soliton.py`)
- **SolitonCell** — cross-chain coupling: self-focusing `−α|ψ|²ψ` (keep identity)
  + consensus coupling `β(Ψ−ψ⁽ⁱ⁾)·exp(−gap/ε)` gated by the per-dimension phase
  gap. Large gap → chains stay distinct; small gap → they exchange information.
- **ε-helix loss** — `L_ε = mean(relu(ε_min − gap)²)`; penalizes chains for getting
  closer than ε_min in phase (ε_min ≈ 1.15, matching an empirically-measured scale).
- **Numerical fixes (important):** `atan2` phase gradients are NaN at |ψ|≈0. Fixes:
  `_safe_phase` (nudge degenerate positions), `detach_gate` (stop-grad the coupling
  gate), staged ε-helix loss (engage after a warmup), and non-finite-gradient
  guards. Without these, the multi-chain model silently fails to train.

### 2.3 Models built on this stack
- **MHCoTEncoder** (`main/model.py`) — scorer: frozen-LLM hidden states →
  down-projection 1536→128 → ComplexLift → N phase-separated chains → semantic
  (attention) + soliton-coupling layers → answer-region-pooled |Ψ| → scalar score.
  ~1M trainable params.
- **MHCoTSolver** (`solver/model.py`) — value heuristic for search: encodes a
  symbolic state → complex chains → interference → V(state).
- **CoEvolveCell** (`solver/coevolve.py`) — recurrent co-evolution step (the
  restart's heart); carries chain state across steps.
- **ChainModel / RealTwoHead** (`solver/stage1_meaning.py`) — per-chain separate
  readout (no summation) + matched real two-head baselines.

---

## 3. Paradigm evolution (and why it matters)

The investigation moved through three paradigms. Recognizing the mismatch between
them and the vision is itself a key finding (§9.2):

| Phase | Paradigm | What MHCoT was | Verdict |
|---|---|---|---|
| 1 | Supervised SCORER | reads a finished reasoning trace, judges it | ties real |
| 2 | Supervised SOLVER | value heuristic guiding search | ties real (N1 narrow win) |
| 3 | "Meaning probe" | per-chain multi-answer readout (no scalar collapse) | ties real |
| (4) | Generative RL agent | *the vision* — never reached (gate failed) | not justified |

Crucially, Phases 1–3 are all **supervised** and (until Phase 3) collapse the
high-dimensional thought-signal to a **scalar** trained on **one bit** — the
structural opposite of the generative, RL-driven vision. Phase 3 was built
specifically to remove the scalar-collapse confound.

---

## 4. Experimental setup

- **Backbone:** DeepSeek-R1-Distill-Qwen-1.5B, **frozen** (hidden size 1536). Only
  the ~1M-param complex head trains; the LLM never enters the loop (precomputed
  hidden-state cache).
- **Preprocessing:** Besta Graph-of-Thoughts (GoT) over GSM8K with the local LLM →
  candidates + a distilled `final_node` per problem.
- **Data:** GSM8K test split (605 problems, 596 usable sequences), train split
  (361). Game-24: exact programmatic puzzles + labels (no LLM).
- **Baselines:** parameter-matched real Transformer encoders; complex N=1 (single
  chain, the DCN-style ablation); for the meaning probe, real-2-head naive and
  real-2-head with an explicit decorrelation (diversity) loss.
- **Metrics:** AUC (accuracy), ECE (calibration), Spearman ρ (difficulty),
  solve-rate / nodes-to-solution (search), coverage@K (multi-answer).
- **Discipline:** multi-seed (≥3), pre-committed bars, confound ablations.

---

## 5. Experiments and results

### 5.1 Methodological: pooling (Finding 0)
Mean-pooling over all ~250 tokens washed out the answer signal (gate AUC 0.59).
**Answer-region pooling (last-16 tokens)** recovered it (AUC 0.69; same-vs-different
answer separation 10× larger). All scorer results use answer-region pooling.

### 5.2 Accuracy: complex ≈ real (tie)
| Model | val AUC |
|---|---|
| complex MHCoT (N=2) | ~0.78–0.80 (peak 0.797) |
| real twin (~1M params) | ~0.78–0.79 (0.784) |
| untrained interference probe | 0.689 |

Complex does **not** beat real on accuracy. Real is more stable and trains ~10×
faster. Both overfit on the small data.

### 5.3 Calibration: the striking result — and the false positive
Initial single-seed (120-val) result:
| Model | ECE | Confound ruled out |
|---|---|---|
| real (raw) | 0.240 | — |
| real (\|features\|) | 0.232 | not the magnitude readout |
| complex N=1 | 0.225 | not general complex processing |
| complex N=2 (task-only) | **0.070** | not the auxiliary losses |
| complex N=2 (full) | 0.072 | — |

This looked like a real, confound-isolated win (≈70% ECE reduction), attributable
to interference between two phase-separated chains. **It was reported, then
falsified by our own replication (§5.4).**

### 5.4 Replication: the calibration win was single-seed noise ★
Multi-seed reproduction under exact original conditions (best-AUC checkpoint), ECE:
```
            seed0   seed1   seed2     mean ± std
real        0.240   0.038   0.192   0.157 ± 0.086
N=1         0.079   0.112   0.182   0.124 ± 0.043
N=2 full    0.075   0.170   0.184   0.143 ± 0.049
```
Seed 0 (the original) was a fluke; seed 1 **reverses** it (real 0.038 ≪ N=2 0.170).
Across seeds N=2 (0.143) ≈ real (0.157); the std dwarfs the gap. On a true held-out
set (train 361 → eval 596), even the fully-differentiated model (chain divergence
1.31) gave ECE 0.188 vs real 0.208 — a tie. **Conclusion: the calibration finding
does not replicate.** ECE on ~120 samples is high-variance; 0.070 fell inside the
noise band.

### 5.5 Difficulty-awareness (also a consequence of §5.4)
Single-seed: complex confidence Spearman ρ = −0.338 (p<0.001) vs real −0.145
(p=0.115). Looked like MHCoT tracks problem hardness. But calibration *implies*
difficulty-tracking, so this is the **same seed-0 artifact** from another angle —
superseded by §5.4. Not an independent finding.

### 5.6 Cross-candidate aggregation (Option 3)
Interference between K=3 genuinely different GoT candidates, predicting consensus:
| Model | AUC | ECE |
|---|---|---|
| complex (interference) | 0.850 | 0.086 |
| real (aggregation) | 0.881 | **0.057** |
| discrete agreement (Self-Consistency) | 0.883 | — |
When real diversity already exists, simple real aggregation calibrates (ensemble
effect); complex interference adds nothing / slightly hurts. **Real wins.**

### 5.7 Solver: Game-24 search-heuristic guidance (3 seeds)
MHCoT rebuilt as a value heuristic on a genuinely multi-path task (exact labels,
no LLM). Param-matched (real 76K, complex 85K). Held-out, mean ± std:
| Model | beam solve | best-first solve | nodes-to-solve |
|---|---|---|---|
| real | 0.488 ± 0.098 | 0.111 ± 0.065 | 41.3 ± 3.4 |
| **N1 (single complex)** | 0.455 ± 0.177 | **0.339 ± 0.135** | **37.2 ± 2.3** |
| N2 (multi-helical) | 0.400 ± 0.083 | 0.119 ± 0.104 | 42.5 ± 5.1 |

- **Multi-helical (N2 vs N1): falsified.** N2 ≤ N1 on every metric, though chains
  did differentiate (div 1.1–1.2). The mechanism engages; it doesn't help.
- **Complex vs real (N1 vs real): the one genuine positive.** N1 beats real on
  best-first in **all 3 seeds** (per-seed: 0.347/0.041, 0.170/0.094, 0.500/0.196)
  and is more node-efficient. Narrow — it does **not** hold on the steadier beam
  metric (real 0.488 ≥ N1 0.455). Honest framing: *a complex single-chain heuristic
  improves greedy best-first search guidance; the advantage is metric-dependent.*

### 5.8 Stage 0 — dynamical probe of co-evolution (taskless)
Before any RL, verify the recurrent co-evolving substrate is healthy:
- **Stability:** amplitude bounded (1.55 → ~6.3 plateau; residual stream runs hot —
  fixed by output normalization downstream).
- **Non-collapse:** under ε-helix, divergence maintained/grew (0.44 → 0.60 trained;
  0.50 → 1.24 free rollout — chains converge early, then specialize).
- **Trainability:** finite gradients through a 20-step recurrent rollout.
- **Richness:** per-dim phase std 1.22 over the trajectory (not frozen).
**Verdict: the substrate is alive — stable, trainable, non-collapsing, signal-rich.
But this proves the medium is healthy, not that the signals are *meaningful*.**

### 5.9 Stage 1 — the meaning probe (the decisive test, 3 seeds) ★
Built to remove the scalar-collapse confound: a **multi-answer** task (predict the
full set of good first-moves), chains read out **separately** (never summed),
metric = coverage@K of the good-move set via a round-robin union of per-chain
rankings. Strong control = real-2-head with an explicit diversity loss.
(Task is genuinely multi-answer: avg 4.6 good moves/puzzle, 33/40 with ≥2.)

| Model | coverage@3 (mean ± std) |
|---|---|
| complex_n2 (ε-helix) | 0.230 ± 0.035 |
| complex_n1 (single chain) | 0.231 ± 0.031 |
| real2_naive (2 heads) | 0.250 ± 0.027 |
| real2_diverse (2 heads + decorrelation) | 0.231 ± 0.029 |

**All four tie within noise; N2 is if anything below the naive real baseline.** On
the setup most favorable to the thesis — no scalar collapse, separate readout,
genuinely multi-answer, ε-helix engaged — the multi-helical mechanism shows no
advantage. This was the **pre-committed cheap gate**: failing it means the
generative-RL stage (Phase 4) is *not* justified, since chains that cannot carry
distinct meaning under direct supervision will not acquire it from RL's weaker
signal.

### 5.10 — Class-conditioned co-evolving manifolds (the last principled test, 3 seeds) ★
§5.9 left one escape hatch: perhaps the chains never carried *genuinely distinct
content*. This experiment removes that hatch. Instead of phase-rotated copies, the
two chains are built from real, different content: chain_pos = the sequence of
CORRECT candidate signals, chain_neg = the INCORRECT ones (cached pooled vectors,
1318 correct / 497 incorrect). Each class's representation space is encoded
(complex attention + pool), the two manifolds are CO-EVOLVED (soliton), and a
held-out trace is classified by which manifold it constructively interferes with.

**Held-out AUC (3-seed mean ± std):**
| Model | AUC | manifold sep (before → after co-evolve) |
|---|---|---|
| complex_coevolve (the idea) | 0.815 ± 0.008 | 2.06 → 2.05 |
| real_proto (real control) | 0.822 ± 0.001 | — |
| plain_mlp (floor) | **0.847 ± 0.006** | — |

- complex_coevolve **ties** real_proto (a hair below).
- Both manifold methods **lose to a plain MLP** — the elaborate machinery subtracts.
- **Co-evolution is inert:** manifold separation does not grow on any seed (tiny
  decrease, all three). The soliton coupling extracts nothing even from genuinely
  distinct chains.

**This closes the last door:** the failure was not "the chains lacked distinct
content" — given real distinct content, co-evolution still adds nothing. Fifth
independent negative, and the most conclusive about the co-evolution dynamic itself.

### 5.11 — The recurrent co-evolution architecture, finally on a task (3 seeds) ★
A gap audit revealed that every prior N1-vs-N2 task comparison used SHALLOW
co-evolution (1–2 soliton coupling layers over depth). The architecture theorized
as *correct* — the recurrent CoEvolveCell (multi-step TEMPORAL co-evolution, state
carried across steps, output-norm fix; verified healthy but taskless in §5.8) —
had never been put on a task. This is that test: multi-answer coverage with the
recurrent cell (6 steps) as the N=2 core, vs matched recurrent N=1 and real
2-head baselines.

**Coverage@3 (3-seed mean ± std):**
| Model | coverage@3 |
|---|---|
| complex_n2_recurrent (the "correct" architecture) | 0.212 ± 0.039 |
| complex_n1_recurrent | 0.207 ± 0.024 |
| real2_recurrent | 0.242 ± 0.028 |
| real2_diverse | 0.246 ± 0.016 |

The proper recurrent co-evolution **ties its single-chain ablation and loses to
both real baselines** — not even the best complex variant. **Sixth independent
negative, and the decisive one: it removes the last untested form of the
architecture.** There is no remaining "but we didn't test the real co-evolution"
gap. The multi-helical / co-evolution thesis is empirically closed in full.

---

## 6. Summary of the verdict

| Claim | Verdict | Evidence |
|---|---|---|
| Accuracy: complex > real | ❌ tie | §5.2 |
| Calibration: N=2 interference > real | ❌ single-seed noise | §5.3–5.4 |
| Difficulty-awareness | ❌ same artifact | §5.5 |
| Cross-candidate interference > real | ❌ real wins | §5.6 |
| Multi-helical (N2 > N1) helps search | ❌ falsified, 3 seeds | §5.7 |
| Multi-helical carries distinct meaning | ❌ falsified, 3 seeds | §5.9 |
| Co-evolving genuinely-distinct manifolds adds value | ❌ falsified, 3 seeds (inert) | §5.10 |
| Recurrent (proper) co-evolution helps | ❌ falsified, 3 seeds (ties N1, loses to real) | §5.11 |
| **Complex N1 > real on best-first search** | ✅ **replicated, narrow** | §5.7 |
| Complex substrate is stable/trainable/rich | ✅ verified | §5.8 |
| Multi-helical mechanism functions (chains differentiate) | ✅ (but no benefit) | §5.7–5.8 |

**The central MHCoT novelty — that the multi-helical complex superposition beats a
matched real network — is not supported across four independent, increasingly
favorable tests.** The complex models are *competitive* (broadly tie, never
catastrophically worse), with one narrow replicated positive (N1 best-first).

---

## 7. What can be legitimately claimed

**Honest, defensible:**
> Designed and implemented a complete complex-valued ("multi-helical") transformer
> with novel cross-chain soliton coupling and ε-helix phase dynamics, verified on
> Apple Silicon. Ran a rigorous multi-seed falsification study across four tasks
> (calibration, search guidance, multi-answer coverage); found the architecture
> competitive with matched real transformers and a replicated complex-representation
> advantage in greedy-search heuristics, while honestly establishing — and catching
> a single-seed false positive en route — that the multi-helical superposition
> yields no robust gain.

**Must NOT be claimed (false / fabrication):** "70% ECE reduction" or "improves
calibration" (the false positive, §5.4); "difficulty-aware" (§5.5); any "under
review" / publication status that does not exist.

---

## 8. Methodology — the durable contribution

1. **Falsification-first.** Pre-committed bars before each experiment; a hypothesis
   must clear them or be reported as failed.
2. **Multi-seed replication as a gate.** The single-seed calibration win was caught
   *by us* via 3-seed reproduction — the most important methodological act here.
3. **Confound ablations.** Magnitude readout, single complex chain (N=1), auxiliary
   losses, and a param-matched real twin, each isolating a possible explanation.
4. **Falsification ladder with cheap gates first** (Stage 0 dynamical → Stage 1
   meaning → Stage 2 RL): fail cheap, early, and diagnosably; never pay for RL
   before the cheap supervised gate clears.
5. **Strong controls.** Not just "real" but "real with the diversity mechanism the
   complex model is supposed to provide for free" (real2_diverse).

---

## 9. Measurement-trap and conceptual findings

### 9.1 Measurement traps (transferable lessons)
- **Small-sample ECE is dangerously noisy** — ~120 samples gave ECE swings of
  0.04–0.24 across seeds. Never report single-seed ECE as a result.
- **Scalar-collapse hides multiplicity** — summing chains (Ψ = ψ⁰+ψ¹) and reading
  |Ψ| is mathematically an ensemble average; it cannot express a multi-hypothesis
  advantage by construction.
- **One-bit supervision under-determines rich structure** — a binary target gives
  the complex/phase machinery neither pressure nor channel to develop meaning; a
  real scalar already saturates the task.

### 9.2 The paradigm diagnosis
Every supervised experiment tested the **structural opposite** of the vision:
static (not generative), scalar (not signal), supervised one-bit (not RL reward).
The negatives are therefore *true about that setup*. Phase 3 (the meaning probe)
removed the scalar-collapse confound specifically — and the thesis still failed,
which is what makes the negative decisive rather than merely a setup artifact.

### 9.3 Conceptual framework (clarified, even though the mechanism failed)
- A thought as a **high-dimensional complex signal**; a chain as a signal; N chains
  as a superposition.
- **Medium vs content:** the model can generate signals with no environment (Stage
  0, "empty thoughts"); the environment supplies the content that makes a signal
  *mean* something (Stage 1).
- The model perceives **states (number configurations), not operations** — operations
  live in the environment's transition function; operational understanding is
  inferred, not perceived.

---

## 10. Limitations and threats to validity

- **Scale.** ~1M-param heads on hundreds of training samples; overfitting present.
  Larger data could change magnitudes (but the multi-seed *ordering* is the claim,
  and it is consistently a tie).
- **Tasks.** GSM8K trace scoring and Game-24. Three different readouts tested
  (calibration, search, coverage); generality beyond these is untested — but four
  consistent negatives make a hidden win increasingly unlikely.
- **The RL paradigm (Phase 4) was never run.** By pre-committed discipline, the
  failed cheap gate (§5.9) makes it unjustified. This is a deliberate *stopping*
  decision, not an omission: the vision's full form remains formally untested, but
  the prerequisite it depends on (chains carrying distinct meaning) failed.
- **Backbone fixed at 1.5B.** A stronger model could change the solver/RL ceiling,
  but would not rescue the multi-helical-vs-single-chain comparison, which is
  internal to the head.

---

## 11. Reusable artifacts (independent of the thesis)

- A complete, unit-tested **complex-valued transformer stack** on MPS
  (`main/complex_ops.py`).
- A working **cross-chain soliton coupling + ε-helix** implementation with the
  `atan2` stability fixes (`main/soliton.py`).
- A verified **recurrent co-evolving complex cell** (`solver/coevolve.py`).
- An **exact Game-24 world + search harness + multi-answer task**
  (`solver/game24.py`, `solver/search.py`, `solver/multianswer.py`).
- The full **experiment/falsification harness** (multi-seed runners, ablations).

---

## 12. Honest future directions (clearly relabeled)

- **Accept and publish the negative + methodology.** The rigorous falsification of
  an attractive hypothesis, with a self-caught false positive, is a respectable and
  useful contribution.
- **Investigate the N1 complex-search edge (§5.7)** as a *small, focused* question:
  do complex-valued heuristics improve greedy search in general? This is real and
  modest — and is about complex representations, **not** the multi-helical thesis.
- **A generative RL reasoning agent** remains a legitimate project — but it would be
  "an RL reasoning agent," not "MHCoT works," since the multi-helical claim has
  failed four times. Pursuing it should not be framed as rescuing MHCoT.

---

## 13. Reproduction

```bash
# Scorer (GSM8K) — encode both splits, then replicate
python main/encoder.py --seq --cache gsm8k_train.jsonl
python main/encoder.py --seq --cache gsm8k_test.jsonl
python experiments/exp_replication.py            # held-out table
python experiments/exp_reproduce_original.py     # multi-seed (exposes the noise)

# Solver (Game-24) — multi-seed
python solver/multiseed.py --seeds 0 1 2

# Stage 0 — dynamical probe
python solver/coevolve.py

# Stage 1 — the decisive meaning probe
python solver/stage1_meaning.py --seeds 0 1 2 --steps 2000 --K 3
```

Supporting records: `RESULTS.md` (summary), `requirements/EMPIRICAL_FINDINGS.md`
(living log), `requirements/MHCoT_paper1_spec.md` (design).

---

## 14. Closing

MHCoT asked a real question — *what is the shape of a thought?* — and answered it
honestly. The specific answer it proposed (multi-helical complex interference)
turned out not to beat a plain real network, across four chances and one tempting
false alarm that we caught ourselves. What remains is real: a sound complex
substrate, a narrow replicated finding, reusable infrastructure, a sharp conceptual
frame, and — above all — a way of testing one's own ideas that refuses to let a
beautiful hypothesis survive on noise. That last part is the science.
