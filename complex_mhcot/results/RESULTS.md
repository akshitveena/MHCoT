# MHCoT — Results Summary (Paper-1 scope)

**Status: honest negative result.** This document is the consolidated, final
record of what was *measured* (not designed) in the Paper-1 effort: MHCoT as a
**scorer** (judging whether a reasoning trace is correct) on GSM8K, on top of a
frozen DeepSeek-R1-Distill-Qwen-1.5B backbone + Graph-of-Thoughts preprocessing.

For the blow-by-blow living record see [`requirements/EMPIRICAL_FINDINGS.md`](requirements/EMPIRICAL_FINDINGS.md).
This file is the conclusion.

---

## 1. What we built (and it works mechanically)

- **Pipeline:** GSM8K problem → Graph-of-Thoughts (local 1.5B) → candidates + one
  distilled `final_node` → frozen backbone per-token hidden states (ℝ¹⁵³⁶) →
  `ComplexLift` (ℝ→ℂ) → **N=2 phase-separated ε-helix chains** → 2 semantic + 2
  soliton-coupling complex layers → answer-region (last-16 token) pooled |Ψ| →
  scorer head. ~1M trainable params; frozen 1.5B never enters the loop.
- **The complex machinery is real and verified:** `ComplexLinear`, `ComplexLift`,
  `ComplexPositionalEncoding`, `modReLU`, `MagnitudeLN`, magnitude-softmax
  `ComplexAttention`, `SolitonCell` cross-chain coupling, the ε-helix phase loss
  and the interference wave loss. All MPS-safe, all self-tested.
- **The multi-helical mechanism engages:** under the full staged losses
  (task → +L_ε → +L_wave) the two chains genuinely differentiate
  (chain divergence ≈ 1.31). This is not a stub — the architecture does what it
  was designed to do.

**Conclusion of §1:** the model is authentic. Whatever the result, it is a result
*about MHCoT*, not about a placeholder.

---

## 2. The headline finding — and why it did NOT survive

### 2a. What the first run showed (single seed, 120-sample val)
On one in-distribution split, the N=2 multi-helical model looked dramatically
better calibrated than a matched real transformer:

| Model | ECE (single-seed) |
|---|---|
| real (raw) | 0.240 |
| complex N=1 (single chain) | 0.225 |
| **complex N=2 (multi-helical)** | **0.070** |

Three confounds were ruled out (magnitude readout, general complex processing,
the auxiliary losses), leaving "interference between two phase-separated chains"
as the apparent cause. This was the exciting result.

### 2b. Replication killed it
**Multi-seed reproduction**, exact original conditions, best-AUC checkpoint:

```
            seed0   seed1   seed2     mean ± std
real        0.240   0.038   0.192   0.157 ± 0.086
N=1         0.079   0.112   0.182   0.124 ± 0.043
N=2 full    0.075   0.170   0.184   0.143 ± 0.049
```

- Seed 0 (the original) was a fluke. Seed 1 **reverses** it (real 0.038 ≪ N=2 0.170).
- Across seeds, N=2 (0.143) ≈ real (0.157); the std dwarfs the gap.
- ECE on 120 samples is inherently high-variance; 0.070 fell inside the noise.

**Full differentiated model on a true held-out set** (train on 361 train-split
traces → eval on 596 test-split traces):

```
complex_N2_full: ECE 0.188  AUC 0.700  div 1.312 (chains DID differentiate)
real 0.208 | N=1 0.205 | N=2-taskonly 0.197       (all ~0.19–0.21, a tie)
```

Even with the chains fully differentiated, **no calibration advantage** on
held-out data.

---

## 3. Final verdict

> **MHCoT does not robustly beat a matched real transformer on GSM8K trace
> scoring — not in accuracy (always a tie) and not in calibration (the apparent
> advantage was single-seed noise). The mechanism works; on this task it does
> not help.**

| Claim | Verdict |
|---|---|
| Accuracy: complex > real | ❌ tie (~0.78–0.80 AUC both; real more stable, trains faster) |
| Calibration: N=2 interference > real | ❌ single-seed noise, does not replicate |
| Difficulty-awareness (ρ=−0.34) | ❌ same seed-0 artifact (consequence of 2b) |
| Cross-candidate (Option 3) | ❌ real aggregation wins (0.057 vs 0.086 ECE) |
| The complex/ε-helix/soliton machinery functions | ✅ verified, chains differentiate |

---

## 4. Why this happened (the honest diagnosis)

The negative is **not** a data bug or a measurement error:
- train (361) and test (605) are disjoint but same-distribution GSM8K
  (train mildly easier: 0.84 vs 0.77 solve rate) — and the *in-distribution*
  multi-seed test controls for this and still shows noise.
- ECE and AUC are the correct metrics; they were computed correctly.

The real reason is **task fit**: GSM8K is single-answer arithmetic. A scorer
only has to read one finished trace and judge it — a problem a real transformer
already solves. There is **no multi-hypothesis structure for the interference to
exploit**. MHCoT's claimed strength (holding several competing reasoning paths
without premature collapse) is never stressed by this task. We tested the
architecture where its advantage cannot appear.

**The "higher-capability / multi-path" hypothesis is therefore untested — not
disproven.** Testing it requires turning MHCoT from a scorer into a *solver* on
a genuinely multi-path task (see [`requirements/MHCoT_paper1_spec.md`](requirements/MHCoT_paper1_spec.md)
§ solver and `RUN_TRACKS.md` Track B).

---

## 5. What is publishable / claimable from this

- ✅ A rigorously-isolated **negative result + methodology**: complex
  multi-helical interference, fully implemented and differentiated, does **not**
  improve calibration or accuracy on single-answer math trace scoring; an
  apparent win was identified as single-seed noise via multi-seed + held-out
  replication. (Honest, and the replication discipline is itself a contribution.)
- ❌ Do **not** claim "70% ECE reduction", "calibrated reasoning from one trace",
  or difficulty-awareness — all trace to the seed-0 artifact. These must be
  removed from any resume / paper / abstract.

---

## 6. Reproduce

```bash
# encode both splits (Mac, ~30 min, one-time)
python main/encoder.py --seq --cache gsm8k_train.jsonl
python main/encoder.py --seq --cache gsm8k_test.jsonl

# the held-out replication
python experiments/exp_replication.py            # full table
python experiments/exp_replication.py --full_only # just the differentiated N=2

# the multi-seed reproduction that exposed the noise
python experiments/exp_reproduce_original.py
```

---

## 7. Option B — MHCoT as a SOLVER (Game-24 neural-guided search)

To test whether the multi-helical mechanism helps where it *should* — a genuinely
multi-path task — MHCoT was rebuilt as a search heuristic on Game-24 (exact
programmatic labels, no LLM; see `solver/`). Real / N=1-complex / N=2-multi-helical
value heuristics, param-matched (~76–85K), guide best-first / beam search on
held-out puzzles. 3 seeds.

**Result (3-seed mean ± std, held-out):**

| Model | beam solve | best-first | nodes-to-solve |
|---|---|---|---|
| real | 0.488 ± 0.098 | 0.111 ± 0.065 | 41.3 ± 3.4 |
| N1 (single complex) | 0.455 ± 0.177 | 0.339 ± 0.135 | 37.2 ± 2.3 |
| N2 (multi-helical) | 0.400 ± 0.083 | 0.119 ± 0.104 | 42.5 ± 5.1 |

- **Multi-helical (N2 vs N1): falsified across seeds.** N2 ≤ N1 on every metric.
  Chains *did* differentiate (div ≈ 1.1–1.2), so the mechanism engaged — it just
  doesn't help. Third independent confirmation (after scorer single- and
  multi-seed).
- **Complex vs real (N1 vs real): mixed.** N1 beats real on best-first (3/3 seeds)
  and is more node-efficient, but ties / slightly loses on the steadier beam
  metric. Weak, metric-dependent — not a clean win.

**Important methodological caveat:** all the above (scorer AND this solver) use a
**scalar-collapse readout** — the two chains are summed (Ψ = ψ⁰+ψ¹), reduced to
|Ψ|, pooled, and trained against a single bit. That sum is mathematically an
ensemble average, so it *cannot* express a multi-hypothesis advantage by
construction. The negatives confirm: averaging-in-complex-space ties real. They
do **not** test reading the chains out as *distinct* answers on a multi-answer
task — the one version of the hypothesis still untested. → next: `solver/`
multi-answer readout (coverage of the valid-first-move set), with a real-2-head
control.

---

## 8. The meaning probe — the decisive test (no scalar collapse)

Built to remove the confound in §7: a multi-answer task (predict the full set of
good first-moves), chains read out SEPARATELY (never summed), measured by
coverage@K of the good-move set. Four-way, 3 seeds. The strong control
(real-2-head with an explicit decorrelation/diversity loss) is the bar that
decides whether the ε-helix is special.

**Coverage@3 (3-seed mean ± std, held-out):**

| Model | coverage@3 |
|---|---|
| complex_n2 (ε-helix) | 0.230 ± 0.035 |
| complex_n1 (single chain) | 0.231 ± 0.031 |
| real2_naive (2 heads, no pressure) | 0.250 ± 0.027 |
| real2_diverse (2 heads + decorrelation) | 0.231 ± 0.029 |

**All four tie within noise; N2 is if anything below the naive real baseline.**
This was the setup most favorable to the thesis — no scalar collapse, separate
readout, genuinely multi-answer, ε-helix engaged — and it showed NO advantage.

**This is the FOURTH independent negative** for the multi-helical idea (after
scorer single-seed, scorer multi-seed, solver multi-seed) and the most decisive,
because it was engineered to fix the one confound that could have hidden a real
effect. It is also the pre-committed CHEAP GATE: we agreed that if the chains
cannot carry distinct meaning under direct supervision, RL (weaker signal) will
not rescue them — so Stage 2 (RL) is NOT justified.

**Overall conclusion: the multi-helical / ε-helix mechanism — the central MHCoT
novelty — is not supported across four independent, increasingly-favorable tests.
The complex substrate is sound (stable, trainable, rich co-evolution — `solver/
coevolve.py` Stage 0); the multi-helical superposition does not add value.** This
is an honest, well-isolated negative result, and the falsification methodology
(self-caught false positive, multi-seed replication, pre-committed gates) is the
contribution that stands.

_Last updated: 2026-06-14._
