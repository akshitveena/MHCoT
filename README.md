# MHCoT → Reasoning Verification & Faithfulness

A multi-seed investigation into **how (mathematical) LLM reasoning can be
verified, reranked, and audited — and where it can't.** The project began as a complex-valued
multi-chain reasoning architecture (**MHCoT**), rigorously **falsified** it, pivoted to a
real-valued reasoning **verifier** that works, and then ran an honest study of reasoning
**faithfulness** that ended in a well-supported negative.

**The throughline is discipline.** Every claim is multi-seed and leak-free. **Two single-seed
false positives were caught and retracted** before they shipped. Negative results are reported
as findings, not buried. Scope is stated plainly: **everything here is mathematical reasoning.**

> - Complete result-by-result record: **[FINDINGS_LEDGER.md](FINDINGS_LEDGER.md)** (source of truth).
> - Negative-results writeup: `complex_mhcot/results/MHCoT_TECHNICAL_REPORT.md`.
> - Wins-only summary: `complex_mhcot/results/POSITIVE_FINDINGS.md`.

---

## The arc (how the project actually went)

1. **Hypothesis — MHCoT.** A complex-valued, multi-chain "helical" reasoning architecture
   (complex attention, soliton cross-chain coupling, ε-helix phase dynamics) claimed to give a
   meta-cognitive per-token uncertainty signal.
2. **Falsification.** Across ~6 increasingly-favorable multi-seed tests, the multi-helical
   mechanism was **null**. A dramatic single-seed calibration win (ECE 0.24→0.07) turned out to
   be **noise** — caught by replication. Retracted.
3. **Pivot — the verifier.** A real-valued **contrastive thought encoder** over *frozen*
   DeepSeek-1.5B representations of GSM8K reasoning candidates: **beats self-consistency (+6 pts)**.
   This is the keeper.
4. **Interpretability.** SAEs on the thought embeddings gave clean features — but an audit
   showed they're **surface-form / stylistic**, not reasoning-validity. Reported as a confound.
5. **Faithfulness.** Chased "right answer, wrong path." The *arithmetic* version is **null**
   (0/531; the apparent cases were parse artifacts — a second caught false positive). On
   **ProcessBench (human labels)** the *conceptual* phenomenon is **real and scales with
   difficulty (4%→52%)** — but **four independent detectors all collapse to surface features**
   (a convergent negative).
6. **The deep limitation.** The "representation of reasoning" we used is a *frozen LLM text
   embedding* — it encodes text/answer, not reasoning structure. That is very likely **why**
   faithfulness kept reducing to surface cues. Naming this gap is the project's deepest finding.

---

## Complete findings

### ✅ Validated (standing)
| Finding | Numbers |
|---|---|
| **Verifier beats self-consistency** (`real_mhcot`) | rerank **0.775 ± 0.029 vs 0.716** self-consistency (+6 pts); verification **AUC 0.819 ± 0.006 vs 0.793**; prototype-cosine alone beats a classifier; oracle 0.837. Leak-free, 3 seeds. |
| **"Right answer, wrong path" scales with difficulty** (`tbert`, ProcessBench **human labels**) | GSM8K **4%** · MATH **19%** · OlympiadBench **32%** · OmniMath **52%** of answer-correct solutions. |
| **Frozen reasoning features > fine-tuned text encoders** | fine-tuned bge-base (full, class-weighted) AUC **0.709**, MiniLM 0.673 — both < frozen-DeepSeek **0.819**; freeze sweep monotonic (unfreezing helps but isn't enough). |
| **Complex nets help on phase-bearing / low-data / federated** (`cvnn`) | L1 **0.625 vs 0.461**; L7 federated **0.714 vs 0.463** — *but prior art*, and the complex substrate (N≈1), not the multi-helical novelty. |

### ⚪ Honest negatives (informative)
| Finding | Detail |
|---|---|
| **MHCoT multi-helical mechanism is null** | ~6 multi-seed tests (calibration, search-guidance, multi-answer coverage, class-conditioned co-evolution, recurrent co-evolution, verification) — all null; the real_mhcot multi-chain variant also trails single-branch (0.800 vs 0.819). |
| **Faithfulness resists cheap detection (convergent, 4 methods)** | R6 SAE features = *style*; C11 text-embedding probe = *length* (AUC 0.70 ≈ 0.72 baseline); C12 frozen-embedding step-localization **F1 0.17**; C13 denoising-AE validity = *norm* (0.65 ≈ 0.63). |
| **Arithmetic "wrong path, right answer" is absent** | **0/531** checkable GSM8K candidates (after fixing a broken checker). |

### ❌ Retracted (kept visible for honesty)
| Claim | Why it's dead |
|---|---|
| "Cut ECE ~70% / calibrated meta-cognitive confidence" | single-seed noise; ECE swings 0.04–0.24 across seeds. |
| "Leaky-label: representation knows genuine vs spurious correctness" (arithmetic) | the step-checker was **100% parse artifacts** (chained sums, algebra); real signal = 0. |

---

## Repository structure

```
README.md               this file
FINDINGS_LEDGER.md      complete labelled inventory of every result

main/                   complex primitives (complex_ops, soliton) — shared, repo root
data/                   caches (git-ignored: hidden_cache/*.pt; got_cache/*.jsonl tracked)

complex_mhcot/          the original complex line (FALSIFIED, archival)
  experiments/          scorer falsification (calibration, difficulty, replication, ablations)
  cvnn/                 complex nets on phase-bearing signals (L1–L8)
  pct_x/                phase-coherent transformer variants
  results/              summaries + MHCoT_TECHNICAL_REPORT.md + POSITIVE_FINDINGS.md

real_mhcot/             the KEEPER — real-valued contrastive verifier
  data.py               load thoughts + labels + leak-free split-by-problem
  encoder.py            contrastive thought encoder (verification AUC 0.82)
  multichain.py         multi-chain ablation (null)
  rerank.py             reranking eval
  answer_rerank.py      final-answer accuracy vs self-consistency (the headline)
  sae.py                sparse autoencoder (interpretable vocabulary)
  interpret.py          SAE feature audit -> surface-form
  denoising_sae.py      denoising-AE validity signal (mostly norm)
  solver/ selfplay/     tested Game-24 self-play reproduction

tbert/                  fine-tuning comparison + the faithfulness study
  data.py               candidate text + labels + F1/P/R/AUC metrics
  finetune.py sweep.py  BERT/MPNet fine-tune + layer-freeze sweep
  quadrants.py          arithmetic step-consistency checker (right/wrong path)
  spotcheck.py          validation of the checker (caught the artifact)
  conceptual.py         gold-step-coverage probe (flags valid alternatives)
  processbench.py       "right answer, wrong path" on human labels (A13)
  faithfulness_probe.py within-OmniMath separability vs length baseline (C11)
  step_localizer.py     ProcessBench step-localization baseline (F1 0.17)
  topics.py             topic modeling of the problems
```

---

## Methodology (the differentiator)

- **Multi-seed everywhere**; **leak-free split-by-problem** (a problem's candidates never straddle
  train/test).
- **Falsification ladder** with pre-committed cheap gates — prevents motivated reasoning.
- **Baselines wired in from the start** (self-consistency, length, norm) so "signal" isn't just a
  surface confound.
- **Two false positives caught by discipline:** the ECE calibration win (replication) and the
  arithmetic checker (a 14-example spot-check → 100% artifacts).
- **Convergent-negative reasoning:** four independent methods agreeing is treated as evidence, and
  a *fifth* proxy was deliberately **not** attempted (that would be motivated reasoning).

---

## Honest scope & limitations

- **Mathematical reasoning only.** GSM8K → competition math (incl. ProcessBench, which is 100%
  math). **No transfer evidence** beyond math — a deliberate trade-off: math is the domain with
  clean, verifiable step-level error labels. Cross-domain (HotpotQA/StrategyQA/ProntoQA) is future
  work and is hard *because* those domains lack such labels.
- **The representation is borrowed, and is of *text*.** Everything runs on frozen LLM *text*
  embeddings (pooled), which encode surface + answer, not reasoning structure — the likely root of
  the surface-form wall. A genuine "representation of reasoning" (step-structured or model-internal)
  is unsolved here and largely open in the field.
- **Compute-gated extensions (deferred):** a proper step-level PRM/critic and model-internal
  faithfulness probing need a GPU/API. The baseline floor and data are staged for pick-up.
- **On publishability:** the positives are largely reproductions; the strongest negative (F6) is
  *under-powered* until strong methods are tested. This is a rigorous **portfolio / negative-results**
  artifact, not (yet) a flagship paper.

---

## Reproducing

Each module self-tests. `real_mhcot` needs the (git-ignored) `data/hidden_cache/` caches;
`tbert` downloads GSM8K / ProcessBench via 🤗 `datasets`.

```bash
# the verifier (headline)
python real_mhcot/answer_rerank.py     # beats self-consistency
python real_mhcot/encoder.py           # verification AUC 0.82
python real_mhcot/interpret.py         # SAE feature audit (surface-form)

# faithfulness
python tbert/processbench.py           # "right answer, wrong path" by difficulty (human labels)
python tbert/faithfulness_probe.py     # convergent negative vs length baseline
python tbert/quadrants.py              # arithmetic-path null; spotcheck.py validates the checker

# the falsification
python complex_mhcot/experiments/exp_replication.py
```

---

_This repository is intentionally an **honest** artifact: the rigor, the retractions, the scope
limits, and the negatives are as much the point as the positive result._
