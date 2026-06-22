# MHCoT → Reasoning Verification & Faithfulness

An honest, multi-seed investigation into **how LLM reasoning can be verified, reranked,
and audited** — and where it *can't*. The project began as a complex-valued multi-chain
reasoning architecture (**MHCoT**), rigorously falsified it, and pivoted to what actually
works: a contrastive reasoning verifier, plus an honest study of reasoning *faithfulness*.

**The throughline is discipline:** every claim is multi-seed and leak-free; two single-seed
false positives were caught and retracted before they shipped; negative results are reported
as findings, not buried.

> Full result-by-result record: **[FINDINGS_LEDGER.md](FINDINGS_LEDGER.md)** (the source of truth).
> Negative-results writeup: `complex_mhcot/results/MHCoT_TECHNICAL_REPORT.md`.

---

## TL;DR — what we found

**✅ Validated (standing):**
- **A reasoning verifier that beats self-consistency** — a contrastive *thought encoder* over
  frozen DeepSeek-1.5B representations of GSM8K Graph-of-Thoughts candidates reranks final
  answers to **0.78 vs 0.72 (+6 pts)**, multi-seed, leak-free; verification **AUC 0.82 vs 0.79**.
- **"Right answer, wrong path" is real and scales with difficulty** — on ProcessBench
  (**human labels**), correct answers reached via a flagged-erroneous reasoning step rise from
  **4% (GSM8K) → 52% (OmniMath)**.
- **Frozen reasoning features > fine-tuned text encoders** for this task (BERT/MPNet fine-tuned,
  full freeze sweep, all < the frozen-DeepSeek probe).

**⚪ Honest negatives (informative):**
- **MHCoT falsification** — the complex-valued multi-helical / ε-helix / co-evolution mechanism
  is null across ~6 increasingly-favorable multi-seed tests.
- **Faithfulness resists cheap detection** — *four* independent representation-based attempts to
  detect reasoning validity all collapse to surface features (style / length / norm / weak
  localization). The phenomenon is real; cheap detection of it is not.

**❌ Retracted (kept visible for honesty):**
- An ECE/"calibrated-confidence" win (single-seed false positive) and an arithmetic
  "leaky-label" finding (a 100%-false-positive regex checker) — both caught and withdrawn.

---

## Repository structure

```
complex_mhcot/   the original complex-valued line (FALSIFIED, archival)
  main/  experiments/  cvnn/  pct_x/  solver/ ...   complex primitives + scorer/solver/signal tests
  results/   summaries, the technical report, POSITIVE_FINDINGS.md
real_mhcot/      the KEEPER — real-valued contrastive thought verifier
  data, encoder, multichain, rerank, sae, answer_rerank, interpret, denoising_sae
  selfplay/   a tested asymmetric self-play reproduction (Game-24)
tbert/           fine-tuning comparison + the faithfulness study (ProcessBench)
  data, finetune, sweep, quadrants, conceptual, processbench, faithfulness_probe,
  step_localizer, topics
FINDINGS_LEDGER.md   complete, labelled inventory of every result
```

Note: `main/` (complex primitives) and `data/` (caches, git-ignored) live at the repo root and
are shared; all import paths resolve to them.

---

## The findings in detail

### 1. The reasoning verifier (`real_mhcot/`) — the headline
A real-valued contrastive **thought encoder** on **frozen** DeepSeek-1.5B representations
(it's a *probe* on the LLM's features — the heavy representation is the LLM's, the projection
+ metric are learned). Leak-free split-by-problem, multi-seed.
- Verification **AUC 0.819 ± 0.006 vs 0.793**; prototype-cosine geometry alone beats a classifier.
- Reranking **beats self-consistency 0.775 ± 0.029 vs 0.716** on final-answer accuracy.
- A multi-chain variant is **null** (0.800 vs 0.819 single-branch) — honest ablation.

### 2. Interpretability — an honest audit, not a claim
SAEs on the thought embeddings reconstruct at **0.999 / 3% sparsity** and surface
correctness-correlated features — but on audit these are **largely surface-form / stylistic**
("First, I need to…" vs "Okay, so…"), not reasoning-validity features. We report the confound.

### 3. Faithfulness (`tbert/`) — the phenomenon is real, cheap detection isn't
- **A13:** "right answer, wrong path" scales 4%→52% with difficulty (ProcessBench human labels).
- **Convergent negative:** four detectors (SAE features, text-embedding probe, frozen-embedding
  step-localization F1 0.17, denoising-AE validity) all reduce to surface features.

### 4. The MHCoT falsification (`complex_mhcot/`) — the original idea, honestly closed
Complex attention, soliton cross-chain coupling, ε-helix phase dynamics. Null on calibration,
search-guidance, multi-answer coverage, and verification — across complex/real, scorer/solver,
with/without co-evolution. The complex *substrate* is sound; the multi-helical *superposition*
adds nothing on non-phase data. (It does help on genuinely phase-bearing signals — `cvnn/` —
but that's prior art.)

---

## Reproducing

Each module is self-testing. Caches (`data/hidden_cache/`, git-ignored) are required for the
`real_mhcot` verifier; `tbert` downloads GSM8K/ProcessBench via 🤗 `datasets`.

```bash
# the verifier (headline)
python real_mhcot/answer_rerank.py        # beats self-consistency
python real_mhcot/encoder.py              # verification AUC
python real_mhcot/interpret.py            # SAE feature audit (surface-form)

# faithfulness
python tbert/processbench.py              # "right answer, wrong path" by difficulty (human labels)
python tbert/faithfulness_probe.py        # convergent negative (vs length baseline)

# the falsification
python complex_mhcot/experiments/exp_replication.py
```

---

## Honest scope & limitations

- The verifier is **solid applied work in an established area** (verifiers / PRMs), not a novel
  architecture.
- "Faithfulness" labels rely on a **necessary-not-sufficient** arithmetic proxy (offline) or
  **human labels** (ProcessBench); the conceptual case genuinely needs a semantic judge.
- Beating the faithfulness-detection negative, and a proper step-level PRM/critic, are
  **compute-gated** (GPU/API) and deferred — the baseline floor and data are staged.

This repository is intentionally an **honest** artifact: the rigor, the retractions, and the
negatives are the point as much as the positive result.
