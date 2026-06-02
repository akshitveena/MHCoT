# MHCoT — Paper 1 Specification

**Multi-Helical Chain-of-Thought: Parallel Complex Reasoning Chains with Phase-Separated Coupling**

Version 1.0 — Realistic scope, designed for one-paper publication. The full
AGI/embodied vision is deferred to future papers and explicitly excluded
from this spec.

---

## 1. One-paragraph pitch

Chain-of-Thought (CoT, Wei 2022), Self-Consistency (SC, Wang 2023), Tree of
Thoughts (ToT, Yao 2023), and Graph of Thoughts (GoT, Besta 2024) all run
real-valued reasoning chains and aggregate via voting or scoring. We extend
this lineage by lifting reasoning into **complex space** and running **N
parallel chains over the same token sequence** with **phase-separated
initializations**. The chains couple through a soliton-inspired forward-pass
rule, remain permanently distinct via an ε-helix constraint, and their
cross-chain interference produces a calibrated per-token uncertainty signal
with zero additional parameters. We add a sparse-autoencoder analysis of
per-chain features to interpret what each chain "thinks" and why chains
agree or disagree.

---

## 1.5 Novelty assessment — honest, on a scale of 10

The original full-vision spec self-scored 9.5/10. That was for the entire
multi-paper, multi-modal, embodied AGI program. For **paper 1 alone**,
realistic component-level novelty:

| Component | Score | Notes |
|---|---|---|
| N parallel complex chains over same tokens | 7 | DCN has complex; SC has parallel; combination + phase-init is new |
| ε-helix constraint (permanent phase separation) | 8 | No prior art on phase-separation as architectural loss |
| Soliton-inspired cross-chain coupling | 7 | Cross-chain coupling exists; this formulation with phase-gated gating is new |
| Interference I(t) as calibrated uncertainty | 8 | Complex superposition for calibration is new |
| SAE analysis on per-chain features | 6 | SAEs are hot; per-chain extension is reasonable |
| Two-level GoT-macro / MHCoT-micro architecture | 7 | Clean lineage from a well-cited paper |

**Overall paper-1 novelty: 7 / 10.**

Calibration anchor:

| Score | Examples | Description |
|---|---|---|
| 9-10 | Transformer, BERT, GPT-3 | Paradigm-shifting |
| 8 | FlashAttention, RLHF, MoE | Major architectural innovation |
| **7** | **MHCoT paper 1** | **Solid new mechanism + clean extension of prior work** |
| 6 | Most LoRA variants, prompting papers | Useful extension but incremental |
| 5 | "We apply X to Y" papers | Minor variation |

7/10 is a strong, defensible novelty for one paper. Sufficient for
ICLR/NeurIPS workshop. Plausible for main-track at ACL/EMNLP/NeurIPS if
experimental results are clean. The 9.5 score returns once future papers
add multi-modal, embodied, and biological-validation contributions on top.

---

## 2. Positioning — what we extend, what we add

| Paper | Year | Contribution | What we keep | What we change |
|---|---|---|---|---|
| **CoT** | 2022 | One reasoning chain | The chain idea | Lift to complex; run N in parallel |
| **SC** | 2023 | N independent chains, majority vote | N parallel chains | Same tokens with phase init (not independent samples); interference-based aggregation |
| **ToT** | 2023 | Tree, backtracking | Multi-path reasoning | No tree topology in paper 1; flat N chains |
| **GoT** | 2024 | Graph of thoughts | Graph structure as input | Each GoT node becomes one MHCoT input |
| **DCN** | 2018 | Complex-valued networks | Complex algebra, modReLU, complex Glorot | N parallel chains, ε-helix, soliton coupling |

**Our claim:** complex chains with phase-separated coupling produce a
calibrated uncertainty signal that real-valued multi-chain methods cannot,
and the per-chain features are mechanistically interpretable via SAEs.

**What we explicitly do NOT claim in this paper:**

- AGI, embodied cognition, or fly-brain validation
- Multi-modal (visual + olfactory + ...) integration
- The Yourdon-Constantine cohesion-coupling agent framework
- Spirograph-Fourier interpretation of the complex lift
- Anything outside text-based reasoning benchmarks

These are future work. This paper does one thing and proves it.

---

## 3. The architecture — concrete modules

### 3.1 The overall pipeline — REVISED to actually use GoT

**Architectural correction (June 2026):** an earlier draft of this spec
proposed using the LLM's raw hidden states as a stand-in for GoT
thought-nodes. That was an approximation, not honest GoT usage. A reviewer
would correctly flag it. The revised pipeline uses Besta 2024's actual GoT
framework as the front-end. This is more expensive (one-time GoT processing
cost) but it is what the paper's "extends GoT" claim requires.

```
Problem text (string)
    ↓
[1]  BESTA GoT FRAMEWORK  (real-valued, runs once per problem)
       Open-source: github.com/spcl/graph-of-thoughts
       Backend: local DeepSeek-R1-Distill-Qwen-1.5B (free, runs on T4)
       Operations:  Generate → Score → Aggregate → Improve
       ↓
       Output: one or more finished thought-nodes as TEXT STRINGS
              e.g., "First, 16 eggs daily, Janet eats 3..."
    ↓
[2]  Cached to disk (one-time cost, ~10-14h for GSM8K train set)
       Reused for every MHCoT training run.
    ↓
[3]  For each thought-node text:
       Tokenize with Qwen tokenizer
       Pass through DeepSeek-R1-Distill-Qwen-1.5B (frozen) → real hidden states h_t ∈ ℝᴰ
    ↓
[4]  ComplexLift (learnable):  h_t → z_t = (W_real h_t) + i (W_imag h_t)
    ↓
[5]  Phase initialization (N=2):
       chain 0: ψ_t^(0) = z_t × exp(i · 0)
       chain 1: ψ_t^(1) = z_t × exp(i · ε_min / 2)
    ↓
[6]  Complex Transformer Stack (K=4 layers, learnable, cfloat throughout):
       Attention is BIDIRECTIONAL (encoder-style — sees full thought-node)
       Layers 1-2  (Stage 1 — semantic):  ComplexAttention + modReLU MLP
       Layers 3-4  (Stage 2 — coupling):  ComplexAttention + SolitonCell
                                          (N=2 chains coupled via EQ4)
    ↓
[7]  Per-token interference:  I(t) = |ψ_t^(0) + ψ_t^(1)|² / D
    ↓
[8]  Pooling — extract a single thought-node summary representation:
       Ψ̄ = mean_t(ψ_t^(0) + ψ_t^(1))     (pooled complex summary)
       Ī = mean_t I(t)                    (mean interference across tokens)
    ↓
[9]  ScoringHead (learnable, real-valued):
       score = Linear( [|Ψ̄|, Ī] ) → scalar confidence in this thought-node
    ↓
[10] Across K candidate thought-nodes from GoT, pick max(score)
       → predicted answer = conclusion of the highest-scoring thought-node
       → per-token confidence trace = I(t) of the chosen thought-node

ARCHITECTURAL TAXONOMY:
  MHCoT is a REPRESENTATION MODEL (encoder).
  Bidirectional attention. Encodes thought-nodes into complex embeddings.
  Trained on thought-node scoring (supervised binary classification:
  "does this thought-node lead to the correct answer?").
  NOT decoder-only. NOT autoregressive. NOT a generation model.
  The full SYSTEM (GoT + MHCoT) produces answers; MHCoT itself encodes
  and scores.
```

**Two-level architecture (matches the spec's original GoT+MHCoT framing):**
- GoT operates at the MACRO level (real space, runs once, produces nodes)
- MHCoT operates at the MICRO level (complex space, runs per node, multi-helical)
- The boundary is clean: one finished GoT node IS the atomic input to MHCoT.

### 3.2 Module specifications

**ComplexLift** (the GoT → MHCoT interface):
```python
class ComplexLift(nn.Module):
    def __init__(self, d):
        self.W_real = nn.Linear(d, d, bias=False)
        self.W_imag = nn.Linear(d, d, bias=False)
        # Init: W_real = I (preserve amplitude), W_imag via DCN complex Glorot
    def forward(self, h_real):  # (B, T, D) real
        return torch.complex(self.W_real(h_real), self.W_imag(h_real))
```

**ComplexAttention** (magnitude-based softmax):
```python
def complex_attention(Q, K, V, mask=None):
    # Q, K, V are complex (B, H, T, d_head)
    scores = torch.einsum("bhid,bhjd->bhij", Q, K.conj()).abs() / sqrt(d_head)
    if mask is not None:
        scores = scores.masked_fill(mask, -inf)
    attn = softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("bhij,bhjd->bhid", attn, V)
```

**modReLU activation** (preserves phase, gates magnitude):
```python
def modrelu(z, b):  # b is a learnable scalar bias
    mag = z.abs()
    phase = z / (mag + 1e-8)
    return F.relu(mag + b) * phase
```

**Magnitude LayerNorm** (normalizes amplitude, preserves phase):
```python
def magnitude_layernorm(z, gamma, beta_phase):
    mag = z.abs()
    phase = z / (mag + 1e-8)
    mag_normalized = (mag - mag.mean(-1, keepdim=True)) / (mag.std(-1, keepdim=True) + 1e-8)
    return (gamma * mag_normalized + beta_phase) * phase
```

**SolitonCell** (the chain coupling — EQ4 from full spec):
```python
class SolitonCell(nn.Module):
    def __init__(self, alpha=0.01, beta=0.05, eps_min=0.25, dt=0.1):
        ...
    def forward(self, psi):  # psi: (B, N=2, T, D) complex
        psi_sum = psi.sum(dim=1, keepdim=True)             # consensus
        phi = psi.angle()                                   # phase per chain
        # phase gap between chain 0 and chain 1, broadcast
        phi_gap = (phi[:, 0:1] - phi[:, 1:2]).norm(dim=-1, keepdim=True)
        gate = torch.exp(-phi_gap / self.eps_min)
        self_focus = -self.alpha * psi.abs().pow(2) * psi
        coupling = self.beta * (psi_sum - psi) * gate
        return psi + self.dt * (self_focus + coupling)
```

### 3.3 Final parameter budget

For paper 1 (small, T4-runnable):
- Frozen backbone: Qwen2.5-1.5B (~1.5B params, frozen)
- ComplexLift: 2 × D² = ~3.3M params (D=1280 for Qwen-1.5B)
- 4 complex transformer layers, d_model=D, 8 heads: ~30M params
- Total learnable: ~33M params. Fits T4 comfortably.

---

## 4. Mathematics (only what we'll implement)

We trim the full spec's 9 equations down to the 5 we actually need.

### EQ-A: Complex Lift

```
z_t = W_real · h_t + i · W_imag · h_t                    ∈ ℂᴰ
```

### EQ-B: Phase Initialization (N=2 chains)

```
ψ_t^(0) = z_t × exp(i · 0)
ψ_t^(1) = z_t × exp(i · ε_min / 2)
```

### EQ-C: Soliton Coupling Forward Rule

```
∂ψ^(i) / ∂d = −α |ψ^(i)|² · ψ^(i) + β (Ψ − ψ^(i)) · exp(−‖φ^(i) − φ^(j)‖ / ε_min)
where Ψ = ψ^(0) + ψ^(1)
```

### EQ-D: Interference (per-token uncertainty)

```
I(t) = |ψ_t^(0) + ψ_t^(1)|² / D                        ∈ ℝ⁺
```

### EQ-E: Training Objective (REVISED — representation/scoring, not generation)

```
L = L_task + λ_ε · L_ε + λ_wave · L_wave + λ_contrast · L_contrast

L_task     = BCEWithLogits(score, label)        ← supervised thought-node scoring
                                                  label = 1 if GoT's thought-node leads
                                                  to correct GSM8K answer, else 0
L_ε        = max(0, ε_min − ‖φ_0 − φ_1‖)²       (helix enforcement)
L_wave     = − E[ Ī · label ] + E[ Ī · (1−label) ]
                                                  reward high mean-interference on correct
                                                  thought-nodes, suppress it on wrong ones
L_contrast = InfoNCE(Ψ̄_correct, Ψ̄_wrong)        (optional — push correct/wrong representations apart)

Starting hyperparameters:
    ε_min        = 0.25
    λ_ε          = 0.10
    λ_wave       = 0.05
    λ_contrast   = 0.05 (optional; off in v1)
```

SAE training is a SEPARATE phase after main model converges, trained on
the complex hidden states. Not in the main loss.

Staged loss introduction (paper 1):
- Steps 0-2K: L_task only (let the complex stack learn basic prediction)
- Steps 2K-5K: + L_ε (enforce helix)
- Steps 5K+:   + L_wave (calibrate interference)
- SAE training is a SEPARATE phase after main model converges.

---

## 5. Datasets

Three benchmarks, all text-only:

| Benchmark | Task | Why |
|---|---|---|
| **GSM8K** | Grade-school math word problems | The universal CoT/SC/ToT/GoT benchmark |
| **Game of 24** | Arithmetic puzzles | ToT's signature benchmark; small enough for fast iteration |
| **ARC-Challenge** | Multi-step science QA | Tests reasoning beyond arithmetic |

Loading (all from HuggingFace):
```python
gsm8k = load_dataset("gsm8k", "main", split="test")        # 1319 problems
game24 = load_dataset("nlile/24-game", split="train")      # custom; or generate
arc   = load_dataset("ai2_arc", "ARC-Challenge", split="test")  # 1172 problems
```

---

## 6. Experiments

### Experiment 1 — Architecture validation (the main result)

Train MHCoT (N=2 complex chains) on GSM8K. Measure:
- Answer accuracy
- Calibration: ECE (Expected Calibration Error) using I(t) at answer token
- AUC of I(t) as predictor of correctness

**Baselines** (all with matched parameter counts):
| Baseline | Description |
|---|---|
| **Backbone only** | Qwen2.5-1.5B with standard CoT prompt |
| **CoT** | Same, with chain-of-thought prompt |
| **SC (K=8)** | Backbone + 8 samples + majority vote |
| **Real-valued multi-chain (ablation)** | Same architecture but real-valued throughout (no phase) |
| **MHCoT (N=2)** | Our model |

**Mandatory ablation:** real-valued multi-chain at matched params answers
"does complex actually help?" — the #1 reviewer question.

### Experiment 2 — Calibration analysis

For each problem in GSM8K test set:
1. Compute mean I(t) over the answer-region tokens
2. Bin problems by I_mean (10 bins)
3. Per bin: measure accuracy and plot calibration curve
4. Compare with SC's confidence (= fraction agreeing in majority vote)

Hypothesis: MHCoT's I-based confidence is better-calibrated than SC's vote-based.

### Experiment 3 — Cross-benchmark generalization

Train on GSM8K; evaluate zero-shot on Game of 24 and ARC-Challenge.
Measure: does the calibration property transfer?

### Experiment 4 — SAE interpretability (the second contribution)

After main model trains:
1. Train one SAE per chain on hidden states (concretely, train an SAE on
   |ψ^(0)| activations and another on |ψ^(1)| activations).
2. Identify features that fire in chain 0 but not chain 1 (and vice versa).
3. For problems where chains agree (high I), identify joint-firing features.
4. For problems where chains disagree (low I), identify divergent features.
5. Use feature-steering to confirm interpretability: clamp a feature on,
   re-run, see if model behavior shifts as predicted.

This produces a story: "MHCoT's chains can be interpreted as alternative
hypotheses the model holds; disagreement maps to specific features."

### Experiment 5 — N=1 ablation (architecture sanity)

Single chain (no helix, no coupling, no interference). Predicts: complex
transformer alone (DCN-equivalent) on top of frozen backbone is NOT
significantly better than backbone alone. This isolates the multi-helical
contribution.

---

## 7. Implementation plan — file structure

```
MHCoT/
├── mhcot/
│   ├── __init__.py
│   ├── config.py            # MHCoTConfig dataclass — all hyperparams
│   ├── got_runner.py        # NEW: wraps Besta GoT, caches outputs to disk
│   ├── complex_ops.py       # ComplexLift, modReLU, MagnitudeLN, ComplexAttention
│   ├── soliton.py           # SolitonCell (EQ-C)
│   ├── model.py             # MHCoTHead — the full complex stack on top of backbone
│   ├── losses.py            # L_task, L_ε, L_wave (EQ-E)
│   ├── sae.py               # Sparse autoencoder for per-chain analysis
│   └── data.py              # GSM8K/Game24/ARC loaders + answer extraction
├── preprocess_got.py        # NEW: run Besta GoT on training/test sets, save to disk
├── train.py                 # main training loop (consumes cached GoT outputs)
├── evaluate.py              # benchmark runner
├── experiments/
│   ├── exp1_main.py         # Experiment 1 — architecture validation
│   ├── exp2_calibration.py  # Experiment 2 — ECE + calibration curves
│   ├── exp3_transfer.py     # Experiment 3 — cross-benchmark
│   ├── exp4_sae.py          # Experiment 4 — SAE interpretability
│   └── exp5_ablation.py     # Experiment 5 — N=1 sanity check
├── results/                 # CSVs, JSON summaries, plots (gitignored)
├── colab/
│   └── train_colab.ipynb    # one-cell Colab runner
├── MHCoT_paper1_spec.md     # this document
├── README.md
└── requirements.txt
```

---

## 8. Training procedure

**Hardware target:** single T4 (Colab free) or single A100 (Colab Pro).

**Schedule:**
- **Phase 0 — Warmup (steps 0-500):** L_task only, learning rate 5e-5,
  ComplexLift trainable, backbone frozen.
- **Phase 1 — Helix enforcement (steps 500-2K):** + L_ε. Verify phase gap
  stays ≥ ε_min in held-out validation.
- **Phase 2 — Wave calibration (steps 2K-10K):** + L_wave. Monitor AUC of
  I as correctness predictor — should rise above 0.6 by step 5K if
  architecture works.
- **Phase 3 — SAE training (separate run, after main model converges):**
  Freeze MHCoT, train SAE on chain-0 and chain-1 hidden states separately,
  use a sparsity coefficient that gives ~50-100 active features per token.

**Compute estimate (Colab Pro A100):**
- Phase 0-2 total training: ~20-40 GPU-hours
- Phase 3 SAE training: ~5-10 GPU-hours
- All experiments + ablations: ~50-100 GPU-hours total
- Fits in $50-100 of Colab Pro compute units.

---

## 9. Risks and what could fail

Be honest. Listing the four most likely failure modes and how we'd respond:

| Risk | Likelihood | If it happens |
|---|---|---|
| Complex stack on frozen backbone doesn't outperform backbone alone | Medium | Unfreeze last 2 backbone layers; if still fails, the architecture genuinely doesn't help on text reasoning — publish negative result honestly |
| ε-helix loss collapses (chains converge despite L_ε) | Low | Increase λ_ε; add explicit phase-projection step at each layer |
| I(t) doesn't correlate with correctness (calibration null) | Medium-high | Diagnose: is it a confound (length, position)? Adjust L_wave; if signal genuinely absent, the architectural claim is unsupported and we say so |
| Real-valued ablation matches MHCoT (no complex advantage) | Medium | Publish honestly. This is the critical experiment and a null is publishable as long as the experiment is rigorous |
| SAEs on complex hidden states don't yield clean features | Medium | Fall back to SAEs on |ψ| alone; if even that fails, drop SAE contribution and keep paper focused on architecture |

We commit upfront: if MHCoT loses to a matched-param real baseline by
more than 0.5 accuracy points, **we publish that** as a negative result.
Reviewers respect honesty more than they respect hype.

---

## 10. Conferences and timeline

**Realistic publication path for a solo researcher with Colab Pro:**

| Phase | Months | Output |
|---|---|---|
| Set up Besta GoT + run preprocessing on GSM8K | 0.5-1 | Cached GoT outputs to disk |
| Implement core architecture | 1-2 | mhcot/ codebase, train.py runs end-to-end on cached GoT outputs |
| Run Experiments 1-2-5 | 2-3 | First arXiv v1 |
| SAE analysis (Experiment 4) | 4 | arXiv v2 with interpretability |
| Cross-benchmark + polish | 5 | Submission-ready draft |
| Workshop submission | 5-6 | NeurIPS / ICLR / ACL workshop |
| Main conference submission | 7-9 | ICLR (Sept deadline) or ACL (Feb deadline) |

**Target venues, in priority order:**

| Venue | Why | Deadline cycle |
|---|---|---|
| **ICLR** | Best for novel reps + interp | Sept submit, May conference |
| **NeurIPS** | Most prestigious | May submit, Dec conference |
| **ACL** | Top NLP venue | Feb submit, summer conference |
| **EMNLP** | Strong applied NLP | June submit, fall conference |
| **COLM** | New, growing, LM-focused | Spring submit |

**Workshops (lower bar, good for first attempt):**
- NeurIPS 2025/2026 — MathAI, Reasoning, Foundation Models workshops
- ICLR 2026 — Reasoning, MechInterp, Tiny Papers
- ACL 2026 — Reasoning, Insights from Negative Results

**My recommendation:** target a **workshop in months 5-6** (low risk,
builds credibility, gives external review feedback), then a **main
conference in months 7-9** (incorporate workshop feedback). arXiv v1
posted in month 3 as the timestamp.

---

## 11. Open design choices for the codebase

Before we write code, these should be locked. My defaults are in bold;
respond if you disagree:

1. **Backbone:** **Qwen2.5-1.5B-Instruct** (~3GB in fp16, fits T4 with room
   for our complex stack on top). Alternatives: Llama-3.2-1B,
   TinyLlama-1.1B.
2. **N for paper 1:** **N=2** (the DNA double-helix minimum).
3. **K complex layers on top:** **K=4** (2 stage-1 + 2 stage-2).
4. **D (hidden dim):** **match backbone's hidden_size** (1536 for Qwen-1.5B).
5. **Number of heads in complex attention:** **8**.
6. **Optimizer:** **AdamW with lr=5e-5, weight_decay=0.01**.
7. **Batch size:** **8 problems, gradient accumulation 4** → effective 32.
8. **Max sequence length:** **512** (covers most GSM8K problems with their
   chain-of-thought).
9. **Soliton hyperparameters:** **α=0.01, β=0.05, dt=0.1**, increased
   over training if stable.
10. **SAE dictionary size:** **8× D** = 12288 features, sparsity ~50 active.

---

## 12. What to build first — Day-1 deliverable

**`mhcot/complex_ops.py`** — the foundational module.

Contains: ComplexLift, modReLU, MagnitudeLN, ComplexAttention, all with
unit tests verifying:
- Output is complex with the right shape
- Gradients flow correctly through complex autograd
- modReLU preserves phase exactly
- MagnitudeLN preserves phase exactly
- ComplexAttention produces a valid attention pattern (rows sum to 1)

Once `complex_ops.py` passes its tests, everything else builds on stable
ground. This is the equivalent of writing the bar before adding plates.

I propose this is the next file we write — together. Then `soliton.py`,
then `model.py`, then `train.py`. One module at a time, each with tests,
before any large training run.

---

## 13. Definitive one-paragraph summary

MHCoT (paper 1) is a learned complex-valued reasoning head that sits on
top of a frozen LLM backbone. It takes the backbone's real-valued hidden
states, lifts them to ℂᴰ via a learnable ComplexLift, runs N=2
phase-separated chains through a small complex transformer with
soliton-style cross-chain coupling, and produces a per-token interference
signal I(t) that we claim is a calibrated uncertainty estimate. The
contribution is twofold: (1) a new architectural primitive for parallel
reasoning that extends the CoT/SC/ToT/GoT lineage into complex space with
a phase-separation constraint, and (2) a sparse-autoencoder analysis that
interprets per-chain features and explains why chains agree or disagree
on specific tokens. We validate on GSM8K, Game of 24, and ARC-Challenge,
with a mandatory real-valued ablation to answer "does complex actually
help?" — the answer to which is the central empirical question of the
paper, regardless of which way it lands.
