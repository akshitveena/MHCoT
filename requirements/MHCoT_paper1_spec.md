# MHCoT — Paper 1 Specification

**Multi-Helical Chain-of-Thought: Parallel Complex Reasoning Chains with Phase-Separated Coupling**

Version 1.4 — Realistic scope, designed for one-paper publication. The full
AGI/embodied vision is deferred to future papers and explicitly excluded
from this spec.

> **Changelog**
> v1.0 → Initial paper-1 spec (encoder framing, GSM8K, ε-helix, SAE)
> v1.1 → Added §3.5: the two chain-init strategies (Option 3 candidate-init,
>        done first; Option 1 ε-helix phase-init, done after), the experiment
>        ordering (Experiment 0 first), the reasoning for why the ε-helix lives
>        in Option 1 (not at candidates/aggregation), and the Paper-2 future
>        unification (natural diversity early + ε-maintained diversity deep).
>        Updated §6 experiments and §7 file structure + GoT cache schema to
>        match the actual implemented code (got_runner produces full artifacts).
> v1.2 → Added the candidate-structure-first plan: §3.5 now measures ε
>        EMPIRICALLY from natural candidate diversity (ε is no longer a guess),
>        and splits "why did the LLM reason this way" into Tier 1 (descriptive,
>        now) vs Tier 2 (mechanistic SAE, later). Added Experiment −1 (candidate
>        structure, no training, GATE 1′) as the true first step. Added §6.5
>        Build & Run Manifest: every file, its purpose, run order, and the
>        critical-path gate diagram.
> v1.3 → Added §4.5 (all training objectives grouped: Group 1 develops the
>        embedding via L_contrast/L_task, Group 2 builds the helix via
>        L_ε/L_wave, Group 3 interprets via SAE; clustering = analysis only,
>        never develops). Added §8 "why the backbone is frozen" rationale +
>        M3 feasibility. Added main/encoder.py (frozen-backbone hidden-state
>        precompute) to the manifest.
> v1.4 → Added §3.6 "how thoughts become numbers": no tokenizer / no
>        thought-vocabulary is built; the only tokenizer is DeepSeek's borrowed
>        subword BPE; meaning is COMPUTED by the frozen LLM (not looked up); a
>        thought-vocabulary is impossible (thoughts are infinite/compositional);
>        the "vocabulary of reasoning concepts" is EMERGENT via the SAE.
> v1.5 → FIRST EMPIRICAL RESULTS (see requirements/EMPIRICAL_FINDINGS.md).
>        Headline: accuracy complex≈real (tie), but CALIBRATION is uniquely the
>        multi-helical interference — ECE 0.070 (N=2) vs 0.225 (N=1) vs 0.240
>        (real), three confounds ruled out (losses, magnitude readout, single
>        complex chain). Option 3: real aggregation wins on multi-candidate, so
>        MHCoT's niche is SINGLE-TRACE calibration (≈ Self-Consistency quality at
>        1/3 cost). #1 open gate: replicate on more data + seeds.

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

## 3.5 Two chain-initialization strategies — Option 3 and Option 1

A GoT run produces both *multiple candidate reasoning chains* (from Generate)
and *one distilled final node* (from Improve). This gives us two distinct ways
to initialize the N chains that MHCoT processes. We pursue them in a deliberate
order, and they form a two-act research narrative.

### Where the N chains come from — the two options

| | **Option 3 (do FIRST)** | **Option 1 (do AFTER)** |
|---|---|---|
| Chains initialized from | Different GoT **candidates** | Phase-shifts of the single **final node** |
| Source of diversity | **Natural** — the inputs genuinely differ | **ε-induced** — the phase law manufactures it |
| ε-helix used? | **No** | **Yes — fully** (the core novelty lives here) |
| Interference granularity | **Pooled, per-problem** (candidates have different lengths) | **Per-token I(t)** (chains are token-aligned) |
| What it proves | "Interference between genuinely different reasoning predicts correctness" — establishes the mechanism | "The ε-helix recovers a comparable signal from ONE node — efficient and elegant" — the contribution |

### Why this order — Act 1 de-risks Act 2

```
ACT 1  (Option 3):  Show interference between different GoT candidates
                    predicts correctness. Signal is real and strong.
                    → Establishes that the interference mechanism is meaningful.

ACT 2  (Option 1):  Show the ε-helix manufactures useful diversity from a
                    SINGLE node, recovering a comparable calibration signal
                    without needing multiple candidates.
                    → This is the novelty (phase-separation constraint).
```

A reviewer asking "why should phase-separated chains carry useful information?"
is answered by Act 1: because interference between reasoning chains
*demonstrably* works, and the ε-helix is how we obtain that signal cheaply,
from one node, in a parameter-efficient way.

### Why the ε-helix belongs in Option 1, NOT in the candidates/aggregation

The ε-helix enforces phase-separation between chains that **share an origin**
— its meaning is "two interpretations of one thing, kept diverse." That only
makes sense when the chains start from a common point.

| Placement | Chains share an origin? | ε-helix fits? |
|---|---|---|
| At candidates (Generate output) | No — genuinely different texts | ❌ already maximally different; no common origin to diverge from |
| At aggregation (merging candidates) | Happens in **text space** inside GoT | ❌ nothing complex to constrain yet |
| At the final node (Option 1) | Yes — one node → phase-shifted chains | ✅ **its natural home** |

So Option 3 and the ε-helix do **not** compete for the same architectural
slot. Option 3 uses natural diversity (different candidates); Option 1 uses
ε-induced diversity (phase-shifted single node). The novelty is fully
preserved — it simply lives in Act 2.

### What transfers from Option 3 → Option 1

- ✅ The **validated mechanism** ("interference predicts correctness")
- ✅ The **shared complex machinery** (`ComplexLift`, `ComplexAttention`,
  encoder layers) — Option 1 can **warm-start** from Option 3's weights
- ✅ The **interference concept** (`Ψ = Σψ, I = |Ψ|²`), pooled vs per-token

Does NOT transfer (and need not):
- Chain initialization (different-candidates → phase-shift): swapped in fresh
- The `L_ε` loss: Option 1 adds it; Option 3 never had it

Concretely, **"tuning Option 3 into Option 1"** = keep the complex encoder,
swap chain-init to phase-shifted-single-node, add the `L_ε` loss.

### Data requirement (already satisfied)

Both options are served by the GoT cache produced by `preprocess_got.py`:
- `candidates` (list of distinct reasoning chains + scores) → feeds Option 3
- `final_node` (distilled output) → feeds Option 1

Saving both costs nothing extra and keeps both acts open. The critical
preprocessing check is that the `candidates` are genuinely **distinct**
(if Generate's sampling collapses them to identical text, Option 3 has no
signal — verify on the smoke test before scaling).

### Measuring ε from natural diversity — making the helix principled

The weakest point of Option 1, and the question a reviewer will attack, is:
*"why should rotating phase by ε produce **meaningful** diversity rather than
noise, and why ε = 0.25?"* Right now ε_min is a guess.

We answer this empirically, **before** building Option 1, using the
candidates we already cache. The candidates are *natural* reasoning diversity
— same LLM, same problem, genuinely different chains. So:

1. Lift each candidate to complex space (the same `ComplexLift`).
2. Measure the **phase gap between candidate representations** — this is the
   empirical scale of real reasoning divergence the LLM exhibits.
3. Set ε_min to match that measured scale.

Then Option 1's ε-helix is no longer arbitrary — it is calibrated to
reproduce the diversity scale the LLM actually exhibits in its own reasoning.
This mirrors the original spec's biological idea (measure ε from connectome
inter-circuit phase gaps), but uses the LLM's own reasoning diversity, which
is more defensible and immediately available from the cache.

This is also the **empirical bridge** from Option 3 → Option 1: characterize
the structure and scale of natural diversity (Option 3 data), then show the
ε-helix reproduces it from a single node (Option 1).

**Two tiers of "why did the LLM reason this way":**
- **Tier 1 (now, on the critical path):** *descriptive* — pairwise candidate
  distances, clustering by final answer, where chains diverge (early framing
  vs late arithmetic), and the natural phase scale that sets ε. No training.
- **Tier 2 (Phase 3, later):** *mechanistic* — SAE features, causal tracing,
  which circuits drove each reasoning style. A research program on its own;
  do NOT start until Option 1 shows signal.

### The future unification (Paper 2 — do NOT build for Paper 1)

There is an elegant way to use **both** kinds of diversity in a single
architecture:

```
lift each candidate to complex space        ← natural diversity (Option 3)
        ↓
aggregate them via interference:  Ψ = Σ_c z_c   ← consensus signal
        ↓
spawn ε-helix phase-chains from that consensus  ← maintained diversity (Option 1)
        ↓
deep complex processing (soliton coupling, etc.)
```

Natural diversity early, ε-maintained diversity deep. This uses everything —
the candidates, the interference, the ε-helix, the soliton coupling — in one
coherent two-stage structure. It is the natural endpoint of this line of work
and a strong Paper 2 architecture. **Explicitly out of scope for Paper 1.**
We note it here only to record where the design leads.

## 3.6 How thoughts become numbers — NO tokenizer, NO thought-vocabulary

A recurring question: do we build a tokenizer / a "vocabulary at the thought
level" to encode reasoning candidates? **No. We build neither.** This section
records why, because it is central to the architecture.

**Tokenization is borrowed, at the subword level only.**
```
candidate text → DeepSeek subword tokenizer (~150k BPE vocab, borrowed)
              → token IDs
              → frozen DeepSeek forward pass
              → hidden states  ∈ ℝ^(T×D)      ← THE MEANING IS COMPUTED HERE
              → pool → reasoning fingerprint ∈ ℝᴰ
              → ComplexLift → ℂᴰ
```
The only tokenizer is DeepSeek's existing subword BPE tokenizer (used in
`main/encoder.py`). We never build, train, or modify a tokenizer.

**Meaning is COMPUTED, not looked up.**
There is no lookup table `thought → vector`. That is the pre-2018 word2vec
paradigm (static per-word vectors). The modern paradigm we use COMPUTES the
representation of a span dynamically by running its subword tokens through the
transformer. The pooled hidden state IS the numeric representation of the
thought — produced by the frozen LLM, not retrieved from a thought-vocabulary.

**A thought-level vocabulary is impossible anyway.**
Subwords are finite and enumerable (~150k). Thoughts are infinite and
compositional — there is no finite list of "all reasoning steps." This is
exactly why the field moved from lookup to computation.

**The "vocabulary of reasoning concepts" is EMERGENT, via the SAE.**
The legitimate goal behind the question — a set of interpretable units that
say *which reasoning type this is and why it happened* — is real, and it is
delivered by the Sparse Autoencoder (Stage E). The SAE learns a sparse,
interpretable feature basis DISCOVERED from the computed representations. That
discovered basis is the closest thing to a "vocabulary of reasoning concepts,"
but it is a RESULT (emergent, learned from data, after representations exist),
not an INPUT (pre-defined, looked up, before tokenization).

| What you might want | How it is actually provided |
|---|---|
| numeric representation of a thought | computed by frozen DeepSeek (pooled hidden state) |
| distinguish reasoning types | geometry of those representations (distances/clustering, Exp −1) |
| which thoughts are similar & why | distances + interference in representation space |
| "what made this reasoning happen" | SAE features (Stage E) — emergent, not pre-built |

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

### 4.5 All training objectives, grouped by stage

Three groups, used at different stages. Group 1 *develops* the embedding;
Group 2 *builds* the helix; Group 3 *interprets* it.

**Group 1 — Develop the embedding (Stage C, Option 3).** Shapes the complex
representation so phase becomes meaningful (the word2vec-analog).

| Objective | Sketch | Role |
|---|---|---|
| **L_contrast** | InfoNCE/triplet: pull together candidates with the SAME final answer, push apart different | PRIMARY embedding-developer. Organizes complex space by reasoning outcome. Uses the answer as a free grouping signal — no manual labels. Makes phase meaningful. |
| **L_task** | `BCE(score, label)`, label = reaches gold | Supervised correctness signal. |

**Group 2 — Shape the helix (Stage D, Option 1).** The MHCoT-specific novelty.

| Objective | Formula | Role |
|---|---|---|
| **L_task** | `BCE(score, label)` | Correctness supervision (carried over) |
| **L_ε** | `max(0, ε_min − ‖φ⁰−φ¹‖)²` | The core novelty — phase-separation ≥ ε_min. Uses the ε MEASURED in Experiment −1, not a guess. |
| **L_wave** | `−E[I·correct] + E[I·wrong]` | Makes interference I(t) a calibrated confidence signal. |

Combined: `L = L_task + λ_ε·L_ε + λ_wave·L_wave`, introduced staged.

**Group 3 — Interpretability (Stage E, SEPARATE training).** Trained after the
main model converges, on its frozen hidden states.

| Objective | Formula | Role |
|---|---|---|
| **L_recon** | `‖h − decode(encode(h))‖²` | SAE reconstructs chain hidden states |
| **L_sparse** | `λ·‖encode(h)‖₁` | Forces sparse interpretable features |

**Deferred:** `L_dim` (thought-tensor entropy, prevents dimensional collapse)
— trimmed for Paper 1; add only if collapse is observed.

**On classical methods (k-means, PCA, clustering):** these are ANALYSIS tools
applied AFTER embeddings exist — they read structure, they do not develop it.
The embedding is developed by Group 1 objectives (representation learning).
Clustering is used twice: on the untrained lift (Exp −1a, baseline) and on the
trained lift (Exp −1b, meaningful reasoning modes). It never trains the
embedding; the contrastive objective does.

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

**Experiment ordering:** Experiment −1 (candidate structure) runs FIRST — it
is pure analysis, needs no training, and produces the measured ε plus the
first correctness signal. Then Experiment 0 (Option 3 interference), then
Experiments 1–5 (Option 1, the ε-helix design).

### Experiment −1 — Candidate structure analysis (do FIRST, no training)

Characterize the diversity *inside* each GoT node before modeling anything.
For each problem's N candidates, encode each through the frozen backbone and:

1. **Pairwise distance:** cosine distance between candidate representations;
   do candidates that reach the *same* final answer cluster together?
2. **Divergence location:** where in the sequence do candidates split —
   early (framing) or late (arithmetic)?
3. **Diversity vs correctness:** tightly-clustered candidates → more often
   correct? Spread-out candidates → the LLM is "unsure" → more often wrong?
4. **Measure natural ε:** lift candidates to complex space, measure the phase
   gap between them → this is the empirical ε_min for Option 1.

- Outputs: distance/clustering plots, divergence histograms, the measured ε.
- **GATE 1′:** Are candidates distinct in *representation* space (not just
  surface text)? Does diversity relate to correctness?
  - Yes → diversity is real, ε is measurable, both options alive; proceed.
  - No (candidates collapse to near-identical representations) → critical
    early warning that Option 3 AND Option 1 stand on sand. Stop and diagnose.

This experiment shares all its infrastructure with Experiment 0 (same loading,
same encoding, same lift) — it is a richer first pass, not a detour.

### Experiment 0 — Option 3: cross-candidate interference

Initialize the N chains from **different GoT candidates** (not phase-shifts).
Pool each candidate to a fixed-size complex summary, compute pooled
interference, and test whether it predicts correctness.

- Signal: pooled `I = |Σ_c pool(z_c)|²` per problem
- Metric: AUC of `I` as predictor of whether the candidates' consensus answer
  is correct
- Label source: does the GoT candidate set lead to the gold answer?
- No ε-helix here (chains are already maximally diverse)

**Decision gate:** if AUC > 0.65, the interference mechanism is real → proceed
to Option 1 with confidence. If AUC ≈ 0.5, diagnose before building Option 1.

### Experiment 1 — Option 1 / Architecture validation (the main result)

Train MHCoT (N=2 complex chains, ε-helix) on GSM8K. Measure:
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

## 6.5 Build & Run Manifest — every file, its purpose, the order

This is the single source of truth for *what to build, what to run, and why*.
Status: ✅ done · 🔜 next · ⬜ later.

### Stage A — Data (produce the GoT nodes)

| File | Run? | Purpose / what it adds |
|---|---|---|
| ✅ `main/got_runner.py` | run for 1-problem smoke test | Besta GoT + local DeepSeek backend. `run_got_full()` returns candidates + scores + aggregated + final_node. Verifies the pipeline on one problem. |
| ✅ `main/preprocess_got.py` | **run for the dataset** | Batch GoT over GSM8K, resumable. Writes `data/got_cache/gsm8k_test.jsonl`. **This produces the dataset everything else consumes.** |

Run order: `got_runner.py` (verify) → `preprocess_got.py --limit 605` (produce).

### Stage B — Foundation (the complex primitives)

| File | Run? | Purpose / what it adds |
|---|---|---|
| ✅ `main/complex_ops.py` | run its self-test | ComplexLinear, ComplexLift, ComplexPositionalEncoding, modReLU, MagnitudeLN, ComplexAttention. The reusable building blocks. MPS-safe, unit-tested. |

### Stage C — Analysis FIRST (no training — the cheap gates)

| File | Run? | Purpose / what it adds |
|---|---|---|
| 🔜 `main/data.py` | imported by experiments | Loads the cache, extracts each candidate/node's predicted answer, compares to gold → **correctness labels**. The bridge from raw text to a trainable dataset. Gates everything. |
| 🔜 `main/encoder.py` | **run once to precompute** | Text → frozen DeepSeek → hidden states `H ∈ ℝ^(T×D)`, **cached to disk**. The frozen-backbone-precompute trick: run the 1.5B model ONCE, then all training reuses the cache. This is what makes M3 training fast (~30-60 min one-time, then the backbone is out of the loop). |
| 🔜 `experiments/exp_candidate_structure.py` | **run — Experiment −1** | Encodes candidates (via cached H), measures pairwise distance/clustering (k-means)/divergence, **measures the natural ε**, correlates diversity with correctness. Two passes: −1a untrained lift (baseline), −1b trained lift (meaningful). → **GATE 1′**. Pure analysis. |
| 🔜 `experiments/exp0_option3.py` | **run — Experiment 0** | Pooled interference between candidates → AUC vs correctness. → **GATE 1** (does interference predict correctness?). Untrained probe first, then trained lift. |

### Stage D — Option 1 model (the ε-helix novelty — build only after gates pass)

| File | Run? | Purpose / what it adds |
|---|---|---|
| ⬜ `main/soliton.py` | run its self-test | SolitonCell (cross-chain coupling, EQ-C) + `L_ε` helix loss. Uses the ε measured in Experiment −1. |
| ⬜ `main/model.py` | imported by training | `MHCoTEncoder`: assembles lift → posenc → phase-init N=2 → Stage1 (attn+modReLU) → Stage2 (attn+soliton) → interference + scoring head. |
| ⬜ `main/losses.py` | imported by training | `L_task` (BCE on correctness) + `L_ε` + `L_wave`. |
| ⬜ `experiments/exp1_main.py` | **run — Experiment 1** | Train Option 1 on final_nodes, N=2 phase-shifted chains, ε-helix. Measure per-token I(t), calibration AUC, ECE. → **GATE 3**. |
| ⬜ `experiments/exp5_ablation.py` | run — Experiment 5 | N=1 + real-valued ablation at matched params. Answers "does complex help?" |

### Stage E — Interpretability + breadth (the second contribution)

| File | Run? | Purpose / what it adds |
|---|---|---|
| ⬜ `main/sae.py` | imported by exp4 | Sparse autoencoder for per-chain feature analysis (Tier 2). |
| ⬜ `experiments/exp4_sae.py` | run — Experiment 4 | Per-chain features, agreement/disagreement features, feature steering. |
| ⬜ `experiments/exp2_calibration.py` | run — Experiment 2 | ECE + calibration curves vs SC's vote-confidence. |
| ⬜ `experiments/exp3_transfer.py` | run — Experiment 3 | Train GSM8K, test zero-shot on Game-of-24 / ARC. |

### The critical path (what actually gates the project)

```
preprocess_got.py  →  data.py  →  exp_candidate_structure.py   [GATE 1′]
                                          ↓ (if diversity real)
                                   exp0_option3.py              [GATE 1]
                                          ↓ (if interference predicts correctness)
                          soliton.py + model.py + losses.py
                                          ↓
                                   exp1_main.py                 [GATE 3]
                                          ↓ (if ε-helix recovers signal)
                              sae.py + remaining experiments → arXiv v1
```

Each gate is go/no-go. We build heavy machinery (Stage D) only after the
cheap analysis (Stage C) shows signal — the same falsification discipline
that killed the wasteful LLM-decoration path early.

---

## 7. Implementation plan — file structure

Actual current layout (code lives under `main/`):

```
MHCoT/
├── main/
│   ├── __init__.py
│   ├── got_runner.py        # ✅ DONE: Besta GoT + local DeepSeek backend.
│   │                        #    run_got_full() returns GoTArtifacts (candidates
│   │                        #    + scores + aggregated + final_node).
│   ├── preprocess_got.py    # ✅ DONE: batch GoT over GSM8K, resumable, caches
│   │                        #    full artifacts to data/got_cache/*.jsonl
│   ├── complex_ops.py       # ✅ DONE: ComplexLinear, ComplexLift,
│   │                        #    ComplexPositionalEncoding, modReLU,
│   │                        #    MagnitudeLN, ComplexAttention (all MPS-safe,
│   │                        #    all unit-tested)
│   ├── soliton.py           # TODO: SolitonCell (EQ-C) + L_ε helix loss
│   ├── model.py             # TODO: MHCoTEncoder (assembles the primitives)
│   ├── losses.py            # TODO: L_task, L_ε, L_wave (EQ-E)
│   ├── sae.py               # TODO: sparse autoencoder for per-chain analysis
│   └── data.py              # TODO: cache loaders + answer extraction
├── experiments/
│   ├── exp0_option3.py      # TODO: Experiment 0 — cross-candidate interference
│   ├── exp1_main.py         # TODO: Experiment 1 — Option 1 validation
│   ├── exp2_calibration.py  # TODO: ECE + calibration curves
│   ├── exp3_transfer.py     # TODO: cross-benchmark
│   ├── exp4_sae.py          # TODO: SAE interpretability
│   └── exp5_ablation.py     # TODO: N=1 sanity check
├── data/got_cache/          # GoT artifacts (gitignored except milestones)
├── colab/                   # Colab notebooks
├── requirements/MHCoT_paper1_spec.md   # this document
├── setup_m3.sh              # ✅ DONE: one-shot M3 conda env rebuild
├── constraints.txt          # pinned versions (torch/numpy frozen)
└── requirements.txt
```

### GoT cache schema (`data/got_cache/gsm8k_{split}.jsonl`)

One JSON object per line:

```
{
  "idx":          int,
  "problem":      str,
  "gold":         str,
  "candidates":   [{"text": str, "score": float|null}, ...],  # → Option 3
  "kept":         [str, ...],
  "aggregated":   str|null,
  "final_node":   str,                                        # → Option 1
  "thought_nodes":[str],         # = [final_node]  (back-compat)
  "scores":       [float|null],
  "phases":       [str],
  "wall_seconds": float,
  "error":        str            # only present on failure
}
```

---

## 8. Training procedure

### Why the DeepSeek backbone is FROZEN (and what we actually train)

We never fine-tune the 1.5B backbone. Only the ~35M-param head trains
(ComplexLift ~4.7M + 4 complex layers ~30M). Reasons, principled first:

1. **Clean attribution (decisive).** With the backbone frozen, ANY calibration
   gain is unambiguously from MHCoT (lift + helix + interference), not from a
   better-tuned language model. A frozen backbone IS the experimental control.
   Fine-tuning would let a reviewer say "maybe the gains are just fine-tuning."
2. **The research question is about MHCoT, not the backbone.** We test the
   processing layer on top of a fixed representation; we are not building a
   better LLM. DeepSeek-R1-Distill already reasons well.
3. **Standard probing methodology.** Freeze a model, train a small head on its
   representations — the correct design for "what can complex multi-helical
   processing extract from a fixed representation?"
4. **Precompute efficiency (the M3 enabler).** Frozen ⇒ run each candidate
   through the backbone ONCE, cache hidden states to disk, train the 35M head
   on the cache forever. The giant model never sits in the training loop.
5. **Reproducibility.** A public frozen checkpoint is fully reproducible.

We would freeze it even on an H100 for Paper 1. LoRA (small adapter, ~1-5M
params, feasible even on M3) is the future option ONCE the mechanism is
proven — a Paper 2 optimization, deliberately out of scope here.

### Can it train on M3? Yes.

Trainable surface is ~35M params (~0.5GB incl. grads + Adam). With cached
hidden states the backbone is out of the loop, so a full run (a few epochs
over ~1815 candidate sequences) is ~10-30 min on MPS. `complex_ops.py` is
real-decomposed and its gradient tests pass on MPS. What is NOT feasible on
M3 — full fine-tuning of the 1.5B backbone — is exactly what we don't do.

**Hardware target:** M3 MPS (Paper 1 head training) · T4/A100 (optional speedup).

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
