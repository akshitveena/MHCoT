# real_mhcot — a real-valued thought encoder (TBERT at the thought level)

The goal, finally framed right: an encoder for **thoughts** (reasoning candidates),
not text. Real-valued (no complex — the imaginary part never earned its place on
text/thoughts). Siamese / contrastive, with the multi-chain idea kept as a
*testable option* (real-valued), and an SAE for an interpretable thought-vocabulary.

**Data = reasoning thoughts** (GSM8K Graph-of-Thoughts candidates, correct/incorrect,
cached as ℝ¹⁵³⁶ vectors — we already have them). **Benchmarks = reasoning-thought
benchmarks** (verification, reranking), NOT GLUE/STS-B (those are text-semantic).

## The ladder (each rung tested before the next)

| Rung | File | Builds | Test gate |
|---|---|---|---|
| **R0** | `data.py` | load thoughts + labels + problem IDs; split by problem; baseline bar | baseline classifier AUC ≈ 0.85 (the bar to beat) |
| **R1** | `encoder.py` | real Siamese contrastive thought encoder (single-branch) | verification AUC ≥ baseline |
| **R2** | `multichain.py` | real multi-chain variant vs single-branch | does multi-chain beat single? (honest test) |
| **R3** | `rerank.py` | reranking eval: pick best candidate → final-answer accuracy | beats majority/random reranking |
| **R4** | `sae.py` | sparse autoencoder → interpretable thought-vocabulary | sparse, reconstructive, inspectable features |

Honest scope: thought/reasoning embedding for verification is adjacent to Process
Reward Models / verifiers (a studied area), so this is a solid, well-scoped
*engineering* project with real numbers — not a novel architecture. The multi-chain
and SAE parts are kept as honest, falsifiable add-ons.

## Results (all rungs tested, multi-seed where noted)

| Rung | Result |
|---|---|
| **R0** baseline | leak-free (split-by-problem) plain-classifier verification AUC **0.793** |
| **R1** contrastive encoder | verification AUC **0.819 ± 0.006** (3 seeds) — **beats** the baseline; prototype-cosine geometry alone beats the classifier |
| **R2** multi-chain | **0.800 ± 0.006** — *trails* single-branch (3/3 seeds) at **2× params**: the multi-chain add-on is null (final confirmation) |
| **R3** reranking | picks a correct candidate **0.80 vs 0.53 random** on mixed problems (+0.27); overall final-answer **0.78 vs 0.71 random**, oracle 0.84 |
| **R4** SAE | recon cosine **0.999**, **3.1%** active features, **25** correctness-*correlated* features (max corr 0.565) — but **largely surface-form on audit** (see R6) |
| **R5** final-answer reranking (`answer_rerank.py`) | **beats self-consistency: 0.775 ± 0.029 vs 0.716 (+5.9 pts), 3 seeds**; recovers ~41% of the random→oracle gap (oracle 0.837) — the headline |
| **R6** feature interpretation (`interpret.py`) | correctness features are interpretable (top exemplars 100% class-coherent, corr up to ±0.89) but appear to track reasoning-**style / surface-form** (formulaic vs exploratory openings) — an honest self-audit flagging a likely stylistic correlate |

**What works (the keeper):** a single-branch, real-valued, contrastive **thought
encoder** whose representation, used as a verifier/reranker, **beats self-consistency
on final-answer accuracy (+6 pts, multi-seed)** and the verification baseline
(AUC 0.82 vs 0.79), and admits a **sparse interpretable thought-vocabulary** (SAE).
**Honest caveats:** the multi-chain/multi-helical add-on is null (trails single-branch);
and the correctness features appear partly **surface-form/style-driven** (R6) — we
surfaced this by auditing our own features rather than overclaiming.

Run: `python real_mhcot/{data,encoder,multichain,rerank,sae,answer_rerank,interpret}.py`
