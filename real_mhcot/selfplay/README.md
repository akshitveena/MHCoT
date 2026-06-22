# Self-Play Reasoning Sandbox (Game-24)

A from-scratch, fully-tested reproduction of **asymmetric self-play** (SQLM /
R-Zero / Absolute-Zero style) on a tractable, exactly-verifiable domain — Game-24
— runnable on a laptop CPU. A **proposer** generates puzzles, a **solver** learns
to solve them, a **verifier** judges correctness, and an **automatic curriculum**
keeps the proposer at the solver's frontier. Includes a reproduction of the
known **self-play collapse** failure and a mitigation.

> Honest scope: this is a **reproduction / learning project**, not novel research.
> The paradigm it reproduces (proposer–solver–verifier self-play with an
> automatic curriculum) is an active 2025–2026 area (SQLM arXiv:2508.03682,
> R-Zero, Absolute Zero Reasoner, "edge of learnability" arXiv:2601.18778). The
> value here is a clean, tested, end-to-end implementation you can run and reason
> about — including the collapse dynamics most write-ups only mention.

## The ladder (each rung independently tested)

| Rung | File | What it demonstrates | Key result |
|---|---|---|---|
| **S0** | `env.py` | Game-24 self-play environment (propose / verify / difficulty) | difficulty spans 0.25–0.97; oracle 50/50 vs blind 0/50 |
| **S1** | `solver.py` | a learnable solver (value-net heuristic guiding search) | heuristic AUC 0.59→0.97; solve-rate 0.00→0.39 |
| **S2** | `proposer.py` | a proposer that controls difficulty | easy 0.33 (100% solved) vs hard 0.88 (18%) |
| **S3** | `loop.py` | the self-play loop (propose→solve→verify→train) | solve-rate 0.10→0.52 over rounds |
| **S4** | `curriculum.py` | edge-seeking curriculum ("asker stays ahead") | difficulty escalates as solver improves; success held near ~50% |
| **S5** | `verifier.py` | learned verifier (the TBERT/PRM reward slot) | acc 0.83 / AUC 0.87 vs the oracle, no oracle at inference |
| **S6** | `collapse.py` | reproduce collapse + mitigate it | naive → difficulty collapses to 0.10 (trivial); edge keeps it meaningful and solves hard puzzles 0.80 vs 0.58 |

## Run it

```bash
python selfplay/env.py          # S0
python selfplay/solver.py       # S1
python selfplay/proposer.py     # S2
python selfplay/loop.py         # S3
python selfplay/curriculum.py   # S4
python selfplay/verifier.py     # S5
python selfplay/collapse.py     # S6
```

## What each component is

- **Proposer** — generates Game-24 puzzles; in S4 it adapts difficulty to keep the
  solver near 50% success (automatic curriculum). The naive variant (S6) is
  rewarded only for solver success and collapses to trivial puzzles.
- **Solver** — a small value network scores states (≈ P(reach 24)) and guides
  best-first search; it learns from self-generated, verified puzzles.
- **Verifier** — the reward signal. Oracle (exact, `game24.reachable`) for
  ground truth; a **learned** verifier (S5) shows self-play can run without the
  oracle (the PRM/TBERT slot; swappable for a contrastive Siamese embedder).

## The headline finding (S6)

Asymmetric self-play **collapses** when the proposer is rewarded naively (it
drifts to trivial or unsolvable tasks and the solver stops learning). An
**edge-seeking curriculum** — keep the solver at ~50% success — prevents the
collapse and yields a solver that handles hard puzzles substantially better
(0.80 vs 0.58 on a hard held-out set). This is the documented self-play
instability, reproduced and fixed in a minimal, inspectable setting.

_Built on the project's Game-24 world (`solver/game24.py`, `solver/search.py`)._
