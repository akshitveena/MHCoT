"""
selfplay/collapse.py  —  S6: reproduce self-play collapse, and mitigate it
==========================================================================

The documented failure of asymmetric self-play: a NAIVE proposer (rewarded simply
for the solver succeeding) drifts to TRIVIAL puzzles — the solver only ever sees
easy tasks, so it never learns the hard ones (collapse). The MITIGATION is the
edge-seeking reward (S4): keep the solver near ~50% success, so difficulty stays
meaningful and the solver keeps improving on hard puzzles.

We run both and compare on a HARD held-out test set.

Run the self-test:
    python selfplay/collapse.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "main"))
sys.path.insert(0, str(_HERE.parent / "solver"))
sys.path.insert(0, str(_HERE))
from env import TARGET, difficulty                             # noqa: E402
from proposer import DifficultyProposer                        # noqa: E402
from curriculum import EdgeSeekingProposer                     # noqa: E402
from loop import _load_agent                                   # noqa: E402
from game24 import gen_puzzles, Puzzle                         # noqa: E402


class NaiveProposer(EdgeSeekingProposer):
    """Rewarded only for solver success → keeps lowering difficulty → collapses
    to trivial puzzles."""
    def update(self, success_rate):
        self.d = max(0.10, self.d - self.step)     # monotonic drift to trivial


def hard_test_set(n=50, seed=777):
    out = []
    for p in gen_puzzles(n * 8, seed=seed, solvable=True):
        d = difficulty(p.numbers)
        if d is not None and d > 0.65:
            out.append(p)
        if len(out) >= n:
            break
    return out


def run(proposer, n_rounds=5, ppr=25, steps=300, budget=100, dev="cpu", seed=0,
        tag=""):
    agent = _load_agent()
    rng = random.Random(seed)
    model = agent.RealSolver(d_model=64).to(dev)
    hard = hard_test_set()
    accumulated, diffs = [], []
    for r in range(n_rounds):
        nums = proposer.propose(rng, ppr)
        batch = [Puzzle(p, TARGET, True) for p in nums]
        succ = agent.solve_rate(model, batch, budget=budget, dev=dev)
        accumulated += batch
        agent.train_solver(model, accumulated, dev=dev, steps=steps, seed=seed)
        proposer.update(succ)
        diffs.append(round(proposer.d, 3))
    hard_sr = agent.solve_rate(model, hard, budget=budget, dev=dev)
    print(f"    [{tag}] difficulty path {diffs}  -> HARD-test solve-rate {hard_sr:.3f}",
          flush=True)
    return diffs, hard_sr


def _test():
    print("[S6] NAIVE proposer (reward = solver success) → expect collapse to trivial ...")
    nd, n_hard = run(NaiveProposer(), tag="naive", seed=0)

    print("[S6] EDGE proposer (target ~50%) → expect difficulty stays meaningful ...")
    ed, e_hard = run(EdgeSeekingProposer(), tag="edge", seed=0)

    print(f"\n[S6] naive final difficulty {nd[-1]} (collapsed to trivial) vs "
          f"edge final difficulty {ed[-1]}")
    print(f"[S6] HARD-test solve-rate: naive {n_hard:.3f}  vs  edge {e_hard:.3f}")
    assert nd[-1] <= 0.20, "naive proposer should collapse difficulty to trivial"
    assert ed[-1] > nd[-1], "edge proposer should keep difficulty higher than naive"
    assert e_hard >= n_hard, "edge curriculum should do at least as well on hard puzzles"
    print("\n[S6 PASSED] collapse reproduced (naive → trivial) and mitigated "
          "(edge keeps difficulty meaningful & solves hard puzzles at least as well).")


if __name__ == "__main__":
    _test()
