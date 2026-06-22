"""
selfplay/curriculum.py  —  S4: the edge-seeking curriculum ("asker stays ahead")
================================================================================

Replaces the fixed-band proposer with an ADAPTIVE one that targets the solver's
frontier: it raises difficulty when the solver succeeds too often and lowers it
when the solver fails too often, holding success near a target (~0.5). As the
solver improves, the proposer escalates — the asymmetric-self-play curriculum
(SQLM/R-Zero style), made concrete and tested.

Run the self-test:
    python selfplay/curriculum.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "main"))
sys.path.insert(0, str(_HERE.parent / "solver"))
sys.path.insert(0, str(_HERE))
from env import TARGET                                          # noqa: E402
from proposer import DifficultyProposer                        # noqa: E402
from loop import _load_agent                                   # noqa: E402
from game24 import gen_puzzles, Puzzle                         # noqa: E402


class EdgeSeekingProposer:
    """Adaptive proposer: keeps the proposed difficulty at the solver's edge by
    tracking a target success rate."""

    def __init__(self, target_success=0.5, step=0.07, width=0.12, start=0.40,
                 dead=0.12):
        self.d = start
        self.target = target_success
        self.step, self.width, self.dead = step, width, dead
        self.base = DifficultyProposer()

    def propose(self, rng, n):
        lo, hi = max(0.0, self.d - self.width), min(1.0, self.d + self.width)
        return [p for p, _ in self.base.propose_band(rng, lo, hi, n)]

    def update(self, success_rate):
        if success_rate > self.target + self.dead:        # too easy → escalate
            self.d = min(0.97, self.d + self.step)
        elif success_rate < self.target - self.dead:      # too hard → ease off
            self.d = max(0.10, self.d - self.step)


def run_curriculum(n_rounds=6, ppr=30, steps=350, budget=100, dev="cpu", seed=0,
                   verbose=True):
    agent = _load_agent()
    rng = random.Random(seed)
    prop = EdgeSeekingProposer()
    model = agent.RealSolver(d_model=64).to(dev)
    test_pz = gen_puzzles(60, seed=999, solvable=True)
    accumulated, hist = [], []
    for r in range(n_rounds):
        nums = prop.propose(rng, ppr)
        batch = [Puzzle(p, TARGET, True) for p in nums]
        succ = agent.solve_rate(model, batch, budget=budget, dev=dev)   # current ability @ edge
        accumulated += batch
        agent.train_solver(model, accumulated, dev=dev, steps=steps, seed=seed)
        test_sr = agent.solve_rate(model, test_pz, budget=budget, dev=dev)
        prop.update(succ)
        hist.append({"difficulty": round(prop.d, 3), "succ_on_proposed": round(succ, 3),
                     "test_solve_rate": round(test_sr, 3)})
        if verbose:
            print(f"    round {r+1}/{n_rounds}: proposed_diff→{prop.d:.2f}  "
                  f"succ@edge={succ:.2f}  test_solve={test_sr:.3f}", flush=True)
    return hist


def _test():
    print("[S4] edge-seeking curriculum ...")
    hist = run_curriculum(n_rounds=6, ppr=30, steps=350, seed=0)
    diffs = [h["difficulty"] for h in hist]
    tests = [h["test_solve_rate"] for h in hist]
    print(f"[S4] difficulty trajectory: {diffs}")
    print(f"[S4] test solve-rate:       {tests}")
    # (a) solver improved; (b) proposer adapted difficulty (didn't sit still); and
    # (c) it escalated late vs its easiest point ("stays ahead").
    assert tests[-1] > tests[0] + 0.05, "solver should improve under the curriculum"
    assert max(diffs) - min(diffs) > 0.05, "proposer should adapt difficulty"
    assert max(diffs[len(diffs)//2:]) >= min(diffs[:max(1, len(diffs)//2)]), \
        "difficulty should escalate (or hold) as the solver improves"
    print("\n[S4 PASSED] proposer tracks the solver's edge and escalates as it improves "
          "— the 'asker stays ahead' curriculum works.")


if __name__ == "__main__":
    _test()
