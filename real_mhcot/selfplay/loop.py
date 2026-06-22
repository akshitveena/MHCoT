"""
selfplay/loop.py  —  S3: the self-play loop (fixed proposer → solver improves)
==============================================================================

Ties proposer + solver + verifier into an iterative loop: each round the proposer
emits puzzles, the solver trains on them (self-generated supervision, verified by
the oracle), and we track the solver's solve-rate on a held-out test set. With a
FIXED-band proposer this shows the loop produces improvement — the scaffolding the
learned curriculum (S4) and collapse study (S6) plug into.

Run the self-test:
    python selfplay/loop.py
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "main"))
sys.path.insert(0, str(_HERE.parent / "solver"))
sys.path.insert(0, str(_HERE))
from env import TARGET                                          # noqa: E402
from proposer import DifficultyProposer                        # noqa: E402
from game24 import gen_puzzles, Puzzle                         # noqa: E402


def _load_agent():
    """Load selfplay/solver.py by path (avoids the 'solver' name clash with the
    solver/ directory)."""
    spec = importlib.util.spec_from_file_location("sp_agent", _HERE / "solver.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_selfplay(n_iters=4, puzzles_per_iter=30, band=(0.40, 0.75),
                 steps_per_iter=400, budget=100, dev="cpu", seed=0, verbose=True):
    agent = _load_agent()
    rng = random.Random(seed)
    prop = DifficultyProposer()
    model = agent.RealSolver(d_model=64).to(dev)
    test_pz = gen_puzzles(60, seed=999, solvable=True)
    accumulated, history = [], []
    for it in range(n_iters):
        batch = prop.propose_band(rng, band[0], band[1], puzzles_per_iter)
        accumulated += [Puzzle(p, TARGET, True) for p, _ in batch]
        agent.train_solver(model, accumulated, dev=dev, steps=steps_per_iter, seed=seed)
        sr = agent.solve_rate(model, test_pz, budget=budget, dev=dev)
        history.append(sr)
        if verbose:
            print(f"    iter {it+1}/{n_iters}: |data|={len(accumulated)}  "
                  f"solve-rate={sr:.3f}", flush=True)
    return history


def _test():
    print("[S3] self-play loop with a fixed-band proposer ...")
    hist = run_selfplay(n_iters=4, puzzles_per_iter=30, steps_per_iter=400, seed=0)
    print(f"[S3] solve-rate trajectory: {[round(h,3) for h in hist]}")
    assert hist[-1] > hist[0] + 0.03, "solver should improve across self-play iters"
    print("\n[S3 PASSED] the self-play loop runs and the solver improves over rounds.")


if __name__ == "__main__":
    _test()
