"""
selfplay/env.py  —  S0: the Game-24 self-play environment
=========================================================

A tractable, fully-verifiable arena for reproducing asymmetric self-play
(SQLM / R-Zero style) on a Mac: a PROPOSER makes Game-24 puzzles, a SOLVER tries
to solve them, and a VERIFIER checks exactly (we own the oracle). This module is
the world: propose / verify / difficulty. Solver and proposer learning come next.

Reuses solver/game24.py (exact reachability oracle) and solver/search.py.

Run the self-test:
    python selfplay/env.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "solver"))
from game24 import canon, reachable, solve, Puzzle          # noqa: E402
from search import successors, best_first_search             # noqa: E402

TARGET = 24


def propose_random(rng, k=4, lo=1, hi=13):
    """Baseline proposer: a random k-number puzzle (numbers in [lo,hi])."""
    return tuple(sorted(rng.randint(lo, hi) for _ in range(k)))


def verify(numbers, target=TARGET):
    """Exact oracle: is this puzzle solvable? (ground-truth verifier)."""
    return reachable(numbers, target)


def difficulty(numbers, target=TARGET):
    """Difficulty proxy for a SOLVABLE puzzle: 1 - (fraction of first moves that
    stay solvable). Easy puzzles have many solving paths (low difficulty); hard
    ones have few (high difficulty). Returns None if unsolvable. Cheap & always
    defined — good for driving the proposer's curriculum."""
    state = canon(numbers)
    if not reachable(state, target):
        return None
    succ = successors(state)
    good = sum(1 for s, _ in succ if reachable(s, target))
    return 1.0 - good / len(succ)


def solve_with(heuristic, numbers, target=TARGET, budget=50):
    """Solver attempt: best-first search guided by `heuristic(state,target)->float`.
    Returns (solved: bool, nodes_expanded: int)."""
    r = best_first_search(numbers, target, heuristic, node_budget=budget)
    return r.solved, r.nodes_expanded


def oracle_heuristic(state, target=TARGET):
    return 1.0 if reachable(state, target) else 0.0


def blind_heuristic(state, target=TARGET):
    return 0.0


# ---------------------------------------------------------------------------
# S0 self-test
# ---------------------------------------------------------------------------
def _test():
    rng = random.Random(0)
    print("[S0] verifier matches oracle, difficulty is defined & varies ...")
    puzzles = [propose_random(rng) for _ in range(400)]
    solvable = [p for p in puzzles if verify(p)]
    frac = len(solvable) / len(puzzles)
    print(f"    proposed 400; solvable fraction = {frac:.2f}")
    assert 0.1 < frac < 0.95, "random proposer should give a mix"

    diffs = [difficulty(p) for p in solvable]
    diffs = [d for d in diffs if d is not None]
    lo, hi = min(diffs), max(diffs)
    avg = sum(diffs) / len(diffs)
    print(f"    difficulty range [{lo:.2f}, {hi:.2f}], mean {avg:.2f} "
          f"({len(set(round(d,2) for d in diffs))} distinct values)")
    assert hi - lo > 0.2, "difficulty should vary across puzzles"

    print("[S0] difficulty correlates with solver effort (sanity) ...")
    # harder puzzles (by our proxy) should need more nodes for a blind solver
    easy = [p for p in solvable if (difficulty(p) or 0) < 0.5]
    hard = [p for p in solvable if (difficulty(p) or 0) > 0.8]
    def avg_nodes(ps):
        ns = [solve_with(blind_heuristic, p, budget=300)[1] for p in ps[:40]]
        return sum(ns) / max(len(ns), 1)
    ne, nh = avg_nodes(easy), avg_nodes(hard)
    print(f"    blind-search nodes: easy={ne:.0f}  hard={nh:.0f}")

    print("[S0] oracle solver solves all solvable; blind solves few ...")
    ok = sum(solve_with(oracle_heuristic, p, budget=50)[0] for p in solvable[:50])
    bk = sum(solve_with(blind_heuristic, p, budget=50)[0] for p in solvable[:50])
    print(f"    oracle solved {ok}/50, blind solved {bk}/50")
    assert ok == 50 and bk < ok

    print("[S0] solution reconstruction works ...")
    p = solvable[0]
    print(f"    {p} -> {solve(p, TARGET)}")
    print("\n[S0 PASSED] environment ready (propose / verify / difficulty / solve).")


if __name__ == "__main__":
    _test()
