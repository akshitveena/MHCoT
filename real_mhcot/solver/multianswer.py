"""
solver/multianswer.py  —  THE MULTI-ANSWER TASK (no scalar collapse)
===================================================================

The test the scalar-readout experiments could never run: a task that REQUIRES
covering MULTIPLE valid answers, scored by coverage — so two differentiated
chains can win by spanning the answer set instead of being averaged away.

The task
--------
Given a solvable start state, several distinct FIRST MOVES keep it solvable.
Each first move leads to a distinct successor state; the "good" successors are
the solvable ones. The model proposes a ranked list of successors; we measure

    coverage@K = |good ∩ top-K proposals| / |good|

A model that proposes K genuinely DIVERSE good moves covers more of the set than
one that piles its K guesses onto the same region. This is exactly where two
epsilon-helix-separated chains (read out SEPARATELY, never summed) should help
over a single chain or a single real head — and where a real-2-head control
tells us whether any gain is "two outputs" or the helix itself.

This module provides only the WORLD side (targets + metric); the per-chain model
readouts live in model.py / the experiment runner.

Run the self-test:
    python solver/multianswer.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game24 import canon, reachable                     # noqa: E402
from search import successors                            # noqa: E402


def good_moves(state, target):
    """Return (all_successors, good_set) for a state:
       all_successors : list[state]  (distinct successor states, ranking universe)
       good_set       : set[state]   (the solvable successors = valid answers)
    """
    tgt = Fraction(target)
    succ = [ns for ns, _ in successors(canon(state))]
    good = {ns for ns in succ if reachable(ns, tgt)}
    return succ, good


def coverage_at_k(ranked_proposals, good_set, k):
    """Fraction of the good_set covered by the top-k distinct proposals.
    ranked_proposals: list[state] best-first. good_set: set[state]."""
    if not good_set:
        return None
    seen, topk = set(), []
    for s in ranked_proposals:                      # dedup, keep order
        if s not in seen:
            seen.add(s); topk.append(s)
        if len(topk) >= k:
            break
    hit = sum(1 for s in topk if s in good_set)
    return hit / len(good_set)


def union_topk(rankings, k):
    """Round-robin union of several rankings (one per chain/head) into a single
    top-k proposal list — the multi-hypothesis readout: take the best of chain 0,
    then chain 1, then chain 0's next, ... so diverse chains contribute distinct
    proposals. rankings: list[list[state]] (each best-first)."""
    out, seen, idx = [], set(), [0] * len(rankings)
    while len(out) < k:
        progressed = False
        for r, rank in enumerate(rankings):
            while idx[r] < len(rank) and rank[idx[r]] in seen:
                idx[r] += 1
            if idx[r] < len(rank):
                s = rank[idx[r]]; idx[r] += 1
                seen.add(s); out.append(s); progressed = True
                if len(out) >= k:
                    break
        if not progressed:
            break
    return out


def build_multianswer_set(puzzles):
    """For each SOLVABLE puzzle, the start state + its successor universe + good
    set. Returns list of dicts. (Only solvable starts have a good set.)"""
    out = []
    for p in puzzles:
        if not p.solvable:
            continue
        succ, good = good_moves(p.numbers, p.target)
        if good:                                    # skip degenerate (terminal) cases
            out.append({"state": canon(p.numbers), "target": Fraction(p.target),
                        "successors": succ, "good": good})
    return out


# ---------------------------------------------------------------------------
# Self-test: the oracle ranking must achieve full coverage; a diversity-blind
# ranking that repeats one region must cover less at small K.
# ---------------------------------------------------------------------------
def _test():
    from game24 import gen_puzzles

    print("[test] good-move sets exist and are multi-valued ...")
    pz = gen_puzzles(40, seed=3, solvable=True)
    data = build_multianswer_set(pz)
    sizes = [len(d["good"]) for d in data]
    multi = sum(1 for s in sizes if s >= 2)
    avg = sum(sizes) / len(sizes)
    print(f"    ok — {len(data)} solvable starts, avg {avg:.1f} good moves, "
          f"{multi} have >=2 (genuinely multi-answer)")
    assert multi > 0, "need multi-answer puzzles for this to be meaningful"

    print("[test] oracle ranking (good first) -> full coverage at K=|good| ...")
    for d in data[:10]:
        good = d["good"]
        # oracle: put good successors first
        ranked = list(good) + [s for s in d["successors"] if s not in good]
        cov = coverage_at_k(ranked, good, k=len(good))
        assert cov == 1.0, f"oracle should fully cover, got {cov}"
    print("    ok — oracle covers 100% at K=|good|")

    print("[test] union_topk diversifies across chains ...")
    # two rankings that each rank a DIFFERENT good move first
    d = next(dd for dd in data if len(dd["good"]) >= 2)
    g = list(d["good"])
    rank_a = [g[0]] + [s for s in d["successors"] if s != g[0]]
    rank_b = [g[1]] + [s for s in d["successors"] if s != g[1]]
    # single ranking A at K=1 covers 1/|good|; union of A,B at K=2 covers 2/|good|
    cov_single = coverage_at_k(rank_a, d["good"], k=2)
    cov_union = coverage_at_k(union_topk([rank_a, rank_b], 2), d["good"], k=2)
    print(f"    ok — single-ranking cov@2={cov_single:.2f}, "
          f"diverse-union cov@2={cov_union:.2f}")
    assert cov_union >= cov_single, "diverse union should cover at least as much"

    print("\n[all multianswer tests passed]")


if __name__ == "__main__":
    _test()
