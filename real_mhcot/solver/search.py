"""
solver/search.py  —  THE SEARCH MDP + HARNESS (no model)
========================================================

Turns Game-24 into a search problem and provides the evaluation harness that a
learned heuristic plugs into. No model lives here: the heuristic is any callable
`state -> float` (higher = "more promising"). The same harness drives the real,
N=1 and N=2 heuristics so the comparison is apples-to-apples.

State / transition
------------------
  state       = canonical sorted tuple of Fractions (from game24.canon)
  successors  = every (state', action) from picking two numbers and one op
  terminal    = len(state) == 1; WIN iff that number == target

What this module gives the rest of the pipeline:
  * successors(state)                  the transition function
  * best_first_search / beam_search    budget-limited search guided by a heuristic
  * label_states(...)                  exact (state, solvable) pairs for training
                                       (the supervised signal, from game24.reachable)

Run the self-test (uses the exact oracle as a perfect heuristic):
    python solver/search.py
"""

from __future__ import annotations

import heapq
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game24 import canon, reachable, gen_puzzles  # noqa: E402

State = tuple
Heuristic = Callable[[State, Fraction], float]


# ---------------------------------------------------------------------------
# Transition function
# ---------------------------------------------------------------------------
def successors(state: State) -> list:
    """All (next_state, action_str) reachable in one move. Deduplicated by
    next_state so the search tree doesn't re-explore identical multisets."""
    out, seen = [], set()
    n = len(state)
    for i, j in combinations(range(n), 2):
        a, b = state[i], state[j]
        rest = tuple(state[k] for k in range(n) if k != i and k != j)
        combos = [(a + b, f"{a}+{b}"), (a * b, f"{a}*{b}"),
                  (a - b, f"{a}-{b}"), (b - a, f"{b}-{a}")]
        if b != 0:
            combos.append((a / b, f"{a}/{b}"))
        if a != 0:
            combos.append((b / a, f"{b}/{a}"))
        for val, act in combos:
            ns = canon(rest + (val,))
            if ns not in seen:
                seen.add(ns)
                out.append((ns, act))
    return out


def is_win(state: State, target: Fraction) -> bool:
    return len(state) == 1 and state[0] == target


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------
class SearchResult:
    __slots__ = ("solved", "nodes_expanded", "path")

    def __init__(self, solved: bool, nodes_expanded: int, path: list | None):
        self.solved = solved
        self.nodes_expanded = nodes_expanded
        self.path = path or []

    def __repr__(self):
        return (f"SearchResult(solved={self.solved}, "
                f"nodes={self.nodes_expanded}, depth={len(self.path)})")


# ---------------------------------------------------------------------------
# Best-first search guided by a heuristic
# ---------------------------------------------------------------------------
def best_first_search(start, target, heuristic: Heuristic,
                      node_budget: int = 50) -> SearchResult:
    """Greedy best-first: always expand the most promising frontier state
    (by `heuristic`). Stops at a win or when `node_budget` expansions are used.
    Returns whether it solved and how many nodes it had to expand — the latter
    is the search-efficiency metric the comparison turns on.

    `heuristic(state, target) -> float`, higher = more promising.
    """
    start = canon(start)
    target = Fraction(target)
    counter = 0  # tie-break / heap stability
    # max-heap via negated score
    frontier = [(-heuristic(start, target), counter, start, [])]
    visited = set()
    nodes = 0
    while frontier and nodes < node_budget:
        _, _, state, path = heapq.heappop(frontier)
        if state in visited:
            continue
        visited.add(state)
        nodes += 1
        if is_win(state, target):
            return SearchResult(True, nodes, path)
        for ns, act in successors(state):
            if ns in visited:
                continue
            counter += 1
            heapq.heappush(frontier,
                           (-heuristic(ns, target), counter, ns, path + [act]))
    return SearchResult(False, nodes, None)


def beam_search(start, target, heuristic: Heuristic,
                beam_width: int = 3, node_budget: int = 50) -> SearchResult:
    """Layer-by-layer beam search: at each depth keep the `beam_width` best
    states by `heuristic`. A diverse, multi-hypothesis heuristic should fill the
    beam with genuinely different promising branches — exactly where N=2's
    anti-collapse is supposed to help over a single ranking."""
    start = canon(start)
    target = Fraction(target)
    beam = [(start, [])]
    nodes = 0
    while beam and nodes < node_budget:
        # check wins in the current beam
        for state, path in beam:
            if is_win(state, target):
                return SearchResult(True, nodes, path)
        # expand the whole beam
        cand = []
        for state, path in beam:
            nodes += 1
            if nodes > node_budget:
                break
            for ns, act in successors(state):
                cand.append((ns, path + [act]))
        if not cand:
            break
        # terminal win produced this layer?
        for ns, path in cand:
            if is_win(ns, target):
                return SearchResult(True, nodes, path)
        # keep the best `beam_width` by heuristic
        cand.sort(key=lambda sp: -heuristic(sp[0], target))
        beam = cand[:beam_width]
    return SearchResult(False, nodes, None)


# ---------------------------------------------------------------------------
# Training-label generation: exact (state, solvable) pairs
# ---------------------------------------------------------------------------
def reachable_states(start, max_states: int | None = None) -> set:
    """All distinct states reachable from `start` by repeated combination
    (the full search tree, deduplicated). These are the states a search would
    encounter and must score."""
    start = canon(start)
    seen = {start}
    stack = [start]
    while stack:
        s = stack.pop()
        if len(s) == 1:
            continue
        for ns, _ in successors(s):
            if ns not in seen:
                seen.add(ns)
                stack.append(ns)
                if max_states and len(seen) >= max_states:
                    return seen
    return seen


def label_states(puzzles: Iterable, max_per_puzzle: int | None = None) -> list:
    """For each puzzle, enumerate reachable intermediate states and label each
    with the EXACT solvability oracle. Returns [(state, target, solvable_int)].
    This is the supervised training set for the heuristic — perfect labels,
    no LLM, no noise."""
    out, seen = [], set()
    for p in puzzles:
        tgt = Fraction(p.target)
        for st in reachable_states(p.numbers, max_per_puzzle):
            key = (st, tgt)
            if key in seen:
                continue
            seen.add(key)
            out.append((st, tgt, int(reachable(st, tgt))))
    return out


# ---------------------------------------------------------------------------
# Self-test: with the EXACT oracle as the heuristic, search must solve every
# solvable puzzle with minimal expansions; with a blind heuristic it expands more.
# ---------------------------------------------------------------------------
def _oracle_heuristic(state, target) -> float:
    return 1.0 if reachable(state, target) else 0.0


def _blind_heuristic(state, target) -> float:
    return 0.0  # no guidance -> essentially BFS order


def _test():
    print("[test] successors / terminal ...")
    s = canon([4, 6, 8, 2])
    succ = successors(s)
    assert all(len(ns) == 3 for ns, _ in succ), "one move drops one number"
    assert len({ns for ns, _ in succ}) == len(succ), "successors deduplicated"
    assert is_win(canon([24]), Fraction(24))
    assert not is_win(canon([24, 1]), Fraction(24))
    print(f"    ok — {len(succ)} unique successors from a 4-number state")

    print("[test] oracle-guided best-first solves solvable puzzles cheaply ...")
    solvable = gen_puzzles(30, seed=1, solvable=True)
    nodes_oracle = []
    for p in solvable:
        r = best_first_search(p.numbers, p.target, _oracle_heuristic, node_budget=50)
        assert r.solved, f"oracle heuristic failed on {p}"
        nodes_oracle.append(r.nodes_expanded)
    avg_oracle = sum(nodes_oracle) / len(nodes_oracle)
    print(f"    ok — solved 30/30, avg nodes expanded = {avg_oracle:.1f}")

    print("[test] blind search needs more nodes (guidance matters) ...")
    nodes_blind, solved_blind = [], 0
    for p in solvable:
        r = best_first_search(p.numbers, p.target, _blind_heuristic, node_budget=50)
        solved_blind += r.solved
        nodes_blind.append(r.nodes_expanded)
    avg_blind = sum(nodes_blind) / len(nodes_blind)
    print(f"    ok — blind solved {solved_blind}/30, avg nodes = {avg_blind:.1f} "
          f"(oracle {avg_oracle:.1f}) -> heuristic quality changes efficiency")
    assert avg_blind > avg_oracle, "a good heuristic should reduce expansions"

    print("[test] beam search with oracle ...")
    bs_solved = sum(beam_search(p.numbers, p.target, _oracle_heuristic,
                                beam_width=3, node_budget=50).solved
                    for p in solvable)
    print(f"    ok — beam(width=3) solved {bs_solved}/30 with oracle")

    print("[test] exact label generation ...")
    labels = label_states(solvable[:5])
    assert all(lab in (0, 1) for _, _, lab in labels)
    # the start states are solvable by construction -> labeled 1
    starts = {canon(p.numbers) for p in solvable[:5]}
    for st, tgt, lab in labels:
        if st in starts:
            assert lab == 1
    pos = sum(lab for _, _, lab in labels)
    print(f"    ok — {len(labels)} labeled states, {pos} solvable "
          f"({pos/len(labels):.0%}) — a real classification signal")

    print("\n[all search tests passed]")


if __name__ == "__main__":
    _test()
