"""
solver/game24.py  —  THE WORLD (no model)
=========================================

Game of 24 (and its harder generalizations) as an EXACT symbolic problem.

Everything here is ground truth: puzzles and their solvability are computed by
brute force over exact rational arithmetic (`fractions.Fraction`), so division
never introduces float error and labels are perfect. This is the whole point of
Option B — the data is free, exact, and needs no LLM.

The game
--------
Start with k numbers (Game-24: k=4, numbers 1..13). Repeatedly pick two numbers
and combine them with +, -, *, / into one number (k -> k-1). You win if the last
remaining number equals the target (Game-24: 24).

Generalizations (to make search actually hard — see search.py):
  * more numbers (k=5, 6)        -> deeper / wider tree
  * arbitrary target (Countdown) -> fewer solution paths

Public API
----------
reachable(numbers, target)      -> bool      exact oracle
solve(numbers, target)          -> str|None  one human-readable solution
gen_puzzles(...)                -> list[Puzzle]
make_split(...)                 -> (train, test) disjoint puzzle lists

Run the self-test:
    python solver/game24.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Optional

Number = Fraction
State = tuple  # sorted tuple[Fraction, ...]


# ---------------------------------------------------------------------------
# Exact arithmetic core
# ---------------------------------------------------------------------------
def _combine(a: Fraction, b: Fraction):
    """All numbers obtainable from a, b with a single operation, each tagged
    with how it was produced (for solution reconstruction)."""
    out = [(a + b, f"({a}+{b})"), (a * b, f"({a}*{b})"),
           (a - b, f"({a}-{b})"), (b - a, f"({b}-{a})")]
    if b != 0:
        out.append((a / b, f"({a}/{b})"))
    if a != 0:
        out.append((b / a, f"({b}/{a})"))
    return out


def canon(numbers) -> State:
    """Canonical state: a sorted tuple of Fractions (order must not matter)."""
    return tuple(sorted(Fraction(x) for x in numbers))


def reachable(numbers, target, _memo: Optional[dict] = None) -> bool:
    """EXACT oracle: can `numbers` be combined down to `target`?"""
    state = canon(numbers)
    tgt = Fraction(target)
    if _memo is None:
        _memo = {}
    return _reachable(state, tgt, _memo)


def _reachable(state: State, tgt: Fraction, memo: dict) -> bool:
    if len(state) == 1:
        return state[0] == tgt
    if state in memo:
        return memo[state]
    res = False
    n = len(state)
    for i, j in combinations(range(n), 2):
        a, b = state[i], state[j]
        rest = tuple(state[k] for k in range(n) if k != i and k != j)
        for val, _ in _combine(a, b):
            if _reachable(canon(rest + (val,)), tgt, memo):
                res = True
                break
        if res:
            break
    memo[state] = res
    return res


def solve(numbers, target) -> Optional[str]:
    """Return ONE solution expression string, or None if unsolvable."""
    state = canon(numbers)
    tgt = Fraction(target)
    exprs = tuple(_fmt(x) for x in state)
    return _solve(state, exprs, tgt)


def _solve(state: State, exprs: tuple, tgt: Fraction) -> Optional[str]:
    if len(state) == 1:
        return exprs[0] if state[0] == tgt else None
    n = len(state)
    for i, j in combinations(range(n), 2):
        a, b = state[i], state[j]
        ea, eb = exprs[i], exprs[j]
        rest = [state[k] for k in range(n) if k != i and k != j]
        rest_e = [exprs[k] for k in range(n) if k != i and k != j]
        combos = [(a + b, f"({ea}+{eb})"), (a * b, f"({ea}*{eb})"),
                  (a - b, f"({ea}-{eb})"), (b - a, f"({eb}-{ea})")]
        if b != 0:
            combos.append((a / b, f"({ea}/{eb})"))
        if a != 0:
            combos.append((b / a, f"({eb}/{ea})"))
        for val, expr in combos:
            # keep state sorted but carry expressions alongside in the same order
            new_pairs = sorted(zip(rest + [val], rest_e + [expr]),
                               key=lambda p: p[0])
            ns = tuple(p[0] for p in new_pairs)
            ne = tuple(p[1] for p in new_pairs)
            got = _solve(ns, ne, tgt)
            if got is not None:
                return got
    return None


def _fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# ---------------------------------------------------------------------------
# Puzzles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Puzzle:
    numbers: tuple          # tuple[int, ...] the starting numbers
    target: int
    solvable: bool

    def __str__(self) -> str:
        nums = ", ".join(str(n) for n in self.numbers)
        return f"Puzzle({nums} -> {self.target}, {'solvable' if self.solvable else 'NO'})"


def gen_puzzles(count: int, k: int = 4, lo: int = 1, hi: int = 13,
                target: int = 24, seed: int = 0,
                solvable: Optional[bool] = None,
                unique: bool = True) -> list:
    """Generate `count` puzzles of k numbers in [lo, hi] for `target`.

    solvable=None  -> natural mix
    solvable=True  -> only solvable puzzles
    solvable=False -> only unsolvable puzzles
    """
    rng = random.Random(seed)
    out, seen, tries = [], set(), 0
    max_tries = count * 2000 + 10000
    while len(out) < count and tries < max_tries:
        tries += 1
        nums = tuple(sorted(rng.randint(lo, hi) for _ in range(k)))
        if unique and nums in seen:
            continue
        s = reachable(nums, target)
        if solvable is not None and s != solvable:
            continue
        seen.add(nums)
        out.append(Puzzle(numbers=nums, target=target, solvable=s))
    return out


def make_split(n_train: int, n_test: int, k: int = 4, lo: int = 1, hi: int = 13,
               target: int = 24, seed: int = 0,
               balance: bool = True) -> tuple:
    """Disjoint train/test puzzle sets. If balance, ~half solvable / half not."""
    if balance:
        half_tr, half_te = n_train // 2, n_test // 2
        pos = gen_puzzles(half_tr + half_te, k, lo, hi, target, seed, solvable=True)
        neg = gen_puzzles(half_tr + half_te, k, lo, hi, target, seed + 1, solvable=False)
        allp = pos + neg
        rng = random.Random(seed + 99)
        rng.shuffle(allp)
    else:
        allp = gen_puzzles(n_train + n_test, k, lo, hi, target, seed)
        rng = random.Random(seed + 99)
        rng.shuffle(allp)
    # de-dup across the split by number tuple
    seen, dedup = set(), []
    for p in allp:
        if p.numbers in seen:
            continue
        seen.add(p.numbers)
        dedup.append(p)
    train = dedup[:n_train]
    test = dedup[n_train:n_train + n_test]
    return train, test


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _test():
    print("[test] exact arithmetic / known cases ...")
    # classic solvable
    assert reachable([4, 6, 8, 2], 24), "(4,6,8,2) is solvable"
    assert reachable([1, 3, 4, 6], 24), "6/(1-3/4)=24"      # famous hard one
    assert reachable([8, 8, 3, 3], 24), "8/(3-8/3)=24 needs exact division"
    # known unsolvable
    assert not reachable([1, 1, 1, 1], 24), "(1,1,1,1) cannot make 24"
    assert not reachable([1, 1, 1, 2], 24)
    print("    ok — solvable/unsolvable oracle correct (incl. exact division)")

    print("[test] solution reconstruction ...")
    for nums in ([4, 6, 8, 2], [1, 3, 4, 6], [8, 8, 3, 3]):
        expr = solve(nums, 24)
        assert expr is not None, f"should solve {nums}"
        # verify the returned expression actually evaluates to 24 (exactly)
        val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - our own strings
        assert Fraction(val).limit_denominator(10**6) == 24, f"{expr} != 24"
        print(f"    {tuple(nums)} -> {expr} = 24")
    assert solve([1, 1, 1, 1], 24) is None
    print("    ok — solutions reconstructed and verified")

    print("[test] permutation invariance ...")
    assert canon([2, 8, 6, 4]) == canon([4, 6, 8, 2])
    assert reachable([2, 8, 6, 4], 24) == reachable([4, 6, 8, 2], 24)
    print("    ok — state is order-independent")

    print("[test] puzzle generation + balanced split ...")
    tr, te = make_split(n_train=40, n_test=20, seed=0, balance=True)
    assert len(tr) == 40 and len(te) == 20
    tr_nums = {p.numbers for p in tr}
    te_nums = {p.numbers for p in te}
    assert tr_nums.isdisjoint(te_nums), "train/test must be disjoint"
    frac_solv = sum(p.solvable for p in tr) / len(tr)
    # every generated solvable flag must match the oracle
    for p in tr + te:
        assert p.solvable == reachable(p.numbers, p.target)
    print(f"    ok — disjoint split, train solvable fraction = {frac_solv:.2f}")
    print(f"    sample: {tr[0]}  |  {tr[1]}")

    print("\n[all game24 tests passed]")


if __name__ == "__main__":
    _test()
