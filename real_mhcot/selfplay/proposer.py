"""
selfplay/proposer.py  —  S2: a proposer that controls difficulty
================================================================

The PROPOSER in the self-play loop. S2 establishes the prerequisite for a
curriculum: the proposer can deliberately produce easy / medium / hard puzzles
(by our difficulty proxy), and that difficulty actually governs how hard the
puzzle is to solve. The LEARNED, edge-seeking proposer comes in S4 — here we
prove difficulty is controllable and meaningful.

Only depends on env.py (which owns propose/verify/difficulty/solve).

Run the self-test:
    python selfplay/proposer.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import (propose_random, verify, difficulty, solve_with,        # noqa: E402
                 blind_heuristic, TARGET)


class DifficultyProposer:
    """Proposes solvable puzzles within a target difficulty band by rejection
    sampling on the difficulty proxy. (S4 will replace this with a learned,
    edge-seeking proposer.)"""

    def propose_at(self, rng, lo, hi, max_tries=2000):
        for _ in range(max_tries):
            p = propose_random(rng)
            if not verify(p):
                continue
            d = difficulty(p)
            if d is not None and lo <= d <= hi:
                return p, d
        return None, None

    def propose_band(self, rng, lo, hi, n):
        out = []
        while len(out) < n:
            p, d = self.propose_at(rng, lo, hi)
            if p is None:
                break
            out.append((p, d))
        return out


# ---------------------------------------------------------------------------
# S2 self-test
# ---------------------------------------------------------------------------
def _avg_blind_nodes(puzzles, budget):
    res = [solve_with(blind_heuristic, p, budget=budget) for p in puzzles]
    solved = [n for ok, n in res if ok]
    rate = len(solved) / len(res)
    nodes = sum(n for _, n in res) / len(res)
    return rate, nodes


def _test():
    rng = random.Random(0)
    prop = DifficultyProposer()

    print("[S2] proposer can target difficulty bands ...")
    easy = prop.propose_band(rng, 0.20, 0.45, 50)
    hard = prop.propose_band(rng, 0.80, 1.00, 50)
    de = sum(d for _, d in easy) / len(easy)
    dh = sum(d for _, d in hard) / len(hard)
    print(f"    easy band: {len(easy)} puzzles, mean difficulty {de:.2f}")
    print(f"    hard band: {len(hard)} puzzles, mean difficulty {dh:.2f}")
    assert dh - de > 0.3, "bands should be clearly separated in difficulty"

    print("[S2] proposed difficulty governs actual solve effort ...")
    re_, ne = _avg_blind_nodes([p for p, _ in easy], budget=400)
    rh, nh = _avg_blind_nodes([p for p, _ in hard], budget=400)
    print(f"    blind solver — easy: solve-rate {re_:.2f}, nodes {ne:.0f}")
    print(f"    blind solver — hard: solve-rate {rh:.2f}, nodes {nh:.0f}")
    assert re_ >= rh and ne <= nh, "harder band should be harder to solve"

    print("\n[S2 PASSED] proposer controls difficulty, and difficulty drives "
          "real solve effort — the curriculum lever works.")


if __name__ == "__main__":
    _test()
