"""
selfplay/solver.py  —  S1: a learnable solver with a measurable solve-rate
==========================================================================

The SOLVER in the self-play loop: a small value net scores Game-24 states
(≈ P(this state can reach 24)) and guides best-first search. It LEARNS from
labeled states, so its solve-rate-at-budget improves with training — the
prerequisite for self-play to have something to improve.

Reuses solver/model.py (RealSolver) + solver/search.py (label_states, search).

Run the self-test:
    python selfplay/solver.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent / "main"))
sys.path.insert(0, str(_ROOT / "solver"))   # solver FIRST: solver/model.py, not main/model.py
from model import RealSolver, make_heuristic                 # noqa: E402
from search import label_states, best_first_search           # noqa: E402
from game24 import gen_puzzles                                # noqa: E402


def new_solver(d_model=64):
    return RealSolver(d_in=None, d_model=d_model) if False else RealSolver(d_model=d_model)


def train_solver(model, puzzles, dev="cpu", steps=600, bs=64, lr=3e-4, seed=0):
    """Train the solver's value net on exact (state -> solvable) labels from the
    given puzzles' reachable states. Balanced batches (solvable states are rare)."""
    labeled = label_states(puzzles)
    states = [s for s, _, _ in labeled]
    targets = [t for _, t, _ in labeled]
    labels = [l for _, _, l in labeled]
    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]
    if not pos or not neg:
        return model, 0.0
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    half = bs // 2
    for _ in range(steps):
        idx = ([rng.choice(pos) for _ in range(half)]
               + [rng.choice(neg) for _ in range(bs - half)])
        rng.shuffle(idx)
        bs_ = [states[i] for i in idx]
        bt = [targets[i] for i in idx]
        by = torch.tensor([float(labels[i]) for i in idx], device=dev)
        out = model(bs_, bt)["score"]
        loss = F.binary_cross_entropy_with_logits(out, by)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model, float(loss.item())


@torch.no_grad()
def solve_rate(model, puzzles, budget=50, dev="cpu"):
    """Fraction of SOLVABLE puzzles the solver cracks within `budget` expansions."""
    model.eval()
    h = make_heuristic(model, device=dev)
    solvable = [p for p in puzzles if p.solvable]
    if not solvable:
        return 0.0
    return sum(best_first_search(p.numbers, p.target, h, node_budget=budget).solved
               for p in solvable) / len(solvable)


# ---------------------------------------------------------------------------
# S1 self-test
# ---------------------------------------------------------------------------
@torch.no_grad()
def heuristic_auc(model, puzzles, dev="cpu"):
    from sklearn.metrics import roc_auc_score
    labeled = label_states(puzzles)
    states = [s for s, _, _ in labeled]; targets = [t for _, t, _ in labeled]
    y = np.array([l for _, _, l in labeled])
    if len(set(y.tolist())) < 2:
        return float("nan")
    s = []
    for i in range(0, len(states), 256):
        s.append(model(states[i:i+256], targets[i:i+256])["score"].cpu().numpy())
    return roc_auc_score(y, np.concatenate(s))


def _test():
    dev = "cpu"
    torch.manual_seed(0); np.random.seed(0)
    train_pz = gen_puzzles(300, seed=0, solvable=True)
    test_pz = gen_puzzles(80, seed=1, solvable=True)
    BUDGET = 100

    model = RealSolver(d_model=64).to(dev)
    print("[S1] untrained solver ...")
    sr0 = solve_rate(model, test_pz, budget=BUDGET, dev=dev)
    print(f"    untrained: solve-rate @ {BUDGET} = {sr0:.3f}, "
          f"heuristic AUC = {heuristic_auc(model, test_pz):.3f}")

    print("[S1] training solver (2000 steps) ...")
    model, last = train_solver(model, train_pz, dev=dev, steps=2000, seed=0)
    auc = heuristic_auc(model, test_pz)
    sr1 = solve_rate(model, test_pz, budget=BUDGET, dev=dev)
    print(f"    trained: solve-rate @ {BUDGET} = {sr1:.3f}, heuristic AUC = {auc:.3f} "
          f"(loss {last:.3f})")

    print(f"[S1] improvement: solve-rate {sr0:.3f} -> {sr1:.3f}, AUC -> {auc:.3f}")
    assert auc > 0.7, "heuristic should learn to rank solvable states (AUC>0.7)"
    assert sr1 > sr0 + 0.05, "training should improve solve-rate"
    print("\n[S1 PASSED] solver is learnable; heuristic discriminates and solve-rate improves.")


if __name__ == "__main__":
    _test()
