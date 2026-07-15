"""
selfplay/verifier.py  —  S5: the learned verifier (the TBERT / PRM slot)
========================================================================

In real self-play there is no oracle — a LEARNED verifier provides the reward
signal. This is where TBERT/PRM plugs into the architecture: a model that judges
a puzzle/answer instead of the exact checker. Here we train a verifier to predict
solvability from the puzzle and measure its accuracy against the oracle; S6 then
runs the loop on the learned signal.

(The verifier is a classifier here — the standard PRM form. It can be swapped for
a contrastive Siamese embedder — "TBERT" proper — without changing its role.)

Run the self-test:
    python selfplay/verifier.py
"""

from __future__ import annotations

import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "main"))
sys.path.insert(0, str(_HERE.parent / "solver"))
sys.path.insert(0, str(_HERE))
from Projects.MHCoT.real_mhcot.selfplay.loop import _load_agent                                    # noqa: E402
from game24 import gen_puzzles, canon                           # noqa: E402


def balanced_puzzles(n_each, seed):
    pos = gen_puzzles(n_each, seed=seed, solvable=True)
    neg = gen_puzzles(n_each, seed=seed + 7, solvable=False)
    allp = pos + neg
    random.Random(seed + 1).shuffle(allp)
    return allp


def train_verifier(model, puzzles, dev="cpu", steps=1500, bs=64, lr=3e-4, seed=0):
    """Train a verifier to predict puzzle solvability from the START state."""
    states = [canon(p.numbers) for p in puzzles]
    targets = [Fraction(p.target) for p in puzzles]
    labels = [int(p.solvable) for p in puzzles]
    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train(); half = bs // 2
    for _ in range(steps):
        idx = ([rng.choice(pos) for _ in range(half)]
               + [rng.choice(neg) for _ in range(bs - half)])
        rng.shuffle(idx)
        out = model([states[i] for i in idx], [targets[i] for i in idx])["score"]
        by = torch.tensor([float(labels[i]) for i in idx], device=dev)
        loss = F.binary_cross_entropy_with_logits(out, by)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def verifier_eval(model, puzzles, dev="cpu"):
    from sklearn.metrics import roc_auc_score
    model.eval()
    states = [canon(p.numbers) for p in puzzles]
    targets = [Fraction(p.target) for p in puzzles]
    y = np.array([int(p.solvable) for p in puzzles])
    s = []
    for i in range(0, len(states), 256):
        s.append(model(states[i:i+256], targets[i:i+256])["score"].cpu().numpy())
    s = np.concatenate(s)
    pred = (1 / (1 + np.exp(-s))) > 0.5
    acc = (pred == y).mean()
    auc = roc_auc_score(y, s) if len(set(y.tolist())) > 1 else float("nan")
    return float(acc), float(auc)


def _test():
    dev = "cpu"
    torch.manual_seed(0); np.random.seed(0)
    agent = _load_agent()
    train_pz = balanced_puzzles(400, seed=0)
    test_pz = balanced_puzzles(150, seed=1)

    verifier = agent.RealSolver(d_model=64).to(dev)
    print("[S5] untrained verifier ...")
    a0, u0 = verifier_eval(verifier, test_pz, dev)
    print(f"    untrained: acc={a0:.3f}  AUC={u0:.3f}")

    print("[S5] training learned verifier (predict solvability, no oracle) ...")
    train_verifier(verifier, train_pz, dev=dev, steps=1500, seed=0)
    a1, u1 = verifier_eval(verifier, test_pz, dev)
    print(f"    trained:   acc={a1:.3f}  AUC={u1:.3f}")

    print(f"[S5] verifier vs oracle: acc {a0:.3f} -> {a1:.3f}, AUC {u0:.3f} -> {u1:.3f}")
    assert a1 > 0.70 and u1 > 0.75, "learned verifier should agree well with the oracle"
    print("\n[S5 PASSED] learned verifier predicts correctness without the oracle — "
          "the TBERT/PRM reward slot works.")


if __name__ == "__main__":
    _test()
