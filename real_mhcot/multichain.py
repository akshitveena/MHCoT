"""
real_mhcot/multichain.py  —  R2: the multi-chain thought encoder (real-valued)
==============================================================================

Your multi-helical idea, kept but real-valued: N branches ("chains") encode the
thought, INTERACT (real cross-branch attention — the honest real-space analog of
"interference"), and combine into one embedding. We test, head-to-head and
param-aware, whether multi-chain beats the single-branch encoder (R1) on thought
verification. Honest prior: multi-chain has tied single elsewhere — we let the
numbers decide on this task.

Run the self-test:
    python real_mhcot/multichain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_thoughts, split_by_problem                    # noqa: E402
from encoder import ThoughtEncoder, train_contrastive, verification_auc  # noqa: E402


class MultiChainThoughtEncoder(nn.Module):
    """N real branches encode the thought; a cross-branch attention lets them
    interact; the interacted branches are pooled into one L2-normalized embedding."""
    def __init__(self, d_in=1536, d=128, n_chains=2, n_heads=4):
        super().__init__()
        self.n_chains = n_chains
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, d))
            for _ in range(n_chains))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = torch.stack([b(x) for b in self.branches], dim=1)   # (B, N, d)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)                                     # chains interact
        return F.normalize(h.mean(1), dim=-1)                    # combined embedding


def _count(m):
    return sum(p.numel() for p in m.parameters())


def _eval(model, Xtr, ytr, Xte, yte, dev, steps, seed):
    train_contrastive(model, Xtr, ytr, dev=dev, steps=steps, seed=seed)
    return verification_auc(model, Xtr, ytr, Xte, yte, dev=dev)


def _test():
    dev = "cpu"
    X, y, pid = load_thoughts("lastk")
    seeds = [0, 1, 2]
    single_aucs, multi_aucs = [], []
    for s in seeds:
        (Xtr, ytr, _), (Xte, yte, _) = split_by_problem(X, y, pid, seed=s)
        torch.manual_seed(s)
        single = ThoughtEncoder(d=128).to(dev)
        ap, _ = _eval(single, Xtr, ytr, Xte, yte, dev, 2000, s)
        torch.manual_seed(s)
        multi = MultiChainThoughtEncoder(d=128, n_chains=2).to(dev)
        mp, _ = _eval(multi, Xtr, ytr, Xte, yte, dev, 2000, s)
        single_aucs.append(ap); multi_aucs.append(mp)
        print(f"    seed{s}: single {ap:.3f}  multi-chain {mp:.3f}  "
              f"(params {_count(single)/1e3:.0f}K vs {_count(multi)/1e3:.0f}K)", flush=True)

    import numpy as np
    sm, mm = np.mean(single_aucs), np.mean(multi_aucs)
    print(f"\n[R2] single {sm:.3f}±{np.std(single_aucs):.3f}  |  "
          f"multi-chain {mm:.3f}±{np.std(multi_aucs):.3f}")
    if mm > sm + 0.01:
        print(f"[R2] multi-chain BEATS single ({mm:.3f} vs {sm:.3f}) — the real "
              f"multi-branch interaction adds value on thought verification.")
    elif abs(mm - sm) <= 0.01:
        print(f"[R2] multi-chain ≈ single ({mm:.3f} vs {sm:.3f}) — no advantage "
              f"(consistent with the multi-chain prior); single-branch is enough.")
    else:
        print(f"[R2] multi-chain trails single ({mm:.3f} vs {sm:.3f}).")
    print("[R2 DONE] honest head-to-head recorded.")


if __name__ == "__main__":
    _test()
