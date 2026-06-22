"""
real_mhcot/data.py  —  R0: thoughts as data + the baseline bar
==============================================================

Loads GSM8K Graph-of-Thoughts candidates as THOUGHTS: each candidate is a real
ℝ¹⁵³⁶ vector (answer-region 'lastk' pooling) with a correctness label and a
problem id. Splits BY PROBLEM (no leakage), and reproduces the plain-classifier
verification AUC — the bar a contrastive thought-encoder must beat (~0.85 from
§5.10).

Run the self-test:
    python real_mhcot/data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
_POOLS = _ROOT / "data" / "hidden_cache" / "gsm8k_test_pools.pt"


def load_thoughts(pool="lastk"):
    """Return X (N,1536) float, y (N,) {1=correct,0=incorrect}, pid (N,) problem id."""
    c = torch.load(_POOLS, weights_only=False)
    X, y, pid = [], [], []
    for idx, e in c.items():
        vecs = e["candidates"][pool] if isinstance(e["candidates"], dict) else e["candidates"]
        for i, corr in enumerate(e["cand_correct"]):
            X.append(vecs[i]); y.append(1 if corr else 0); pid.append(idx)
    return (torch.stack(X).float(),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(pid, dtype=torch.long))


def split_by_problem(X, y, pid, seed=0, frac_train=0.7):
    """Split so a problem's thoughts are entirely in train OR test (no leakage)."""
    probs = torch.unique(pid).tolist()
    rng = np.random.RandomState(seed); rng.shuffle(probs)
    cut = int(frac_train * len(probs))
    tr_set = set(probs[:cut])
    tr = torch.tensor([i for i in range(len(pid)) if pid[i].item() in tr_set])
    te = torch.tensor([i for i in range(len(pid)) if pid[i].item() not in tr_set])
    return (X[tr], y[tr], pid[tr]), (X[te], y[te], pid[te])


class MLP(nn.Module):
    def __init__(self, d_in=1536, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def baseline_auc(Xtr, ytr, Xte, yte, dev="cpu", steps=1500, bs=128, lr=3e-4, seed=0):
    """Plain classifier verification AUC — the bar to beat."""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1]).to(dev)
    Xtr, ytr = Xtr.to(dev), ytr.float().to(dev)
    pos = (ytr == 1).nonzero().squeeze(-1); neg = (ytr == 0).nonzero().squeeze(-1)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed); half = bs // 2
    for _ in range(steps):
        bi = torch.cat([pos[torch.randint(len(pos), (half,), generator=g)],
                        neg[torch.randint(len(neg), (bs - half,), generator=g)]])
        loss = F.binary_cross_entropy_with_logits(m(Xtr[bi]), ytr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        s = m(Xte.to(dev)).cpu().numpy()
    return roc_auc_score(yte.numpy(), s)


def _test():
    print("[R0] loading thoughts ...")
    X, y, pid = load_thoughts("lastk")
    n_prob = len(torch.unique(pid))
    print(f"    {len(y)} thoughts over {n_prob} problems; "
          f"correct={int(y.sum())} incorrect={int((1-y).sum())} "
          f"({y.float().mean():.2%} correct); dim={X.shape[1]}")

    (Xtr, ytr, ptr), (Xte, yte, pte) = split_by_problem(X, y, pid, seed=0)
    assert set(ptr.tolist()).isdisjoint(set(pte.tolist())), "problem leakage!"
    print(f"[R0] split by problem (no leakage): train {len(ytr)} / test {len(yte)} thoughts")

    print("[R0] baseline plain-classifier verification AUC (the bar) ...")
    auc = baseline_auc(Xtr, ytr, Xte, yte)
    print(f"    baseline verification AUC = {auc:.3f}")
    assert auc > 0.75, "baseline should clearly separate correct/incorrect thoughts"
    print(f"\n[R0 PASSED] thoughts loaded, split clean, baseline bar = {auc:.3f} "
          f"(the contrastive thought-encoder must beat this).")


if __name__ == "__main__":
    _test()
