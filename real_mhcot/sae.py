"""
real_mhcot/sae.py  —  R4: sparse autoencoder → interpretable thought-vocabulary
===============================================================================

A sparse autoencoder on the thought embeddings: it reconstructs each embedding
from a SPARSE code over an overcomplete dictionary — the "emergent vocabulary" of
thought-features. We check it (a) reconstructs well, (b) is genuinely sparse, and
(c) yields interpretable features — e.g., features that fire more for CORRECT
reasoning (a learned "correctness vocabulary").

Run the self-test:
    python real_mhcot/sae.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_thoughts, split_by_problem                    # noqa: E402
from encoder import ThoughtEncoder, train_contrastive               # noqa: E402


class SAE(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d, bias=False)

    def forward(self, x):
        f = F.relu(self.enc(x))          # sparse code (the active thought-features)
        return self.dec(f), f


def train_sae(sae, Z, dev="cpu", steps=3000, bs=128, lr=1e-3, l1=2e-3, seed=0):
    torch.manual_seed(seed)
    Z = Z.to(dev)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    sae.train()
    for _ in range(steps):
        bi = torch.randint(len(Z), (bs,), generator=g)
        x = Z[bi]
        xr, f = sae(x)
        loss = F.mse_loss(xr, x) + l1 * f.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return sae


@torch.no_grad()
def sae_eval(sae, Z, y, dev="cpu"):
    sae.eval()
    xr, f = sae(Z.to(dev))
    xr, f = xr.cpu(), f.cpu()
    recon_cos = F.cosine_similarity(Z, xr, dim=-1).mean().item()
    l0 = (f > 1e-6).float().sum(-1).mean().item()                 # active features/sample
    # feature ↔ correctness: which sparse features track correct reasoning?
    yv = y.numpy().astype(float)
    fa = f.numpy()
    active = (fa > 1e-6).mean(0) > 0.02                            # features used at all
    corrs = []
    for k in np.where(active)[0]:
        c = np.corrcoef(fa[:, k], yv)[0, 1]
        if np.isfinite(c):
            corrs.append(abs(c))
    max_corr = max(corrs) if corrs else 0.0
    n_interp = sum(c > 0.2 for c in corrs)
    return recon_cos, l0, max_corr, n_interp, int(active.sum())


def _test():
    dev = "cpu"
    X, y, pid = load_thoughts("lastk")
    (Xtr, ytr, _), (Xte, yte, _) = split_by_problem(X, y, pid, seed=0)
    torch.manual_seed(0)
    enc = ThoughtEncoder(d=128).to(dev)
    train_contrastive(enc, Xtr, ytr, dev=dev, steps=2000, seed=0)
    with torch.no_grad():
        Ztr = enc(Xtr.to(dev)).cpu()
        Zte = enc(Xte.to(dev)).cpu()

    print("[R4] training SAE (dict=512) on thought embeddings ...")
    sae = SAE(d=128, m=512).to(dev)
    train_sae(sae, Ztr, dev=dev, steps=3000, l1=2e-3, seed=0)
    rc, l0, mc, ni, na = sae_eval(sae, Zte, yte, dev=dev)
    print(f"    reconstruction cosine = {rc:.3f}")
    print(f"    sparsity: {l0:.1f} active features / 512 ({l0/512:.1%})")
    print(f"    interpretable vocabulary: {na} features used; "
          f"max feature↔correctness corr = {mc:.3f}; {ni} features with |corr|>0.2")

    assert rc > 0.80, "SAE should reconstruct the thought embeddings well"
    assert l0 < 256, "SAE code should be sparse"
    assert mc > 0.20, "at least one feature should track correctness (interpretable)"
    print(f"\n[R4 PASSED] SAE gives a sparse, reconstructive, interpretable "
          f"thought-vocabulary — incl. {ni} features that track correct reasoning.")


if __name__ == "__main__":
    _test()
