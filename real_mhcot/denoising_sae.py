"""
real_mhcot/denoising_sae.py  —  custom denoising sparse AE on thought embeddings.

Your TSDAE instinct, made coherent and at the EMBEDDING level (a sibling of sae.py):
  * corrupt a thought embedding (mask dims)  -> "damage"
  * encode to a SPARSE bottleneck (no context leak — your fix)
  * decode reconstructs the CLEAN embedding

Trained on CORRECT-ONLY thoughts, so it models the manifold of VALID reasoning.
Two outputs, extending the prior interpretability model:
  (F1/F2 extension) reconstruction-error VALIDITY signal: off-manifold (incorrect)
                    thoughts should reconstruct worse -> AUC vs verifier (0.82) and a
                    NORM baseline (the surface trap we pre-check).
  (F3 extension)    interpretable sparse features, audited for surface-form.

    python real_mhcot/denoising_sae.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_thoughts, split_by_problem                  # noqa: E402


class DenoisingSAE(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d)

    def forward(self, x):
        f = F.relu(self.enc(x))
        return self.dec(f), f


def train(model, Xclean, steps=3000, bs=128, lr=1e-3, l1=1e-3, noise=0.3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        bi = torch.randint(len(Xclean), (bs,), generator=g)
        x = Xclean[bi]
        mask = (torch.rand(x.shape, generator=g) > noise).float()
        xr, f = model(x * mask)                       # reconstruct CLEAN from corrupted
        loss = F.mse_loss(xr, x) + l1 * f.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def recon_err(model, X):
    model.eval()
    xr, _ = model(X)
    return ((X - xr) ** 2).mean(-1)


def _auc(y, score):
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y, score)
    return max(a, 1 - a)                               # best-direction (fair baseline)


def _test():
    from sklearn.metrics import roc_auc_score
    X, y, pid = load_thoughts("lastk")
    print(f"[d-sae] {len(y)} thoughts ({int(y.sum())} correct), dim={X.shape[1]}")

    val_auc, norm_auc, err_norm_corr, feat_corr = [], [], [], []
    for seed in [0, 1, 2]:
        (Xtr, ytr, _), (Xte, yte, _) = split_by_problem(X, y, pid, seed=seed)
        model = DenoisingSAE(d=X.shape[1], m=512)
        train(model, Xtr[ytr == 1], seed=seed)        # CORRECT-ONLY manifold

        err = recon_err(model, Xte).numpy()
        yv = yte.numpy()
        val_auc.append(roc_auc_score(yv, -err))       # low err -> correct
        # surface baseline: embedding norm
        norm = Xte.norm(dim=-1).numpy()
        norm_auc.append(_auc(yv, norm))
        err_norm_corr.append(np.corrcoef(err, norm)[0, 1])
        # feature audit
        with torch.no_grad():
            _, f = model(Xte)
        f = f.numpy(); active = (f > 1e-6).mean(0) > 0.02
        cs = [abs(np.corrcoef(f[:, k], yv)[0, 1]) for k in np.where(active)[0]]
        feat_corr.append(max([c for c in cs if np.isfinite(c)] or [0]))

    print(f"\n[RESULT] denoising-AE on correct-only manifold (3 seeds, leak-free):")
    print(f"  reconstruction-error validity AUC = {np.mean(val_auc):.3f} ± {np.std(val_auc):.3f}")
    print(f"  norm-only baseline AUC            = {np.mean(norm_auc):.3f}")
    print(f"  reference: contrastive verifier AUC = 0.819")
    print(f"  [surface audit] corr(recon-error, embedding-norm) = {np.mean(err_norm_corr):+.3f}")
    print(f"  [features] max feature↔correctness corr = {np.mean(feat_corr):.3f}")

    v, n = np.mean(val_auc), np.mean(norm_auc)
    if v > n + 0.03 and v > 0.6:
        print(f"\n[FINDING] reconstruction error detects invalid reasoning ({v:.2f}) beyond norm ({n:.2f}) "
              "— the valid-reasoning manifold carries a real validity signal.")
    elif v <= n + 0.03:
        print(f"\n[HONEST] recon-error AUC ({v:.2f}) ≈ norm baseline ({n:.2f}) — the signal is mostly "
              "surface (embedding magnitude), not a learned validity manifold. Report as-is.")
    else:
        print(f"\n[HONEST] recon-error AUC {v:.2f} below the verifier (0.82) — weaker validity signal; report exactly this.")


if __name__ == "__main__":
    _test()
