"""
real_mhcot/encoder.py  —  R1: real Siamese contrastive thought encoder
======================================================================

A real-valued thought encoder trained Siamese/contrastively: same-correctness
thoughts are pulled together, different-correctness pushed apart (cosine
embedding loss). We then read verification straight from the embedding geometry
(cosine to the correct-prototype) and via a linear probe, and compare to the R0
baseline bar (~0.79).

Honest note: for *verification* a plain classifier is a strong baseline, so the
encoder's job here is to MATCH it while producing a reusable embedding (the real
payoff is reranking/retrieval — R3). Beating it would be a bonus.

Run the self-test:
    python real_mhcot/encoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_thoughts, split_by_problem, baseline_auc      # noqa: E402


class ThoughtEncoder(nn.Module):
    """Real thought encoder: ℝ¹⁵³⁶ → L2-normalized ℝ^d embedding."""
    def __init__(self, d_in=1536, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, d))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def train_contrastive(model, Xtr, ytr, dev="cpu", steps=2000, bs=128, lr=3e-4,
                      margin=0.2, seed=0):
    torch.manual_seed(seed)
    Xtr, ytr = Xtr.to(dev), ytr.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CosineEmbeddingLoss(margin=margin)
    g = torch.Generator().manual_seed(seed)
    n = len(ytr)
    model.train()
    for _ in range(steps):
        a = torch.randint(n, (bs,), generator=g)
        b = torch.randint(n, (bs,), generator=g)
        pair = torch.where(ytr[a] == ytr[b], 1, -1).float().to(dev)   # +1 same class
        za, zb = model(Xtr[a]), model(Xtr[b])
        loss = lossf(za, zb, pair)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def verification_auc(model, Xtr, ytr, Xte, yte, dev="cpu"):
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    model.eval()
    Ztr = model(Xtr.to(dev)).cpu().numpy()
    Zte = model(Xte.to(dev)).cpu().numpy()
    # (a) prototype geometry: cosine to mean correct-thought embedding
    proto = Ztr[ytr.numpy() == 1].mean(0)
    proto = proto / (np.linalg.norm(proto) + 1e-9)
    proto_score = Zte @ proto
    auc_proto = roc_auc_score(yte.numpy(), proto_score)
    # (b) linear probe on the (frozen) embedding
    clf = LogisticRegression(max_iter=2000).fit(Ztr, ytr.numpy())
    auc_probe = roc_auc_score(yte.numpy(), clf.decision_function(Zte))
    return auc_proto, auc_probe


def _test():
    dev = "cpu"
    X, y, pid = load_thoughts("lastk")
    (Xtr, ytr, _), (Xte, yte, _) = split_by_problem(X, y, pid, seed=0)
    bar = baseline_auc(Xtr, ytr, Xte, yte, dev=dev)
    print(f"[R1] baseline bar (plain classifier) = {bar:.3f}")

    model = ThoughtEncoder().to(dev)
    print("[R1] training Siamese contrastive thought encoder ...")
    train_contrastive(model, Xtr, ytr, dev=dev, steps=2000, seed=0)
    auc_proto, auc_probe = verification_auc(model, Xtr, ytr, Xte, yte, dev=dev)
    print(f"[R1] verification AUC — prototype-cosine {auc_proto:.3f}, "
          f"linear-probe {auc_probe:.3f}  (baseline {bar:.3f})")

    best = max(auc_proto, auc_probe)
    assert best > 0.72, "encoder should learn a verification-useful thought space"
    verdict = ("BEATS" if best > bar + 0.01 else
               "matches" if best > bar - 0.03 else "trails")
    print(f"\n[R1 PASSED] contrastive thought encoder {verdict} the baseline "
          f"({best:.3f} vs {bar:.3f}); embedding is reusable for R3 reranking.")


if __name__ == "__main__":
    _test()
