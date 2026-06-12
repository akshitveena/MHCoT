"""
experiments/exp_magnitude_ablation.py  —  COMPLEX vs MAGNITUDE-READOUT
=====================================================================

The last loophole on the calibration claim: is the complex model's calibration
from the COMPLEX arithmetic, or just from its MAGNITUDE readout (it scores from
|Ψ|)?  Test: a REAL twin that scores from |pooled features| (element-wise
magnitude) — same everything else, task-loss only.

  real  (raw readout)        ECE 0.240   ← over-confident
  complex (|Ψ| readout)      ECE 0.070   ← calibrated
  real  (|features| readout)  ???  ← THIS script

If abs-real ECE ≈ 0.07  → it's the MAGNITUDE READOUT (complex isn't the cause).
If abs-real ECE ≈ 0.24  → it's genuinely the COMPLEX architecture.

Run:  python experiments/exp_magnitude_ablation.py        (~4 min)
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "main"))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

from encoder import load_sequences                  # noqa: E402
from train import SeqDataset, collate, _device      # noqa: E402
from exp_calibration import ece                     # noqa: E402

REAL_RAW_ECE = 0.240      # real twin, raw readout
COMPLEX_ECE = 0.070       # complex, |Ψ| readout


class RealEncoderAbs(nn.Module):
    """Real twin identical to exp5's RealEncoder, but scores from |pooled|."""
    def __init__(self, d_in, d_model=128, n_layers=4, n_heads=8, d_ff=512,
                 answer_tail=16, dropout=0.1):
        super().__init__()
        self.answer_tail = answer_tail
        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def forward(self, h, key_padding_mask=None, lengths=None):
        h = self.in_proj(h)
        z = self.encoder(h, src_key_padding_mask=key_padding_mask)
        B, T, _ = z.shape
        k = min(self.answer_tail, T)
        idx = torch.arange(T, device=z.device).unsqueeze(0)
        lo = (lengths - k).clamp(min=0).unsqueeze(1)
        hi = lengths.unsqueeze(1)
        region = ((idx >= lo) & (idx < hi)).float()
        denom = region.sum(1, keepdim=True).clamp(min=1.0)
        pooled = (z * region.unsqueeze(-1)).sum(1) / denom
        pooled = pooled.abs()                          # ← the ONLY change: magnitude readout
        return {"score": self.scorer(pooled).squeeze(-1)}


def main():
    dev = _device(); print(f"[mag-ablation] device={dev}")
    cache = load_sequences()
    items, labels = [], []
    for e in cache.values():
        items.append((e["h"], e["length"], e["label"])); labels.append(e["label"])
    d_in = items[0][0].shape[1]
    tr, va = train_test_split(items, test_size=0.2, random_state=0, stratify=labels)
    coll = partial(collate, max_seq_len=384)
    tl = DataLoader(SeqDataset(tr), batch_size=8, shuffle=True, collate_fn=coll)
    vl = DataLoader(SeqDataset(va), batch_size=8, shuffle=False, collate_fn=coll)

    torch.manual_seed(0); np.random.seed(0)
    model = RealEncoderAbs(d_in=d_in).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    for epoch in range(1, 41):
        model.train()
        for H, mask, lengths, y in tl:
            H, mask, lengths, y = H.to(dev), mask.to(dev), lengths.to(dev), y.to(dev)
            out = model(H, key_padding_mask=mask, lengths=lengths)
            loss = F.binary_cross_entropy_with_logits(out["score"], y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    # evaluate ECE + AUC on val
    model.eval(); s_all, y_all = [], []
    with torch.no_grad():
        for H, mask, lengths, y in vl:
            H, mask, lengths = H.to(dev), mask.to(dev), lengths.to(dev)
            s_all.append(model(H, key_padding_mask=mask, lengths=lengths)["score"].cpu())
            y_all.append(y)
    s = torch.cat(s_all).numpy(); y = torch.cat(y_all).numpy()
    p = 1 / (1 + np.exp(-s))
    abs_ece = ece(p, y)
    auc = roc_auc_score(y, s)

    print("=" * 64)
    print(f"  real  (raw readout)        ECE {REAL_RAW_ECE:.3f}")
    print(f"  complex (|Ψ| readout)      ECE {COMPLEX_ECE:.3f}")
    print(f"  real  (|features| readout) ECE {abs_ece:.3f}   AUC {auc:.3f}  ← THIS")
    print("-" * 64)
    if abs_ece < 0.13:
        print(f"  → MAGNITUDE READOUT explains the calibration. The complex")
        print(f"    arithmetic is NOT the cause — a real model with |·| readout")
        print(f"    is also calibrated. Honest finding.")
    elif abs_ece > 0.18:
        print(f"  → COMPLEX is genuinely the cause. A real magnitude readout")
        print(f"    does NOT calibrate; the complex architecture does.")
    else:
        print(f"  → PARTIAL: magnitude readout helps but doesn't fully explain it.")
    print("=" * 64)


if __name__ == "__main__":
    main()
