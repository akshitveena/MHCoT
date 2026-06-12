"""
experiments/exp_n1_ablation.py  —  MULTI-HELICAL vs GENERAL COMPLEX
==================================================================

The calibration win is the complex architecture (we ruled out losses and
magnitude readout). But WHICH part? The task-only N=2 model stayed calibrated
even though its chains never differentiated — hinting the calibration might
come from complex processing IN GENERAL (the lift), not the MULTI-HELICAL
structure specifically.

This trains an N=1 complex model (single chain — no interference, no helix,
no soliton coupling) task-only, and measures its calibration.

  real            ECE 0.240   (no complex at all)
  complex N=2     ECE 0.070   (multi-helical)
  complex N=1     ???  ← THIS  (complex but single chain)

If N=1 ECE ≈ 0.07 → calibration is GENERAL COMPLEX processing (DCN-adjacent);
                    the multi-helical structure is not the cause.
If N=1 ECE ≈ 0.24 → calibration REQUIRES the multi-helical structure;
                    it is uniquely MHCoT.
In between → both contribute.

This decides what we can claim NOVELTY for. Run: ~30 min on Mac.
    python experiments/exp_n1_ablation.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "main"))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

from encoder import load_sequences                 # noqa: E402
from model import MHCoTEncoder                       # noqa: E402
from train import SeqDataset, collate, _device       # noqa: E402
from exp_calibration import ece                      # noqa: E402

REAL_ECE = 0.240
COMPLEX_N2_ECE = 0.070


def main():
    dev = _device()
    print(f"[n1-ablation] device={dev}")
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
    # N=1 complex: single chain, no interference/helix/coupling. Task-only.
    model = MHCoTEncoder(d_in=d_in, d_model=128, n_chains=1).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[n1-ablation] N=1 MHCoTEncoder params = {nparams/1e6:.2f}M (task-only)\n")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    for epoch in range(1, 41):
        model.train()
        for H, mask, lengths, y in tl:
            H, mask, lengths, y = H.to(dev), mask.to(dev), lengths.to(dev), y.to(dev)
            out = model(H, key_padding_mask=mask, lengths=lengths)
            loss = F.binary_cross_entropy_with_logits(out["score"], y)  # task only
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if epoch % 10 == 0:
            print(f"  ...epoch {epoch}/40")

    model.eval(); s_all, y_all = [], []
    with torch.no_grad():
        for H, mask, lengths, y in vl:
            H, mask, lengths = H.to(dev), mask.to(dev), lengths.to(dev)
            s_all.append(model(H, key_padding_mask=mask, lengths=lengths)["score"].cpu())
            y_all.append(y)
    s = torch.cat(s_all).numpy(); y = torch.cat(y_all).numpy()
    p = 1 / (1 + np.exp(-s))
    n1_ece = ece(p, y); auc = roc_auc_score(y, s)

    print("\n" + "=" * 64)
    print(f"  real          ECE {REAL_ECE:.3f}")
    print(f"  complex N=2   ECE {COMPLEX_N2_ECE:.3f}")
    print(f"  complex N=1   ECE {n1_ece:.3f}   AUC {auc:.3f}  ← THIS")
    print("-" * 64)
    if n1_ece < 0.13:
        print("  → GENERAL COMPLEX processing calibrates (N=1 already does it).")
        print("    The multi-helical structure is NOT the cause of calibration.")
        print("    Claim: 'complex reasoning representations are better calibrated'")
        print("    (DCN-adjacent; multi-helical adds chain diversity, not calibration).")
    elif n1_ece > 0.18:
        print("  → MULTI-HELICAL structure is REQUIRED. N=1 does not calibrate;")
        print("    only N=2 interference does. Calibration is uniquely MHCoT.")
    else:
        print("  → PARTIAL: single complex chain helps, multi-helical adds more.")
    print("=" * 64)


if __name__ == "__main__":
    main()
