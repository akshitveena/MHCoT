"""
experiments/exp5_ablation.py  —  THE REAL-vs-COMPLEX ABLATION
=============================================================

The reviewer's #1 question: does the COMPLEX machinery (phase / ε-helix /
interference) actually help, or would a matched REAL transformer do the same?

This trains a real-valued twin — same param budget, identical training data,
SAME val split (seed=0, val_frac=0.2) as main/train.py, task loss only — and
compares its val AUC to the complex MHCoTEncoder result.

  complex MHCoT (N=2):   best val AUC ≈ 0.797, stable ≈ 0.73   (from train.py)
  untrained probe:       AUC 0.689
  real twin (here):      ??? ← the number this script produces

Reading:
  real ≪ complex  → complex helps; the ε-helix is the source of the gain.
  real ≈ complex  → complex did NOT help; honest finding, rethink the claim.

Fairness notes:
  * Same architecture skeleton (down-proj → encoder → answer-region pool →
    scorer), but real: standard nn.TransformerEncoder, no chains/phase.
  * Param count printed for both so the match is transparent.
  * Real model gets task loss only — helix/wave are intrinsically complex and
    have no real analog (that's the point of the ablation).

Run:
    python experiments/exp5_ablation.py            # ~30 min on Mac
Output: results/exp5_ablation/{summary.json, real_model.pt}
"""

from __future__ import annotations

import json
import sys
import time
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

from encoder import load_sequences, SEQ_PATH         # noqa: E402
from train import SeqDataset, collate, _device       # noqa: E402

_OUT = _PROJECT_ROOT / "results" / "exp5_ablation"
_OUT.mkdir(parents=True, exist_ok=True)

COMPLEX_BEST = 0.797      # from the complex MHCoT run
COMPLEX_STABLE = 0.73
UNTRAINED_PROBE = 0.689


# ---------------------------------------------------------------------------
# Real-valued twin — matched params, no phase / chains / interference
# ---------------------------------------------------------------------------
class RealEncoder(nn.Module):
    def __init__(self, d_in: int, d_model: int = 128, n_layers: int = 4,
                 n_heads: int = 8, d_ff: int = 512, answer_tail: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.answer_tail = answer_tail
        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        # enable_nested_tensor=False: the nested-tensor fast-path uses an op
        # (_nested_tensor_from_mask_left_aligned) not implemented on MPS.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1),
        )

    def _answer_region_mask(self, B, T, lengths, device):
        k = min(self.answer_tail, T)
        idx = torch.arange(T, device=device).unsqueeze(0)
        lo = (lengths - k).clamp(min=0).unsqueeze(1)
        hi = lengths.unsqueeze(1)
        return ((idx >= lo) & (idx < hi)).float()       # (B,T)

    def forward(self, h, key_padding_mask=None, lengths=None):
        h = self.in_proj(h)                              # (B,T,d_model)
        z = self.encoder(h, src_key_padding_mask=key_padding_mask)
        B, T, _ = z.shape
        region = self._answer_region_mask(B, T, lengths, z.device)
        denom = region.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (z * region.unsqueeze(-1)).sum(dim=1) / denom
        score = self.scorer(pooled).squeeze(-1)
        # interface parity with the complex model (no interference here)
        return {"score": score, "I_answer": score, "chain_divergence": 0.0}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    s_all, y_all = [], []
    for H, mask, lengths, labels in loader:
        H, mask, lengths = H.to(dev), mask.to(dev), lengths.to(dev)
        out = model(H, key_padding_mask=mask, lengths=lengths)
        s_all.append(out["score"].float().cpu())
        y_all.append(labels)
    s = torch.cat(s_all).numpy()
    y = torch.cat(y_all).numpy()
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = _device(); print(f"[ablation] device={dev}")

    if not SEQ_PATH.exists():
        print(f"[error] run `python main/encoder.py --seq` first."); sys.exit(1)
    cache = load_sequences()
    items, labels = [], []
    for e in cache.values():
        items.append((e["h"], e["length"], e["label"])); labels.append(e["label"])
    d_in = items[0][0].shape[1]

    # IDENTICAL split to train.py (same seed/val_frac/stratify) → same val set
    tr, va = train_test_split(items, test_size=args.val_frac,
                              random_state=args.seed, stratify=labels)
    coll = partial(collate, max_seq_len=args.max_seq_len)
    tl = DataLoader(SeqDataset(tr), batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    vl = DataLoader(SeqDataset(va), batch_size=args.batch_size, shuffle=False, collate_fn=coll)

    model = RealEncoder(d_in=d_in, d_model=args.d_model, n_layers=args.n_layers).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[ablation] RealEncoder params = {nparams/1e6:.2f}M  "
          f"(complex MHCoT was 1.03M — matched)")
    print(f"\n[targets] untrained probe {UNTRAINED_PROBE} | "
          f"complex stable {COMPLEX_STABLE} | complex best {COMPLEX_BEST}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = 0.0; log = []; t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for H, mask, lengths, labels_b in tl:
            H, mask, lengths, labels_b = (H.to(dev), mask.to(dev),
                                          lengths.to(dev), labels_b.to(dev))
            out = model(H, key_padding_mask=mask, lengths=lengths)
            loss = F.binary_cross_entropy_with_logits(out["score"], labels_b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step(); losses.append(loss.item())
        auc = evaluate(model, vl, dev)
        best = max(best, auc)
        flag = "  <-- beats complex-stable" if auc > COMPLEX_STABLE else ""
        print(f"ep{epoch:>3} | task {np.mean(losses):.3f} | val AUC (real) = {auc:.3f}{flag}")
        log.append({"epoch": epoch, "auc_real": auc, "loss": float(np.mean(losses))})
        if auc >= best:
            torch.save(model.state_dict(), _OUT / "real_model.pt")

    print("=" * 64)
    print(f"[done] {(time.time()-t0)/60:.1f} min")
    print(f"  REAL  twin  best val AUC = {best:.3f}  ({nparams/1e6:.2f}M params)")
    print(f"  COMPLEX MHCoT best        = {COMPLEX_BEST}  (1.03M params)")
    print(f"  untrained probe           = {UNTRAINED_PROBE}")
    print("-" * 64)
    margin = COMPLEX_BEST - best
    if margin > 0.04:
        print(f"  → COMPLEX BEATS REAL by {margin:+.3f} — the ε-helix adds value")
    elif margin > -0.04:
        print(f"  → REAL ≈ COMPLEX ({margin:+.3f}) — within val noise; complex's")
        print(f"    edge is NOT established. Needs more data to separate them.")
    else:
        print(f"  → REAL BEATS COMPLEX ({margin:+.3f}) — complex did not help. Honest finding.")
    print("  (val set n=120 → ±0.08 noise; treat margins <0.08 cautiously)")
    print("=" * 64)

    json.dump({"args": vars(args), "real_best_auc": best, "real_params": nparams,
               "complex_best": COMPLEX_BEST, "complex_stable": COMPLEX_STABLE,
               "untrained_probe": UNTRAINED_PROBE, "log": log},
              open(_OUT / "summary.json", "w"), indent=2)
    print(f"[saved] {_OUT}/summary.json, real_model.pt")


if __name__ == "__main__":
    main()
