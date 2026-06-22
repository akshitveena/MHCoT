"""
experiments/exp_difficulty.py  —  DIFFICULTY / OOD DETECTION  (capability #3)
===========================================================================

Does MHCoT KNOW which problems are hard? A well-calibrated model should be LESS
confident on harder problems; an over-confident model stays confident regardless.

Difficulty proxy: number of calculation steps in the GSM8K gold solution
(count of '<<...>>' annotations = number of operations).

Test (on the SAME val split as everything else):
  * confidence = max(p, 1-p) from each model's score
  * Spearman correlation( confidence , difficulty )  → expect NEGATIVE
    (harder problem → lower confidence) for a difficulty-aware model.
  * Compare complex N=2 vs real. Also correlate MHCoT's INTERFERENCE with difficulty.
  * Sanity: accuracy by difficulty bin (harder → lower accuracy).

Hypothesis: complex (calibrated) tracks difficulty (more negative correlation)
better than real (over-confident, flat high confidence).

Run:  python experiments/exp_difficulty.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent / "main"))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

from encoder import load_sequences                  # noqa: E402
from model import MHCoTEncoder                        # noqa: E402
from train import SeqDataset, collate, _device        # noqa: E402
from exp5_ablation import RealEncoder                  # noqa: E402

_COMPLEX_CKPT = _PROJECT_ROOT / "results" / "train" / "best_model.pt"
_REAL_CKPT = _PROJECT_ROOT / "results" / "exp5_ablation" / "real_model.pt"


def gsm8k_difficulty():
    """idx → number of calculation steps in the gold solution."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    return {i: ds[i]["answer"].count("<<") for i in range(len(ds))}


@torch.no_grad()
def collect(model, loader, dev):
    model.eval()
    s, ia, y = [], [], []
    for H, mask, lengths, yy in loader:
        H, mask, lengths = H.to(dev), mask.to(dev), lengths.to(dev)
        out = model(H, key_padding_mask=mask, lengths=lengths)
        s.append(out["score"].float().cpu())
        ia.append(out["I_answer"].float().cpu())
        y.append(yy)
    return (torch.cat(s).numpy(), torch.cat(ia).numpy(), torch.cat(y).numpy())


def main():
    dev = _device(); print(f"[difficulty] device={dev}")
    cache = load_sequences()
    idxs = list(cache.keys())
    items = [(cache[i]["h"], cache[i]["length"], cache[i]["label"]) for i in idxs]
    labels = [cache[i]["label"] for i in idxs]

    # SAME split (seed 0), but track which problem idx each val item is
    pos = np.arange(len(idxs))
    _, va_pos = train_test_split(pos, test_size=0.2, random_state=0, stratify=labels)
    val_items = [items[p] for p in va_pos]
    val_idxs = [idxs[p] for p in va_pos]

    diff_map = gsm8k_difficulty()
    difficulty = np.array([diff_map.get(i, 0) for i in val_idxs])
    print(f"[difficulty] val n={len(val_idxs)} | "
          f"steps: min={difficulty.min()} median={int(np.median(difficulty))} "
          f"max={difficulty.max()}")

    coll = partial(collate, max_seq_len=384)
    vl = DataLoader(SeqDataset(val_items), batch_size=8, shuffle=False, collate_fn=coll)
    d_in = items[0][0].shape[1]

    cx = MHCoTEncoder(d_in=d_in, d_model=128).to(dev)
    cx.load_state_dict(torch.load(_COMPLEX_CKPT, map_location=dev))
    rl = RealEncoder(d_in=d_in, d_model=128, n_layers=4).to(dev)
    rl.load_state_dict(torch.load(_REAL_CKPT, map_location=dev))

    cx_s, cx_i, y = collect(cx, vl, dev)
    rl_s, _, _ = collect(rl, vl, dev)

    def conf(s):
        p = 1 / (1 + np.exp(-s)); return np.maximum(p, 1 - p)

    # correlations (negative = confidence drops on harder problems = good)
    rho_cx, p_cx = spearmanr(conf(cx_s), difficulty)
    rho_rl, p_rl = spearmanr(conf(rl_s), difficulty)
    rho_int, p_int = spearmanr(cx_i, difficulty)   # interference vs difficulty

    print("=" * 66)
    print("Spearman( confidence , difficulty )  — NEGATIVE = knows what's hard")
    print(f"  complex confidence : rho {rho_cx:+.3f}  (p={p_cx:.3f})")
    print(f"  real    confidence : rho {rho_rl:+.3f}  (p={p_rl:.3f})")
    print(f"  complex interference: rho {rho_int:+.3f}  (p={p_int:.3f})")
    print("-" * 66)
    # sanity: accuracy by difficulty tertile
    print("Accuracy by difficulty (sanity — harder should be lower):")
    order = np.argsort(difficulty)
    thirds = np.array_split(order, 3)
    for name, g in zip(["easy", "med", "hard"], thirds):
        acc = ((cx_s[g] > 0) == y[g]).mean()
        print(f"  {name:<6} steps≈{int(np.median(difficulty[g]))}  "
              f"complex-pred acc {acc:.3f}  (n={len(g)})")
    print("=" * 66)
    better = "complex" if rho_cx < rho_rl else "real"
    print(f"VERDICT: {better} confidence tracks difficulty better "
          f"(more negative rho).")
    if rho_cx < rho_rl - 0.05:
        print("  → MHCoT is more difficulty-aware — its uncertainty rises on")
        print("    harder problems more than the real model's. Extends calibration.")
    elif rho_rl < rho_cx - 0.05:
        print("  → Real tracks difficulty better here. Honest read.")
    else:
        print("  → Comparable (small data). Not decisive.")
    print("=" * 66)


if __name__ == "__main__":
    main()
