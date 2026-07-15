"""
experiments/exp_reproduce_original.py  —  faithful reproduction of the 0.070
============================================================================

Reproduces the EXACT conditions that produced the original ECE 0.070, across
multiple seeds, to settle: was the calibration gap real-under-those-conditions,
or a single-run fluke?

Original conditions (exactly):
  * 605 test-split sequences (gsm8k_test_seq.pt), split 80/20 IN-DISTRIBUTION
  * real / N=1 trained task-only; N=2 trained with FULL staged losses
    (task → +L_ε → +L_wave) — as the headline run did
  * ECE measured on the BEST-AUC checkpoint (not the final epoch)
  * val = the 120 in-distribution held-out slice

Runs seeds 0,1,2 and reports ECE mean ± std per model.
  - N=2 consistently ~0.07  → real under these conditions (held-out run then
    shows it doesn't GENERALIZE — an honest, different finding)
  - N=2 swings wildly        → the 0.070 was single-run noise

Prereq:  python main/encoder.py --seq --cache gsm8k_test.jsonl   (already done)
Run:     python experiments/exp_reproduce_original.py
"""

from __future__ import annotations

import copy
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

from model import MHCoTEncoder                     # noqa: E402
from train import SeqDataset, collate, _device      # noqa: E402
from Projects.MHCoT.complex_mhcot.experiments.exp5_ablation import RealEncoder               # noqa: E402
from Projects.MHCoT.complex_mhcot.experiments.exp_calibration import ece                     # noqa: E402
from losses import compute_losses, LossConfig        # noqa: E402

_HID = _PROJECT_ROOT / "data" / "hidden_cache"
SEEDS = [0, 1, 2]
EPOCHS = 40


def load_items(path):
    cache = torch.load(path)
    items = [(e["h"], e["length"], e["label"]) for e in cache.values()]
    labels = [e["label"] for e in cache.values()]
    return items, labels


@torch.no_grad()
def _auc_ece(model, loader, dev):
    model.eval(); s, y = [], []
    for H, mask, lengths, yy in loader:
        H, mask, lengths = H.to(dev), mask.to(dev), lengths.to(dev)
        s.append(model(H, key_padding_mask=mask, lengths=lengths)["score"].float().cpu())
        y.append(yy)
    s = torch.cat(s).numpy(); y = torch.cat(y).numpy()
    p = 1 / (1 + np.exp(-s))
    auc = roc_auc_score(y, s) if len(set(y.tolist())) > 1 else float("nan")
    return auc, ece(p, y)


def train_track_best(model, tl, vl, dev, full_losses):
    """Train; keep the BEST-AUC checkpoint (as the original did). Return its ECE+AUC."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    cfg = LossConfig(eps_min=1.15, lambda_eps=0.10, lambda_wave=0.05,
                     stage_eps_step=500, stage_wave_step=1500)  # original schedule
    step = 0; best_auc = -1; best_ece = None; best_state = None
    for _ in range(EPOCHS):
        model.train()
        for H, mask, lengths, y in tl:
            H, mask, lengths, y = H.to(dev), mask.to(dev), lengths.to(dev), y.to(dev)
            out = model(H, key_padding_mask=mask, lengths=lengths)
            if full_losses:
                loss, _ = compute_losses(out, y, step, cfg)
            else:
                loss = F.binary_cross_entropy_with_logits(out["score"], y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
        auc, e = _auc_ece(model, vl, dev)
        if auc > best_auc:                              # best-AUC checkpoint (as original)
            best_auc, best_ece = auc, e
            best_state = copy.deepcopy(model.state_dict())
    return best_auc, best_ece


def main():
    dev = _device(); print(f"[reproduce] device={dev}")
    te = _HID / "gsm8k_test_seq.pt"
    if not te.exists():
        print(f"[error] missing {te}. Run: python main/encoder.py --seq --cache gsm8k_test.jsonl")
        sys.exit(1)
    items, labels = load_items(te)
    d_in = items[0][0].shape[1]
    print(f"[reproduce] {len(items)} sequences, 80/20 in-distribution split, "
          f"best-AUC checkpoint, seeds={SEEDS}\n")

    coll = partial(collate, max_seq_len=384)
    out = {"real": [], "complex_N1": [], "complex_N2_full": []}

    for seed in SEEDS:
        tr, va = train_test_split(items, test_size=0.2, random_state=seed, stratify=labels)
        tl = DataLoader(SeqDataset(tr), batch_size=8, shuffle=True, collate_fn=coll)
        vl = DataLoader(SeqDataset(va), batch_size=8, shuffle=False, collate_fn=coll)
        print(f"--- seed {seed}: train={len(tr)} val={len(va)} ---")

        torch.manual_seed(seed); np.random.seed(seed)
        _, e = train_track_best(RealEncoder(d_in=d_in, d_model=128, n_layers=4).to(dev),
                                tl, vl, dev, full_losses=False)
        out["real"].append(e); print(f"  real        ECE {e:.3f}")

        torch.manual_seed(seed); np.random.seed(seed)
        _, e = train_track_best(MHCoTEncoder(d_in=d_in, d_model=128, n_chains=1).to(dev),
                                tl, vl, dev, full_losses=False)
        out["complex_N1"].append(e); print(f"  N=1         ECE {e:.3f}")

        torch.manual_seed(seed); np.random.seed(seed)
        _, e = train_track_best(MHCoTEncoder(d_in=d_in, d_model=128, n_chains=2).to(dev),
                                tl, vl, dev, full_losses=True)
        out["complex_N2_full"].append(e); print(f"  N=2 (full)  ECE {e:.3f}")

    print("\n" + "=" * 60)
    print("EXACT-ORIGINAL conditions, ECE across seeds (best-AUC checkpoint):")
    for name, es in out.items():
        es = np.array(es)
        print(f"  {name:<16} {es.mean():.3f} ± {es.std():.3f}   (per-seed: "
              f"{', '.join(f'{x:.3f}' for x in es)})")
    print("-" * 60)
    n2 = np.array(out["complex_N2_full"])
    print(f"Original headline: N=2 = 0.070")
    if n2.mean() < 0.12 and n2.std() < 0.05:
        print("  → REPRODUCIBLE: N=2 ~0.07 across seeds. Real under these conditions.")
        print("    (The held-out run then shows it does not GENERALIZE — report both.)")
    elif n2.std() >= 0.05:
        print(f"  → NOISE: N=2 ECE swings (std {n2.std():.3f}). The 0.070 was a fluke.")
    else:
        print(f"  → N=2 mean {n2.mean():.3f} — did not reproduce the 0.070 even here.")
    print("=" * 60)


if __name__ == "__main__":
    main()
