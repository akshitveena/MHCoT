"""
experiments/exp_replication.py  —  THE REPLICATION (the #1 rigor gate)
=====================================================================

Trains real / N=1-complex / N=2-complex on the 361 GSM8K TRAIN-split traces,
then measures calibration (ECE) + accuracy (AUC) on the 605 GSM8K TEST-split
traces — a properly held-out eval set 5× larger than the noisy 120-val we used
before. All three trained TASK-ONLY (identical objective; only architecture
differs) for a clean apples-to-apples comparison.

Replicates if the ordering holds on the big held-out set:
    real ≈ N=1 (ECE ~0.22–0.24)  ≫  N=2 (ECE ~0.07)

Prereqs (encode both splits — run on Mac, ~30 min total, no Colab):
    python main/encoder.py --seq --cache gsm8k_train.jsonl
    python main/encoder.py --seq --cache gsm8k_test.jsonl

Run:
    python experiments/exp_replication.py
Out: results/exp_replication/summary.json
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "main"))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

from model import MHCoTEncoder                     # noqa: E402
from train import SeqDataset, collate, _device      # noqa: E402
from exp5_ablation import RealEncoder               # noqa: E402
from exp_calibration import ece                     # noqa: E402

_HID = _PROJECT_ROOT / "data" / "hidden_cache"
_OUT = _PROJECT_ROOT / "results" / "exp_replication"
_OUT.mkdir(parents=True, exist_ok=True)


def load_items(path):
    cache = torch.load(path)
    return [(e["h"], e["length"], e["label"]) for e in cache.values()]


def train_taskonly(model, loader, dev, epochs=40, lr=3e-4, wd=0.05):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        for H, mask, lengths, y in loader:
            H, mask, lengths, y = H.to(dev), mask.to(dev), lengths.to(dev), y.to(dev)
            out = model(H, key_padding_mask=mask, lengths=lengths)
            loss = F.binary_cross_entropy_with_logits(out["score"], y)  # TASK ONLY
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); s, y = [], []
    for H, mask, lengths, yy in loader:
        H, mask, lengths = H.to(dev), mask.to(dev), lengths.to(dev)
        s.append(model(H, key_padding_mask=mask, lengths=lengths)["score"].float().cpu())
        y.append(yy)
    s = torch.cat(s).numpy(); y = torch.cat(y).numpy()
    p = 1 / (1 + np.exp(-s))
    auc = roc_auc_score(y, s) if len(set(y.tolist())) > 1 else float("nan")
    return ece(p, y), auc, float(y.mean())


def main():
    dev = _device(); print(f"[replication] device={dev}")
    tr_path, te_path = _HID / "gsm8k_train_seq.pt", _HID / "gsm8k_test_seq.pt"
    for p in (tr_path, te_path):
        if not p.exists():
            print(f"[error] missing {p.name}. Run:")
            print("   python main/encoder.py --seq --cache gsm8k_train.jsonl")
            print("   python main/encoder.py --seq --cache gsm8k_test.jsonl")
            sys.exit(1)

    train_items = load_items(tr_path)
    test_items = load_items(te_path)
    d_in = train_items[0][0].shape[1]
    print(f"[replication] train={len(train_items)} (train split) | "
          f"eval={len(test_items)} (test split, held-out) | d_in={d_in}")

    coll = partial(collate, max_seq_len=384)
    tl = DataLoader(SeqDataset(train_items), batch_size=8, shuffle=True, collate_fn=coll)
    vl = DataLoader(SeqDataset(test_items), batch_size=8, shuffle=False, collate_fn=coll)

    configs = {
        "real":        lambda: RealEncoder(d_in=d_in, d_model=128, n_layers=4),
        "complex_N1":  lambda: MHCoTEncoder(d_in=d_in, d_model=128, n_chains=1),
        "complex_N2":  lambda: MHCoTEncoder(d_in=d_in, d_model=128, n_chains=2),
    }
    results = {}
    t0 = time.time()
    for name, build in configs.items():
        torch.manual_seed(0); np.random.seed(0)
        model = build().to(dev)
        np_ = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] params={np_/1e6:.2f}M — training task-only on {len(train_items)} ...")
        train_taskonly(model, tl, dev)
        e, auc, base = evaluate(model, vl, dev)
        results[name] = {"ece": e, "auc": auc, "params": np_}
        print(f"[{name}] held-out ECE={e:.3f}  AUC={auc:.3f}")

    print("\n" + "=" * 60)
    print(f"REPLICATION on {len(test_items)} held-out test traces "
          f"(trained on {len(train_items)} train traces)")
    print(f"{'model':<14}{'ECE':>8}{'AUC':>8}")
    for name in configs:
        r = results[name]
        print(f"{name:<14}{r['ece']:>8.3f}{r['auc']:>8.3f}")
    print("-" * 60)
    re_, n1, n2 = results["real"]["ece"], results["complex_N1"]["ece"], results["complex_N2"]["ece"]
    print("Original (120-val): real 0.240 | N=1 0.225 | N=2 0.070")
    print(f"Now    (605 held-out): real {re_:.3f} | N=1 {n1:.3f} | N=2 {n2:.3f}")
    if n2 < re_ - 0.05 and n2 < n1 - 0.05:
        print("  → REPLICATED: N=2 multi-helical interference calibrates on the")
        print("    big held-out set; real and single-chain complex do not. Bulletproof.")
    elif n2 < re_ - 0.02:
        print("  → Partial: N=2 still best-calibrated but margin smaller — report honestly.")
    else:
        print("  → DID NOT replicate at scale. The 120-val result was likely noise.")
    print("=" * 60)
    json.dump({"n_train": len(train_items), "n_eval": len(test_items),
               "results": results, "elapsed_min": (time.time()-t0)/60},
              open(_OUT / "summary.json", "w"), indent=2)
    print(f"[saved] {_OUT}/summary.json  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
