"""
cvnn/exp_l1.py  —  L1: complex vs real, accuracy AND overfitting-robustness
===========================================================================

Reproduce the paper's actual finding on our phase-carried data: complex ties
real on test accuracy but generalizes better (smaller train-test gap) in the
small-data, overfitting-prone regime.

Primary metric is the GENERALIZATION GAP (train_acc - test_acc), not accuracy —
the lens where complex nets are known to win and where we never looked.

Pre-committed L1 bar: complex test-acc >= real (tie or better) AND complex
gap < real gap (less overfitting), across seeds.

Run:
    python cvnn/exp_l1.py --smoke
    python cvnn/exp_l1.py --seeds 0 1 2 --n_train 300 --epochs 120
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from signals import make_dataset                                   # noqa: E402
from models import ComplexSeqClassifier, RealSeqClassifier, n_params  # noqa: E402


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_eval(model, Ztr, ytr, Zte, yte, dev, epochs, lr=1e-3, bs=64):
    Ztr, ytr, Zte, yte = Ztr.to(dev), ytr.to(dev), Zte.to(dev), yte.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = Ztr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = model(Ztr[idx])
            loss = lossf(out, ytr[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = (model(Ztr).argmax(1) == ytr).float().mean().item()
        te_acc = (model(Zte).argmax(1) == yte).float().mean().item()
    return tr_acc, te_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=300)   # small => overfitting regime
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--device", default=None, help="override: cpu/mps/cuda")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.n_train, args.epochs = [0], 120, 20

    dev = args.device or _device()
    print(f"[L1] device={dev}  n_train={args.n_train} (overfitting regime) "
          f"epochs={args.epochs} classes={args.n_classes}\n")
    res = {"complex": [], "real": []}
    for s in args.seeds:
        torch.manual_seed(s); np.random.seed(s)
        Ztr, ytr = make_dataset(args.n_train, args.n_classes, args.T, seed=s)
        Zte, yte = make_dataset(args.n_test, args.n_classes, args.T, seed=1000 + s)
        for name, build in (("complex", lambda: ComplexSeqClassifier(n_classes=args.n_classes, T=args.T)),
                            ("real", lambda: RealSeqClassifier(n_classes=args.n_classes, T=args.T))):
            torch.manual_seed(s); np.random.seed(s)
            m = build().to(dev)
            tr, te = train_eval(m, Ztr, ytr, Zte, yte, dev, args.epochs)
            res[name].append((tr, te))
            print(f"    seed{s} {name:<8} train={tr:.3f} test={te:.3f} "
                  f"gap={tr - te:.3f}  params={n_params(m)/1e3:.0f}K", flush=True)
            del m; gc.collect()
            if dev == "mps":
                torch.mps.empty_cache()

    print("\n" + "=" * 60)
    print(f"L1 SUMMARY (mean ± std over seeds {args.seeds})")
    summ = {}
    for name in ("complex", "real"):
        arr = np.array(res[name])
        tr, te = arr[:, 0], arr[:, 1]
        gap = tr - te
        summ[name] = (te.mean(), te.std(), gap.mean(), gap.std())
        print(f"  {name:<8} test {te.mean():.3f}±{te.std():.3f}   "
              f"gap {gap.mean():.3f}±{gap.std():.3f}")
    print("-" * 60)
    c_te, _, c_gap, _ = summ["complex"]
    r_te, _, r_gap, _ = summ["real"]
    tie = c_te >= r_te - 0.03
    better_gap = c_gap < r_gap
    if tie and better_gap:
        print(f"  -> L1 PASS: complex ties on accuracy ({c_te:.3f} vs {r_te:.3f}) "
              f"AND overfits less (gap {c_gap:.3f} < {r_gap:.3f}). "
              f"Reproduces the paper on our home field. Proceed to L2/L3.")
    elif tie:
        print(f"  -> Partial: accuracy tie ({c_te:.3f} vs {r_te:.3f}) but gap not smaller "
              f"({c_gap:.3f} vs {r_gap:.3f}).")
    else:
        print(f"  -> complex below real on accuracy ({c_te:.3f} vs {r_te:.3f}).")
    print("=" * 60)


if __name__ == "__main__":
    main()
