"""
cvnn/exp_l2.py  —  L2: characterize the L1 win (data-efficiency curve)
=====================================================================

L1 showed complex >> real on phase-carried data at n_train=300. L2 maps the
WHOLE curve: test accuracy (and overfitting gap) for complex vs real across
training-set sizes. This shows where the complex advantage lives — the cleanest
way to SEE the win, and whether it widens in the small-data regime.

Chunk-friendly (the background runtime cap kills long jobs): run a few sizes /
seeds at a time and aggregate the printed lines.

Run:
    python cvnn/exp_l2.py --sizes 100 200 400 800 1600 --seeds 0
    python cvnn/exp_l2.py --sizes 100 200 400 800 1600 --seeds 1
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "main"))
from signals import make_dataset                                         # noqa: E402
from models import ComplexSeqClassifier, RealSeqClassifier, n_params     # noqa: E402
from exp_l1 import train_eval, _device                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or _device()
    nc, T = args.n_classes, args.T
    print(f"[L2] device={dev} sizes={args.sizes} seeds={args.seeds} "
          f"epochs={args.epochs}\n")
    rows = []
    for size in args.sizes:
        for s in args.seeds:
            torch.manual_seed(s); np.random.seed(s)
            Ztr, ytr = make_dataset(size, nc, T, seed=s)
            Zte, yte = make_dataset(args.n_test, nc, T, seed=1000 + s)
            for name, build in (("complex", lambda: ComplexSeqClassifier(n_classes=nc, T=T)),
                                ("real", lambda: RealSeqClassifier(n_classes=nc, T=T))):
                torch.manual_seed(s); np.random.seed(s)
                m = build().to(dev)
                tr, te = train_eval(m, Ztr, ytr, Zte, yte, dev, args.epochs)
                rows.append((size, s, name, tr, te))
                print(f"    size={size:<5} seed{s} {name:<8} test={te:.3f} "
                      f"gap={tr - te:.3f}", flush=True)
                del m; gc.collect()
                if dev == "mps":
                    torch.mps.empty_cache()

    print("\n" + "=" * 56)
    print(f"L2 — test accuracy by train size (seeds {args.seeds})")
    print(f"{'size':<8}{'complex':>10}{'real':>10}{'Δ':>10}")
    for size in args.sizes:
        cte = np.mean([te for (sz, s, n, tr, te) in rows if sz == size and n == "complex"])
        rte = np.mean([te for (sz, s, n, tr, te) in rows if sz == size and n == "real"])
        print(f"{size:<8}{cte:>10.3f}{rte:>10.3f}{cte - rte:>+10.3f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
