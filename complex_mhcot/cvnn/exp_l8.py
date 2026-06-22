"""
cvnn/exp_l8.py  —  L8: does the federated advantage GROW as silos shrink?
=========================================================================

Sweeps the number of silos M at fixed total N. Prediction from the data-
efficiency curve: the more fragmented the federation (smaller per-silo data),
the larger complex's advantage over real — because real collapses on tiny silos
while complex still learns.

Δ(complex_fed − real_fed) vs M is the money figure for the federated thesis.

Chunk-friendly: run a few M / seeds per job (runtime cap), aggregate the lines.
Run:
    python cvnn/exp_l8.py --N 800 --M_list 4 8 16 --seeds 0 --device cpu
    python cvnn/exp_l8.py --N 800 --M_list 4 8 16 --seeds 1 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "main"))
from signals import make_dataset                                         # noqa: E402
from models import ComplexSeqClassifier, RealSeqClassifier              # noqa: E402
from exp_l7 import federated                                            # noqa: E402
from exp_l1 import _device                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--M_list", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or _device()
    nc, T, N = args.n_classes, args.T, args.N
    cbuild = lambda: ComplexSeqClassifier(n_classes=nc, T=T)
    rbuild = lambda: RealSeqClassifier(n_classes=nc, T=T)
    print(f"[L8] device={dev} N={N} M_list={args.M_list} seeds={args.seeds} "
          f"epochs={args.epochs}\n")

    rec = {}  # (M) -> list of (cf, rf)
    for M in args.M_list:
        silo = N // M
        rec[M] = []
        for s in args.seeds:
            torch.manual_seed(s); np.random.seed(s)
            Ztr, ytr = make_dataset(N, nc, T, seed=s)
            Zte, yte = make_dataset(args.n_test, nc, T, seed=1000 + s)
            cf, _ = federated(cbuild, Ztr, ytr, Zte, yte, dev, M, silo, args.epochs, s)
            rf, _ = federated(rbuild, Ztr, ytr, Zte, yte, dev, M, silo, args.epochs, s)
            rec[M].append((cf, rf))
            print(f"    M={M:<3} ({silo}/silo) seed{s}  complex_fed={cf:.3f}  "
                  f"real_fed={rf:.3f}  Δ={cf - rf:+.3f}", flush=True)

    print("\n" + "=" * 58)
    print(f"L8 — federated advantage vs silo count (N={N}, seeds {args.seeds})")
    print(f"{'M':<5}{'silo':<8}{'complex':>9}{'real':>9}{'Δ':>9}")
    for M in args.M_list:
        arr = np.array(rec[M]); cf = arr[:, 0].mean(); rf = arr[:, 1].mean()
        print(f"{M:<5}{N // M:<8}{cf:>9.3f}{rf:>9.3f}{cf - rf:>+9.3f}")
    print("=" * 58)


if __name__ == "__main__":
    main()
