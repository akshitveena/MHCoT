"""
cvnn/exp_l7.py  —  L7: federated framing (per-silo, no pooling allowed)
======================================================================

The honest home for the data-efficiency edge. Data is split across M silos that
CANNOT be pooled (privacy / on-device / distributed). Each silo trains its own
local expert; a server aggregates predictions (federated ensemble — no raw data
or gradients shared). Complex vs real under the IDENTICAL protocol.

Here joint training is not an available baseline — the only comparison is
complex-federated vs real-federated. L6 (seed 0) showed complex experts crush
real experts on tiny silos (0.70 vs 0.43); this confirms it multi-seed and
reports both the federated-ensemble accuracy and the average single-silo accuracy.

Run (1 seed per job is safest under the runtime cap):
    python cvnn/exp_l7.py --N 800 --M 5 --seeds 0 --device cpu
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
from models import ComplexSeqClassifier, RealSeqClassifier              # noqa: E402
from exp_l6 import train_model, get_probs, acc                          # noqa: E402
from exp_l1 import _device                                              # noqa: E402


def federated(build, Ztr, ytr, Zte, yte, dev, M, silo, epochs, seed):
    """Train M local experts (one per silo, no pooling); return federated-ensemble
    accuracy and the average single-silo accuracy."""
    member_probs, member_accs = [], []
    for m in range(M):
        torch.manual_seed(seed * 100 + m); np.random.seed(seed * 100 + m)
        idx = slice(m * silo, (m + 1) * silo)
        model = train_model(build, Ztr[idx], ytr[idx], dev, epochs)
        p = get_probs(model, Zte, dev)
        member_probs.append(p); member_accs.append(acc(p, yte))
        del model; gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()
    fed = torch.stack(member_probs).mean(0)
    return acc(fed, yte), float(np.mean(member_accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--M", type=int, default=5, help="number of silos")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or _device()
    nc, T, N, M = args.n_classes, args.T, args.N, args.M
    silo = N // M
    cbuild = lambda: ComplexSeqClassifier(n_classes=nc, T=T)
    rbuild = lambda: RealSeqClassifier(n_classes=nc, T=T)
    print(f"[L7] device={dev} FEDERATED: N={N} across M={M} silos ({silo}/silo, "
          f"no pooling) epochs={args.epochs} seeds={args.seeds}\n")

    agg = {"complex_fed": [], "real_fed": [], "complex_silo": [], "real_silo": []}
    for s in args.seeds:
        torch.manual_seed(s); np.random.seed(s)
        Ztr, ytr = make_dataset(N, nc, T, seed=s)
        Zte, yte = make_dataset(args.n_test, nc, T, seed=1000 + s)
        cf, cs = federated(cbuild, Ztr, ytr, Zte, yte, dev, M, silo, args.epochs, s)
        rf, rs = federated(rbuild, Ztr, ytr, Zte, yte, dev, M, silo, args.epochs, s)
        agg["complex_fed"].append(cf); agg["complex_silo"].append(cs)
        agg["real_fed"].append(rf); agg["real_silo"].append(rs)
        print(f"    seed{s}  complex: fed={cf:.3f} (avg silo {cs:.3f})  |  "
              f"real: fed={rf:.3f} (avg silo {rs:.3f})", flush=True)

    print("\n" + "=" * 58)
    print(f"L7 FEDERATED SUMMARY (N={N}, M={M} silos, seeds {args.seeds})")
    for kk in ("complex_fed", "real_fed", "complex_silo", "real_silo"):
        v = np.array(agg[kk]); print(f"  {kk:<14} {v.mean():.3f} ± {v.std():.3f}")
    print("-" * 58)
    cf, rf = np.mean(agg["complex_fed"]), np.mean(agg["real_fed"])
    print(f"complex-federated {cf:.3f}  vs  real-federated {rf:.3f}  "
          f"(Δ {cf - rf:+.3f})")
    if cf > rf + 0.02:
        print("  -> complex experts WIN the federated/no-pooling setting. The "
              "data-efficiency edge is a real advantage where data can't be pooled.")
    else:
        print("  -> complex does not clearly beat real in this federated config.")
    print("=" * 58)


if __name__ == "__main__":
    main()
