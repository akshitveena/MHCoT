"""
cvnn/exp_l5.py  —  L5: the AMBIGUOUS-signals home field (per-chain readout)
===========================================================================

The second multi-hypothesis home field (the one I owed you): each signal is a
single chirp whose rate sits between two adjacent class prototypes, so it is
genuinely ambiguous between {c1, c2}. The win is REPRESENTING both alternatives
rather than collapsing to one — the purest anti-collapse test for the ε-helix.

Reuses the L4b per-chain soft-OR readout and harness; only the dataset changes
(make_ambiguous instead of make_superposition).

Configs: real_1head / real_2head / complex_n1 / complex_n2 (ε-helix).
Metric: recall@2 / exact-pair of the ambiguity set {c1,c2}.
Pre-committed bar: complex_n2 > complex_n1 AND > real_2head, across seeds.

Run:
    python cvnn/exp_l5.py --smoke --device cpu
    python cvnn/exp_l5.py --seeds 0 1 2 --device cpu
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
from signals import make_ambiguous                                       # noqa: E402
from models import MultiHelicalPerChain, RealPerHead, n_params           # noqa: E402
from exp_l4b import train_eval, _device                                  # noqa: E402 (reuse harness)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.n_train = [0], 20, 300

    dev = args.device or _device()
    nc, T, k = args.n_classes, args.T, 2
    CONFIGS = {
        "real_1head": (lambda: RealPerHead(n_heads=1, n_classes=nc, T=T), "real"),
        "real_2head": (lambda: RealPerHead(n_heads=2, n_classes=nc, T=T), "real"),
        "complex_n1": (lambda: MultiHelicalPerChain(n_chains=1, n_classes=nc, T=T), "complex_n1"),
        "complex_n2": (lambda: MultiHelicalPerChain(n_chains=2, n_classes=nc, T=T), "complex_n2"),
    }
    print(f"[L5] device={dev} AMBIGUOUS signals (between adjacent classes), "
          f"n_train={args.n_train} epochs={args.epochs}\n")
    res = {c: [] for c in CONFIGS}
    for s in args.seeds:
        torch.manual_seed(s); np.random.seed(s)
        Ztr, Ytr = make_ambiguous(args.n_train, nc, T, seed=s)
        Zte, Yte = make_ambiguous(args.n_test, nc, T, seed=1000 + s)
        for name, (build, kind) in CONFIGS.items():
            torch.manual_seed(s); np.random.seed(s)
            m = build().to(dev)
            ex, rc = train_eval(m, Ztr, Ytr, Zte, Yte, dev, args.epochs, kind, k)
            res[name].append((ex, rc))
            print(f"    seed{s} {name:<11} exact={ex:.3f} recall@{k}={rc:.3f}  "
                  f"params={n_params(m)/1e3:.0f}K", flush=True)
            del m; gc.collect()
            if dev == "mps":
                torch.mps.empty_cache()

    print("\n" + "=" * 62)
    print(f"L5 SUMMARY (mean ± std over seeds {args.seeds}) — AMBIGUOUS / anti-collapse")
    summ = {}
    for name in CONFIGS:
        arr = np.array(res[name]); ex = arr[:, 0]; rc = arr[:, 1]
        summ[name] = (ex.mean(), ex.std(), rc.mean(), rc.std())
        print(f"  {name:<11} exact {ex.mean():.3f}±{ex.std():.3f}   "
              f"recall {rc.mean():.3f}±{rc.std():.3f}")
    print("-" * 62)
    n2, n2s = summ["complex_n2"][0], summ["complex_n2"][1]
    n1, n1s = summ["complex_n1"][0], summ["complex_n1"][1]
    r2 = summ["real_2head"][0]
    beats = n2 > n1 and n2 > r2
    robust = (n2 - max(n1, r2)) > (n2s + n1s) / 2
    if beats and robust:
        print(f"  -> L5: N=2 ({n2:.3f}) ROBUSTLY beats N=1 ({n1:.3f}) & real_2head "
              f"({r2:.3f}). The anti-collapse claim holds — FIRST real positive.")
    elif beats:
        print(f"  -> L5: N=2 ({n2:.3f}) > N=1 ({n1:.3f}), real_2head ({r2:.3f}) but "
              f"within noise — promising, needs more seeds.")
    else:
        print(f"  -> L5: N=2 ({n2:.3f}) does NOT beat both (N=1 {n1:.3f}, "
              f"real_2head {r2:.3f}). Anti-collapse adds nothing even here.")
    print("=" * 62)


if __name__ == "__main__":
    main()
