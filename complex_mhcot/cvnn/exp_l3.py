"""
cvnn/exp_l3.py  —  L3: the multi-helical layer, on the home field
=================================================================

The actual MHCoT question, finally tested where complex demonstrably matters
(L1 established complex >> real on phase-carried data). Does N=2 (multi-helical,
co-evolving chains + interference) beat N=1 (single complex chain) and real?

Models (param-comparable, multi-seed, small-data overfitting regime):
    real        real transformer (sees re/im)
    complex_n1  single complex chain (the L1 winner)
    complex_n2  multi-helical: 2 chains + soliton coupling + ε-helix + interference

N=2 trained with CE + staged ε-helix loss (engaged after warmup; atan2 fixes baked
into soliton.py). Primary metric: test accuracy + train-test gap.

Pre-committed L3 bar: complex_n2 > complex_n1 (the multi-helical adds value over a
single complex chain) AND >= real, across seeds. Otherwise the multi-helical layer
adds nothing even on the home field — but the complex foundation (L1) still stands.

Run:
    python cvnn/exp_l3.py --smoke
    python cvnn/exp_l3.py --seeds 0 1 2 --device cpu
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
sys.path.insert(0, str(_HERE.parent.parent / "main"))
from signals import make_dataset                                          # noqa: E402
from models import (ComplexSeqClassifier, RealSeqClassifier,             # noqa: E402
                    MultiHelicalSeqClassifier, n_params)
from soliton import epsilon_helix_loss                                    # noqa: E402


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_eval(model, Ztr, ytr, Zte, yte, dev, epochs, kind,
               lr=1e-3, bs=64, lam_eps=0.05, warmup_frac=0.4):
    Ztr, ytr, Zte, yte = Ztr.to(dev), ytr.to(dev), Zte.to(dev), yte.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = Ztr.shape[0]
    warmup = int(warmup_frac * epochs)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if kind == "complex_n2":
                out, psi = model(Ztr[idx], return_psi=True)
                loss = lossf(out, ytr[idx])
                if ep >= warmup:
                    loss = loss + lam_eps * epsilon_helix_loss(psi)
            else:
                loss = lossf(model(Ztr[idx]), ytr[idx])
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                opt.zero_grad(); continue
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        tr = (model(Ztr).argmax(1) == ytr).float().mean().item()
        te = (model(Zte).argmax(1) == yte).float().mean().item()
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = [0], 20

    dev = args.device or _device()
    nc, T = args.n_classes, args.T
    CONFIGS = {
        "real":       (lambda: RealSeqClassifier(n_classes=nc, T=T), "real"),
        "complex_n1": (lambda: MultiHelicalSeqClassifier(n_chains=1, n_classes=nc, T=T), "complex_n1"),
        "complex_n2": (lambda: MultiHelicalSeqClassifier(n_chains=2, n_classes=nc, T=T), "complex_n2"),
    }
    print(f"[L3] device={dev} n_train={args.n_train} epochs={args.epochs} "
          f"classes={nc}  (multi-helical on the home field)\n")
    res = {k: [] for k in CONFIGS}
    for s in args.seeds:
        torch.manual_seed(s); np.random.seed(s)
        Ztr, ytr = make_dataset(args.n_train, nc, T, seed=s)
        Zte, yte = make_dataset(args.n_test, nc, T, seed=1000 + s)
        for name, (build, kind) in CONFIGS.items():
            torch.manual_seed(s); np.random.seed(s)
            m = build().to(dev)
            tr, te = train_eval(m, Ztr, ytr, Zte, yte, dev, args.epochs, kind)
            res[name].append((tr, te))
            print(f"    seed{s} {name:<11} train={tr:.3f} test={te:.3f} "
                  f"gap={tr - te:.3f}  params={n_params(m)/1e3:.0f}K", flush=True)
            del m; gc.collect()
            if dev == "mps":
                torch.mps.empty_cache()

    print("\n" + "=" * 60)
    print(f"L3 SUMMARY (mean ± std over seeds {args.seeds})")
    summ = {}
    for name in CONFIGS:
        arr = np.array(res[name]); te = arr[:, 1]; gap = arr[:, 0] - arr[:, 1]
        summ[name] = (te.mean(), te.std(), gap.mean())
        print(f"  {name:<11} test {te.mean():.3f}±{te.std():.3f}   gap {gap.mean():.3f}")
    print("-" * 60)
    n2, n1, r = summ["complex_n2"][0], summ["complex_n1"][0], summ["real"][0]
    if n2 > n1 + 0.01 and n2 >= r:
        print(f"  -> L3: N=2 ({n2:.3f}) BEATS N=1 ({n1:.3f}) and real ({r:.3f}). "
              f"The multi-helical layer adds value on the home field. INVESTIGATE.")
    elif abs(n2 - n1) <= 0.01:
        print(f"  -> L3: N=2 ({n2:.3f}) ≈ N=1 ({n1:.3f}) — multi-helical adds nothing "
              f"over a single complex chain (complex foundation still wins vs real {r:.3f}).")
    else:
        print(f"  -> L3: N=2 ({n2:.3f}) vs N=1 ({n1:.3f}), real ({r:.3f}). Report as-is.")
    print("=" * 60)


if __name__ == "__main__":
    main()
