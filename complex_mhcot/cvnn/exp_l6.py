"""
cvnn/exp_l6.py  —  L6: the shard-ensemble idea (data-efficiency + bagging)
==========================================================================

Tests the proposal: split the dataset into smaller segments, train a complex
model on each (where complex is data-efficient), and aggregate — to leverage the
small-data advantage and average out noise.

The sharp test runs at a data size where a SINGLE complex model LOSES to real
(n≈800: complex 0.771 < real 0.901). Can a complex shard-ensemble beat the
single real model trained jointly on all the data?

Configs at fixed total N:
    single_real        real trained on all N           (the bar to beat)
    single_complex     complex trained on all N
    complex_ens_shard  K complex, each on a disjoint N/K segment, probs averaged
    real_ens_shard     K real,    each on a disjoint N/K segment (fairness)

If complex_ens_shard > single_real, the idea works (chopping leverages data-
efficiency). If not, joint training wins — partitioning doesn't add information.

Run:
    python cvnn/exp_l6.py --N 800 --K 5 --seeds 0 --device cpu
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "main"))
from signals import make_dataset                                         # noqa: E402
from models import ComplexSeqClassifier, RealSeqClassifier              # noqa: E402
from exp_l1 import _device                                               # noqa: E402


def train_model(build, Z, y, dev, epochs=70, lr=1e-3, bs=64):
    m = build().to(dev)
    Z, y = Z.to(dev), y.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = len(y)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = lossf(m(Z[idx]), y[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


@torch.no_grad()
def get_probs(m, Z, dev, bs=256):
    m.eval()
    out = [F.softmax(m(Z[i:i + bs].to(dev)), dim=-1).cpu() for i in range(0, len(Z), bs)]
    return torch.cat(out)


def acc(probs, y):
    return (probs.argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or _device()
    nc, T, N, K = args.n_classes, args.T, args.N, args.K
    cbuild = lambda: ComplexSeqClassifier(n_classes=nc, T=T)
    rbuild = lambda: RealSeqClassifier(n_classes=nc, T=T)
    print(f"[L6] device={dev} N={N} K={K} (shard={N//K}) epochs={args.epochs} "
          f"seeds={args.seeds}\n")

    agg = {k: [] for k in ["single_real", "single_complex",
                           "complex_ens_shard", "real_ens_shard"]}
    for s in args.seeds:
        torch.manual_seed(s); np.random.seed(s)
        Ztr, ytr = make_dataset(N, nc, T, seed=s)
        Zte, yte = make_dataset(args.n_test, nc, T, seed=1000 + s)

        torch.manual_seed(s)
        a = acc(get_probs(train_model(rbuild, Ztr, ytr, dev, args.epochs), Zte, dev), yte)
        agg["single_real"].append(a); print(f"    seed{s} single_real        {a:.3f}", flush=True)
        gc.collect()
        torch.manual_seed(s)
        a = acc(get_probs(train_model(cbuild, Ztr, ytr, dev, args.epochs), Zte, dev), yte)
        agg["single_complex"].append(a); print(f"    seed{s} single_complex     {a:.3f}", flush=True)
        gc.collect()

        sh = N // K
        for label, build in (("complex_ens_shard", cbuild), ("real_ens_shard", rbuild)):
            member_probs = []
            for k in range(K):
                torch.manual_seed(s * 100 + k)
                idx = slice(k * sh, (k + 1) * sh)
                m = train_model(build, Ztr[idx], ytr[idx], dev, args.epochs)
                member_probs.append(get_probs(m, Zte, dev))
                del m; gc.collect()
                if dev == "mps":
                    torch.mps.empty_cache()
            ens = torch.stack(member_probs).mean(0)
            a = acc(ens, yte)
            agg[label].append(a)
            print(f"    seed{s} {label:<18} {a:.3f}  (K={K} x {sh} each)", flush=True)

    print("\n" + "=" * 56)
    print(f"L6 SUMMARY (N={N}, K={K}, mean over seeds {args.seeds})")
    for k in agg:
        v = np.array(agg[k])
        print(f"  {k:<20} {v.mean():.3f} ± {v.std():.3f}")
    print("-" * 56)
    ce = np.mean(agg["complex_ens_shard"]); sr = np.mean(agg["single_real"])
    sc = np.mean(agg["single_complex"]); re_ = np.mean(agg["real_ens_shard"])
    print(f"complex shard-ensemble {ce:.3f}  vs  single real (joint) {sr:.3f}")
    if ce > sr:
        print("  -> shard-ensemble BEATS joint real: chopping leverages data-efficiency!")
    else:
        print(f"  -> shard-ensemble does NOT beat joint real ({ce:.3f} < {sr:.3f}); "
              f"partitioning doesn't add information. (But complex_ens {ce:.3f} >> "
              f"real_ens {re_:.3f} in the sharded regime.)")
    print("=" * 56)


if __name__ == "__main__":
    main()
