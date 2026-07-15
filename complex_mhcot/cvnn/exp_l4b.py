"""
cvnn/exp_l4b.py  —  L4b: per-chain readout on the superposition home field
==========================================================================

The sharpest possible test of the multi-helical claim. Each chain decodes its
OWN component (no summation); a soft-OR over chains marks a class present if ANY
chain flags it:  p(c) = 1 - Π_i (1 - σ(logit_i[c])).

If the ε-helix makes chain⁰ lock onto one present chirp and chain¹ onto the
other, the union covers BOTH → higher recall@2 / exact-pair than a single chain.

Configs (param-comparable, multi-seed):
    real_1head    real, 1 head
    real_2head    real, 2 heads + soft-OR        (the 'two outputs' control)
    complex_n1    1 complex chain
    complex_n2    2 complex chains + ε-helix + per-chain soft-OR   (the claim)

Pre-committed bar: complex_n2 > complex_n1 AND > real_2head on exact-pair, across
seeds. This is the last faithful variant of the multiplicity hypothesis.

Run:
    python cvnn/exp_l4b.py --smoke --device cpu
    python cvnn/exp_l4b.py --seeds 0 1 2 --device cpu
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
sys.path.insert(0, str(_HERE.parent / "main"))
from Projects.MHCoT.complex_mhcot.cvnn.signals import make_superposition                                   # noqa: E402
from models import MultiHelicalPerChain, RealPerHead, n_params           # noqa: E402
from soliton import epsilon_helix_loss                                    # noqa: E402


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def soft_or(logits):
    """logits (B, H, C) -> combined present-probability (B, C) via soft-OR."""
    p = torch.sigmoid(logits)
    return (1.0 - torch.prod(1.0 - p, dim=1)).clamp(1e-6, 1 - 1e-6)


@torch.no_grad()
def metrics(model, Z, Y, dev, k, bs=256):
    probs = []
    for i in range(0, Z.shape[0], bs):
        out = model(Z[i:i + bs].to(dev))
        probs.append(soft_or(out).cpu())
    p = torch.cat(probs)
    topk = p.topk(k, dim=1).indices
    exact = recall = 0.0
    Yb = Y.bool()
    for i in range(Z.shape[0]):
        pred = set(topk[i].tolist())
        true = set(torch.where(Yb[i])[0].tolist())
        recall += len(pred & true) / len(true)
        exact += float(pred == true)
    n = Z.shape[0]
    return exact / n, recall / n


def train_eval(model, Ztr, Ytr, Zte, Yte, dev, epochs, kind, k,
               lr=1e-3, bs=64, lam_eps=0.05, warmup_frac=0.4):
    Ztr, Ytr = Ztr.to(dev), Ytr.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = Ztr.shape[0]
    warmup = int(warmup_frac * epochs)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if kind == "complex_n2":
                logits, psi = model(Ztr[idx], return_psi=True)
                loss = F.binary_cross_entropy(soft_or(logits), Ytr[idx])
                if ep >= warmup:
                    loss = loss + lam_eps * epsilon_helix_loss(psi)
            else:
                loss = F.binary_cross_entropy(soft_or(model(Ztr[idx])), Ytr[idx])
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                opt.zero_grad(); continue
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    ex_te, rc_te = metrics(model, Zte, Yte, dev, k)
    return ex_te, rc_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--k_present", type=int, default=2)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.n_train = [0], 20, 300

    dev = args.device or _device()
    nc, T, k = args.n_classes, args.T, args.k_present
    CONFIGS = {
        "real_1head": (lambda: RealPerHead(n_heads=1, n_classes=nc, T=T), "real"),
        "real_2head": (lambda: RealPerHead(n_heads=2, n_classes=nc, T=T), "real"),
        "complex_n1": (lambda: MultiHelicalPerChain(n_chains=1, n_classes=nc, T=T), "complex_n1"),
        "complex_n2": (lambda: MultiHelicalPerChain(n_chains=2, n_classes=nc, T=T), "complex_n2"),
    }
    print(f"[L4b] device={dev} per-chain soft-OR, superposition k={k}-of-{nc}, "
          f"n_train={args.n_train} epochs={args.epochs}\n")
    res = {c: [] for c in CONFIGS}
    for s in args.seeds:
        torch.manual_seed(s); np.random.seed(s)
        Ztr, Ytr = make_superposition(args.n_train, nc, T, k, seed=s)
        Zte, Yte = make_superposition(args.n_test, nc, T, k, seed=1000 + s)
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
    print(f"L4b SUMMARY (mean ± std over seeds {args.seeds}) — per-chain readout")
    summ = {}
    for name in CONFIGS:
        arr = np.array(res[name]); ex = arr[:, 0]; rc = arr[:, 1]
        summ[name] = (ex.mean(), ex.std(), rc.mean(), rc.std())
        print(f"  {name:<11} exact {ex.mean():.3f}±{ex.std():.3f}   "
              f"recall {rc.mean():.3f}±{rc.std():.3f}")
    print("-" * 62)
    n2, n2s = summ["complex_n2"][0], summ["complex_n2"][1]
    n1 = summ["complex_n1"][0]
    r2 = summ["real_2head"][0]
    beats = n2 > n1 and n2 > r2
    robust = (n2 - max(n1, r2)) > (n2s + summ["complex_n1"][1]) / 2
    if beats and robust:
        print(f"  -> L4b: N=2 ({n2:.3f}) ROBUSTLY beats N=1 ({n1:.3f}) and real_2head "
              f"({r2:.3f}). FIRST real positive for the multi-helical novelty.")
    elif beats:
        print(f"  -> L4b: N=2 ({n2:.3f}) > N=1 ({n1:.3f}), real_2head ({r2:.3f}) but "
              f"within noise — promising, needs more seeds.")
    else:
        print(f"  -> L4b: N=2 ({n2:.3f}) does NOT beat both (N=1 {n1:.3f}, "
              f"real_2head {r2:.3f}). Multi-helical adds nothing even here.")
    print("=" * 62)


if __name__ == "__main__":
    main()
