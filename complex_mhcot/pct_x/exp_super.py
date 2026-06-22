"""
pct_x/exp_super.py  —  genuine_interp on its HOME FIELD (steady-tone superposition)
==================================================================================

The fair test for the genuine-interpretation architecture: signals that are a SUM
of steady tones (multiple coherent components to decompose), multi-label "which
tones are present". Here coherent pooling per-component can REINFORCE (unlike a
chirp, whose rotating phase cancels). K interpretations = K components to lock onto.

Models (multi-label, BCE; metric = recall@k and exact-set match):
    real                 standard real transformer
    cplx_single          PCT, 1 chain
    cplx_multi_crossterm PCT, 2 rotated chains + interference readout
    genuine_interp       distinct interpretations, coherent strand pooling (K=k)

THE question: does genuine_interp beat the others *here*, where its core operation
(coherent pooling) finally matches the data?

Run it yourself:
    python pct_x/exp_super.py --seeds 0 1 2 --epochs 50 --device cpu
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
sys.path.insert(0, str(_HERE.parent / "cvnn"))
sys.path.insert(0, str(_HERE.parent.parent / "main"))
from pct import PCTClassifier, n_params                                  # noqa: E402
from interp import GenuineInterpClassifier                               # noqa: E402
from models import RealSeqClassifier                                     # noqa: E402
from signals import make_multitone                                       # noqa: E402
from soliton import epsilon_helix_loss                                   # noqa: E402


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def metrics(model, Z, Y, dev, k, bs=256):
    logits = []
    for i in range(0, len(Z), bs):
        logits.append(model(Z[i:i + bs].to(dev)).cpu())
    logits = torch.cat(logits)
    topk = logits.topk(k, dim=1).indices
    Yb = Y.bool(); exact = recall = 0.0
    for i in range(len(Z)):
        pred = set(topk[i].tolist()); true = set(torch.where(Yb[i])[0].tolist())
        recall += len(pred & true) / len(true); exact += float(pred == true)
    return exact / len(Z), recall / len(Z)


def train_eval(model, Ztr, Ytr, Zte, Yte, dev, epochs, multi, k, lr=1e-3, bs=64,
               lam_eps=0.05, warmup_frac=0.4):
    Ztr, Ytr = Ztr.to(dev), Ytr.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Ztr); warm = int(warmup_frac * epochs)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if multi:
                out, psi = model(Ztr[idx], return_psi=True)
                loss = F.binary_cross_entropy_with_logits(out, Ytr[idx])
                if ep >= warm:
                    loss = loss + lam_eps * epsilon_helix_loss(psi)
            else:
                loss = F.binary_cross_entropy_with_logits(model(Ztr[idx]), Ytr[idx])
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                opt.zero_grad(); continue
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return metrics(model, Zte, Yte, dev, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--k_present", type=int, default=2)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--noise", type=float, default=0.15, help="raise to harden the task")
    ap.add_argument("--configs", default=None, help="comma list to subset (fit the cap)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or _device()
    nc, T, k = args.n_classes, args.T, args.k_present
    CONFIGS = {
        "real":                 (lambda: RealSeqClassifier(n_classes=nc, T=T), False),
        "cplx_single":          (lambda: PCTClassifier(n_chains=1, n_classes=nc, T=T), False),
        "cplx_multi_crossterm": (lambda: PCTClassifier(n_chains=2, n_classes=nc, T=T, readout="crossterm"), True),
        "genuine_interp":       (lambda: GenuineInterpClassifier(K=k, n_classes=nc, T=T, coevolve=True), True),
    }
    if args.configs:
        keep = set(args.configs.split(","))
        CONFIGS = {k_: v for k_, v in CONFIGS.items() if k_ in keep}
    print(f"[exp_super] STEADY-TONE superposition: k={k}-of-{nc} present, T={T}, "
          f"n_train={args.n_train} epochs={args.epochs} (genuine_interp's home field)\n")
    res = {c: [] for c in CONFIGS}
    for s in args.seeds:
        torch.manual_seed(s); np.random.seed(s)
        Ztr, Ytr = make_multitone(args.n_train, nc, T, k, noise=args.noise, seed=s)
        Zte, Yte = make_multitone(args.n_test, nc, T, k, noise=args.noise, seed=1000 + s)
        for name, (build, multi) in CONFIGS.items():
            torch.manual_seed(s); np.random.seed(s)
            m = build().to(dev)
            ex, rc = train_eval(m, Ztr, Ytr, Zte, Yte, dev, args.epochs, multi, k)
            res[name].append((ex, rc))
            print(f"    seed{s} {name:<22} exact={ex:.3f} recall@{k}={rc:.3f}  "
                  f"params={n_params(m)/1e3:.0f}K", flush=True)
            del m; gc.collect()
            if dev == "mps":
                torch.mps.empty_cache()

    print("\n" + "=" * 62)
    print(f"RESULTS (mean ± std over seeds {args.seeds}) — steady-tone superposition")
    summ = {}
    for c in CONFIGS:
        a = np.array(res[c]); summ[c] = (a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean())
        print(f"  {c:<22} exact {a[:,0].mean():.3f}±{a[:,0].std():.3f}  "
              f"recall {a[:,1].mean():.3f}")
    print("-" * 62)
    gi = summ["genuine_interp"][0]
    best_other = max(summ["cplx_single"][0], summ["cplx_multi_crossterm"][0], summ["real"][0])
    if gi > best_other + 0.01:
        print(f"  -> genuine_interp ({gi:.3f}) BEATS the best baseline ({best_other:.3f}) "
              f"on its home field. The architecture works where its assumptions hold.")
    elif gi >= best_other - 0.02:
        print(f"  -> genuine_interp ({gi:.3f}) ties the baselines ({best_other:.3f}) — "
              f"works now (unlike chirps) but no clear advantage.")
    else:
        print(f"  -> genuine_interp ({gi:.3f}) still below baselines ({best_other:.3f}).")
    print("=" * 62)


if __name__ == "__main__":
    main()
