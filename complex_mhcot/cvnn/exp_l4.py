"""
cvnn/exp_l4.py  —  L4: multi-helical on its OWN home field (superposition)
==========================================================================

The first test where the multi-helical mechanism has a genuine mechanistic
reason to win (gate 4 of the anatomy: the task requires holding MULTIPLE
hypotheses). Each signal is a sum of TWO chirps; the model must recover BOTH
rate-classes (multi-label). Two co-evolving chains could each lock onto a
different component — something a single chain cannot do by construction.

Models: real / complex_n1 / complex_n2 (multi-helical, ε-helix engaged).
Readout: interference Ψ → multi-label logits (same as L3, now multi-label).
Metrics: exact-pair match (both components correct) + recall@k (fraction of
true components recovered).

Pre-committed L4 bar: complex_n2 > complex_n1 on exact-pair match, across seeds.
THIS is the multi-helical claim's real test. If it ties here too, the mechanism
genuinely adds nothing even where it should; if it wins, it's the first positive
evidence for the novelty — to be replicated before believing.

Run:
    python cvnn/exp_l4.py --smoke --device cpu
    python cvnn/exp_l4.py --seeds 0 1 2 --device cpu
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
from signals import make_superposition                                   # noqa: E402
from models import (RealSeqClassifier, MultiHelicalSeqClassifier, n_params)  # noqa: E402
from soliton import epsilon_helix_loss                                    # noqa: E402


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def metrics(model, Z, Y, dev, k):
    Z = Z.to(dev)
    logits = model(Z).cpu()
    topk = logits.topk(k, dim=1).indices                      # (n,k)
    exact, recall = 0.0, 0.0
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
    bce = nn.BCEWithLogitsLoss()
    n = Ztr.shape[0]
    warmup = int(warmup_frac * epochs)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if kind == "complex_n2":
                out, psi = model(Ztr[idx], return_psi=True)
                loss = bce(out, Ytr[idx])
                if ep >= warmup:
                    loss = loss + lam_eps * epsilon_helix_loss(psi)
            else:
                loss = bce(model(Ztr[idx]), Ytr[idx])
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                opt.zero_grad(); continue
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    ex_tr, _ = metrics(model, Ztr.cpu(), Ytr.cpu(), dev, k)
    ex_te, rc_te = metrics(model, Zte, Yte, dev, k)
    return ex_tr, ex_te, rc_te


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
        "real":       (lambda: RealSeqClassifier(n_classes=nc, T=T), "real"),
        "complex_n1": (lambda: MultiHelicalSeqClassifier(n_chains=1, n_classes=nc, T=T), "complex_n1"),
        "complex_n2": (lambda: MultiHelicalSeqClassifier(n_chains=2, n_classes=nc, T=T), "complex_n2"),
    }
    print(f"[L4] device={dev} superposition k={k}-of-{nc}, n_train={args.n_train} "
          f"epochs={args.epochs}  (multi-helical's HOME FIELD)\n")
    res = {c: [] for c in CONFIGS}
    for s in args.seeds:
        torch.manual_seed(s); np.random.seed(s)
        Ztr, Ytr = make_superposition(args.n_train, nc, T, k, seed=s)
        Zte, Yte = make_superposition(args.n_test, nc, T, k, seed=1000 + s)
        for name, (build, kind) in CONFIGS.items():
            torch.manual_seed(s); np.random.seed(s)
            m = build().to(dev)
            ex_tr, ex_te, rc_te = train_eval(m, Ztr, Ytr, Zte, Yte, dev, args.epochs, kind, k)
            res[name].append((ex_te, rc_te))
            print(f"    seed{s} {name:<11} exact={ex_te:.3f} recall@{k}={rc_te:.3f} "
                  f"(train_exact={ex_tr:.3f})", flush=True)
            del m; gc.collect()
            if dev == "mps":
                torch.mps.empty_cache()

    print("\n" + "=" * 60)
    print(f"L4 SUMMARY (mean ± std over seeds {args.seeds})")
    summ = {}
    for name in CONFIGS:
        arr = np.array(res[name]); ex = arr[:, 0]; rc = arr[:, 1]
        summ[name] = (ex.mean(), ex.std(), rc.mean())
        print(f"  {name:<11} exact {ex.mean():.3f}±{ex.std():.3f}   recall {rc.mean():.3f}")
    print("-" * 60)
    n2, n1 = summ["complex_n2"][0], summ["complex_n1"][0]
    if n2 > n1 + 0.01:
        print(f"  -> L4: N=2 ({n2:.3f}) BEATS N=1 ({n1:.3f}) on the multi-hypothesis "
              f"task. FIRST positive for the multi-helical novelty — replicate & dig in.")
    elif abs(n2 - n1) <= 0.01:
        print(f"  -> L4: N=2 ({n2:.3f}) ≈ N=1 ({n1:.3f}) — multi-helical adds nothing "
              f"even on its home field. The strongest evidence yet that interference "
              f"itself is not the mechanism.")
    else:
        print(f"  -> L4: N=2 ({n2:.3f}) < N=1 ({n1:.3f}). Report as-is.")
    print("=" * 60)


if __name__ == "__main__":
    main()
