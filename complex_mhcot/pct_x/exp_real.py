"""
pct_x/exp_real.py  —  the readout test, on REAL data, runnable by you
=====================================================================

Tests the fix for "the answers always tie": does exposing the interference
CROSS-TERM (the only genuinely multi-chain signal) + preserving PHASE at the
readout let multi-chain beat single-chain — where the old summing/magnitude
readout could not, by construction?

Models (PCT phase-coherent attention throughout; param counts printed):
    real                 standard real transformer (sees I,Q)        [reference]
    cplx_single          PCT, 1 chain, phase-preserving readout
    cplx_multi_sum       PCT, 2 chains, OLD summing readout (≈ average) → expect TIE
    cplx_multi_crossterm PCT, 2 chains, NEW cross-term readout       → the test

Data: RadioML (real) via --path, else synthetic fallback.

Run it yourself:
    # real data:
    python pct_x/exp_real.py --path /path/to/RML2016.10a_dict.pkl --seeds 0 1 2
    # or synthetic, to see the pipeline immediately:
    python pct_x/exp_real.py --data synth --seeds 0
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
sys.path.insert(0, str(_HERE.parent / "cvnn"))
sys.path.insert(0, str(_HERE.parent.parent / "main"))
from pct import PCTClassifier, n_params                                  # noqa: E402
from interp import GenuineInterpClassifier                               # noqa: E402
from models import RealSeqClassifier                                     # noqa: E402 (cvnn)
from data_real import load_radioml, load_synth_fallback                  # noqa: E402
from soliton import epsilon_helix_loss                                   # noqa: E402


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_eval(model, Ztr, ytr, Zte, yte, dev, epochs, multi, lr=1e-3, bs=64,
               lam_eps=0.05, warmup_frac=0.4):
    Ztr, ytr, Zte, yte = Ztr.to(dev), ytr.to(dev), Zte.to(dev), yte.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(); n = len(ytr); warm = int(warmup_frac * epochs)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if multi:
                out, psi = model(Ztr[idx], return_psi=True)
                loss = lossf(out, ytr[idx])
                if ep >= warm:
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
        te = (model(Zte).argmax(1) == yte).float().mean().item()
    return te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["radioml", "synth"], default="radioml")
    ap.add_argument("--path", default=None)
    ap.add_argument("--snr_min", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.data == "radioml" and not args.path:
        print("[info] no --path given; falling back to synthetic. "
              "Pass --path RML2016.10a_dict.pkl for the real-data test.")
        args.data = "synth"
    dev = args.device or _device()

    if args.data == "radioml":
        X, y, names = load_radioml(args.path, args.snr_min)
        T, ncl = X.shape[1], len(names)
        print(f"[exp_real] REAL RadioML: {len(y)} samples, T={T}, {ncl} classes "
              f"(SNR>={args.snr_min})")
    else:
        Ztr0, ytr0, Zte0, yte0, names = load_synth_fallback(T=128)
        T, ncl = 128, len(names)
        print(f"[exp_real] SYNTHETIC fallback: T={T}, {ncl} classes")

    CONFIGS = {
        "real":                 (lambda: RealSeqClassifier(n_classes=ncl, T=T), False),
        "cplx_single":          (lambda: PCTClassifier(n_chains=1, n_classes=ncl, T=T), False),
        "cplx_multi_sum":       (lambda: PCTClassifier(n_chains=2, n_classes=ncl, T=T, readout="sum"), True),
        "cplx_multi_phase":     (lambda: PCTClassifier(n_chains=2, n_classes=ncl, T=T, readout="phase"), True),
        "cplx_multi_crossterm": (lambda: PCTClassifier(n_chains=2, n_classes=ncl, T=T, readout="crossterm"), True),
        "genuine_interp":       (lambda: GenuineInterpClassifier(K=2, n_classes=ncl, T=T, coevolve=True), True),
    }
    res = {k: [] for k in CONFIGS}
    for s in args.seeds:
        torch.manual_seed(s); np.random.seed(s)
        if args.data == "radioml":
            g = np.random.RandomState(s); perm = g.permutation(len(y))
            cut = int(0.7 * len(perm))
            tr, te = perm[:cut][:args.max_train], perm[cut:]
            Ztr, ytr, Zte, yte = X[tr], y[tr], X[te], y[te]
        else:
            Ztr, ytr, Zte, yte = Ztr0, ytr0, Zte0, yte0
        for name, (build, multi) in CONFIGS.items():
            torch.manual_seed(s); np.random.seed(s)
            m = build().to(dev)
            te = train_eval(m, Ztr, ytr, Zte, yte, dev, args.epochs, multi)
            res[name].append(te)
            print(f"    seed{s} {name:<22} test={te:.3f}  params={n_params(m)/1e3:.0f}K",
                  flush=True)
            del m; gc.collect()
            if dev == "mps":
                torch.mps.empty_cache()

    print("\n" + "=" * 60)
    print(f"RESULTS (mean ± std over seeds {args.seeds})")
    summ = {}
    for k in CONFIGS:
        v = np.array(res[k]); summ[k] = (v.mean(), v.std())
        print(f"  {k:<22} {v.mean():.3f} ± {v.std():.3f}")
    print("-" * 60)
    cs = summ["cplx_single"][0]; sm = summ["cplx_multi_sum"][0]
    ct = summ["cplx_multi_crossterm"][0]
    print(f"THE TEST: cross-term {ct:.3f}  vs  single {cs:.3f}  vs  sum {sm:.3f}")
    if ct > cs + 0.01 and ct > sm + 0.01:
        print("  -> cross-term readout BEATS single-chain AND the old sum readout: "
              "exposing interference (not averaging) makes multiplicity pay. NOVEL.")
    elif abs(sm - cs) < 0.01 and ct <= cs + 0.01:
        print("  -> sum readout ties single (as predicted); cross-term doesn't beat "
              "single either → multi-chain adds nothing even with the right readout.")
    else:
        print("  -> mixed; read the numbers.")
    print("=" * 60)


if __name__ == "__main__":
    main()
