"""
pct_x/data_real.py  —  REAL complex-valued data (RadioML) + synthetic fallback
==============================================================================

RadioML 2016.10a is the standard real complex-valued benchmark: raw IQ radio
signals (128 complex samples each), 11 modulation classes, where PHASE is
physically meaningful. This is the honest "real data" test.

Download (one-time, ~600 MB):
    1. https://www.deepsig.ai/datasets  → "RML2016.10a"  (RML2016.10a.tar.bz2)
    2. tar xjf RML2016.10a.tar.bz2      → gives RML2016.10a_dict.pkl
    3. put the .pkl somewhere and pass --path /path/to/RML2016.10a_dict.pkl

If you don't have it yet, the synthetic superposition data (cvnn/signals.py) is a
drop-in fallback so the pipeline runs immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cvnn"))


def load_radioml(path, snr_min=6, normalize=True):
    """RML2016.10a pickle -> (X complex (N,128), y (N,), class_names).
    Keeps samples with SNR >= snr_min (higher SNR = cleaner; lower = harder)."""
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in d.keys()})
    idx = {m: i for i, m in enumerate(mods)}
    X, y = [], []
    for (mod, snr), arr in d.items():
        if snr < snr_min:
            continue
        iq = np.asarray(arr, dtype=np.float32)        # (n, 2, 128) = I, Q
        comp = iq[:, 0, :] + 1j * iq[:, 1, :]         # (n, 128) complex
        X.append(comp.astype(np.complex64)); y += [idx[mod]] * len(comp)
    X = np.concatenate(X, 0); y = np.array(y, dtype=np.int64)
    X = torch.from_numpy(X); y = torch.from_numpy(y)
    if normalize:                                      # per-sample power normalisation
        rms = X.abs().pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        X = X / rms
    return X, y, mods


def load_synth_fallback(n_train=2000, n_test=2000, n_classes=8, T=128, seed=0):
    """Synthetic complex signals as a fallback so the experiment runs without the
    RadioML download. Single-chirp classification (phase-carried)."""
    from signals import make_dataset
    Ztr, ytr = make_dataset(n_train, n_classes, T, seed=seed)
    Zte, yte = make_dataset(n_test, n_classes, T, seed=1000 + seed)
    return Ztr, ytr, Zte, yte, [f"class{i}" for i in range(n_classes)]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="RML2016.10a_dict.pkl")
    ap.add_argument("--snr_min", type=int, default=6)
    args = ap.parse_args()
    if args.path:
        X, y, mods = load_radioml(args.path, args.snr_min)
        print(f"[radioml] X {tuple(X.shape)} {X.dtype}, {len(mods)} classes: {mods}")
        print(f"[radioml] samples={len(y)}, class balance={np.bincount(y.numpy())}")
    else:
        Ztr, ytr, Zte, yte, names = load_synth_fallback()
        print(f"[synth fallback] train {tuple(Ztr.shape)}, test {tuple(Zte.shape)}, "
              f"{len(names)} classes")
        print("Provide --path to RML2016.10a_dict.pkl for the real-data test.")
