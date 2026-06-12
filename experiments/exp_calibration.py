"""
experiments/exp_calibration.py  —  CALIBRATION + SELECTIVE PREDICTION
====================================================================

Tests MHCoT's LITERAL headline claim: does complex interference give better /
more useful uncertainty than a real model's confidence?

Loads the trained models (same val split as train.py / exp5):
    complex MHCoT : results/train/best_model.pt
    real twin     : results/exp5_ablation/real_model.pt

Compares three confidence signals on the SAME 120 val traces:
    A. real model    — sigmoid(score)
    B. complex model — sigmoid(score)
    C. complex model — INTERFERENCE (I_answer)   ← MHCoT's "free uncertainty"

Two questions:
  1. CALIBRATION (ECE): when a model says 80% confident, is it right 80%?
     Real models are famously over-confident — if complex is better-calibrated,
     that's a genuine win at equal accuracy.
  2. SELECTIVE TRACE-QUALITY: keep the top-k% traces by each confidence signal —
     what fraction are actually correct? (base rate ≈ 78%.) Does interference
     let us pick out correct reasoning better than a real model's score?

PRE-COMMITTED BAR (set before seeing results):
  complex "wins calibration" if its ECE is lower than real by >0.03, OR
  interference selects correct traces better than real-score at 50% coverage
  by >3 points (outside the ~±4pt noise on 120 samples).

Run:  python experiments/exp_calibration.py
Out:  results/exp_calibration/{summary.json, reliability.png, risk_coverage.png}
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "main"))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

from encoder import load_sequences, SEQ_PATH          # noqa: E402
from model import MHCoTEncoder                          # noqa: E402
from train import SeqDataset, collate, _device         # noqa: E402
from exp5_ablation import RealEncoder                   # noqa: E402

_OUT = _PROJECT_ROOT / "results" / "exp_calibration"
_OUT.mkdir(parents=True, exist_ok=True)
_COMPLEX_CKPT = _PROJECT_ROOT / "results" / "train" / "best_model.pt"
_REAL_CKPT = _PROJECT_ROOT / "results" / "exp5_ablation" / "real_model.pt"


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error (max-confidence binning)."""
    conf = np.maximum(probs, 1 - probs)
    pred = (probs > 0.5).astype(int)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            e += abs(correct[m].mean() - conf[m].mean()) * m.sum()
    return e / len(probs)


def brier(probs, labels):
    return float(np.mean((probs - labels) ** 2))


def precision_at_coverage(confidence, label, cov):
    """Keep top `cov` fraction by confidence; return fraction correct (label=1)."""
    k = max(1, int(round(len(label) * cov)))
    order = np.argsort(-confidence)[:k]
    return float(label[order].mean())


def risk_coverage(confidence, label):
    """Return (coverages, precision) sweeping coverage from high→low confidence."""
    order = np.argsort(-confidence)
    lab = label[order]
    n = len(lab)
    cov = np.arange(1, n + 1) / n
    prec = np.cumsum(lab) / np.arange(1, n + 1)
    return cov, prec


@torch.no_grad()
def collect(model, loader, dev, is_complex):
    model.eval()
    scores, interf, labels = [], [], []
    for H, mask, lengths, y in loader:
        H, mask, lengths = H.to(dev), mask.to(dev), lengths.to(dev)
        out = model(H, key_padding_mask=mask, lengths=lengths)
        scores.append(out["score"].float().cpu())
        interf.append(out["I_answer"].float().cpu())
        labels.append(y)
    s = torch.cat(scores).numpy()
    i = torch.cat(interf).numpy()
    y = torch.cat(labels).numpy()
    return s, i, y


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex_ckpt", type=str, default=str(_COMPLEX_CKPT),
                    help="complex model checkpoint (default: full-loss model)")
    ap.add_argument("--real_ckpt", type=str, default=str(_REAL_CKPT))
    args = ap.parse_args()
    complex_ckpt = Path(args.complex_ckpt)
    real_ckpt = Path(args.real_ckpt)

    dev = _device()
    print(f"[calib] device={dev}")
    print(f"[calib] complex ckpt: {complex_ckpt.name}  |  real ckpt: {real_ckpt.name}")
    for p in (complex_ckpt, real_ckpt):
        if not p.exists():
            print(f"[error] missing checkpoint: {p}")
            print("        run main/train.py and experiments/exp5_ablation.py first.")
            sys.exit(1)

    # SAME val split as train.py / exp5 (seed=0, val_frac=0.2, stratify)
    cache = load_sequences()
    items, labels = [], []
    for e in cache.values():
        items.append((e["h"], e["length"], e["label"]))
        labels.append(e["label"])
    d_in = items[0][0].shape[1]
    _, va = train_test_split(items, test_size=0.2, random_state=0, stratify=labels)
    coll = partial(collate, max_seq_len=384)
    vl = DataLoader(SeqDataset(va), batch_size=8, shuffle=False, collate_fn=coll)

    # load models
    cx = MHCoTEncoder(d_in=d_in, d_model=128).to(dev)
    cx.load_state_dict(torch.load(complex_ckpt, map_location=dev))
    rl = RealEncoder(d_in=d_in, d_model=128, n_layers=4).to(dev)
    rl.load_state_dict(torch.load(real_ckpt, map_location=dev))

    cx_s, cx_i, y = collect(cx, vl, dev, True)
    rl_s, _, y2 = collect(rl, vl, dev, False)
    assert np.array_equal(y, y2), "val labels mismatch"
    base = float(y.mean())

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    # confidence signals → probabilities (interference standardized→sigmoid)
    p_real = sig(rl_s)
    p_cx = sig(cx_s)
    i_norm = (cx_i - cx_i.mean()) / (cx_i.std() + 1e-8)
    p_interf = sig(i_norm)

    # ---- calibration ----
    res = {
        "base_rate_correct": base, "n_val": int(len(y)),
        "ece": {"real_score": ece(p_real, y), "complex_score": ece(p_cx, y),
                "complex_interference": ece(p_interf, y)},
        "brier": {"real_score": brier(p_real, y), "complex_score": brier(p_cx, y)},
    }

    # ---- selective trace-quality (precision @ coverage) ----
    res["precision_at_coverage"] = {}
    for cov in (0.25, 0.5, 0.75):
        res["precision_at_coverage"][f"{int(cov*100)}%"] = {
            "real_score": precision_at_coverage(rl_s, y, cov),
            "complex_score": precision_at_coverage(cx_s, y, cov),
            "complex_interference": precision_at_coverage(cx_i, y, cov),
        }

    # ---- report ----
    print("=" * 70)
    print(f"val n={len(y)} | base rate correct = {base:.3f}")
    print("-" * 70)
    print("CALIBRATION — Expected Calibration Error (lower = better):")
    print(f"  real  score        : ECE {res['ece']['real_score']:.3f}  "
          f"Brier {res['brier']['real_score']:.3f}")
    print(f"  complex score      : ECE {res['ece']['complex_score']:.3f}  "
          f"Brier {res['brier']['complex_score']:.3f}")
    print(f"  complex interference: ECE {res['ece']['complex_interference']:.3f}")
    print("-" * 70)
    print(f"SELECTIVE TRACE-QUALITY — % correct among top-k by confidence "
          f"(base {base:.2f}):")
    print(f"  {'coverage':<10}{'real-score':>12}{'cx-score':>12}{'cx-interf':>12}")
    for cov in ("25%", "50%", "75%"):
        d = res["precision_at_coverage"][cov]
        print(f"  {cov:<10}{d['real_score']:>12.3f}{d['complex_score']:>12.3f}"
              f"{d['complex_interference']:>12.3f}")
    print("=" * 70)

    # ---- verdict vs pre-committed bar ----
    ece_gain = res["ece"]["real_score"] - res["ece"]["complex_score"]
    sel_gain = (res["precision_at_coverage"]["50%"]["complex_interference"]
                - res["precision_at_coverage"]["50%"]["real_score"])
    print("VERDICT (pre-committed bar):")
    print(f"  complex ECE better than real by {ece_gain:+.3f} "
          f"({'WIN >0.03' if ece_gain > 0.03 else 'not decisive'})")
    print(f"  interference selects better than real-score @50% by {sel_gain:+.3f} "
          f"({'WIN >0.03' if sel_gain > 0.03 else 'not decisive'})")
    if ece_gain > 0.03 or sel_gain > 0.03:
        print("  → COMPLEX shows a real uncertainty advantage on this task.")
    else:
        print("  → No decisive uncertainty advantage yet (needs more data, or")
        print("    the advantage isn't here). Honest read.")
    print("=" * 70)

    # ---- reliability diagram ----
    plt.figure(figsize=(6, 6))
    for name, p, c in [("real score", p_real, "tab:blue"),
                       ("complex score", p_cx, "tab:orange"),
                       ("complex interf", p_interf, "tab:green")]:
        conf = np.maximum(p, 1 - p)
        pred = (p > 0.5).astype(int)
        corr = (pred == y).astype(float)
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for k in range(10):
            m = (conf > edges[k]) & (conf <= edges[k + 1])
            if m.sum() > 0:
                xs.append(conf[m].mean()); ys.append(corr[m].mean())
        plt.plot(xs, ys, "o-", label=name, color=c)
    plt.plot([0.5, 1], [0.5, 1], "k--", alpha=0.4, label="perfect")
    plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.legend()
    plt.title("Reliability diagram"); plt.tight_layout()
    plt.savefig(_OUT / "reliability.png", dpi=120); plt.close()

    # ---- risk-coverage ----
    plt.figure(figsize=(7, 4))
    for name, conf, c in [("real score", rl_s, "tab:blue"),
                          ("complex score", cx_s, "tab:orange"),
                          ("complex interference", cx_i, "tab:green")]:
        cov, prec = risk_coverage(conf, y)
        plt.plot(cov, prec, label=name, color=c)
    plt.axhline(base, color="k", ls="--", alpha=0.4, label=f"base {base:.2f}")
    plt.xlabel("coverage (fraction kept, high→low confidence)")
    plt.ylabel("precision (% correct among kept)")
    plt.legend(); plt.title("Selective trace-quality"); plt.tight_layout()
    plt.savefig(_OUT / "risk_coverage.png", dpi=120); plt.close()

    json.dump(res, open(_OUT / "summary.json", "w"), indent=2)
    print(f"[saved] {_OUT}/summary.json, reliability.png, risk_coverage.png")


if __name__ == "__main__":
    main()
