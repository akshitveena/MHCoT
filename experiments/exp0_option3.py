"""
experiments/exp0_option3.py  —  EXPERIMENT 0  (Option 3 interference probe)
==========================================================================

THE GATE-1 QUESTION:
    Does cross-candidate INTERFERENCE predict correctness?

For each problem we lift its candidate pooled vectors to complex space and
superpose them:
        Ψ = Σ_i ẑ_i           (ẑ_i = unit-normalized complex lift of candidate i)
        I = |Ψ|² / n²          ∈ [0, 1]   (1 = all aligned/agree, 0 = cancel)
Then measure AUC of I as a predictor of correctness. High I (constructive
interference = candidates agree) should predict correct answers.

SCOPE / honesty
---------------
With only ~10²–10³ problems, training a rich D×D W_imag (~4.7M params) would
overfit catastrophically, and calibrating a scalar (a·I+b) does NOT change AUC
(threshold-invariant). So the ROBUST gate-1 test is the UNTRAINED probe with a
fixed-seed lift. The TRAINED Option-3 (contrastive embedding development on the
full dataset) is a separate, larger build — deferred until preprocessing yields
the full set and the contrastive objective is implemented.

We also report a pure-REAL agreement baseline. At an untrained lift the complex
and real numbers should be close (W_imag is small) — that is expected and is a
sanity check, not a failure. Complex's advantage is a TRAINED phenomenon.

Prereqs:
    python main/encoder.py        # pooled hidden cache must exist

Outputs: results/exp0_option3/{summary.json, interference_scatter.png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "main"))

from data import load_dataset                  # noqa: E402
from encoder import load_pooled                # noqa: E402
from complex_ops import ComplexLift            # noqa: E402

_OUT = _PROJECT_ROOT / "results" / "exp0_option3"
_OUT.mkdir(parents=True, exist_ok=True)


def normalized_interference(Z: torch.Tensor) -> float:
    """
    Z: (n, D) complex candidate lifts. Returns |Σ ẑ_i|² / n²  in [0,1],
    where ẑ_i is unit-normalized. 1 = perfect agreement, 0 = cancellation.
    """
    n = Z.shape[0]
    if n == 0:
        return float("nan")
    norms = Z.abs().pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-8
    Zn = Z / norms
    Psi = Zn.sum(dim=0)
    return (Psi.abs().pow(2).sum() / (n * n)).item()


def normalized_real_agreement(V: torch.Tensor) -> float:
    """Pure-real baseline: |Σ v̂_i|² / n²  on unit-normalized REAL vectors."""
    n = V.shape[0]
    Vn = F.normalize(V, dim=-1)
    s = Vn.sum(dim=0)
    return (s.pow(2).sum() / (n * n)).item()


def majority_correct(preds, gold) -> int:
    vals = [p for p in preds if p is not None]
    if not vals:
        return 0
    counts: dict[str, int] = {}
    for p in vals:
        counts[p] = counts.get(p, 0) + 1
    maj = max(counts, key=counts.get)
    return int(maj == gold)


def safe_auc(y, x):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, x))


def main() -> None:
    pooled_path = _PROJECT_ROOT / "data" / "hidden_cache" / "gsm8k_test_pooled.pt"
    if not pooled_path.exists():
        print(f"[error] pooled cache not found: {pooled_path}")
        print("        run `python main/encoder.py` first.")
        sys.exit(1)
    pooled = load_pooled(pooled_path)
    records = {r.idx: r for r in load_dataset()}
    print(f"[exp0] {len(pooled)} problems with pooled vectors")

    D = next(iter(pooled.values()))["candidates"].shape[-1]
    torch.manual_seed(0)
    lift = ComplexLift(D).eval()

    I_complex, I_real = [], []
    final_correct, consensus_correct = [], []

    with torch.no_grad():
        for idx, e in pooled.items():
            V = e["candidates"]                    # (n, D) real
            if V.shape[0] < 2:
                continue
            Z = lift(V)                            # (n, D) complex
            I_complex.append(normalized_interference(Z))
            I_real.append(normalized_real_agreement(V))
            final_correct.append(int(e["final_correct"]))
            rec = records.get(idx)
            preds = [c.pred for c in rec.candidates] if rec else []
            consensus_correct.append(majority_correct(preds, e["gold"]))

    I_complex = np.array(I_complex)
    I_real = np.array(I_real)
    final_correct = np.array(final_correct)
    consensus_correct = np.array(consensus_correct)

    auc_cx_final = safe_auc(final_correct, I_complex)
    auc_cx_cons = safe_auc(consensus_correct, I_complex)
    auc_re_final = safe_auc(final_correct, I_real)
    auc_re_cons = safe_auc(consensus_correct, I_real)

    best_cx = np.nanmax([auc_cx_final, auc_cx_cons])

    print("=" * 64)
    print(f"n problems (≥2 candidates): {len(I_complex)}")
    print(f"mean interference: complex {I_complex.mean():.3f} | real {I_real.mean():.3f}")
    print("-" * 64)
    print(f"{'predictor':<22}{'vs final':>10}{'vs consensus':>14}")
    print(f"{'complex interference':<22}{auc_cx_final:>10.3f}{auc_cx_cons:>14.3f}")
    print(f"{'real agreement (base)':<22}{auc_re_final:>10.3f}{auc_re_cons:>14.3f}")
    print("=" * 64)
    print("GATE 1 verdict (untrained probe):")
    if best_cx > 0.70:
        print(f"  STRONG — best AUC {best_cx:.3f} > 0.70. Interference predicts correctness.")
    elif best_cx > 0.60:
        print(f"  PASS — best AUC {best_cx:.3f} > 0.60. Worth pursuing Option 1.")
    elif best_cx > 0.55:
        print(f"  WEAK — best AUC {best_cx:.3f}. Re-check on full dataset before deciding.")
    else:
        print(f"  NULL — best AUC {best_cx:.3f} ≈ 0.5. Interference does NOT predict correctness.")
    print(f"  (complex ≈ real at untrained lift is EXPECTED; complex's edge is trained.)")
    print("=" * 64)

    # scatter
    plt.figure(figsize=(7, 4))
    jit = (np.random.rand(len(consensus_correct)) - 0.5) * 0.12
    plt.scatter(I_complex, consensus_correct + jit, s=20, alpha=0.6)
    plt.xlabel("normalized complex interference  I = |Σẑ|²/n²")
    plt.ylabel("consensus correct (0/1, jittered)")
    plt.title(f"Exp 0 — interference vs correctness  (AUC {auc_cx_cons:.3f}, N={len(I_complex)})")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(_OUT / "interference_scatter.png", dpi=120); plt.close()

    summary = {
        "n_problems": int(len(I_complex)),
        "mean_interference_complex": float(I_complex.mean()),
        "mean_interference_real": float(I_real.mean()),
        "auc_complex_vs_final": auc_cx_final,
        "auc_complex_vs_consensus": auc_cx_cons,
        "auc_real_vs_final": auc_re_final,
        "auc_real_vs_consensus": auc_re_cons,
        "best_complex_auc": float(best_cx),
        "gate1_pass": bool(best_cx > 0.60),
        "note": "untrained probe; trained Option-3 (contrastive, full dataset) deferred",
    }
    with open(_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {_OUT}/summary.json, interference_scatter.png")


if __name__ == "__main__":
    main()
