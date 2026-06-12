"""
experiments/exp_option3_trained.py  —  OPTION 3: cross-candidate interference
=============================================================================

Option 1 showed: interference between phase-separated chains of ONE node
calibrates. Option 3 asks the natural question: does interference between
GENUINELY DIFFERENT reasoning chains (the GoT candidates) do the same — and
better, since the chains are really distinct?

Setup (operates on the answer-region pooled candidate vectors):
  * Each problem has K=3 GoT candidate vectors (lastk pool).
  * Complex model: lift each to ℂ, superpose Ψ = Σ_c ẑ_c, score from |Ψ|.
    → the score uses cross-candidate INTERFERENCE.
  * Real baseline: same skeleton, real aggregation (no complex lift / no
    interference). Matched params.
  * Target: consensus correctness (is the candidates' majority answer right?).

Metrics: AUC (does it predict correctness?) and ECE (is it calibrated?).
The calibration comparison is the key one — it tests whether the Option-1
interference→calibration finding extends to cross-candidate interference.

Baselines printed for context:
  discrete agreement (SC): do candidates reach the same answer? (the classic vote)

Run:  python experiments/exp_option3_trained.py     (~3 min, pooled vectors)
Out:  results/exp_option3/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "main"))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

from data import load_dataset                     # noqa: E402
from encoder import load_pooled, MULTIPOOL_PATH    # noqa: E402
from complex_ops import ComplexLift                # noqa: E402
from train import _device                          # noqa: E402
from exp_calibration import ece                    # noqa: E402

_OUT = _PROJECT_ROOT / "results" / "exp_option3"
_OUT.mkdir(parents=True, exist_ok=True)
K = 3   # candidates per problem


class Option3Complex(nn.Module):
    """Score from cross-candidate complex interference |Σ_c lift(h_c)|."""
    def __init__(self, d_in, d_model=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_model),
                                  nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.lift = ComplexLift(d_model)
        self.scorer = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                    nn.Linear(d_model // 2, 1))

    def forward(self, cands):                       # (B, K, d_in)
        h = self.proj(cands)                        # (B, K, d_model)
        z = self.lift(h)                            # (B, K, d_model) complex
        Psi = z.sum(dim=1)                          # (B, d_model) superposition
        score = self.scorer(Psi.abs()).squeeze(-1)
        I = (Psi.abs() ** 2).sum(-1) / Psi.shape[-1]
        return {"score": score, "I": I}


class Option3Real(nn.Module):
    """Matched real baseline: real aggregation of candidates, no interference."""
    def __init__(self, d_in, d_model=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_model),
                                  nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.scorer = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                    nn.Linear(d_model // 2, 1))

    def forward(self, cands):
        h = self.proj(cands)                        # (B, K, d_model)
        pooled = h.sum(dim=1)                        # (B, d_model) real sum
        score = self.scorer(pooled.abs()).squeeze(-1)
        return {"score": score, "I": pooled.abs().mean(-1)}


def majority_correct(preds, gold):
    vals = [p for p in preds if p is not None]
    if not vals:
        return 0
    c = {}
    for p in vals:
        c[p] = c.get(p, 0) + 1
    return int(max(c, key=c.get) == gold)


def build(dev):
    """Return X:(N,K,d_in), y:(N,) consensus-correct, and the SC agreement flag."""
    pooled = load_pooled(MULTIPOOL_PATH, pool="lastk")
    records = {r.idx: r for r in load_dataset()}
    X, y, agree = [], [], []
    for idx, e in pooled.items():
        V = e["candidates"]                         # (n_cand, d_in)
        if V.shape[0] < K:
            continue
        rec = records.get(idx)
        preds = [c.pred for c in rec.candidates] if rec else []
        X.append(V[:K])
        y.append(majority_correct(preds, e["gold"]))
        agree.append(int(len(set(p for p in preds[:K] if p is not None)) == 1))
    return torch.stack(X).float(), np.array(y), np.array(agree)


def run(model_cls, Xtr, ytr, Xva, yva, dev, epochs=60):
    model = model_cls(d_in=Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xtr, Xva = Xtr.to(dev), Xva.to(dev)
    for _ in range(epochs):
        model.train()
        out = model(Xtr)
        loss = F.binary_cross_entropy_with_logits(out["score"], ytr_t)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        s = model(Xva)["score"].cpu().numpy()
    p = 1 / (1 + np.exp(-s))
    auc = roc_auc_score(yva, s) if len(set(yva.tolist())) > 1 else float("nan")
    return auc, ece(p, yva), sum(pp.numel() for pp in model.parameters())


def main():
    dev = _device(); print(f"[option3] device={dev}")
    X, y, agree = build(dev)
    print(f"[option3] {len(y)} problems, K={K} candidates each | "
          f"consensus-correct base rate = {y.mean():.3f}")

    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
    Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]

    torch.manual_seed(0)
    cx_auc, cx_ece, cx_p = run(Option3Complex, Xtr, ytr, Xva, yva, dev)
    torch.manual_seed(0)
    rl_auc, rl_ece, rl_p = run(Option3Real, Xtr, ytr, Xva, yva, dev)

    # discrete SC baseline: does "candidates agree" predict consensus correct?
    sc_auc = roc_auc_score(yva, agree[va]) if len(set(yva.tolist())) > 1 else float("nan")

    print("=" * 64)
    print(f"{'model':<28}{'AUC':>8}{'ECE':>8}{'params':>10}")
    print(f"{'complex (interference)':<28}{cx_auc:>8.3f}{cx_ece:>8.3f}{cx_p/1e6:>9.2f}M")
    print(f"{'real (aggregation)':<28}{rl_auc:>8.3f}{rl_ece:>8.3f}{rl_p/1e6:>9.2f}M")
    print(f"{'discrete agreement (SC)':<28}{sc_auc:>8.3f}{'—':>8}")
    print("-" * 64)
    print(f"  AUC: complex {cx_auc:.3f} vs real {rl_auc:.3f}  "
          f"({cx_auc-rl_auc:+.3f})")
    print(f"  ECE: complex {cx_ece:.3f} vs real {rl_ece:.3f}  "
          f"({rl_ece-cx_ece:+.3f} better)")
    if (rl_ece - cx_ece) > 0.03:
        print("  → Cross-candidate INTERFERENCE calibrates better than real")
        print("    aggregation — the Option-1 calibration finding EXTENDS to")
        print("    genuinely different reasoning chains. Stronger story.")
    else:
        print("  → No decisive calibration edge for cross-candidate interference")
        print("    here (small data). Honest read.")
    print("=" * 64)

    json.dump({"n": int(len(y)), "base_rate": float(y.mean()),
               "complex": {"auc": cx_auc, "ece": cx_ece},
               "real": {"auc": rl_auc, "ece": rl_ece},
               "sc_agreement_auc": sc_auc}, open(_OUT / "summary.json", "w"), indent=2)
    print(f"[saved] {_OUT}/summary.json")


if __name__ == "__main__":
    main()
