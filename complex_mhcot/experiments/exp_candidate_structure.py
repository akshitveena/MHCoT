"""
experiments/exp_candidate_structure.py  —  EXPERIMENT −1
========================================================

Characterize the diversity INSIDE each GoT node before modeling anything.
Pure analysis, no training. This is GATE 1′.

Prereqs (run these first):
    python main/preprocess_got.py --split test --limit N   # produces GoT cache
    python main/encoder.py                                   # produces pooled hidden cache

Analyses
--------
1. Per-problem spread vs correctness
   spread = mean pairwise cosine distance among a problem's candidate vectors.
   Hypothesis: low spread (candidates agree in representation space) → correct.
   Metric: AUC of (−spread) predicting correctness. This is the continuous,
   representation-space version of the discrete agreement signal.

2. Same-answer vs different-answer distance
   Are candidates that reach the SAME final answer closer in representation
   space than those that disagree? Validates that the embedding captures
   reasoning outcome (justifies the later contrastive objective).

3. Natural ε (the bridge to Option 1)
   Lift candidate vectors to complex space and measure the phase gap between
   candidate pairs → an empirical ε scale. NOTE: with an UNTRAINED lift the
   phase is near-trivial, so this is a BASELINE; the meaningful ε is measured
   again after training (Exp −1b).

4. Global structure
   PCA of all candidate vectors colored by correctness; k-means cluster purity.
   Does correct vs incorrect reasoning separate in representation space?

GATE 1′ verdict
---------------
- candidates distinct in representation space?  (mean spread > ~0)
- does diversity relate to correctness?         (Analysis 1 AUC > 0.6)
- same-answer closer than different-answer?     (Analysis 2 effect)
If yes → diversity is real, both options alive. If no → stop and diagnose.

Outputs: results/exp_candidate_structure/{summary.json, pca.png, spread_hist.png}
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- make main/ importable -------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent / "main"))

from data import load_dataset                      # noqa: E402
from encoder import load_pooled                    # noqa: E402
from complex_ops import ComplexLift                # noqa: E402

_OUT = _PROJECT_ROOT / "results" / "exp_candidate_structure"
_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cosine_distance_matrix(V: torch.Tensor) -> torch.Tensor:
    """V: (n, D) → (n, n) cosine-distance matrix."""
    Vn = F.normalize(V, dim=-1)
    sim = Vn @ Vn.T
    return 1.0 - sim


def mean_pairwise_distance(V: torch.Tensor) -> float:
    n = V.shape[0]
    if n < 2:
        return 0.0
    D = cosine_distance_matrix(V)
    iu = torch.triu_indices(n, n, offset=1)
    return D[iu[0], iu[1]].mean().item()


def wrapped_phase_gap(za: torch.Tensor, zb: torch.Tensor) -> float:
    """L2 norm of per-dim phase difference (wrapped to [-π,π]), per-dim scale."""
    dphi = torch.angle(za) - torch.angle(zb)
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi      # wrap
    return (dphi.norm() / math.sqrt(za.numel())).item()    # per-dimension scale


def majority_answer(preds: list[str | None]) -> str | None:
    vals = [p for p in preds if p is not None]
    if not vals:
        return None
    counts: dict[str, int] = {}
    for p in vals:
        counts[p] = counts.get(p, 0) + 1
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    records = load_dataset()
    rec_by_idx = {r.idx: r for r in records}

    import os
    POOL = os.environ.get("MHCOT_POOL", "lastk")   # mean | lastk | last
    pooled_path = _PROJECT_ROOT / "data" / "hidden_cache" / "gsm8k_test_pools.pt"
    if not pooled_path.exists():
        print(f"[error] multi-pool cache not found: {pooled_path}")
        print("        run `python main/encoder.py` first (after preprocessing).")
        sys.exit(1)
    pooled = load_pooled(pooled_path, pool=POOL)
    print(f"[exp-1] {len(pooled)} problems | POOL = {POOL}")

    # Determine D and an untrained ComplexLift (fixed seed for reproducibility)
    any_e = next(iter(pooled.values()))
    D = any_e["candidates"].shape[-1]
    torch.manual_seed(0)
    lift = ComplexLift(D)
    lift.eval()

    spreads: list[float] = []
    final_correct: list[int] = []
    consensus_correct: list[int] = []
    same_ans_dists: list[float] = []
    diff_ans_dists: list[float] = []
    eps_gaps: list[float] = []

    all_vecs: list[np.ndarray] = []
    all_correct: list[int] = []

    for idx, e in pooled.items():
        V = e["candidates"]                       # (n_cand, D) float
        cand_correct = e["cand_correct"]
        rec = rec_by_idx.get(idx)
        preds = [c.pred for c in rec.candidates] if rec else [None] * V.shape[0]

        # Analysis 1: spread
        spread = mean_pairwise_distance(V)
        spreads.append(spread)
        final_correct.append(int(e["final_correct"]))
        maj = majority_answer(preds)
        consensus_correct.append(int(maj is not None and maj == e["gold"]))

        # Analysis 2: same vs different answer distances
        Dmat = cosine_distance_matrix(V)
        n = V.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                d = Dmat[i, j].item()
                if preds[i] is not None and preds[j] is not None:
                    (same_ans_dists if preds[i] == preds[j] else diff_ans_dists).append(d)

        # Analysis 3: natural ε via untrained complex lift
        with torch.no_grad():
            Z = lift(V)                            # (n_cand, D) complex
        for i in range(n):
            for j in range(i + 1, n):
                eps_gaps.append(wrapped_phase_gap(Z[i], Z[j]))

        # Analysis 4: collect for global PCA / kmeans
        for i in range(n):
            all_vecs.append(V[i].numpy())
            all_correct.append(int(cand_correct[i]))

    spreads = np.array(spreads)
    final_correct = np.array(final_correct)
    consensus_correct = np.array(consensus_correct)

    # ---- metrics ----------------------------------------------------------
    def safe_auc(y, x):
        y = np.asarray(y)
        if len(set(y.tolist())) < 2:
            return float("nan")
        return float(roc_auc_score(y, x))

    # low spread → correct, so predictor is (−spread)
    auc_final = safe_auc(final_correct, -spreads)
    auc_consensus = safe_auc(consensus_correct, -spreads)

    same_mean = float(np.mean(same_ans_dists)) if same_ans_dists else float("nan")
    diff_mean = float(np.mean(diff_ans_dists)) if diff_ans_dists else float("nan")
    eps_arr = np.array(eps_gaps)

    # ---- global structure -------------------------------------------------
    X = np.stack(all_vecs)
    yc = np.array(all_correct)
    pca = PCA(n_components=2).fit(X)
    X2 = pca.transform(X)

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    # cluster purity wrt correctness
    purity = 0.0
    for c in range(2):
        mask = km.labels_ == c
        if mask.sum() > 0:
            maj = max(yc[mask].mean(), 1 - yc[mask].mean())
            purity += maj * mask.sum()
    purity /= len(yc)

    # ---- report -----------------------------------------------------------
    print("=" * 66)
    print(f"ANALYSIS 1 — spread vs correctness")
    print(f"  mean spread (cosine dist among candidates) : {spreads.mean():.4f}")
    print(f"  AUC( low spread → final_node correct )      : {auc_final:.3f}")
    print(f"  AUC( low spread → consensus correct )       : {auc_consensus:.3f}")
    print(f"ANALYSIS 2 — same vs different answer distance")
    print(f"  mean dist  same-answer pairs : {same_mean:.4f}")
    print(f"  mean dist  diff-answer pairs : {diff_mean:.4f}")
    print(f"  separation (diff − same)     : {diff_mean - same_mean:+.4f}  (want > 0)")
    print(f"ANALYSIS 3 — natural ε (UNTRAINED lift — baseline only)")
    print(f"  per-dim phase gap: mean {eps_arr.mean():.4f}  "
          f"median {np.median(eps_arr):.4f}  std {eps_arr.std():.4f}")
    print(f"ANALYSIS 4 — global structure")
    print(f"  PCA explained var (2 comp) : {pca.explained_variance_ratio_.sum():.3f}")
    print(f"  k-means(2) purity wrt correctness : {purity:.3f}")
    print("=" * 66)
    print("GATE 1′ verdict:")
    g_distinct = spreads.mean() > 0.01
    g_signal = (not math.isnan(auc_consensus)) and max(auc_final, auc_consensus) > 0.60
    g_sameans = (diff_mean - same_mean) > 0
    print(f"  candidates distinct in repr space? {'YES' if g_distinct else 'NO'}")
    print(f"  diversity predicts correctness?    {'YES' if g_signal else 'NO'}  "
          f"(best AUC {max(auc_final, auc_consensus):.3f})")
    print(f"  same-answer closer than different? {'YES' if g_sameans else 'NO'}")
    verdict = g_distinct and g_signal and g_sameans
    print(f"  → {'PASS — proceed to Option 3 / Option 1' if verdict else 'INVESTIGATE before proceeding'}")
    print("=" * 66)

    # ---- plots ------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    for lab, col, name in [(1, "tab:green", "correct"), (0, "tab:red", "wrong")]:
        m = yc == lab
        plt.scatter(X2[m, 0], X2[m, 1], s=14, alpha=0.5, c=col, label=name)
    plt.legend(); plt.title("Candidate vectors (PCA), colored by correctness")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    plt.savefig(_OUT / "pca.png", dpi=120); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(spreads[final_correct == 1], bins=20, alpha=0.6, label="correct", color="tab:green")
    plt.hist(spreads[final_correct == 0], bins=20, alpha=0.6, label="wrong", color="tab:red")
    plt.legend(); plt.xlabel("per-problem candidate spread"); plt.ylabel("count")
    plt.title("Spread distribution by correctness"); plt.tight_layout()
    plt.savefig(_OUT / "spread_hist.png", dpi=120); plt.close()

    summary = {
        "n_problems": len(spreads),
        "mean_spread": float(spreads.mean()),
        "auc_spread_final": auc_final,
        "auc_spread_consensus": auc_consensus,
        "same_answer_mean_dist": same_mean,
        "diff_answer_mean_dist": diff_mean,
        "answer_distance_separation": diff_mean - same_mean,
        "natural_eps_mean": float(eps_arr.mean()),
        "natural_eps_median": float(np.median(eps_arr)),
        "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
        "kmeans_purity": float(purity),
        "gate_distinct": bool(g_distinct),
        "gate_signal": bool(g_signal),
        "gate_same_answer": bool(g_sameans),
        "gate_pass": bool(verdict),
    }
    with open(_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {_OUT}/summary.json, pca.png, spread_hist.png")


if __name__ == "__main__":
    main()
