"""
real_mhcot/rerank.py  —  R3: reranking (where a thought encoder earns its name)
===============================================================================

The real payoff of a thought EMBEDDER (vs a scalar classifier): use it to pick
the best candidate per problem. For each test problem we score its candidates by
cosine to the correct-prototype and select the top one; the chosen candidate's
correctness is the final-answer accuracy. We compare to random selection (the
per-problem base rate) and the oracle ceiling (any-correct).

Run the self-test:
    python real_mhcot/rerank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_thoughts, split_by_problem                    # noqa: E402
from encoder import ThoughtEncoder, train_contrastive               # noqa: E402


@torch.no_grad()
def rerank_eval(model, Xtr, ytr, Xte, yte, pte, dev="cpu"):
    model.eval()
    Ztr = model(Xtr.to(dev)).cpu().numpy()
    proto = Ztr[ytr.numpy() == 1].mean(0)
    proto = proto / (np.linalg.norm(proto) + 1e-9)
    Zte = model(Xte.to(dev)).cpu().numpy()
    scores = Zte @ proto
    yte = yte.numpy(); pte = pte.numpy()

    rerank, rnd, oracle, mixed_rerank, mixed_rnd, n_mixed = [], [], [], [], [], 0
    for p in np.unique(pte):
        idx = np.where(pte == p)[0]
        labels = yte[idx]
        pick = idx[np.argmax(scores[idx])]
        rerank.append(int(yte[pick]))
        rnd.append(labels.mean())            # expected accuracy of a random pick
        oracle.append(float(labels.any()))   # ceiling: a correct candidate exists
        if 0 < labels.sum() < len(labels):   # mixed problem (reranking can matter)
            n_mixed += 1
            mixed_rerank.append(int(yte[pick]))
            mixed_rnd.append(labels.mean())
    return {
        "rerank": float(np.mean(rerank)), "random": float(np.mean(rnd)),
        "oracle": float(np.mean(oracle)),
        "mixed_rerank": float(np.mean(mixed_rerank)) if mixed_rerank else float("nan"),
        "mixed_random": float(np.mean(mixed_rnd)) if mixed_rnd else float("nan"),
        "n_mixed": n_mixed,
    }


def _test():
    dev = "cpu"
    X, y, pid = load_thoughts("lastk")
    seeds = [0, 1, 2]
    rr, rnd, orc, mrr, mrnd = [], [], [], [], []
    for s in seeds:
        (Xtr, ytr, _), (Xte, yte, pte) = split_by_problem(X, y, pid, seed=s)
        torch.manual_seed(s)
        m = ThoughtEncoder(d=128).to(dev)
        train_contrastive(m, Xtr, ytr, dev=dev, steps=2000, seed=s)
        r = rerank_eval(m, Xtr, ytr, Xte, yte, pte, dev=dev)
        rr.append(r["rerank"]); rnd.append(r["random"]); orc.append(r["oracle"])
        mrr.append(r["mixed_rerank"]); mrnd.append(r["mixed_random"])
        print(f"    seed{s}: rerank={r['rerank']:.3f} random={r['random']:.3f} "
              f"oracle={r['oracle']:.3f} | mixed: rerank={r['mixed_rerank']:.3f} "
              f"random={r['mixed_random']:.3f} ({r['n_mixed']} mixed)", flush=True)

    print(f"\n[R3] ALL problems:   rerank {np.mean(rr):.3f}  vs random {np.mean(rnd):.3f}  "
          f"(oracle {np.mean(orc):.3f})")
    print(f"[R3] MIXED problems: rerank {np.nanmean(mrr):.3f}  vs random {np.nanmean(mrnd):.3f}")
    assert np.mean(rr) > np.mean(rnd) + 0.01, "reranker should beat random selection"
    print(f"\n[R3 PASSED] the thought encoder reranks candidates better than random "
          f"— it improves final-answer accuracy toward the oracle.")


if __name__ == "__main__":
    _test()
