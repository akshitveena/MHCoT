"""
tbert/genuine_vs_spurious.py  —  the heart of the idea.

Both Q1 (answer ✓, steps ✓ = genuine) and Q2 (answer ✓, steps ✗ = spurious) are
ANSWER-CORRECT — so the answer and its surface-form correlates are held fixed.
Whatever separates them is closer to genuine reasoning validity (this defeats the
R6 surface-form confound).

Two claims, tested on the frozen-DeepSeek embeddings (the representation that won),
leak-free (split by problem), 3 seeds:

  (1) SEPARABLE?  a probe on the representation tells genuine (Q1) from spurious (Q2).
  (2) FOOLED?     an ANSWER-correctness verifier scores Q2 ~ Q1 (can't tell them apart),
                  while the validity-probe can -> the standard signal is blind to it.

    python tbert/genuine_vs_spurious.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import extract_answer, _to_float, _eq                  # noqa: E402
from quadrants import step_check                                 # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_POOLS = _ROOT / "data" / "hidden_cache" / "gsm8k_test_pools.pt"
_GOT = _ROOT / "data" / "got_cache" / "gsm8k_test.jsonl"


def load_joined():
    pools = torch.load(_POOLS, weights_only=False)
    got = {json.loads(l)["idx"]: json.loads(l) for l in open(_GOT)}
    sample = next(iter(pools.values()))["candidates"]
    pool = "lastk"
    if isinstance(sample, dict):
        pool = "mean" if "mean" in sample else next(iter(sample))   # whole-trace if available
    X, ac, sv, ck, pid = [], [], [], [], []
    for idx, e in pools.items():
        if idx not in got:
            continue
        ce = e["candidates"]
        vecs = ce[pool] if isinstance(ce, dict) else ce
        cands = got[idx].get("candidates", [])
        gold = _to_float(got[idx].get("gold"))
        if gold is None or len(cands) != len(vecs):
            continue
        for i, c in enumerate(cands):
            t = c["text"]
            ne, nb = step_check(t)
            X.append(vecs[i]); pid.append(idx)
            ac.append(int(_eq(extract_answer(t), gold)))
            ck.append(ne > 0); sv.append(ne > 0 and nb == 0)
    return (torch.stack(X).float().numpy(), np.array(ac), np.array(sv, bool),
            np.array(ck, bool), np.array(pid), pool)


def _probe_auc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    s = clf.decision_function(sc.transform(Xte))
    return s, (roc_auc_score(yte, s) if len(set(yte)) > 1 else float("nan"))


def _test():
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score
    X, ac, sv, ck, pid, pool = load_joined()
    print(f"[g-vs-s] {len(ac)} candidates, pooling='{pool}', dim={X.shape[1]}")
    corr_ck = ac.astype(bool) & ck                       # answer-correct & checkable = Q1 ∪ Q2
    print(f"[g-vs-s] answer-correct & checkable: {corr_ck.sum()}  "
          f"(Q1 genuine={int((corr_ck & sv).sum())}, Q2 spurious={int((corr_ck & ~sv).sum())})\n")

    sep, fooled, ans_overall = [], [], []
    for seed in [0, 1, 2]:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        tr, te = next(gss.split(X, ac, groups=pid))
        trs, tes = set(tr), set(te)

        # (2a) answer-correctness verifier: trained on ALL candidates (label = answer-correct)
        ans_s, ans_auc = _probe_auc(X[tr], ac[tr], X[te], ac[te])
        ans_overall.append(ans_auc)

        # restrict test to answer-correct & checkable -> Q1 vs Q2
        te_mask = corr_ck[te]
        y_gen = sv[te][te_mask].astype(int)              # 1 = Q1 genuine, 0 = Q2 spurious
        # answer-verifier's score on those: can IT tell Q1 from Q2? (expect ~0.5 = fooled)
        f_auc = roc_auc_score(y_gen, ans_s[te_mask]) if len(set(y_gen)) > 1 else float("nan")
        fooled.append(f_auc)

        # (1) dedicated validity probe: trained on Q1-vs-Q2 in TRAIN, tested on Q1-vs-Q2 in TEST
        tr_mask = corr_ck[tr]
        _, sep_auc = _probe_auc(X[tr][tr_mask], sv[tr][tr_mask].astype(int),
                                X[te][te_mask], y_gen)
        sep.append(sep_auc)
        print(f"  seed{seed}: validity-probe Q1-vs-Q2 AUC={sep_auc:.3f} | "
              f"answer-verifier on Q1-vs-Q2 AUC={f_auc:.3f} | (answer-verifier overall AUC={ans_auc:.3f})")

    print(f"\n[RESULT] genuine(Q1) vs spurious(Q2) — both answer-correct:")
    print(f"  validity-probe AUC      = {np.nanmean(sep):.3f} ± {np.nanstd(sep):.3f}   (can the representation tell them apart?)")
    print(f"  answer-verifier AUC     = {np.nanmean(fooled):.3f} ± {np.nanstd(fooled):.3f}   (does the standard signal tell them apart? ~0.5 = FOOLED)")
    print(f"  [context] answer-verifier overall correct/incorrect AUC = {np.nanmean(ans_overall):.3f}")

    gap = np.nanmean(sep) - np.nanmean(fooled)
    if np.nanmean(sep) > 0.6 and np.nanmean(fooled) < 0.6:
        print(f"\n[NOVEL + HONEST] The representation encodes genuine-vs-spurious correctness "
              f"(probe AUC {np.nanmean(sep):.2f}), but the answer-trained verifier is largely "
              f"BLIND to it (AUC {np.nanmean(fooled):.2f}). Answer-correctness is a leaky label; "
              f"the validity signal exists but standard verifiers don't use it.")
    else:
        print(f"\n[HONEST] separation gap = {gap:+.3f}; report exactly what we see "
              f"(if the probe also can't separate, the distinction isn't in this pooling).")


if __name__ == "__main__":
    _test()
