"""
tbert/validity_features.py  —  what separates GENUINE (Q1) from SPURIOUS (Q2)?

Both are answer-correct, so this is a confound-controlled look at the validity
direction. Two lenses:

  (A) exemplars — the answer-correct candidates the validity-probe scores most
      "genuine" vs most "spurious"; read them.
  (B) interpretable correlates — do simple TEXT features (length, #steps,
      verification language, #equations) separate Q1 from Q2, and which ones does
      the probe lean on? Honest surface-form audit.

We deliberately EXCLUDE n_bad_equations (that IS the Q1/Q2 definition — circular).

    python tbert/validity_features.py
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

_STEP = re.compile(r"\b(first|next|then|so|therefore|finally|after|now)\b", re.I)
_VERIFY = re.compile(r"\b(check|verif|confirm|double|make sure|recalculat|recheck|ensure)\b", re.I)


def _feats(text):
    return {
        "length": len(text.split()),
        "n_equations": step_check(text)[0],
        "n_step_words": len(_STEP.findall(text)),
        "n_verify_words": len(_VERIFY.findall(text)),
    }


def load():
    pools = torch.load(_POOLS, weights_only=False)
    got = {json.loads(l)["idx"]: json.loads(l) for l in open(_GOT)}
    sample = next(iter(pools.values()))["candidates"]
    pool = "mean" if isinstance(sample, dict) and "mean" in sample else "lastk"
    X, txt, ac, sv, ck = [], [], [], [], []
    for idx, e in pools.items():
        if idx not in got:
            continue
        ce = e["candidates"]; vecs = ce[pool] if isinstance(ce, dict) else ce
        cands = got[idx].get("candidates", []); gold = _to_float(got[idx].get("gold"))
        if gold is None or len(cands) != len(vecs):
            continue
        for i, c in enumerate(cands):
            t = c["text"]; ne, nb = step_check(t)
            X.append(vecs[i]); txt.append(t)
            ac.append(_eq(extract_answer(t), gold)); ck.append(ne > 0); sv.append(ne > 0 and nb == 0)
    return torch.stack(X).float().numpy(), txt, np.array(ac, bool), np.array(sv, bool), np.array(ck, bool)


def _test():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, txt, ac, sv, ck = load()
    m = ac & ck
    Xm = X[m]; tm = [t for t, k in zip(txt, m) if k]; y = sv[m].astype(int)   # 1=Q1 genuine,0=Q2
    print(f"[validity-features] Q1∪Q2 = {len(y)} (Q1 genuine={int(y.sum())}, Q2 spurious={int((1-y).sum())})\n")

    sc = StandardScaler().fit(Xm)
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(sc.transform(Xm), y)
    score = clf.decision_function(sc.transform(Xm))      # higher = more 'genuine' (descriptive)

    # (A) exemplars
    order = np.argsort(-score)
    print("=== most 'GENUINE'-scored answer-correct candidates ===")
    for i in order[:2]:
        print(f"   [{'Q1' if y[i] else 'Q2'}] " + re.sub(r'\s+', ' ', tm[i])[:200] + "…")
    print("\n=== most 'SPURIOUS'-scored answer-correct candidates ===")
    for i in order[-2:]:
        print(f"   [{'Q1' if y[i] else 'Q2'}] " + re.sub(r'\s+', ' ', tm[i])[:200] + "…")

    # (B) interpretable correlates
    feats = {k: np.array([_feats(t)[k] for t in tm], float) for k in _feats(tm[0])}
    print("\n=== interpretable text features: correlation with genuine(Q1)=1 and with probe score ===")
    print(f"  {'feature':16} | corr w/ Q1-label | corr w/ probe-score")
    for k, v in feats.items():
        c_lab = np.corrcoef(v, y)[0, 1]
        c_scr = np.corrcoef(v, score)[0, 1]
        print(f"  {k:16} | {c_lab:+.3f}           | {c_scr:+.3f}")

    # honest verdict: is the probe leaning on a surface feature?
    max_surface = max(abs(np.corrcoef(v, score)[0, 1]) for v in feats.values())
    print(f"\n[honest read] strongest surface-feature ↔ probe-score correlation = {max_surface:.2f}.")
    if max_surface < 0.4:
        print("  The probe is NOT explained by these surface features — the genuine/spurious signal "
              "it uses is mostly NON-surface (more credibly about reasoning validity than R6's style cues).")
    else:
        print("  A surface feature carries much of the signal — report it honestly "
              "(the distinction may be partly surface-level, like R6).")


if __name__ == "__main__":
    _test()
