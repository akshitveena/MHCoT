"""
tbert/conceptual.py  —  offline probe of the CONCEPTUAL "right answer, didn't earn it" case.

Uses the gold GSM8K solutions (cached) as a reference path: each gold solution's
<<a op b = c>> annotations give the ESSENTIAL intermediate quantities a valid path
computes. For each ANSWER-CORRECT candidate we measure how many of those gold
intermediates it actually computes ('gold-step coverage'). Low coverage + right answer
= reached the answer WITHOUT the essential intermediate reasoning = candidate for a
conceptual short-circuit.

HONEST SCOPE: a divergent path is NOT necessarily a wrong one (valid alternatives exist).
This flags candidates to INSPECT; it does not by itself label reasoning wrong. We spot-check.

    python tbert/conceptual.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import extract_answer, _to_float, _eq                  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
_GOT = _ROOT / "data" / "got_cache" / "gsm8k_test.jsonl"


def _norm(q):
    return re.sub(r"\s+", " ", q).strip().lower()[:120]


def gold_intermediates():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")["test"]
    out = {}
    for ex in ds:
        ann = re.findall(r"<<(.+?)>>", ex["answer"])
        inter = []
        for a in ann:
            if "=" in a:
                inter.append(_to_float(a.split("=")[-1]))
        out[_norm(ex["question"])] = [v for v in inter if v is not None]
    return out


def _nums(text):
    return {round(_to_float(m), 3) for m in re.findall(r"-?\d[\d,]*\.?\d*", text)
            if _to_float(m) is not None}


def _test():
    import os
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    gold = gold_intermediates()
    print(f"[conceptual] gold solutions loaded: {len(gold)} problems")

    rows = []   # (coverage, n_gold_inter, answer_correct, text, gold_ans, q)
    for line in open(_GOT):
        d = json.loads(line)
        g = gold.get(_norm(d.get("problem", "")))
        ga = _to_float(d.get("gold"))
        if not g or ga is None:
            continue
        inter = [v for v in g if not _eq(v, ga)]        # essential intermediates (drop final answer)
        if not inter:
            continue
        for c in d.get("candidates", []):
            t = c["text"]
            if not _eq(extract_answer(t), ga):           # only answer-correct candidates
                continue
            cand = _nums(t)
            cov = np.mean([any(_eq(v, x) for x in cand) for v in inter])
            rows.append((cov, len(inter), t, ga, d.get("problem", ""), inter))

    cov = np.array([r[0] for r in rows])
    print(f"[conceptual] {len(rows)} answer-correct candidates with checkable gold intermediates")
    print(f"            gold-step coverage: mean {cov.mean():.2f}, "
          f"%full(=1.0) {np.mean(cov==1):.0%}, %low(<0.5) {np.mean(cov<0.5):.0%}\n")

    low = [r for r in rows if r[0] < 0.5]
    print(f"[conceptual] {len(low)} answer-correct candidates compute <50% of the gold's essential "
          f"intermediates — the set to INSPECT (divergent path, may be valid-alt or genuinely-wrong):\n")
    rng = np.random.RandomState(0); idx = list(range(len(low))); rng.shuffle(idx)
    for j in idx[:4]:
        c0, ni, t, ga, q, inter = low[j]
        present = [v for v in inter if any(_eq(v, x) for x in _nums(t))]
        print(f"  Q: {re.sub(r'\\s+',' ',q)[:110]}")
        print(f"     gold essential intermediates={inter}  | candidate computes {present}  (answer {ga:g} ✓)")
        print(f"     candidate: {re.sub(r'\\s+',' ',t)[:200]}…\n")


if __name__ == "__main__":
    _test()
