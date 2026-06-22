"""
real_mhcot/answer_rerank.py  —  the headline: does the verifier beat self-consistency?
=====================================================================================

Joins the GoT candidate *answers* (parsed from text) with their cached embeddings,
trains the thought-verifier, and reranks each problem's candidates to pick a final
answer. Compares FINAL-ANSWER accuracy to the standard strong baseline
(self-consistency / majority vote), plus random and oracle.

This turns "AUC 0.82" into the metric that matters: reasoning final-answer accuracy.

Run:
    python real_mhcot/answer_rerank.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder import ThoughtEncoder, train_contrastive             # noqa: E402

_POOLS = _ROOT / "data" / "hidden_cache" / "gsm8k_test_pools.pt"
_GOT = _ROOT / "data" / "got_cache" / "gsm8k_test.jsonl"


def _to_float(s):
    if s is None:
        return None
    m = re.search(r"-?\d+\.?\d*", str(s).replace(",", "").replace("$", ""))
    return float(m.group(0)) if m else None


def extract_answer(text):
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        return _to_float(boxes[-1])
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return _to_float(nums[-1]) if nums else None


def _eq(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-4


def load_joined(pool="lastk"):
    pools = torch.load(_POOLS, weights_only=False)
    got = {}
    for line in open(_GOT):
        d = json.loads(line); got[d["idx"]] = d
    problems, mism = [], 0
    for idx, e in pools.items():
        if idx not in got:
            continue
        vecs = e["candidates"][pool] if isinstance(e["candidates"], dict) else e["candidates"]
        cc = [int(x) for x in e["cand_correct"]]
        g = got[idx]
        cands = g.get("candidates", [])
        if len(cands) != len(vecs):
            mism += 1; continue
        gold = _to_float(g.get("gold"))
        answers = [extract_answer(c["text"]) for c in cands]
        # sanity: extracted-correct vs cached cand_correct
        problems.append({"emb": vecs, "answers": answers, "correct": cc,
                         "gold": gold, "pid": idx})
    return problems, mism


def split(problems, seed, frac=0.7):
    rng = np.random.RandomState(seed)
    order = list(range(len(problems))); rng.shuffle(order)
    cut = int(frac * len(order))
    tr = [problems[i] for i in order[:cut]]
    te = [problems[i] for i in order[cut:]]
    return tr, te


@torch.no_grad()
def _score(model, emb, proto):
    z = model(emb).numpy()
    return z @ proto


def evaluate(seed):
    problems, mism = load_joined()
    tr, te = split(problems, seed)
    # train verifier on train candidates
    Xtr = torch.cat([p["emb"] for p in tr])
    ytr = torch.tensor([c for p in tr for c in p["correct"]], dtype=torch.long)
    torch.manual_seed(seed)
    model = ThoughtEncoder(d=128)
    train_contrastive(model, Xtr, ytr, steps=2000, seed=seed)
    with torch.no_grad():
        Ztr = model(Xtr).numpy()
    proto = Ztr[ytr.numpy() == 1].mean(0); proto /= np.linalg.norm(proto) + 1e-9

    rerank = sc = rnd = oracle = 0
    n = 0
    for p in te:
        if p["gold"] is None:
            continue
        n += 1
        ans = p["answers"]; gold = p["gold"]
        sc_pick = _score(model, p["emb"], proto)
        # reranker: top-scored candidate's answer
        rerank += int(_eq(ans[int(np.argmax(sc_pick))], gold))
        # self-consistency: majority answer
        valid = [a for a in ans if a is not None]
        if valid:
            vals, counts = np.unique(np.round(valid, 4), return_counts=True)
            maj = vals[int(np.argmax(counts))]
            sc += int(_eq(maj, gold))
        # random pick (expected) and oracle
        rnd += np.mean([int(_eq(a, gold)) for a in ans])
        oracle += int(any(_eq(a, gold) for a in ans))
    return {"rerank": rerank / n, "self_consistency": sc / n,
            "random": rnd / n, "oracle": oracle / n, "n": n, "mism": mism}


def _test():
    seeds = [0, 1, 2]
    rr, sc, rn, orc = [], [], [], []
    for s in seeds:
        r = evaluate(s)
        rr.append(r["rerank"]); sc.append(r["self_consistency"])
        rn.append(r["random"]); orc.append(r["oracle"])
        print(f"    seed{s}: rerank={r['rerank']:.3f}  self-consistency={r['self_consistency']:.3f}"
              f"  random={r['random']:.3f}  oracle={r['oracle']:.3f}  (n={r['n']})", flush=True)
    print(f"\n[ANSWER-RERANK] final-answer accuracy (mean of {seeds}):")
    print(f"  verifier-rerank   {np.mean(rr):.3f} ± {np.std(rr):.3f}")
    print(f"  self-consistency  {np.mean(sc):.3f} ± {np.std(sc):.3f}")
    print(f"  random pick       {np.mean(rn):.3f}")
    print(f"  oracle ceiling    {np.mean(orc):.3f}")
    delta = np.mean(rr) - np.mean(sc)
    print(f"\n  reranker vs self-consistency: {delta:+.3f}")
    if delta > 0.005:
        print("  -> verifier-reranking BEATS self-consistency on final-answer accuracy.")
    elif abs(delta) <= 0.005:
        print("  -> verifier-reranking ≈ self-consistency (ties the strong baseline).")
    else:
        print("  -> self-consistency wins; report honestly.")


if __name__ == "__main__":
    _test()
