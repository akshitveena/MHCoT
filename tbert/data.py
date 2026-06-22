"""
tbert/data.py  —  Step 1: data + leak-free split + metrics for the FINE-TUNED
thought encoder (BERT-family, fine-tuned — NOT frozen DeepSeek features).

This is the "do it right" track:
  * input  = the raw candidate TEXT (a reasoning trace), not a cached embedding
  * label  = is the candidate's final answer correct? (boxed/last-number vs gold)
  * split  = leak-free, by problem (a problem's candidates never straddle train/test)
  * metric = F1 / precision / recall / accuracy / AUC (compute_metrics, HF-style)

No DeepSeek embeddings involved — we will fine-tune an actual encoder on text.

Run the self-test:
    python tbert/data.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
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


def load_candidates():
    """-> (texts, labels, pids). label = candidate answer matches gold."""
    texts, labels, pids = [], [], []
    for line in open(_GOT):
        d = json.loads(line)
        gold = _to_float(d.get("gold"))
        if gold is None:
            continue
        for c in d.get("candidates", []):
            t = c.get("text", "").strip()
            if not t:
                continue
            texts.append(t)
            labels.append(int(_eq(extract_answer(t), gold)))
            pids.append(d["idx"])
    return texts, labels, pids


def split_by_problem(texts, labels, pids, seed=0, frac=0.7):
    """Leak-free: partition PROBLEMS, not candidates."""
    uniq = sorted(set(pids))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    cut = int(frac * len(uniq))
    train_p = set(uniq[:cut])
    tr = [i for i in range(len(pids)) if pids[i] in train_p]
    te = [i for i in range(len(pids)) if pids[i] not in train_p]
    pick = lambda idx: ([texts[i] for i in idx], [labels[i] for i in idx], [pids[i] for i in idx])
    return pick(tr), pick(te)


def compute_metrics(eval_pred):
    """HF-style. Accepts (logits, labels); reports F1/precision/recall/acc/AUC."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    logits, labels = eval_pred
    logits = np.asarray(logits)
    if logits.ndim == 2 and logits.shape[1] == 2:               # 2-logit classifier
        prob1 = np.exp(logits[:, 1] - logits.max(1)) / np.exp(logits - logits.max(1, keepdims=True)).sum(1)
        preds = logits.argmax(1)
    else:                                                       # single score
        prob1 = 1 / (1 + np.exp(-logits.ravel()))
        preds = (prob1 >= 0.5).astype(int)
    labels = np.asarray(labels).ravel()
    out = {
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }
    try:
        out["auc"] = roc_auc_score(labels, prob1)
    except ValueError:
        out["auc"] = float("nan")
    return out


def _test():
    texts, labels, pids = load_candidates()
    n, pos = len(labels), int(np.sum(labels))
    print(f"[tbert.data] {n} candidates over {len(set(pids))} problems; "
          f"positive (correct) = {pos} ({pos/n:.1%})")
    assert n > 1000 and len(set(pids)) > 400, "expected the full GSM8K GoT candidate set"

    (xtr, ytr, ptr), (xte, yte, pte) = split_by_problem(texts, labels, pids, seed=0)
    print(f"[tbert.data] split-by-problem: train {len(xtr)} cands / {len(set(ptr))} probs"
          f"  |  test {len(xte)} cands / {len(set(pte))} probs")
    leak = set(ptr) & set(pte)
    assert not leak, f"LEAK: {len(leak)} problems in both splits"
    print(f"[tbert.data] leak-free OK (0 shared problems); "
          f"train pos {np.mean(ytr):.1%}, test pos {np.mean(yte):.1%}")

    # compute_metrics sanity on dummy 2-logit predictions
    rng = np.random.RandomState(0)
    fake_logits = rng.randn(len(yte), 2)
    m = compute_metrics((fake_logits, np.array(yte)))
    print(f"[tbert.data] compute_metrics on random preds: "
          + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    assert set(m) == {"f1", "precision", "recall", "accuracy", "auc"}
    print("\n[STEP 1 PASSED] text+label loaded, leak-free split, F1/P/R/AUC metrics ready.")


if __name__ == "__main__":
    _test()
