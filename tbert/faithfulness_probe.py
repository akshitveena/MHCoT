"""
tbert/faithfulness_probe.py  —  does the representation know GENUINE vs SPURIOUS
correctness?  (A11 redone on TRUSTWORTHY human labels, difficulty controlled.)

Within OmniMath (balanced: ~259 wrong-path-right vs ~241 clean, all answer-correct,
all the same difficulty tier), embed each solution's text and probe:
  label 1 = wrong-path-right (human-flagged error step, answer still correct)
  label 0 = clean (no error, answer correct)

We compare the representation probe to a LENGTH-ONLY baseline — if the representation
only matches length, it's a surface cue, not faithfulness.

    python tbert/faithfulness_probe.py
"""

from __future__ import annotations

import numpy as np


def load_omnimath_ac():
    from datasets import load_dataset
    ds = load_dataset("Qwen/ProcessBench")["omnimath"]
    txt, y, length, nsteps = [], [], [], []
    for e in ds:
        if not e["final_answer_correct"]:
            continue
        s = "\n".join(e["steps"])
        txt.append(s); y.append(1 if int(e["label"]) != -1 else 0)
        length.append(len(s.split())); nsteps.append(len(e["steps"]))
    return txt, np.array(y), np.array(length, float), np.array(nsteps, float)


def _test():
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score

    txt, y, length, nsteps = load_omnimath_ac()
    print(f"[faithfulness] OmniMath answer-correct: {len(y)} "
          f"(wrong-path-right={int(y.sum())}, clean={int((1-y).sum())})")
    print(f"[faithfulness] embedding solutions (bge-base, offline; truncated to 512 tok) ...")
    enc = SentenceTransformer("BAAI/bge-base-en-v1.5")
    E = enc.encode(txt, normalize_embeddings=True, show_progress_bar=False, batch_size=32)

    rep_auc, len_auc, rep_f1 = [], [], []
    for s in [0, 1, 2]:
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.3, random_state=s, stratify=y)
        # representation probe
        sc = StandardScaler().fit(E[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(sc.transform(E[tr]), y[tr])
        sco = clf.decision_function(sc.transform(E[te]))
        rep_auc.append(roc_auc_score(y[te], sco)); rep_f1.append(f1_score(y[te], (sco > 0).astype(int)))
        # length-only baseline
        L = length.reshape(-1, 1)
        scl = StandardScaler().fit(L[tr])
        cl = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scl.transform(L[tr]), y[tr])
        len_auc.append(roc_auc_score(y[te], cl.decision_function(scl.transform(L[te]))))

    print(f"\n[RESULT] within-OmniMath (difficulty fixed), 3 seeds:")
    print(f"  representation probe AUC = {np.mean(rep_auc):.3f} ± {np.std(rep_auc):.3f}   (F1 {np.mean(rep_f1):.3f})")
    print(f"  length-only baseline AUC = {np.mean(len_auc):.3f} ± {np.std(len_auc):.3f}")
    print(f"  [surface] corr(length, wrong-path) = {np.corrcoef(length, y)[0,1]:+.3f}; "
          f"corr(#steps, wrong-path) = {np.corrcoef(nsteps, y)[0,1]:+.3f}")

    gap = np.mean(rep_auc) - np.mean(len_auc)
    if np.mean(rep_auc) > 0.6 and gap > 0.05:
        print(f"\n[FINDING] the representation separates genuine from wrong-path-right correctness "
              f"(AUC {np.mean(rep_auc):.2f}) BEYOND length ({np.mean(len_auc):.2f}, +{gap:.2f}) — "
              f"on human labels, difficulty fixed. The faithfulness signal is in the representation.")
    elif np.mean(rep_auc) <= 0.6:
        print(f"\n[HONEST NULL] the representation does NOT separate them (AUC {np.mean(rep_auc):.2f}) — "
              f"reasoning faithfulness is not linearly readable from this text embedding.")
    else:
        print(f"\n[HONEST] representation AUC {np.mean(rep_auc):.2f} ≈ length baseline "
              f"{np.mean(len_auc):.2f}: the apparent signal is mostly surface length.")


if __name__ == "__main__":
    _test()
