"""
tbert/step_localizer.py  —  ProcessBench's REAL task: locate the first erroneous step.
Step 1 of the heavier effort: a frozen-embedding, value-function process-reward baseline.

Idea (PRM value-function framing):
  represent each PREFIX (problem + steps[0..i]); label y_i = 1 if the reasoning is still
  valid through step i, 0 if step i is the first error. (Steps after the first error are
  dropped from TRAINING — their validity is undefined.)
  Inference: scan a solution's steps; the first one scored 'not valid' is the predicted
  error step (predict -1 if none).

Evaluated with ProcessBench's metric:
  acc_correct  = of no-error solutions, fraction predicted -1
  acc_error    = of erroneous solutions, fraction whose predicted step == gold first-error
  F1           = harmonic mean (the headline ProcessBench number)

HONEST: this is a weak frozen-embedding baseline, not a fine-tuned PRM. It establishes a
floor on M3; beating it later (model-internal reps / fine-tuning) needs a GPU.

    python tbert/step_localizer.py [gsm8k|math|...]
"""

from __future__ import annotations

import sys

import numpy as np


def build_prefixes(subset="gsm8k"):
    from datasets import load_dataset
    ds = load_dataset("Qwen/ProcessBench")[subset]
    pref, y, inc, sid, sti, gold = [], [], [], [], [], []
    for s, e in enumerate(ds):
        steps, lab = e["steps"], int(e["label"])
        for i in range(len(steps)):
            pref.append(e["problem"] + "\n" + "\n".join(steps[: i + 1]))
            valid = (lab == -1) or (i < lab)
            y.append(1 if valid else 0)
            inc.append((lab == -1) or (i <= lab))      # train only up to (incl.) the error
            sid.append(s); sti.append(i); gold.append(lab)
    return np.array(y), np.array(inc, bool), np.array(sid), np.array(sti), np.array(gold), pref


def _test():
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sub = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
    y, inc, sid, sti, gold, pref = build_prefixes(sub)
    n_sol = len(set(sid))
    print(f"[localizer] {sub}: {n_sol} solutions, {len(pref)} step-prefixes "
          f"({inc.sum()} usable for training)")
    print(f"[localizer] embedding prefixes (bge-base, offline) ...")
    enc = SentenceTransformer("BAAI/bge-base-en-v1.5")
    E = enc.encode(pref, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

    sols = np.array(sorted(set(sid)))
    f1s, cas, eas = [], [], []
    for seed in [0, 1, 2]:
        rng = np.random.RandomState(seed); order = sols.copy(); rng.shuffle(order)
        tr_sol = set(order[: int(0.7 * len(order))])
        tr = np.array([k for k in range(len(sid)) if sid[k] in tr_sol and inc[k]])
        te_sol = [s for s in sols if s not in tr_sol]

        sc = StandardScaler().fit(E[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(sc.transform(E[tr]), y[tr])

        # localize per test solution
        n_corr_ok = n_corr = n_err_ok = n_err = 0
        for s in te_sol:
            ks = [k for k in range(len(sid)) if sid[k] == s]
            ks.sort(key=lambda k: sti[k])
            probs = clf.predict_proba(sc.transform(E[ks]))[:, 1]   # P(valid)
            flagged = [sti[k] for k, p in zip(ks, probs) if p < 0.5]
            pred = flagged[0] if flagged else -1
            g = gold[ks[0]]
            if g == -1:
                n_corr += 1; n_corr_ok += int(pred == -1)
            else:
                n_err += 1; n_err_ok += int(pred == g)
        ca = n_corr_ok / max(1, n_corr); ea = n_err_ok / max(1, n_err)
        f1 = 2 * ca * ea / (ca + ea) if (ca + ea) else 0.0
        cas.append(ca); eas.append(ea); f1s.append(f1)
        print(f"  seed{seed}: acc_correct={ca:.3f}  acc_error={ea:.3f}  F1={f1:.3f}")

    print(f"\n[ProcessBench-{sub}] frozen-embedding PRM baseline (3 seeds):")
    print(f"  acc_correct {np.mean(cas):.3f} | acc_error {np.mean(eas):.3f} | "
          f"F1 {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"  (reference: prompted strong LLMs ~0.5-0.8 F1 on ProcessBench; this is a floor.)")


if __name__ == "__main__":
    _test()
