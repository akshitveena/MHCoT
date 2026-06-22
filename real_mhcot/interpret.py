"""
real_mhcot/interpret.py  —  #2: what do the correctness features fire on?
========================================================================

Light interpretability pass on the SAE thought-vocabulary: for the SAE features
most associated with reasoning correctness, surface the candidate thoughts that
most activate them (with text + correctness), so we can describe what each feature
captures. Turns "25 features track correctness" into "here is what the model's
internal notion of good reasoning looks like."

Honest scope: this is a *descriptive* pass (top-activating exemplars + activation
contrast), not a causal claim — feature-realness is contested (see mirage/), so we
keep it qualitative + report the contrast statistics, not strong causal assertions.

Run:
    python real_mhcot/interpret.py
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
from sae import SAE, train_sae                                     # noqa: E402

_POOLS = _ROOT / "data" / "hidden_cache" / "gsm8k_test_pools.pt"
_GOT = _ROOT / "data" / "got_cache" / "gsm8k_test.jsonl"


def load_with_text(pool="lastk"):
    pools = torch.load(_POOLS, weights_only=False)
    got = {json.loads(l)["idx"]: json.loads(l) for l in open(_GOT)}
    X, txt, y, pid = [], [], [], []
    for idx, e in pools.items():
        if idx not in got:
            continue
        vecs = e["candidates"][pool] if isinstance(e["candidates"], dict) else e["candidates"]
        cc = e["cand_correct"]; cands = got[idx].get("candidates", [])
        if len(cands) != len(vecs):
            continue
        for i in range(len(vecs)):
            X.append(vecs[i]); txt.append(cands[i]["text"])
            y.append(int(cc[i])); pid.append(idx)
    return torch.stack(X).float(), txt, torch.tensor(y), torch.tensor(pid)


def snippet(t, n=160):
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:n] + "…") if len(t) > n else t


def _test():
    torch.manual_seed(0); np.random.seed(0)
    X, txt, y, pid = load_with_text()
    print(f"[interpret] {len(y)} candidate thoughts ({int(y.sum())} correct)")

    # train encoder + SAE
    probs = torch.unique(pid).tolist(); rng = np.random.RandomState(0); rng.shuffle(probs)
    tr_set = set(probs[: int(0.7 * len(probs))])
    tr = torch.tensor([i for i in range(len(pid)) if pid[i].item() in tr_set])
    enc = ThoughtEncoder(d=128)
    train_contrastive(enc, X[tr], y[tr], steps=2000, seed=0)
    with torch.no_grad():
        Z = enc(X)
    sae = SAE(d=128, m=512)
    train_sae(sae, Z[tr], steps=3000, l1=2e-3, seed=0)
    with torch.no_grad():
        _, F = sae(Z)
    F = F.numpy(); yv = y.numpy()

    # rank features by correctness association (on active features)
    active = (F > 1e-6).mean(0) > 0.02
    feats = []
    for k in np.where(active)[0]:
        c = np.corrcoef(F[:, k], yv)[0, 1]
        if np.isfinite(c):
            contrast = F[yv == 1, k].mean() - F[yv == 0, k].mean()
            feats.append((abs(c), c, contrast, k))
    feats.sort(reverse=True)

    print(f"[interpret] {int(active.sum())} active features; "
          f"top correctness-associated features:\n")
    for absc, c, contrast, k in feats[:4]:
        align = "CORRECT-aligned" if c > 0 else "INCORRECT-aligned"
        print(f"=== feature #{k}: corr={c:+.3f} ({align}), "
              f"act(correct)-act(incorrect)={contrast:+.3f} ===")
        top = np.argsort(-F[:, k])[:3]
        for j in top:
            tag = "✓correct" if yv[j] == 1 else "✗wrong"
            print(f"   [{tag}, act={F[j,k]:.2f}] {snippet(txt[j])}")
        print()

    best = feats[0]
    assert best[0] > 0.25, "should find a feature clearly associated with correctness"
    # top-activating exemplars of the best feature should skew to its aligned class
    top = np.argsort(-F[:, best[3]])[:10]
    skew = yv[top].mean() if best[1] > 0 else 1 - yv[top].mean()
    print(f"[interpret] best feature #{best[3]}: top-10 activations are "
          f"{skew:.0%} of its aligned class (interpretable signal).")
    assert skew > 0.6, "top-activating exemplars should be coherent (one class)"
    print("\n[#2 PASSED] correctness features are interpretable: top-activating thoughts "
          "are coherent and class-aligned — a describable 'good-reasoning' vocabulary.")


if __name__ == "__main__":
    _test()
