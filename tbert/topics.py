"""
tbert/topics.py  —  STEP 1 of the conceptual pipeline (offline): topic-model the problems.

Embed the GSM8K problems (cached encoder, offline), cluster by concept, label each
cluster by its distinctive terms. Saves problem->topic so the semantic-judge step can
ask WHERE the 'right-for-wrong-reasons' phenomenon concentrates.

    python tbert/topics.py [k]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_GOT = _ROOT / "data" / "got_cache" / "gsm8k_test.jsonl"


def load_problems():
    seen, idxs, probs = set(), [], []
    for line in open(_GOT):
        d = json.loads(line)
        if d["idx"] in seen:
            continue
        seen.add(d["idx"]); idxs.append(d["idx"]); probs.append(d.get("problem", ""))
    return idxs, probs


def _test():
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    idxs, probs = load_problems()
    print(f"[topics] embedding {len(probs)} problems (MiniLM, offline) ...")
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E = enc.encode(probs, normalize_embeddings=True, show_progress_bar=False)

    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(E)
    lab = km.labels_

    # distinctive terms per cluster via tf-idf
    tf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    M = tf.fit_transform(probs); vocab = np.array(tf.get_feature_names_out())
    print(f"[topics] {k} concept clusters:\n")
    for c in range(k):
        m = lab == c
        centroid = np.asarray(M[m].mean(0)).ravel()
        top = vocab[np.argsort(-centroid)[:6]]
        ex = probs[np.where(m)[0][0]]
        print(f"  topic {c} (n={int(m.sum())}): {', '.join(top)}")
        print(f"      e.g. {ex[:90]}…")

    out = _ROOT / "tbert" / "topics.json"
    out.write_text(json.dumps({str(i): int(t) for i, t in zip(idxs, lab)}))
    print(f"\n[topics] saved problem->topic map -> {out.name} "
          f"({len(idxs)} problems, {k} topics) for the judge step.")
    assert len(set(lab)) == k


if __name__ == "__main__":
    _test()
