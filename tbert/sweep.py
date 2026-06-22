"""
tbert/sweep.py  —  Step 3: freeze-vs-F1/AUC sweep + epoch balance.

Answers your two questions directly:
  * how many encoder layers should we freeze before performance stabilizes?
  * how do fine-tuned encoders compare to the frozen-DeepSeek probe (AUC 0.82)
    and to the self-consistency-era numbers?

Runs `finetune` across n_freeze levels, prints the curve, saves json + a plot.

    python tbert/sweep.py minilm     # fast 6-layer sweep
    python tbert/sweep.py bge         # strong 12-layer sweep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import finetune, MINILM, BGE                       # noqa: E402

# reference bars (from our own prior runs / the data)
REF = {"majority (predict-correct)": {"f1": 0.837, "auc": 0.500, "acc": 0.720},
       "frozen-DeepSeek probe": {"f1": None, "auc": 0.819, "acc": None}}


def run_sweep(model_name, freezes, epochs=3, seed=0, class_weight=False):
    tag = model_name.split("/")[-1]
    rows = []
    for nf in freezes:
        r = finetune(model_name, n_freeze=nf, epochs=epochs, seed=seed,
                     class_weight=class_weight, verbose=True)
        rows.append(r)
        print(f"  -> n_freeze={nf}: F1={r['f1']:.3f} AUC={r['auc']:.3f} "
              f"acc={r['accuracy']:.3f} prec={r['precision']:.3f} rec={r['recall']:.3f}",
              flush=True)
    print(f"\n=== {tag} freeze sweep ({epochs} epochs) ===")
    print(f"{'n_freeze':>10} | {'F1':>6} | {'AUC':>6} | {'acc':>6}")
    for r in rows:
        print(f"{str(r['n_freeze']):>10} | {r['f1']:.3f} | {r['auc']:.3f} | {r['accuracy']:.3f}")
    print("\nreference:  frozen-DeepSeek probe AUC=0.819 | majority F1=0.837/acc=0.720")

    out = Path(__file__).resolve().parent / f"sweep_{tag}.json"
    out.write_text(json.dumps({"model": model_name, "epochs": epochs, "rows": rows}, indent=2))
    print(f"saved -> {out.name}")

    # plot F1 & AUC vs n_freeze
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [len(__import__("transformers").AutoModel.from_pretrained(model_name).base_model.encoder.layer)
              if r["n_freeze"] == "all" else int(r["n_freeze"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [r["f1"] for r in rows], "o-", label="F1")
        ax.plot(xs, [r["auc"] for r in rows], "s-", label="AUC")
        ax.axhline(0.819, ls="--", c="gray", label="frozen-DeepSeek AUC 0.819")
        ax.set_xlabel("# frozen encoder layers"); ax.set_ylabel("held-out score")
        ax.set_title(f"{tag}: performance vs frozen layers"); ax.legend(); fig.tight_layout()
        p = Path(__file__).resolve().parent / f"sweep_{tag}.png"; fig.savefig(p, dpi=120)
        print(f"plot  -> {p.name}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "minilm"
    if which == "minilm":
        run_sweep(MINILM, freezes=[0, 2, 4, "all"], epochs=3)
    elif which == "bge":
        run_sweep(BGE, freezes=[0, 6, "all"], epochs=3, class_weight=True)
    else:
        print("usage: python tbert/sweep.py [minilm|bge]")
