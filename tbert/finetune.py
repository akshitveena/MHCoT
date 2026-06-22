"""
tbert/finetune.py  —  Step 2: fine-tune a real ENCODER on reasoning-candidate text.

The "do it right" pipeline you asked for, end to end:
  AutoModelForSequenceClassification (BERT-family encoder, cached/offline)
  + DataCollatorWithPadding + TrainingArguments + Trainer + compute_metrics
  + a layer-FREEZING utility so encoder blocks genuinely learn (or not) on demand.

`finetune(model_name, n_freeze, epochs, ...)` returns held-out F1/P/R/acc/AUC and is
the unit reused by the freeze-vs-F1 sweep (Step 3).

Run the smoke self-test (quick MiniLM, 1 epoch):
    python tbert/finetune.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_candidates, split_by_problem, compute_metrics      # noqa: E402

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE = "BAAI/bge-base-en-v1.5"


def _freeze(model, n_freeze):
    """Freeze embeddings + the first n_freeze encoder layers; rest stays trainable.
    n_freeze=0 -> full fine-tune; n_freeze='all' -> only the classifier head trains."""
    base = model.base_model                       # the BertModel under the head
    layers = base.encoder.layer
    n = len(layers) if n_freeze == "all" else int(n_freeze)
    for p in base.embeddings.parameters():
        p.requires_grad = (n == 0)
    for i, layer in enumerate(layers):
        train = i >= n
        for p in layer.parameters():
            p.requires_grad = train
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def finetune(model_name=MINILM, n_freeze=0, epochs=2, bs=16, lr=2e-5,
             max_len=256, seed=0, class_weight=False, verbose=True):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    from datasets import Dataset
    import torch.nn.functional as Fnn

    class WeightedTrainer(Trainer):
        def __init__(self, *a, class_weights=None, **k):
            super().__init__(*a, **k); self.cw = class_weights
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            w = self.cw.to(out.logits.device) if self.cw is not None else None
            loss = Fnn.cross_entropy(out.logits, labels, weight=w)
            return (loss, out) if return_outputs else loss

    torch.manual_seed(seed); np.random.seed(seed)
    texts, labels, pids = load_candidates()
    (xtr, ytr, _), (xte, yte, _) = split_by_problem(texts, labels, pids, seed=seed)

    tok = AutoTokenizer.from_pretrained(model_name)
    def make(ds_x, ds_y):
        d = Dataset.from_dict({"text": ds_x, "labels": ds_y})
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                     batched=True, remove_columns=["text"])
    dtr, dte = make(xtr, ytr), make(xte, yte)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    trn, tot = _freeze(model, n_freeze)
    if verbose:
        print(f"  [{model_name.split('/')[-1]}] n_freeze={n_freeze}: "
              f"trainable {trn/1e6:.1f}M / {tot/1e6:.1f}M ({trn/tot:.0%})")

    args = TrainingArguments(
        output_dir=str(Path(__file__).resolve().parent / "_runs"),
        num_train_epochs=epochs, per_device_train_batch_size=bs,
        per_device_eval_batch_size=64, learning_rate=lr, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="no", logging_strategy="no",
        report_to="none", seed=seed, disable_tqdm=True, use_cpu=False,
    )
    cw = None
    if class_weight:
        p = float(np.mean(ytr))                       # fraction positive
        cw = torch.tensor([1.0 / (1 - p), 1.0 / p], dtype=torch.float)
        cw = cw / cw.sum() * 2
    trainer = WeightedTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dte,
                      data_collator=DataCollatorWithPadding(tok),
                      processing_class=tok, compute_metrics=compute_metrics,
                      class_weights=cw)
    trainer.train()
    m = trainer.evaluate()
    return {k.replace("eval_", ""): v for k, v in m.items()
            if k.startswith("eval_") and k != "eval_loss"} | {"n_freeze": n_freeze}


def _test():
    # majority/random baselines for context (72% positive)
    _, yte_all, _ = split_by_problem(*load_candidates(), seed=0)[1] if False else (None, None, None)
    print("[STEP 2] smoke fine-tune: MiniLM, full fine-tune (n_freeze=0), 1 epoch ...")
    r = finetune(MINILM, n_freeze=0, epochs=1, seed=0)
    print("  held-out: " + ", ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "n_freeze"))
    assert r["auc"] > 0.55, "a fine-tuned encoder should beat random (AUC>0.55)"
    print("\n[STEP 2 PASSED] full fine-tuning pipeline trains end-to-end "
          "(Trainer + DataCollator + compute_metrics + layer-freeze) and learns signal.")


if __name__ == "__main__":
    _test()
