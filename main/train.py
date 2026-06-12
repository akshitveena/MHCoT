"""
main/train.py  —  Option 1 training loop (the ε-helix model)
============================================================

Trains the MHCoTEncoder on cached per-token final-node sequences to predict
correctness, with the staged objective (L_task → +L_ε → +L_wave).

THE TWO QUESTIONS THIS RUN ANSWERS:
  1. Does validation AUC (score or interference) BEAT the untrained
     interference probe (AUC 0.689)? If it only ties, complex added nothing.
  2. Does `chain_divergence` rise from ~0.57 toward ε_min (1.15)? If the chains
     stay collapsed, MHCoT = single chain.

Prereqs:
    python main/encoder.py --seq      # caches final-node per-token sequences

Run (Colab CUDA recommended; works on MPS/CPU):
    python main/train.py                       # default 40 epochs
    python main/train.py --epochs 2            # fast pipeline smoke test

Outputs: results/train/{best_model.pt, training_log.json}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder import load_sequences, SEQ_PATH        # noqa: E402
from model import MHCoTEncoder                       # noqa: E402
from losses import compute_losses, LossConfig        # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUT = _PROJECT_ROOT / "results" / "train"
_OUT.mkdir(parents=True, exist_ok=True)

BASELINE_AUC = 0.689   # untrained interference probe — the bar to beat


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SeqDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, max_seq_len: int = 384):
    """
    Pad variable-length sequences; build padding mask + lengths.
    Sequences longer than max_seq_len are truncated to their LAST max_seq_len
    tokens — this keeps the answer region (at the end) while bounding the
    O(T²) attention memory that caused MPS OOM on long traces.
    """
    proc = []
    for h, length, label in batch:
        if length > max_seq_len:
            h = h[-max_seq_len:]
            length = max_seq_len
        proc.append((h, length, label))
    batch = proc

    maxT = max(it[1] for it in batch)
    B = len(batch)
    D = batch[0][0].shape[1]
    H = torch.zeros(B, maxT, D)
    mask = torch.ones(B, maxT, dtype=torch.bool)        # True = padding
    lengths = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B)
    for i, (h, length, label) in enumerate(batch):
        H[i, :length] = h.float()
        mask[i, :length] = False
        lengths[i] = length
        labels[i] = float(label)
    return H, mask, lengths, labels


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    s_all, i_all, y_all, divs = [], [], [], []
    for H, mask, lengths, labels in loader:
        H, mask, lengths = H.to(dev), mask.to(dev), lengths.to(dev)
        out = model(H, key_padding_mask=mask, lengths=lengths)
        s_all.append(out["score"].float().cpu())
        i_all.append(out["I_answer"].float().cpu())
        y_all.append(labels)
        divs.append(out["chain_divergence"])
    s = torch.cat(s_all).numpy()
    ia = torch.cat(i_all).numpy()
    y = torch.cat(y_all).numpy()

    def auc(x):
        return float(roc_auc_score(y, x)) if len(set(y.tolist())) > 1 else float("nan")

    return auc(s), auc(ia), float(np.mean(divs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_semantic", type=int, default=2)
    ap.add_argument("--n_coupling", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=384,
                    help="truncate sequences to their last N tokens (bounds attn memory)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--eps_min", type=float, default=1.15)
    ap.add_argument("--lambda_eps", type=float, default=0.10)
    ap.add_argument("--lambda_wave", type=float, default=0.05)
    ap.add_argument("--stage_eps", type=int, default=100)
    ap.add_argument("--stage_wave", type=int, default=300)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = _device()
    print(f"[train] device = {dev}")

    if not SEQ_PATH.exists():
        print(f"[error] sequence cache missing: {SEQ_PATH}")
        print("        run:  python main/encoder.py --seq")
        sys.exit(1)

    cache = load_sequences()
    items, labels = [], []
    for e in cache.values():
        items.append((e["h"], e["length"], e["label"]))
        labels.append(e["label"])
    d_in = items[0][0].shape[1]
    n_pos = int(sum(labels))
    print(f"[train] {len(items)} sequences | d_in={d_in} | "
          f"correct={n_pos} wrong={len(labels)-n_pos}")

    tr, va = train_test_split(items, test_size=args.val_frac,
                              random_state=args.seed, stratify=labels)
    print(f"[train] split: train={len(tr)} val={len(va)}")

    from functools import partial
    coll = partial(collate, max_seq_len=args.max_seq_len)
    tl = DataLoader(SeqDataset(tr), batch_size=args.batch_size,
                    shuffle=True, collate_fn=coll)
    vl = DataLoader(SeqDataset(va), batch_size=args.batch_size,
                    shuffle=False, collate_fn=coll)

    model = MHCoTEncoder(
        d_in=d_in, d_model=args.d_model,
        n_semantic=args.n_semantic, n_coupling=args.n_coupling,
        eps_min=args.eps_min,
    ).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[train] MHCoTEncoder params = {nparams/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    cfg = LossConfig(eps_min=args.eps_min, lambda_eps=args.lambda_eps,
                     lambda_wave=args.lambda_wave,
                     stage_eps_step=args.stage_eps, stage_wave_step=args.stage_wave)

    print(f"\n[baseline] untrained interference probe AUC = {BASELINE_AUC:.3f} "
          f"(the bar to beat)\n")

    step = 0
    best_auc = 0.0
    log = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = {"L_task": [], "L_eps": [], "L_wave": [], "div": []}
        for H, mask, lengths, labels_b in tl:
            H, mask, lengths, labels_b = (H.to(dev), mask.to(dev),
                                          lengths.to(dev), labels_b.to(dev))
            out = model(H, key_padding_mask=mask, lengths=lengths)
            loss, comp = compute_losses(out, labels_b, step, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            step += 1
            agg["L_task"].append(comp["L_task"])
            agg["L_eps"].append(comp["L_eps"])
            agg["L_wave"].append(comp["L_wave"])
            agg["div"].append(comp["chain_divergence"])

        auc_s, auc_i, div = evaluate(model, vl, dev)
        best_this = max(auc_s, auc_i)
        flag = "  <-- beats baseline!" if best_this > BASELINE_AUC else ""
        print(f"ep{epoch:>3} | task {np.mean(agg['L_task']):.3f} "
              f"eps {np.mean(agg['L_eps']):.3f} wave {np.mean(agg['L_wave']):.3f} "
              f"| div {np.mean(agg['div']):.3f} "
              f"| val AUC: score={auc_s:.3f} interf={auc_i:.3f}{flag}")
        log.append({
            "epoch": epoch, "auc_score": auc_s, "auc_interf": auc_i,
            "divergence": float(np.mean(agg["div"])),
            "L_task": float(np.mean(agg["L_task"])),
            "L_eps": float(np.mean(agg["L_eps"])),
            "L_wave": float(np.mean(agg["L_wave"])),
        })
        if best_this > best_auc:
            best_auc = best_this
            torch.save(model.state_dict(), _OUT / "best_model.pt")

    json.dump({"args": vars(args), "baseline_auc": BASELINE_AUC, "log": log},
              open(_OUT / "training_log.json", "w"), indent=2)

    print(f"\n[done] {(time.time()-t0)/60:.1f} min | best val AUC = {best_auc:.3f} "
          f"vs baseline {BASELINE_AUC:.3f}")
    verdict = ("BEATS the baseline — the trained complex model adds value"
               if best_auc > BASELINE_AUC + 0.01
               else "does NOT beat the baseline yet — diagnose (chain divergence? data size?)")
    print(f"  → {verdict}")
    final_div = log[-1]["divergence"] if log else 0.0
    print(f"  final chain divergence = {final_div:.3f} "
          f"(ε_min={args.eps_min}; rising = chains differentiating)")
    print(f"  saved → {_OUT}/best_model.pt, training_log.json")


if __name__ == "__main__":
    main()
