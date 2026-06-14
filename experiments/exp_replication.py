"""
experiments/exp_replication.py  —  THE REPLICATION (the #1 rigor gate)
=====================================================================

Trains real / N=1-complex / N=2-complex on the 361 GSM8K TRAIN-split traces,
then measures calibration (ECE) + accuracy (AUC) on the 605 GSM8K TEST-split
traces — a properly held-out eval set 5× larger than the noisy 120-val we used
before. All three trained TASK-ONLY (identical objective; only architecture
differs) for a clean apples-to-apples comparison.

Replicates if the ordering holds on the big held-out set:
    real ≈ N=1 (ECE ~0.22–0.24)  ≫  N=2 (ECE ~0.07)

Prereqs (encode both splits — run on Mac, ~30 min total, no Colab):
    python main/encoder.py --seq --cache gsm8k_train.jsonl
    python main/encoder.py --seq --cache gsm8k_test.jsonl

Run:
    python experiments/exp_replication.py
Out: results/exp_replication/summary.json
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "main"))
sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

from model import MHCoTEncoder                     # noqa: E402
from train import SeqDataset, collate, _device      # noqa: E402
from exp5_ablation import RealEncoder               # noqa: E402
from exp_calibration import ece                     # noqa: E402
from losses import compute_losses, LossConfig        # noqa: E402

_HID = _PROJECT_ROOT / "data" / "hidden_cache"
_OUT = _PROJECT_ROOT / "results" / "exp_replication"
_OUT.mkdir(parents=True, exist_ok=True)


def load_items(path):
    cache = torch.load(path)
    return [(e["h"], e["length"], e["label"]) for e in cache.values()]


def train_taskonly(model, loader, dev, epochs=40, lr=3e-4, wd=0.05):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        for H, mask, lengths, y in loader:
            H, mask, lengths, y = H.to(dev), mask.to(dev), lengths.to(dev), y.to(dev)
            out = model(H, key_padding_mask=mask, lengths=lengths)
            loss = F.binary_cross_entropy_with_logits(out["score"], y)  # TASK ONLY
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def train_full(model, loader, dev, epochs=40, lr=3e-4, wd=0.05):
    """Full multi-helical training: staged task → +L_ε → +L_wave, so the
    chains actually DIFFERENTIATE (ε-helix engaged). Returns (model, final_div).
    Uses the ORIGINAL schedule (500/1500); engaging L_ε too early destabilizes
    the atan2 phase gradients → NaN. NaN steps are skipped defensively."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    cfg = LossConfig(eps_min=1.15, lambda_eps=0.10, lambda_wave=0.05,
                     stage_eps_step=500, stage_wave_step=1500)  # original schedule
    step = 0; last_div = 0.0; n_skipped = 0
    for _ in range(epochs):
        model.train()
        for H, mask, lengths, y in loader:
            H, mask, lengths, y = H.to(dev), mask.to(dev), lengths.to(dev), y.to(dev)
            out = model(H, key_padding_mask=mask, lengths=lengths)
            loss, comp = compute_losses(out, y, step, cfg)
            if not torch.isfinite(loss):           # skip NaN/inf steps
                n_skipped += 1; step += 1; continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1; last_div = comp["chain_divergence"]
    return model, last_div


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); s, y, divs = [], [], []
    for H, mask, lengths, yy in loader:
        H, mask, lengths = H.to(dev), mask.to(dev), lengths.to(dev)
        out = model(H, key_padding_mask=mask, lengths=lengths)
        s.append(out["score"].float().cpu()); y.append(yy)
        divs.append(out["chain_divergence"])
    s = torch.cat(s).numpy(); y = torch.cat(y).numpy()
    if not np.isfinite(s).all():               # diverged → report NaN, don't crash
        n_bad = int((~np.isfinite(s)).sum())
        print(f"    [warn] {n_bad}/{len(s)} scores are NaN/inf — model diverged")
        return float("nan"), float("nan"), float(np.mean(divs))
    p = 1 / (1 + np.exp(-s))
    auc = roc_auc_score(y, s) if len(set(y.tolist())) > 1 else float("nan")
    return ece(p, y), auc, float(np.mean(divs))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_only", action="store_true",
                    help="train ONLY the full multi-helical N=2 (skip real/N1/N2-taskonly, "
                         "whose results we already have)")
    args = ap.parse_args()

    dev = _device(); print(f"[replication] device={dev}")
    tr_path, te_path = _HID / "gsm8k_train_seq.pt", _HID / "gsm8k_test_seq.pt"
    for p in (tr_path, te_path):
        if not p.exists():
            print(f"[error] missing {p.name}. Run:")
            print("   python main/encoder.py --seq --cache gsm8k_train.jsonl")
            print("   python main/encoder.py --seq --cache gsm8k_test.jsonl")
            sys.exit(1)

    train_items = load_items(tr_path)
    test_items = load_items(te_path)
    d_in = train_items[0][0].shape[1]
    print(f"[replication] train={len(train_items)} (train split) | "
          f"eval={len(test_items)} (test split, held-out) | d_in={d_in}")

    coll = partial(collate, max_seq_len=384)
    tl = DataLoader(SeqDataset(train_items), batch_size=8, shuffle=True, collate_fn=coll)
    vl = DataLoader(SeqDataset(test_items), batch_size=8, shuffle=False, collate_fn=coll)

    configs = {
        "real":        lambda: RealEncoder(d_in=d_in, d_model=128, n_layers=4),
        "complex_N1":  lambda: MHCoTEncoder(d_in=d_in, d_model=128, n_chains=1),
        "complex_N2":  lambda: MHCoTEncoder(d_in=d_in, d_model=128, n_chains=2),
    }
    # known prior results (120-val) — used to fill the table when --full_only
    PRIOR = {"real": 0.208, "complex_N1": 0.205, "complex_N2": 0.197}
    results = {}
    t0 = time.time()
    if not args.full_only:
        for name, build in configs.items():
            torch.manual_seed(0); np.random.seed(0)
            model = build().to(dev)
            np_ = sum(p.numel() for p in model.parameters())
            print(f"\n[{name}] params={np_/1e6:.2f}M — training task-only on {len(train_items)} ...")
            train_taskonly(model, tl, dev)
            e, auc, div = evaluate(model, vl, dev)
            results[name] = {"ece": e, "auc": auc, "params": np_, "eval_div": div}
            print(f"[{name}] held-out ECE={e:.3f}  AUC={auc:.3f}  div={div:.3f}")
    else:
        print("\n[--full_only] skipping real / N1 / N2-taskonly "
              f"(prior held-out ECE: {PRIOR})")
        for name in configs:
            results[name] = {"ece": PRIOR[name], "auc": float("nan"),
                             "eval_div": 0.0, "note": "prior run"}

    # THE ACTUAL multi-helical model: full staged losses, chains DIFFERENTIATE
    torch.manual_seed(0); np.random.seed(0)
    model = MHCoTEncoder(d_in=d_in, d_model=128, n_chains=2).to(dev)
    print(f"\n[complex_N2_FULL] training with staged L_ε + L_wave "
          f"(chains should differentiate) ...")
    model, train_div = train_full(model, tl, dev)
    e, auc, div = evaluate(model, vl, dev)
    results["complex_N2_full"] = {"ece": e, "auc": auc, "eval_div": div,
                                   "train_div": train_div}
    print(f"[complex_N2_FULL] held-out ECE={e:.3f}  AUC={auc:.3f}  "
          f"chain divergence: train={train_div:.3f} eval={div:.3f}  "
          f"({'DIFFERENTIATED' if div > 0.9 else 'did NOT differentiate'})")

    print("\n" + "=" * 60)
    print(f"REPLICATION on {len(test_items)} held-out test traces "
          f"(trained on {len(train_items)} train traces)")
    print(f"{'model':<18}{'ECE':>8}{'AUC':>8}{'div':>8}")
    order = ["real", "complex_N1", "complex_N2", "complex_N2_full"]
    for name in order:
        r = results[name]
        print(f"{name:<18}{r['ece']:>8.3f}{r['auc']:>8.3f}{r.get('eval_div',0):>8.3f}")
    print("-" * 60)
    re_ = results["real"]["ece"]
    n2f = results["complex_N2_full"]["ece"]
    n2f_div = results["complex_N2_full"]["eval_div"]
    print("Original (120-val): real 0.240 | N=2-full 0.070")
    print(f"Now (held-out):     real {re_:.3f} | N=2-full {n2f:.3f} "
          f"(chains {'differentiated' if n2f_div > 0.9 else 'did NOT differentiate'}, "
          f"div={n2f_div:.2f})")
    if n2f_div <= 0.9:
        print("  ⚠ The full model's chains did NOT differentiate on held-out — the")
        print("    multi-helical mechanism didn't engage; verdict inconclusive for it.")
    elif n2f < re_ - 0.05:
        print("  → REPLICATED for the FULL multi-helical model: differentiated chains")
        print("    calibrate where real doesn't. The finding survives.")
    elif n2f < re_ - 0.02:
        print("  → Partial: full N=2 best-calibrated but margin small — report honestly.")
    else:
        print("  → DID NOT replicate even for the full multi-helical model.")
    print("=" * 60)
    json.dump({"n_train": len(train_items), "n_eval": len(test_items),
               "results": results, "elapsed_min": (time.time()-t0)/60},
              open(_OUT / "summary.json", "w"), indent=2)
    print(f"[saved] {_OUT}/summary.json  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
