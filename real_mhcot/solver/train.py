"""
solver/train.py  —  SUPERVISED HEURISTIC LEARNING
=================================================

Trains each architecture (real / N=1 / N=2) to predict the EXACT solvability of
a search state. Targets come from the brute-force oracle (game24.reachable via
search.label_states) -> perfect labels, no LLM, no noise.

Deliberately NO reinforcement learning for this first probe: deterministic exact
targets are the lowest-variance way to ask "does the multi-helical inductive bias
produce a better search heuristic?". RL is a later step only if this clears.

Two things the foundation tests told us to handle:
  * ~1% of states are solvable -> we train on BALANCED batches (half solvable,
    half not) so the net can't win by always predicting "dead end".
  * the real baseline must be PARAM-MATCHED to the complex models (no free
    capacity advantage); we shrink it and print the counts to prove it.

Run:
    python solver/train.py                  # full run, saves checkpoints
    python solver/train.py --smoke          # tiny end-to-end sanity run
Out: solver/ckpt/{real,N1,N2}.pt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "main"))
sys.path.insert(0, str(_HERE))   # solver FIRST: solver/model.py, not main/model.py

from game24 import make_split                              # noqa: E402
from search import label_states                            # noqa: E402
from model import MHCoTSolver, RealSolver                  # noqa: E402
from soliton import epsilon_helix_loss                     # noqa: E402

_CKPT = _HERE / "ckpt"
_CKPT.mkdir(exist_ok=True)


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_dataset(puzzles):
    """puzzles -> (states, targets, labels) over all reachable intermediate
    states, with exact solvability labels."""
    labeled = label_states(puzzles)
    states = [st for st, _, _ in labeled]
    targets = [tg for _, tg, _ in labeled]
    labels = [lab for _, _, lab in labeled]
    return states, targets, labels


def balanced_batches(states, targets, labels, batch_size, steps, rng):
    """Yield `steps` batches, each ~half solvable / half not (oversampling the
    rare positive class)."""
    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]
    half = batch_size // 2
    for _ in range(steps):
        idx = ([rng.choice(pos) for _ in range(half)] +
               [rng.choice(neg) for _ in range(batch_size - half)])
        rng.shuffle(idx)
        yield ([states[i] for i in idx], [targets[i] for i in idx],
               torch.tensor([float(labels[i]) for i in idx]))


def train_model(model, data, dev, steps=2000, batch_size=64, lr=3e-4,
                lambda_eps=0.05, warmup_frac=0.4, seed=0, log_every=400):
    states, targets, labels = data
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bce = nn.BCEWithLogitsLoss()
    is_complex = isinstance(model, MHCoTSolver)
    n_chains = getattr(model, "n_chains", 0)
    warmup = int(warmup_frac * steps)   # engage L_eps only after warmup
    model.train()
    t0 = time.time(); n_skipped = 0
    for step, (bs, bt, by) in enumerate(
            balanced_batches(states, targets, labels, batch_size, steps, rng)):
        by = by.to(dev)
        out = model(bs, bt)
        loss = bce(out["score"], by)
        # stage the eps-helix loss: atan2 phase grads are unstable while
        # magnitudes are still near zero, so wait until the head has warmed up
        if is_complex and n_chains >= 2 and lambda_eps > 0 and step >= warmup:
            loss = loss + lambda_eps * epsilon_helix_loss(out["psi"])
        if not torch.isfinite(loss):
            n_skipped += 1; continue
        opt.zero_grad(); loss.backward()
        # guard: a NaN/inf gradient (atan2 at |psi|~0) must NOT corrupt weights
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters())
        if not finite:
            opt.zero_grad(); n_skipped += 1; continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % log_every == 0:
            div = out.get("chain_divergence", torch.zeros(())).item()
            print(f"    step {step+1:>5}/{steps}  loss {loss.item():.4f}  "
                  f"div {div:.3f}  ({(time.time()-t0):.0f}s)")
    return model


@torch.no_grad()
def quick_auc(model, data, dev, n=2000):
    from sklearn.metrics import roc_auc_score
    states, targets, labels = data
    n = min(n, len(states))
    idx = list(range(len(states)))
    random.Random(0).shuffle(idx); idx = idx[:n]
    model.eval()
    scores = []
    for i in range(0, n, 128):
        chunk = idx[i:i + 128]
        out = model([states[j] for j in chunk], [targets[j] for j in chunk])
        scores.append(out["score"].float().cpu())
    s = torch.cat(scores).numpy()
    y = np.array([labels[j] for j in idx])
    return roc_auc_score(y, s) if len(set(y.tolist())) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--target", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.steps = 40, 150

    dev = _device(); print(f"[train] device={dev}")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    train_pz, test_pz = make_split(args.n_train, max(50, args.n_train // 4),
                                   k=args.k, target=args.target, seed=args.seed)
    print(f"[train] puzzles: {len(train_pz)} train / {len(test_pz)} test "
          f"(k={args.k}, target={args.target})")
    data = build_dataset(train_pz)
    pos = sum(data[2]); print(f"[train] labeled states: {len(data[2])} "
                              f"({pos} solvable = {pos/len(data[2]):.1%})")
    # save the test puzzles so evaluate.py uses the SAME held-out set
    torch.save({"test_puzzles": [(p.numbers, p.target, p.solvable) for p in test_pz],
                "args": vars(args)}, _CKPT / "split.pt")

    # param-matched configs (real shrunk to ~complex size; printed to prove it)
    configs = {
        "real": lambda: RealSolver(d_model=64, n_heads=4, n_layers=2),
        "N1":   lambda: MHCoTSolver(d_model=64, n_heads=4, n_chains=1),
        "N2":   lambda: MHCoTSolver(d_model=64, n_heads=4, n_chains=2),
    }
    for name, build in configs.items():
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        model = build().to(dev)
        nparam = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] params={nparam/1e3:.1f}K — training {args.steps} steps ...")
        train_model(model, data, dev, steps=args.steps, seed=args.seed)
        auc = quick_auc(model, data, dev)
        torch.save(model.state_dict(), _CKPT / f"{name}.pt")
        print(f"[{name}] train-set heuristic AUC={auc:.3f}  -> saved {name}.pt")

    print(f"\n[done] checkpoints in {_CKPT}/  — now run: python solver/evaluate.py")


if __name__ == "__main__":
    main()
