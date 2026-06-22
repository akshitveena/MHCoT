"""
solver/multiseed.py  —  THE MULTI-SEED VERDICT
==============================================

Trains and evaluates real / N=1 / N=2 across several seeds, each on its own
independent train/test split, and aggregates mean ± std. This is the rigor gate
the scorer taught us to never skip: a single seed already lied to us once.

Two questions it settles:
  Q1  multi-helical:   does N=2 beat N=1 ?   (the MHCoT novelty)
  Q2  complex vs real: does N=1 beat real ?  (the DCN-style baseline effect)

Metrics per seed (held-out puzzles):
  * beam_solve   (primary, steady)
  * bestfirst_solve + nodes-to-solution (efficiency, harsher)

Run:
    python solver/multiseed.py                       # seeds 0,1,2
    python solver/multiseed.py --seeds 0 1 2 3 4
Out: solver/results/multiseed.json
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "main"))
sys.path.insert(0, str(_HERE))   # solver FIRST: solver/model.py, not main/model.py

from game24 import make_split                                      # noqa: E402
from model import MHCoTSolver, RealSolver, make_heuristic         # noqa: E402
from train import build_dataset, train_model, _device             # noqa: E402
from evaluate import evaluate_heuristic                            # noqa: E402

_OUT = _HERE / "results"
_OUT.mkdir(exist_ok=True)

CONFIGS = {
    "real": lambda: RealSolver(d_model=64, n_heads=4, n_layers=2),
    "N1":   lambda: MHCoTSolver(d_model=64, n_heads=4, n_chains=1),
    "N2":   lambda: MHCoTSolver(d_model=64, n_heads=4, n_chains=2),
}


def run_seed(seed, n_train, n_test, steps, budget, beam, dev):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    train_pz, test_pz = make_split(n_train, n_test, k=4, target=24, seed=seed)
    data = build_dataset(train_pz)
    out = {}
    for name, build in CONFIGS.items():
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = build().to(dev)
        train_model(model, data, dev, steps=steps, seed=seed, log_every=steps)
        h = make_heuristic(model, device=dev)
        m = evaluate_heuristic(h, test_pz, budget, beam)
        out[name] = m
        print(f"    seed{seed} {name:<5} beam {m['beam_solve_rate']:.3f}  "
              f"best-first {m['bestfirst_solve_rate']:.3f}  "
              f"nodes {None if m['bestfirst_avg_nodes'] is None else round(m['bestfirst_avg_nodes'],1)}")
    return out


def agg(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--beam", type=int, default=3)
    args = ap.parse_args()

    dev = _device()
    print(f"[multiseed] device={dev}  seeds={args.seeds}  "
          f"n_train={args.n_train} steps={args.steps} budget={args.budget}\n")
    t0 = time.time()
    per_seed = {}
    for seed in args.seeds:
        print(f"--- seed {seed} ---")
        per_seed[seed] = run_seed(seed, args.n_train, args.n_test,
                                  args.steps, args.budget, args.beam, dev)

    # aggregate
    print("\n" + "=" * 66)
    print(f"AGGREGATE over seeds {args.seeds}  (mean ± std)")
    print(f"{'model':<6}{'beam_solve':>16}{'bestfirst':>16}{'nodes':>14}")
    summary = {}
    for name in CONFIGS:
        beam_m, beam_s = agg([per_seed[s][name]["beam_solve_rate"] for s in args.seeds])
        bf_m, bf_s = agg([per_seed[s][name]["bestfirst_solve_rate"] for s in args.seeds])
        nd_m, nd_s = agg([per_seed[s][name]["bestfirst_avg_nodes"] for s in args.seeds])
        summary[name] = {"beam": (beam_m, beam_s), "bestfirst": (bf_m, bf_s),
                         "nodes": (nd_m, nd_s)}
        nd = "n/a" if nd_m is None else f"{nd_m:.1f}±{nd_s:.1f}"
        print(f"{name:<6}{beam_m:>9.3f}±{beam_s:<5.3f}"
              f"{bf_m:>9.3f}±{bf_s:<5.3f}{nd:>14}")
    print("-" * 66)

    # verdicts on the PRIMARY metric (beam), with std awareness
    b = {n: summary[n]["beam"] for n in CONFIGS}
    def beats(x, y):  # mean higher AND gap exceeds combined std (rough)
        return b[x][0] > b[y][0]
    def robust(x, y):
        return b[x][0] - b[y][0] > (b[x][1] + b[y][1]) / 2
    print("Q1 multi-helical  (N2 vs N1):")
    if beats("N2", "N1"):
        tag = "ROBUST" if robust("N2", "N1") else "within noise"
        print(f"   N2 {b['N2'][0]:.3f} > N1 {b['N1'][0]:.3f}  -> N2 helps ({tag})")
    else:
        print(f"   N2 {b['N2'][0]:.3f} <= N1 {b['N1'][0]:.3f}  -> multi-helical "
              f"does NOT help (interference adds nothing over a single chain)")
    print("Q2 complex vs real (N1 vs real):")
    if beats("N1", "real"):
        tag = "ROBUST" if robust("N1", "real") else "within noise"
        print(f"   N1 {b['N1'][0]:.3f} > real {b['real'][0]:.3f}  -> complex helps ({tag})")
    else:
        print(f"   N1 {b['N1'][0]:.3f} <= real {b['real'][0]:.3f}  -> no complex advantage")
    print("=" * 66)

    json.dump({"args": vars(args), "per_seed": per_seed, "summary": summary,
               "elapsed_min": (time.time() - t0) / 60},
              open(_OUT / "multiseed.json", "w"), indent=2)
    print(f"[saved] {_OUT}/multiseed.json  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
