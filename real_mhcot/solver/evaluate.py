"""
solver/evaluate.py  —  THE PRE-COMMITTED TEST
=============================================

Plugs each trained heuristic (real / N=1 / N=2) into the SAME search harness on
the SAME held-out puzzles and measures, at a fixed search budget:

  * solve-rate           : fraction of SOLVABLE puzzles actually solved
  * nodes-to-solution    : avg expansions before the first solution (efficiency;
                           lower = the heuristic guided the search better)

Both best-first and beam search are reported. Run across multiple seeds (the
checkpoints are per-seed) for variance.

THE BAR (committed before looking at numbers):
  N=2 must beat BOTH real AND N=1 on solve-rate-at-budget (and ideally on
  nodes-to-solution) on held-out puzzles, consistently. Otherwise the
  multi-path hypothesis is dead — cheaply and honestly.

Run (after train.py):
    python solver/evaluate.py
    python solver/evaluate.py --budget 30 --beam 3
Out: solver/results/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "main"))
sys.path.insert(0, str(_HERE))   # solver FIRST: solver/model.py, not main/model.py

from game24 import Puzzle                                   # noqa: E402
from search import best_first_search, beam_search          # noqa: E402
from model import MHCoTSolver, RealSolver, make_heuristic  # noqa: E402

_CKPT = _HERE / "ckpt"
_OUT = _HERE / "results"
_OUT.mkdir(exist_ok=True)


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(dev):
    builders = {
        "real": lambda: RealSolver(d_model=64, n_heads=4, n_layers=2),
        "N1":   lambda: MHCoTSolver(d_model=64, n_heads=4, n_chains=1),
        "N2":   lambda: MHCoTSolver(d_model=64, n_heads=4, n_chains=2),
    }
    models = {}
    for name, build in builders.items():
        ck = _CKPT / f"{name}.pt"
        if not ck.exists():
            print(f"[error] missing {ck} — run train.py first."); sys.exit(1)
        m = build().to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev))
        m.eval()
        models[name] = m
    return models


def evaluate_heuristic(h, puzzles, budget, beam_width):
    """Return metrics over the SOLVABLE puzzles (the ones with a findable goal)."""
    solvable = [p for p in puzzles if p.solvable]
    bf_solved, bf_nodes = 0, []
    bm_solved = 0
    for p in solvable:
        r = best_first_search(p.numbers, p.target, h, node_budget=budget)
        if r.solved:
            bf_solved += 1; bf_nodes.append(r.nodes_expanded)
        rb = beam_search(p.numbers, p.target, h, beam_width=beam_width,
                         node_budget=budget)
        bm_solved += rb.solved
    n = len(solvable)
    return {
        "n_solvable": n,
        "bestfirst_solve_rate": bf_solved / n if n else 0.0,
        "bestfirst_avg_nodes": (sum(bf_nodes) / len(bf_nodes)) if bf_nodes else None,
        "beam_solve_rate": bm_solved / n if n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=30,
                    help="node-expansion budget (lower = harder, sharper test)")
    ap.add_argument("--beam", type=int, default=3)
    args = ap.parse_args()

    dev = _device(); print(f"[eval] device={dev}  budget={args.budget}  beam={args.beam}")
    split = torch.load(_CKPT / "split.pt", weights_only=False)
    puzzles = [Puzzle(tuple(n), t, s) for (n, t, s) in split["test_puzzles"]]
    n_solv = sum(p.solvable for p in puzzles)
    print(f"[eval] held-out: {len(puzzles)} puzzles ({n_solv} solvable)\n")

    models = load_models(dev)
    results = {}
    for name, m in models.items():
        h = make_heuristic(m, device=dev)
        results[name] = evaluate_heuristic(h, puzzles, args.budget, args.beam)
        r = results[name]
        nodes = r["bestfirst_avg_nodes"]
        print(f"  {name:<5}  best-first solve {r['bestfirst_solve_rate']:.3f}  "
              f"nodes {nodes if nodes is None else round(nodes,1)}  |  "
              f"beam solve {r['beam_solve_rate']:.3f}")

    print("\n" + "=" * 60)
    bf = {k: results[k]["bestfirst_solve_rate"] for k in results}
    print("VERDICT (best-first solve-rate @ budget {}):".format(args.budget))
    print(f"  real={bf['real']:.3f}  N1={bf['N1']:.3f}  N2={bf['N2']:.3f}")
    if bf["N2"] > bf["real"] and bf["N2"] > bf["N1"]:
        margin = bf["N2"] - max(bf["real"], bf["N1"])
        print(f"  -> N=2 beats BOTH real and N=1 by {margin:.3f}. Hypothesis SURVIVES "
              f"this seed -> run more seeds, then scale to a harder regime / LLM-backed.")
    else:
        print("  -> N=2 does NOT beat both. On this run the multi-path hypothesis "
              "does not hold. (Check more seeds before final verdict.)")
    print("=" * 60)
    json.dump({"budget": args.budget, "beam": args.beam, "results": results},
              open(_OUT / "eval.json", "w"), indent=2)
    print(f"[saved] {_OUT}/eval.json")


if __name__ == "__main__":
    main()
