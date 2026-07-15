"""
preprocess_game24.py  —  Track B data generation (Game of 24)
=============================================================

Generates solvable 24-game puzzles, runs GoT on each (local DeepSeek), and
caches candidates + final expression with PROGRAMMATIC correctness labels
(verify() — no LLM needed for ground truth, unlike GSM8K).

Output: data/got_cache/game24.jsonl  (one JSON per puzzle)
    { idx, numbers, candidates:[{text,score,expr,valid}], final_node,
      final_expr, final_valid, any_candidate_valid, wall_seconds }

Run (Colab, after the env is set up):
    python main/preprocess_game24.py --n 600
Resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from Projects.MHCoT.main.got_game24 import generate_puzzles, run_game24, extract_expression, verify  # noqa: E402
from tqdm import tqdm                                                            # noqa: E402

CACHE_DIR = _ROOT / "data" / "got_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_done(path):
    if not path.exists():
        return set()
    done = set()
    for line in open(path):
        try:
            done.add(json.loads(line)["idx"])
        except Exception:
            continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="number of puzzles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_initial", type=int, default=3)
    ap.add_argument("--num_keep", type=int, default=2)
    args = ap.parse_args()

    out = CACHE_DIR / "game24.jsonl"
    puzzles = generate_puzzles(args.n, seed=args.seed)
    print(f"[game24] {len(puzzles)} solvable puzzles → {out}")
    done = load_done(out)
    todo = [i for i in range(len(puzzles)) if i not in done]
    print(f"[game24] already done {len(done)}, remaining {len(todo)}")

    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for idx in tqdm(todo, desc="GoT game24"):
            nums = puzzles[idx]
            t = time.time()
            try:
                art = run_game24(nums, args.num_initial, args.num_keep)
            except Exception as e:
                print(f"\n[error] idx={idx}: {e}", file=sys.stderr)
                continue
            cands = []
            for c in art.candidates:
                expr = extract_expression(c["text"])
                cands.append({"text": c["text"], "score": c["score"],
                              "expr": expr, "valid": bool(expr and verify(expr, nums))})
            fexpr = extract_expression(art.final_node)
            rec = {
                "idx": idx, "numbers": nums,
                "candidates": cands,
                "final_node": art.final_node,
                "final_expr": fexpr,
                "final_valid": bool(fexpr and verify(fexpr, nums)),
                "any_candidate_valid": any(c["valid"] for c in cands),
                "wall_seconds": time.time() - t,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[game24] done in {(time.time()-t0)/60:.1f} min → {out}")


if __name__ == "__main__":
    main()
