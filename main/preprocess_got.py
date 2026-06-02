"""
preprocess_got.py
=================

Runs Besta GoT (via mhcot.got_runner) over a GSM8K split and caches the
thought-node outputs to disk for downstream MHCoT training.

Output: data/got_cache/gsm8k_{split}.jsonl
    One JSON object per line, in problem order:
    {
      "idx": int,                   index in the original split
      "problem": str,               GSM8K question
      "gold": str,                  GSM8K gold answer (the number)
      "thought_nodes": list[str],   GoT outputs (typically 1)
      "scores": list[float|null],
      "phases": list[str],
      "wall_seconds": float
    }

USAGE
-----
Smoke test first (5 problems, ~5-10 minutes on M3):
    python preprocess_got.py --split test --limit 5

Verify the output looks reasonable, then full run:
    Full test split (~1.3K problems, ~10-22 hours on M3):
        python preprocess_got.py --split test
    Full train split (~7.5K problems, multiple days on M3 — use Colab):
        python preprocess_got.py --split train

Resume after interruption (re-run the same command — already-cached problems
are skipped automatically).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Ensure both this file's folder (for `got_runner` sibling) and the project
# root (for cache paths) are on sys.path regardless of cwd.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import load_dataset
from tqdm import tqdm

from got_runner import run_got_on_problem


# ---------------------------------------------------------------------------
# Paths and gold-answer extraction
# ---------------------------------------------------------------------------
CACHE_DIR = _PROJECT_ROOT / "data" / "got_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GOLD_RE = re.compile(r"####\s*(-?\d[\d,]*\.?\d*)")


def extract_gold(answer_field: str) -> str:
    m = GOLD_RE.search(answer_field)
    return m.group(1).replace(",", "") if m else ""


# ---------------------------------------------------------------------------
# Resume support — read what's already processed
# ---------------------------------------------------------------------------
def load_existing(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    with open(path) as f:
        for line in f:
            try:
                seen.add(json.loads(line)["idx"])
            except Exception:
                continue
    return seen


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N problems (for smoke tests)")
    ap.add_argument("--num_initial", type=int, default=3,
                    help="GoT Generate: number of initial candidates")
    ap.add_argument("--num_keep", type=int, default=2,
                    help="GoT KeepBestN: number kept after scoring")
    args = ap.parse_args()

    out_path = CACHE_DIR / f"gsm8k_{args.split}.jsonl"
    print(f"[preprocess] split={args.split}  output={out_path}")

    ds = load_dataset("gsm8k", "main", split=args.split)
    n_total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"[preprocess] total problems to consider: {n_total}")

    already = load_existing(out_path)
    print(f"[preprocess] already cached: {len(already)}  (will skip these)")

    todo = [i for i in range(n_total) if i not in already]
    print(f"[preprocess] remaining: {len(todo)}")
    if not todo:
        print("[preprocess] nothing to do.")
        return 0

    t_start = time.time()
    n_done = 0
    n_errors = 0

    # Open in append mode so resume works
    with open(out_path, "a", encoding="utf-8") as f:
        for idx in tqdm(todo, desc=f"GoT on {args.split}"):
            problem = ds[idx]["question"]
            gold = extract_gold(ds[idx]["answer"])
            t0 = time.time()

            try:
                nodes = run_got_on_problem(
                    problem,
                    num_initial=args.num_initial,
                    num_keep=args.num_keep,
                )
            except Exception as e:
                n_errors += 1
                err = f"{type(e).__name__}: {e}"
                print(f"\n[error] idx={idx}: {err}", file=sys.stderr)
                # Write a placeholder so we can analyze failures later
                record = {
                    "idx": idx, "problem": problem, "gold": gold,
                    "thought_nodes": [], "scores": [], "phases": [],
                    "wall_seconds": time.time() - t0, "error": err,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                continue

            record = {
                "idx": idx,
                "problem": problem,
                "gold": gold,
                "thought_nodes": [n.text for n in nodes],
                "scores": [n.score for n in nodes],
                "phases": [n.phase for n in nodes],
                "wall_seconds": time.time() - t0,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1

    elapsed = time.time() - t_start
    avg = elapsed / max(n_done, 1)
    print(f"\n[preprocess] done.")
    print(f"  processed:    {n_done}")
    print(f"  errors:       {n_errors}")
    print(f"  elapsed:      {elapsed/60:.1f} min")
    print(f"  avg/problem:  {avg:.1f} s")
    print(f"  output:       {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
