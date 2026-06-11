"""
main/data.py
============

Loads the GoT cache, extracts predicted answers, and produces correctness
labels — the bridge from raw GoT text to a trainable dataset.

Every downstream experiment depends on this:
  * correctness label  → supervision signal (L_task)
  * same-answer groups  → contrastive signal (L_contrast)
  * candidate disagreement → the Option-3 diversity signal

Public API
----------
load_dataset(path) -> list[ProblemRecord]
extract_answer(text) -> str | None        # robust to <think>, \\boxed, LaTeX
summary_stats(records) -> dict            # quick health/diversity report
same_answer_groups(record) -> dict        # {answer: [candidate indices]}

Run directly to print a summary over the current cache:
    python main/data.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = _PROJECT_ROOT / "data" / "got_cache" / "gsm8k_test.jsonl"


# ---------------------------------------------------------------------------
# Answer extraction — robust to DeepSeek's <think> blocks and LaTeX formatting
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize_number(s: str) -> str | None:
    """Strip LaTeX/currency artifacts and return a canonical numeric string."""
    if s is None:
        return None
    # remove LaTeX thin-spaces and stray backslashes, currency, commas, spaces
    s = s.replace("\\!", "").replace("\\,", "").replace("\\", "")
    s = s.replace("$", "").replace(",", "").replace(" ", "").strip()
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    val = m.group()
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def extract_answer(text: str) -> str | None:
    """
    Extract the final numeric answer from a reasoning trace.
    Priority: \\boxed{...}  →  'Answer: X'  →  last number.
    The <think> block is stripped first (it cites the problem's numbers).
    """
    if not text:
        return None
    # 1. Drop the internal monologue — the answer is in the final portion
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # 2. Prefer the last \boxed{...}
    boxes = _BOXED_RE.findall(text)
    if boxes:
        n = _normalize_number(boxes[-1])
        if n is not None:
            return n

    # 3. 'Answer: X' / 'Final Answer: X'
    m = re.search(r"(?:final\s+answer|answer)\s*[:*]*\s*\$?([^\n]*)",
                  text, re.IGNORECASE)
    if m:
        n = _normalize_number(m.group(1))
        if n is not None:
            return n

    # 4. Last number anywhere in the final portion
    nums = _NUM_RE.findall(text)
    if nums:
        return _normalize_number(nums[-1])

    return None


def is_correct(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    text: str
    pred: str | None
    correct: bool
    score: float | None = None


@dataclass
class ProblemRecord:
    idx: int
    problem: str
    gold: str
    candidates: list[Candidate]
    final_text: str
    final_pred: str | None
    final_correct: bool
    error: str | None = None

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def distinct_answers(self) -> set[str]:
        """Set of distinct predicted answers among candidates (None dropped)."""
        return {c.pred for c in self.candidates if c.pred is not None}

    @property
    def candidates_agree(self) -> bool:
        """True if all candidates reached the same answer."""
        return len(self.distinct_answers) <= 1

    @property
    def any_candidate_correct(self) -> bool:
        return any(c.correct for c in self.candidates)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dataset(path: str | Path = _DEFAULT_CACHE) -> list[ProblemRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cache not found: {path}")

    records: list[ProblemRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gold = r.get("gold", "")

            cands: list[Candidate] = []
            raw_candidates = r.get("candidates") or []
            raw_scores = r.get("scores") or []
            for i, c in enumerate(raw_candidates):
                ctext = c.get("text", "") if isinstance(c, dict) else str(c)
                cscore = c.get("score") if isinstance(c, dict) else None
                pred = extract_answer(ctext)
                cands.append(Candidate(
                    text=ctext, pred=pred,
                    correct=is_correct(pred, gold), score=cscore,
                ))

            final_text = r.get("final_node", "") or ""
            if not final_text and r.get("thought_nodes"):
                final_text = r["thought_nodes"][0]
            final_pred = extract_answer(final_text)

            records.append(ProblemRecord(
                idx=r.get("idx", -1),
                problem=r.get("problem", ""),
                gold=gold,
                candidates=cands,
                final_text=final_text,
                final_pred=final_pred,
                final_correct=is_correct(final_pred, gold),
                error=r.get("error"),
            ))
    return records


# ---------------------------------------------------------------------------
# Contrastive grouping — same predicted answer = positive pairs
# ---------------------------------------------------------------------------
def same_answer_groups(record: ProblemRecord) -> dict[str, list[int]]:
    """Map each predicted answer → list of candidate indices that produced it."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(record.candidates):
        if c.pred is None:
            continue
        groups.setdefault(c.pred, []).append(i)
    return groups


# ---------------------------------------------------------------------------
# Summary / health report — a mini-preview of Experiment -1
# ---------------------------------------------------------------------------
def summary_stats(records: list[ProblemRecord]) -> dict:
    n = len(records)
    ok = [r for r in records if not r.error and r.candidates]
    n_ok = len(ok)

    final_acc = sum(r.final_correct for r in ok) / max(n_ok, 1)

    cand_total = sum(r.n_candidates for r in ok)
    cand_correct = sum(c.correct for r in ok for c in r.candidates)
    cand_acc = cand_correct / max(cand_total, 1)

    n_extracted = sum(1 for r in ok for c in r.candidates if c.pred is not None)
    extract_rate = n_extracted / max(cand_total, 1)

    agree = [r for r in ok if r.candidates_agree]
    disagree = [r for r in ok if not r.candidates_agree]

    # The Option-3 signal preview: does agreement relate to correctness?
    acc_when_agree = (sum(r.final_correct for r in agree) / max(len(agree), 1))
    acc_when_disagree = (sum(r.final_correct for r in disagree) / max(len(disagree), 1))

    return {
        "n_records": n,
        "n_usable": n_ok,
        "n_errors": n - n_ok,
        "final_node_accuracy": round(final_acc, 3),
        "candidate_accuracy": round(cand_acc, 3),
        "answer_extraction_rate": round(extract_rate, 3),
        "frac_candidates_agree": round(len(agree) / max(n_ok, 1), 3),
        "frac_candidates_disagree": round(len(disagree) / max(n_ok, 1), 3),
        "acc_when_candidates_agree": round(acc_when_agree, 3),
        "acc_when_candidates_disagree": round(acc_when_disagree, 3),
    }


def _print_summary() -> None:
    records = load_dataset()
    stats = summary_stats(records)
    print(f"[data] loaded {stats['n_records']} records "
          f"({stats['n_usable']} usable, {stats['n_errors']} errors)\n")
    print(f"  answer extraction rate : {stats['answer_extraction_rate']:.3f}  "
          f"(want > 0.95 — else the parser is missing answers)")
    print(f"  final_node accuracy    : {stats['final_node_accuracy']:.3f}")
    print(f"  candidate accuracy     : {stats['candidate_accuracy']:.3f}")
    print()
    print(f"  candidates AGREE       : {stats['frac_candidates_agree']:.3f} of problems")
    print(f"  candidates DISAGREE    : {stats['frac_candidates_disagree']:.3f} of problems")
    print()
    print("  --- Option-3 signal preview (agreement vs correctness) ---")
    print(f"  acc when candidates agree    : {stats['acc_when_candidates_agree']:.3f}")
    print(f"  acc when candidates disagree : {stats['acc_when_candidates_disagree']:.3f}")
    print(f"  → if agree >> disagree, candidate consensus predicts correctness")
    print()

    # Show one disagreeing example so we can eyeball extraction
    for r in records:
        if not r.error and not r.candidates_agree:
            print(f"  example idx={r.idx} gold={r.gold}")
            for i, c in enumerate(r.candidates):
                mark = "✓" if c.correct else "✗"
                print(f"    cand{i}: pred={c.pred!s:>10} {mark}")
            fm = "✓" if r.final_correct else "✗"
            print(f"    final: pred={r.final_pred!s:>10} {fm}")
            break


if __name__ == "__main__":
    _print_summary()
