"""
tbert/quadrants.py  —  de-risk the core idea: can we MEASURE "right answer, wrong path"?

Splits every reasoning candidate by two independent signals:
  * answer-correct?  (boxed/last-number vs gold)         -- we already have this
  * steps-valid?     (do the stated arithmetic equations check out?)  -- NEW, self-contained

  Q1 ✓/✓  genuinely correct
  Q2 ✓/✗  RIGHT ANSWER, WRONG PATH   <-- the phenomenon the interpretability idea needs
  Q3 ✗/✓  valid steps, wrong answer
  Q4 ✗/✗  wrong both ways

No external model — just arithmetic verification of the equations the candidate states.
Honest limitation: only catches errors in EXPLICIT 'a op b = c' steps; implicit reasoning
isn't checked, so 'steps-valid' is a NECESSARY-not-sufficient proxy for reasoning validity.

    python tbert/quadrants.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_candidates                                  # noqa: E402

_NUM = r"-?\d[\d,]*\.?\d*"
# a FULL arithmetic expression (>=2 operands) followed by '= number'
_EXPR_EQ = re.compile(r"((?:" + _NUM + r"\s*[-+*x×/÷]\s*)+" + _NUM + r")\s*=\s*(" + _NUM + r")"
                      r"(?![\d/×÷*=])")          # RHS not part of a bigger expr (e.g. 42/x)


def _f(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def _eval_expr(expr):
    s = expr.replace(",", "").replace("×", "*").replace("÷", "/").replace("x", "*")
    if not re.fullmatch(r"[\d.\s+\-*/()]+", s):
        return None
    try:
        return float(eval(s, {"__builtins__": {}}, {}))
    except Exception:
        return None


def find_equations(text, tol=1e-2):
    """-> list of (display, lhs_value, stated, is_bad, span). Evaluates the FULL
    leading expression and skips chained tails / fractions / algebra via a prefix guard."""
    res = []
    for m in _EXPR_EQ.finditer(text):
        s = m.start(); j = s - 1
        while j >= 0 and text[j] == " ":
            j -= 1
        if j >= 0 and (text[j].isalpha() or text[j] in "+-*/×÷=."):   # tail/algebra/fraction
            continue
        e = m.end()
        tail = text[e:e + 6]
        if re.match(r"[A-Za-z]", tail) or re.match(r"\s*[*/×÷+\-]", tail):  # RHS is a var/expr (algebra)
            continue
        expr, c = m.group(1), m.group(2)
        lhs, cf = _eval_expr(expr), _f(c)
        if lhs is None or cf is None:
            continue
        bad = abs(lhs - cf) > tol * max(1, abs(cf))
        res.append((f"{expr}={c}", lhs, cf, bad, m.span()))
    return res


def step_check(text, tol=1e-2):
    """-> (n_equations, n_bad). Verifies the full expression, not isolated pairs."""
    eqs = find_equations(text, tol)
    return len(eqs), sum(1 for *_, bad, _ in eqs if bad)


def _test():
    texts, ans_correct, pids = load_candidates()
    rows = []
    for t, ac in zip(texts, ans_correct):
        n_eq, n_bad = step_check(t)
        rows.append((ac, n_eq, n_bad))
    rows = np.array(rows)
    ac = rows[:, 0].astype(bool)
    has_eq = rows[:, 1] > 0
    steps_valid = has_eq & (rows[:, 2] == 0)        # has equations and none wrong

    # only judge step-validity where we actually found equations
    judg = has_eq
    print(f"[quadrants] {len(rows)} candidates; {judg.mean():.0%} have explicit equations we can check")
    print(f"            (the rest are 'unknown steps' — no explicit arithmetic to verify)\n")

    a, s = ac[judg], steps_valid[judg]
    q1 = int(np.sum(a & s)); q2 = int(np.sum(a & ~s))
    q3 = int(np.sum(~a & s)); q4 = int(np.sum(~a & ~s))
    tot = q1 + q2 + q3 + q4
    print("                     steps ✓        steps ✗")
    print(f"  answer ✓ (right)   Q1 {q1:4d}      Q2 {q2:4d}  <- right answer, WRONG PATH")
    print(f"  answer ✗ (wrong)   Q3 {q3:4d}      Q4 {q4:4d}")
    print(f"\n  Q2 'right answer, wrong path' = {q2}/{tot} = {q2/tot:.1%} of checkable candidates")

    # show a few Q2 exemplars
    q2_idx = [i for i in np.where(judg)[0] if ac[i] and not steps_valid[i]][:2]
    for i in q2_idx:
        ne, nb = step_check(texts[i])
        print(f"\n  [Q2 example] answer ✓ but {nb}/{ne} stated equations wrong:")
        print("   " + re.sub(r"\s+", " ", texts[i])[:240] + "…")

    assert tot > 100, "need a reasonable checkable set"
    if q2 >= 30:
        print(f"\n[VIABLE] Q2 is populated ({q2} cases) — 'right answer, wrong path' is measurable; "
              "the interpretability pipeline (topic-model -> within-topic quadrant features) has data.")
    else:
        print(f"\n[CAUTION] Q2 is small ({q2}) — the phenomenon may be too rare here to learn from; "
              "we'd need more candidates or a stronger validity signal (e.g. an LLM judge / ProcessBench).")


if __name__ == "__main__":
    _test()
