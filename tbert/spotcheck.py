"""
tbert/spotcheck.py  —  validate the step-checker: are Q2 flags REAL errors or artifacts?

For a seeded sample of Q2 candidates (answer ✓, >=1 flagged bad equation), print each
flagged equation as `stated a op b = c | actual = r` with surrounding context, so we
can classify real arithmetic errors vs parse artifacts (esp. chained expressions like
'5 + 3 + 2 = 10' where the regex grabs '3 + 2 = 10').

    python tbert/spotcheck.py [N]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_candidates                                  # noqa: E402
from quadrants import find_equations, step_check                  # noqa: E402


def bad_eqs_with_context(text, tol=1e-2, ctx=45):
    out = []
    for display, lhs, stated, bad, (s, e) in find_equations(text, tol):
        if bad:
            before = text[max(0, s - ctx):s].replace("\n", " ")
            after = text[e:e + ctx].replace("\n", " ")
            out.append((display, lhs, before, after))
    return out


def _test():
    n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    texts, ac, _ = load_candidates()
    q2 = [i for i, a in enumerate(ac) if a and step_check(texts[i])[1] > 0]
    rng = np.random.RandomState(0); rng.shuffle(q2)
    print(f"[spotcheck] {len(q2)} Q2 candidates; showing {n_show} (seed 0)\n")
    for k, i in enumerate(q2[:n_show], 1):
        for stated, actual, before, after in bad_eqs_with_context(texts[i])[:1]:
            print(f"{k:2d}. flagged: {stated}   (actual {actual:g})")
            print(f"    …{before}[{stated}]{after}…\n")


if __name__ == "__main__":
    _test()
