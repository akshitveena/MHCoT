"""
main/got_game24.py  —  Track B: Game of 24 (the multi-path frontier test)
========================================================================

Game of 24: given 4 numbers, find an expression using each exactly once with
+ − × ÷ that equals 24. It is the canonical MULTI-PATH benchmark (ToT) — the
solution space branches and greedy single-path search fails. This is where
MHCoT's anti-collapse / multi-hypothesis structure should give a STRUCTURAL
advantage, unlike single-answer GSM8K.

Reuses the GoT machinery + local DeepSeek backend from got_runner.py. Provides:
  * generate_puzzles(n)         — self-contained solvable-puzzle generator
  * verify(expr, numbers)       — programmatic ground truth (no LLM needed)
  * Game24Prompter / Parser     — GoT operations for the 24-game
  * run_game24(numbers)         — full GoT run → candidate expressions

Smoke test (generator/verifier only — no model):
    python -m main.got_game24 --selftest
Full GoT smoke (needs the model):
    python -m main.got_game24
"""

from __future__ import annotations

import itertools
import random
import re
import sys

# graph_of_thoughts (needs openai<1.0) is only required for the FULL GoT run,
# not for the generator / verifier. Import it lazily so this module — and its
# self-test — works even where GoT isn't installed (e.g. system python3).
try:
    from graph_of_thoughts import controller, operations, parser, prompter
    _GOT_OK = True
except Exception as _e:                                      # noqa: BLE001
    _GOT_OK = False
    _GOT_ERR = _e
    prompter = parser = None  # type: ignore


# ---------------------------------------------------------------------------
# Ground truth — solve / verify a 24-game puzzle programmatically
# ---------------------------------------------------------------------------
_OPS = ["+", "-", "*", "/"]


def _solvable(nums) -> bool:
    """Brute-force: is there an expression over these 4 numbers equal to 24?"""
    for perm in set(itertools.permutations(nums)):
        a, b, c, d = perm
        for o1, o2, o3 in itertools.product(_OPS, repeat=3):
            exprs = [
                f"(({a}{o1}{b}){o2}{c}){o3}{d}",
                f"({a}{o1}({b}{o2}{c})){o3}{d}",
                f"({a}{o1}{b}){o2}({c}{o3}{d})",
                f"{a}{o1}(({b}{o2}{c}){o3}{d})",
                f"{a}{o1}({b}{o2}({c}{o3}{d}))",
            ]
            for e in exprs:
                try:
                    if abs(eval(e) - 24) < 1e-6:
                        return True
                except ZeroDivisionError:
                    continue
    return False


def verify(expr: str, numbers) -> bool:
    """
    True iff `expr` uses exactly the given numbers (each once) and equals 24.
    Ground truth for the 24-game — no LLM involved.
    """
    if not expr:
        return False
    used = [int(x) for x in re.findall(r"\d+", expr)]
    if sorted(used) != sorted(int(n) for n in numbers):
        return False
    safe = re.fullmatch(r"[0-9+\-*/().\s]+", expr)
    if not safe:
        return False
    try:
        return abs(eval(expr) - 24) < 1e-6
    except (ZeroDivisionError, SyntaxError):
        return False


def generate_puzzles(n: int, seed: int = 0, lo: int = 1, hi: int = 13):
    """Generate n distinct SOLVABLE 4-number puzzles."""
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        nums = tuple(sorted(rng.randint(lo, hi) for _ in range(4)))
        if nums in seen:
            continue
        seen.add(nums)
        if _solvable(nums):
            out.append(list(nums))
    return out


# ---------------------------------------------------------------------------
# GoT prompter / parser for the 24-game
# ---------------------------------------------------------------------------
def _nums_str(s):
    return s.get("numbers", "") if isinstance(s, dict) else ""


_PBase = prompter.Prompter if _GOT_OK else object
_ParBase = parser.Parser if _GOT_OK else object


class Game24Prompter(_PBase):
    def generate_prompt(self, num_branches, original="", current="", **kw):
        nums = original
        base = (f"Use the numbers {nums} exactly once each, with + - * / and "
                f"parentheses, to make 24. Think step by step, then end with "
                f"'Answer: <expression>'.")
        if current:
            return f"{base}\n\nProblem numbers: {nums}\nSo far:\n{current}\n\nContinue:"
        return f"{base}\n\nNumbers: {nums}\n\nReasoning:"

    def score_prompt(self, state_dicts, **kw):
        s = state_dicts[0]
        return (f"Rate this attempt to make 24 from {s.get('original','')} for "
                f"correctness. End with 'Score: X/10'.\n\n"
                f"Attempt:\n{s.get('current','')}\n\nFinish with 'Score: X/10'.")

    def aggregation_prompt(self, state_dicts, **kw):
        cands = "\n\n".join(f"Candidate {i+1}:\n{s.get('current','')}"
                            for i, s in enumerate(state_dicts))
        return (f"Combine the best ideas from these attempts to make 24 from "
                f"{state_dicts[0].get('original','')}. End with 'Answer: <expression>'.\n\n"
                f"{cands}\n\nSynthesized:")

    def improve_prompt(self, original="", current="", **kw):
        return (f"Improve this attempt to make 24 from {original}, fixing any "
                f"arithmetic error. End with 'Answer: <expression>'.\n\n"
                f"Current:\n{current}\n\nImproved:")

    def validation_prompt(self, **kw):
        return self.score_prompt([kw])


class Game24Parser(_ParBase):
    def parse_generate_answer(self, state, texts):
        return [{**state, "current": t.strip(), "phase": "generated"} for t in texts]

    def parse_aggregation_answer(self, states, texts):
        base = states[0] if states else {}
        return [{**base, "current": t.strip(), "phase": "aggregated"} for t in texts]

    def parse_improve_answer(self, state, texts):
        return {**state, "current": texts[0].strip(), "phase": "improved"}

    def parse_validation_answer(self, state, texts):
        return True

    def parse_score_answer(self, states, texts):
        # reuse the robust 1-10 extractor from the GSM8K parser
        from got_runner import GSM8KParser
        return [GSM8KParser._extract_score(t) for t in texts]


def build_game24_graph(num_initial=3, num_keep=2):
    g = operations.GraphOfOperations()
    g.append_operation(operations.Generate(num_branches_prompt=1,
                                           num_branches_response=num_initial))
    g.append_operation(operations.Score(num_samples=1, combined_scoring=False,
                                        scoring_function=None))
    g.append_operation(operations.KeepBestN(n=num_keep, higher_is_better=True))
    g.append_operation(operations.Aggregate(num_responses=1))
    g.append_operation(operations.Improve())
    return g


def extract_expression(text: str):
    """Pull the final expression from a reasoning trace."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = re.search(r"answer\s*[:=]?\s*([0-9+\-*/().\s]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cands = re.findall(r"[0-9][0-9+\-*/().\s]*[0-9)]", text)
    return cands[-1].strip() if cands else None


def run_game24(numbers, num_initial=3, num_keep=2):
    if not _GOT_OK:
        raise RuntimeError(
            f"graph_of_thoughts unavailable ({_GOT_ERR}). Install 'openai<1.0' "
            f"and graph_of_thoughts (done automatically in the Colab setup).")
    from got_runner import get_llm, _extract_artifacts  # lazy (needs GoT)
    lm = get_llm()
    g = build_game24_graph(num_initial, num_keep)
    nums_str = " ".join(str(n) for n in numbers)
    init = {"original": nums_str, "numbers": nums_str, "current": "", "phase": "initial"}
    ctrl = controller.Controller(lm, g, Game24Prompter(), Game24Parser(), init)
    ctrl.run()
    return _extract_artifacts(g, nums_str)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest():
    print("[selftest] generator + verifier (no model) ...")
    puz = generate_puzzles(5, seed=0)
    print(f"  generated {len(puz)} solvable puzzles: {puz}")
    # verifier sanity
    assert verify("(1+2+3)*4", [1, 2, 3, 4]) is True      # 6*4=24
    assert verify("1*2*3*4", [1, 2, 3, 4]) is True         # 24
    assert verify("8*3*1*1", [8, 3, 1, 1]) is True         # 24
    assert verify("1+2+3+4", [1, 2, 3, 4]) is False        # 10, not 24
    assert verify("(1+2+3)*4", [1, 2, 3, 5]) is False      # wrong numbers
    print("  verifier OK")
    print("[selftest passed]")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        nums = [4, 6, 7, 9]
        print(f"[smoke] GoT on Game-of-24 puzzle {nums} (needs model) ...")
        art = run_game24(nums)
        print(f"  {len(art.candidates)} candidates, final: {art.final_node[:200]}")
        for i, c in enumerate(art.candidates):
            expr = extract_expression(c["text"])
            print(f"  cand{i}: expr={expr!r}  valid={verify(expr, nums) if expr else False}")
