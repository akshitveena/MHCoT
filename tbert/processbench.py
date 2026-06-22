"""
tbert/processbench.py  —  the CONCEPTUAL phenomenon, on HUMAN labels (ProcessBench).

ProcessBench gives, per model-generated solution:
  final_answer_correct : did it reach the right answer?
  label                : index of the FIRST human-annotated erroneous step (-1 = none)

Your phenomenon, human-verified:
  RIGHT ANSWER, WRONG PATH  =  final_answer_correct == True  AND  label != -1
  (the answer is correct, yet annotators marked a genuine reasoning error along the way)

No proxy, no LLM judge — these are human labels. We just count and inspect.

    python tbert/processbench.py
"""

from __future__ import annotations

import re


def _test():
    from datasets import load_dataset
    ds = load_dataset("Qwen/ProcessBench")

    print(f"{'subset':14} {'n':>5} {'ans✓':>6} {'clean(✓,no err)':>16} {'WRONG-PATH-RIGHT':>18}")
    keep_examples = []
    for sub in ds.keys():
        d = ds[sub]
        ac = [bool(e["final_answer_correct"]) for e in d]
        lab = [int(e["label"]) for e in d]
        clean = sum(1 for a, l in zip(ac, lab) if a and l == -1)
        phen = sum(1 for a, l in zip(ac, lab) if a and l != -1)
        n_ac = sum(ac)
        rate = phen / n_ac if n_ac else 0
        print(f"{sub:14} {len(d):5d} {n_ac:6d} {clean:16d} {phen:8d} ({rate:.0%} of ans✓)")
        if sub == "gsm8k":
            keep_examples = [e for a, l, e in zip(ac, lab, d) if a and l != -1]

    print(f"\n[gsm8k] {len(keep_examples)} human-verified 'right answer, wrong path' cases. Examples:\n")
    for e in keep_examples[:3]:
        steps = e["steps"]; k = int(e["label"])
        print(f"  problem: {re.sub(r'\\s+',' ', e['problem'])[:110]}")
        print(f"  generator: {e['generator']}  | first ERROR at step {k} (of {len(steps)}); answer is correct")
        if 0 <= k < len(steps):
            print(f"  flagged erroneous step {k}: {re.sub(r'\\s+',' ', steps[k])[:200]}")
        print()

    total_phen = len(keep_examples)
    assert total_phen >= 0
    print("[ProcessBench] the conceptual phenomenon is now measured on HUMAN labels — "
          "no proxy. Next: embed these solutions and test whether the representation "
          "separates clean-correct from wrong-path-right (the A11 test on trustworthy labels).")


if __name__ == "__main__":
    _test()
