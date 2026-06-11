"""
mhcot/got_runner.py
===================

Wraps Besta 2024's Graph of Thoughts framework so it can produce thought-node
text strings using a local DeepSeek-R1-Distill-Qwen-1.5B model as the LLM
backend (instead of OpenAI's API, which the published Besta code defaults to).

Outputs from this module are the atomic input units for MHCoT:
    Problem text  ──>  run_got_on_problem()  ──>  list[ThoughtNode]
    Each ThoughtNode.text is a self-contained reasoning trace that becomes
    one input to the MHCoT encoder downstream.

Smoke test (M3):
    conda activate mhcot
    python -m mhcot.got_runner

The first run downloads the DeepSeek-Distill model (~3GB) if not cached.
End-to-end runtime per problem on M3: ~30-90 seconds.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Besta imports.
# Their package eagerly loads `openai` via controller/__init__.py — that's why
# we pip-installed openai<1.0. We are NOT calling OpenAI; the import is purely
# to satisfy Besta's module loader.
# ---------------------------------------------------------------------------
from graph_of_thoughts import controller, operations, parser, prompter

# The base LLM class lives in either `language_models` or `controller`
# depending on Besta's version. Try both.
try:
    from graph_of_thoughts.language_models import AbstractLanguageModel
except ImportError:
    from graph_of_thoughts.controller.abstract_language_model import AbstractLanguageModel  # type: ignore


# ---------------------------------------------------------------------------
# Module-level config
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Local LLM backend — replaces OpenAI's ChatGPT in the Besta paper
# ---------------------------------------------------------------------------
class LocalDeepSeekLLM(AbstractLanguageModel):
    """
    Plugs DeepSeek-R1-Distill-Qwen-1.5B into Besta's GoT framework.

    Besta's `AbstractLanguageModel` typically expects:
      - .query(prompt: str, num_responses: int) -> list[Any]
      - .get_response_texts(query_response) -> list[str]
      - some accounting attributes (cost, tokens) — we set them as 0

    We bypass the parent __init__ that loads a config.json (we don't use any
    OpenAI config), and we initialize the bookkeeping attributes ourselves.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        # IMPORTANT: skip parent __init__ that would try to read a config file.
        # We populate the attributes the parent and Besta's controller expect.
        self.config: dict[str, Any] = {}
        self.model_id: str = model_name
        self.cache: bool = False
        self.response_cache: dict[str, Any] = {}
        self.cost: float = 0.0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        # Some Besta versions look for these too; harmless if unused
        self.prompt_token_cost: float = 0.0
        self.response_token_cost: float = 0.0

        print(f"[LocalLLM] loading {model_name} on {device} (dtype={dtype}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        print(f"[LocalLLM] ready.")

    @torch.no_grad()
    def query(self, query: str, num_responses: int = 1) -> list[dict[str, str]]:
        """
        Generate `num_responses` independent completions for the same prompt.
        Returns a list of {"text": <generated_string>}.
        """
        # Wrap in DeepSeek-Distill's chat template
        messages = [{"role": "user", "content": query}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)

        responses: list[dict[str, str]] = []
        # If only one response asked, use greedy (deterministic); else sample
        do_sample = num_responses > 1
        for _ in range(num_responses):
            out = self.model.generate(
                **ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else 1.0,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            gen_ids = out[0][ids.input_ids.shape[1]:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append({"text": gen_text})

        # Token accounting (rough — only used for logging)
        self.prompt_tokens += int(ids.input_ids.shape[1]) * num_responses
        self.completion_tokens += sum(
            len(self.tokenizer(r["text"], add_special_tokens=False)["input_ids"])
            for r in responses
        )
        return responses

    def get_response_texts(self, query_response: Any) -> list[str]:
        """Extract the plain text from one or more query() responses."""
        if isinstance(query_response, dict):
            return [query_response.get("text", "")]
        if isinstance(query_response, list):
            return [r.get("text", "") if isinstance(r, dict) else str(r)
                    for r in query_response]
        return [str(query_response)]


# ---------------------------------------------------------------------------
# GSM8K-specific Prompter and Parser
#
# Besta's operations call into these classes to (a) construct the LLM prompt
# for each operation type and (b) parse the LLM's response into a state dict
# that the next operation consumes.
# ---------------------------------------------------------------------------
class GSM8KPrompter(prompter.Prompter):
    def generate_prompt(self, num_branches: int, original: str = "",
                        current: str = "", **kwargs) -> str:
        if not current:
            return (
                f"Solve this math word problem step by step. Show your "
                f"reasoning. End with 'Answer: <number>'.\n\n"
                f"Problem: {original}\n\nReasoning:"
            )
        return (
            f"Continue this partial reasoning, then end with 'Answer: <number>'.\n\n"
            f"Problem: {original}\nPartial reasoning so far:\n{current}\n\nContinue:"
        )

    def score_prompt(self, state_dicts: list[dict], **kwargs) -> str:
        s = state_dicts[0]
        return (
            f"Rate the following math reasoning for correctness and clarity. "
            f"After any thinking, end your reply with the exact line "
            f"'Score: X/10' where X is a number from 1 to 10.\n\n"
            f"Problem: {s.get('original', '')}\n"
            f"Reasoning:\n{s.get('current', '')}\n\n"
            f"Finish with 'Score: X/10'."
        )

    def aggregation_prompt(self, state_dicts: list[dict], **kwargs) -> str:
        candidates = "\n\n".join(
            f"Candidate {i+1}:\n{s.get('current', '')}"
            for i, s in enumerate(state_dicts)
        )
        return (
            f"You are given multiple candidate solutions to a math problem. "
            f"Synthesize the best parts of each into a single improved "
            f"reasoning trace. End with 'Answer: <number>'.\n\n"
            f"Problem: {state_dicts[0].get('original', '')}\n\n"
            f"{candidates}\n\nSynthesized reasoning:"
        )

    def improve_prompt(self, original: str = "", current: str = "",
                       **kwargs) -> str:
        return (
            f"Improve the following math reasoning by fixing errors and "
            f"clarifying steps. End with 'Answer: <number>'.\n\n"
            f"Problem: {original}\nCurrent reasoning:\n{current}\n\n"
            f"Improved reasoning:"
        )

    def validation_prompt(self, **kwargs) -> str:
        return self.score_prompt([kwargs])


class GSM8KParser(parser.Parser):
    def parse_generate_answer(self, state: dict, texts: list[str]) -> list[dict]:
        return [{**state, "current": t.strip(), "phase": "generated"} for t in texts]

    def parse_aggregation_answer(self, states: list[dict], texts: list[str]) -> list[dict]:
        base = states[0] if states else {}
        return [{**base, "current": t.strip(), "phase": "aggregated"} for t in texts]

    def parse_improve_answer(self, state: dict, texts: list[str]) -> dict:
        return {**state, "current": texts[0].strip(), "phase": "improved"}

    @staticmethod
    def _extract_score(text: str) -> float:
        """
        Robustly pull a 1-10 quality score from a DeepSeek-R1-Distill response.

        The naive 'first number' approach fails because the model emits a
        <think>...</think> block that cites the PROBLEM's numbers (e.g. "16
        eggs"), so the first number is never the score. Strategy:
          1. Drop the <think> block — the score lives in the final answer.
          2. Prefer explicit 'X/10' or 'X out of 10' patterns.
          3. Then 'Score: X' / 'rating: X' patterns.
          4. Fall back to the LAST number that lies in [0, 10].
          5. Default to a neutral 5.0 if nothing usable is found.
        Result is always clamped to [0, 10].
        """
        # 1. Strip the internal monologue
        if "</think>" in text:
            text = text.split("</think>")[-1]

        def _clamp(v: float) -> float:
            return max(0.0, min(10.0, v))

        # 2. 'X/10' or 'X out of 10'
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*10", text, re.IGNORECASE)
        if m:
            return _clamp(float(m.group(1)))

        # 3. 'Score: X' / 'rating: X' / 'rate: X'
        m = re.search(r"(?:score|rating|rate)\s*[:=]?\s*(\d+(?:\.\d+)?)",
                      text, re.IGNORECASE)
        if m:
            return _clamp(float(m.group(1)))

        # 4. Last number in [0, 10] within the final answer
        in_range = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)
                    if 0.0 <= float(x) <= 10.0]
        if in_range:
            return in_range[-1]

        # 5. Neutral default
        return 5.0

    def parse_validation_answer(self, state: dict, texts: list[str]) -> bool:
        return self._extract_score(texts[0]) >= 5.0

    def parse_score_answer(self, states: list[dict], texts: list[str]) -> list[float]:
        return [self._extract_score(t) for t in texts]


# ---------------------------------------------------------------------------
# GoT operation graph for GSM8K
# ---------------------------------------------------------------------------
def build_gsm8k_graph(
    num_initial: int = 3,
    num_keep: int = 2,
) -> operations.GraphOfOperations:
    """
    A minimal-but-faithful GoT pipeline for GSM8K:

        Generate num_initial reasoning candidates
        ──> Score each
        ──> KeepBestN keep the top num_keep
        ──> Aggregate them into one synthesized trace
        ──> Improve that synthesized trace

    Final output: one polished thought-node per problem.
    """
    g = operations.GraphOfOperations()
    g.append_operation(operations.Generate(num_branches_prompt=1,
                                           num_branches_response=num_initial))
    g.append_operation(operations.Score(num_samples=1,
                                        combined_scoring=False,
                                        scoring_function=None))
    g.append_operation(operations.KeepBestN(n=num_keep, higher_is_better=True))
    g.append_operation(operations.Aggregate(num_responses=1))
    g.append_operation(operations.Improve())
    return g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class ThoughtNode:
    problem: str
    text: str
    score: float | None = None
    phase: str = "improved"


@dataclass
class GoTArtifacts:
    """
    Full record of one GoT graph run — not just the distilled final node.
    Needed so we can later pursue:
      * Option 1: phase-shifted chains over `final_node` (the ε-helix design)
      * Option 3: N chains initialized from different `candidates`
                  (interference between genuinely different reasoning approaches)
    Saving candidates costs nothing now and preserves both options.
    """
    problem: str
    candidates: list[dict]              # [{"text": str, "score": float|None}, ...] from Generate
    kept: list[str]                    # texts that survived KeepBestN
    aggregated: str | None             # Aggregate output
    final_node: str                    # Improve output — the distilled node
    final_score: float | None = None


# --- helpers to read text/score out of a Besta "thought" defensively --------
def _thought_text(th) -> str:
    state = getattr(th, "state", None)
    if isinstance(state, dict):
        return state.get("current", "") or ""
    return ""


def _thought_score(th):
    s = getattr(th, "score", None)
    try:
        return float(s) if s is not None else None
    except (TypeError, ValueError):
        return None


def _extract_artifacts(graph, problem: str) -> GoTArtifacts:
    """
    Walk the graph's operations and pull out per-stage artifacts.
    Written defensively: any stage that fails to yield thoughts is skipped,
    so a Besta API quirk degrades gracefully rather than crashing the run.
    """
    candidates: list[dict] = []
    kept: list[str] = []
    aggregated: str | None = None
    final_node: str = ""
    final_score = None

    for op in graph.operations:
        op_name = type(op).__name__
        try:
            thoughts = op.get_thoughts()
        except Exception:
            continue
        if not thoughts:
            continue

        if op_name == "Generate":
            candidates = [
                {"text": _thought_text(t), "score": _thought_score(t)}
                for t in thoughts
            ]
        elif op_name == "Score":
            # Score attaches scores to the thoughts flowing through it;
            # align positionally with the candidates if possible.
            for i, t in enumerate(thoughts):
                s = _thought_score(t)
                if i < len(candidates) and s is not None:
                    candidates[i]["score"] = s
        elif op_name == "KeepBestN":
            kept = [_thought_text(t) for t in thoughts]
        elif op_name == "Aggregate":
            aggregated = _thought_text(thoughts[0])
        elif op_name == "Improve":
            final_node = _thought_text(thoughts[0])
            final_score = _thought_score(thoughts[0])

    # Fallback: if Improve yielded nothing, use the terminal op's thought
    if not final_node:
        try:
            term = graph.operations[-1].get_thoughts()
            if term:
                final_node = _thought_text(term[0])
                final_score = _thought_score(term[0])
        except Exception:
            pass

    return GoTArtifacts(
        problem=problem,
        candidates=candidates,
        kept=kept,
        aggregated=aggregated,
        final_node=final_node,
        final_score=final_score,
    )


_LLM_SINGLETON: LocalDeepSeekLLM | None = None


def get_llm() -> LocalDeepSeekLLM:
    """Lazy-load the LLM once and reuse across calls."""
    global _LLM_SINGLETON
    if _LLM_SINGLETON is None:
        _LLM_SINGLETON = LocalDeepSeekLLM()
    return _LLM_SINGLETON


def run_got_full(
    problem: str,
    num_initial: int = 3,
    num_keep: int = 2,
) -> GoTArtifacts:
    """
    Run the full Besta GoT pipeline on one problem and return ALL graph
    artifacts (candidates + scores + aggregated + final node), not just the
    distilled output. This is the function preprocessing should call.
    """
    lm = get_llm()
    g = build_gsm8k_graph(num_initial=num_initial, num_keep=num_keep)
    initial_state = {"original": problem, "current": "", "phase": "initial"}

    ctrl = controller.Controller(
        lm,
        g,
        GSM8KPrompter(),
        GSM8KParser(),
        initial_state,
    )
    ctrl.run()
    return _extract_artifacts(g, problem)


def run_got_on_problem(
    problem: str,
    num_initial: int = 3,
    num_keep: int = 2,
) -> list[ThoughtNode]:
    """
    Backward-compatible wrapper — returns just the final node as a list of
    ThoughtNode. Prefer run_got_full() for new code.
    """
    art = run_got_full(problem, num_initial=num_initial, num_keep=num_keep)
    return [ThoughtNode(
        problem=problem,
        text=art.final_node,
        score=art.final_score,
        phase="improved",
    )]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    problem = (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
        "every morning and bakes muffins for her friends every day with four. "
        "She sells the remainder at the farmers' market daily for $2 per "
        "fresh duck egg. How much in dollars does she make every day?"
    )
    print(f"\n[smoke] problem:\n{problem}\n")
    art = run_got_full(problem, num_initial=3, num_keep=2)
    print(f"\n[smoke] captured {len(art.candidates)} candidate(s), "
          f"{len(art.kept)} kept, aggregated={'yes' if art.aggregated else 'no'}\n")
    for i, c in enumerate(art.candidates):
        print(f"{'='*70}")
        print(f"CANDIDATE {i}   score={c['score']}")
        print(f"{'='*70}")
        print(c["text"][:600])
        print()
    print(f"{'#'*70}")
    print(f"FINAL NODE   score={art.final_score}")
    print(f"{'#'*70}")
    print(art.final_node[:800])
    print()
    # The critical check for Option 3: are the candidates actually different?
    if len(art.candidates) >= 2:
        a, b = art.candidates[0]["text"], art.candidates[1]["text"]
        identical = (a.strip() == b.strip())
        print(f"[check] candidate 0 vs 1 identical? {identical}  "
              f"(want False — distinct reasoning approaches for Option 3)")


if __name__ == "__main__":
    try:
        _smoke_test()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
