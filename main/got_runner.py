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
            f"Rate the following math reasoning on a scale of 1-10 for "
            f"correctness and clarity. Respond with ONLY the number.\n\n"
            f"Problem: {s.get('original', '')}\n"
            f"Reasoning:\n{s.get('current', '')}\n\nScore (1-10):"
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

    def parse_validation_answer(self, state: dict, texts: list[str]) -> bool:
        try:
            score = float(re.search(r"-?\d+\.?\d*", texts[0]).group())
            return score >= 5.0
        except Exception:
            return False

    def parse_score_answer(self, states: list[dict], texts: list[str]) -> list[float]:
        scores: list[float] = []
        for t in texts:
            m = re.search(r"-?\d+\.?\d*", t)
            scores.append(float(m.group()) if m else 0.0)
        return scores


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


_LLM_SINGLETON: LocalDeepSeekLLM | None = None


def get_llm() -> LocalDeepSeekLLM:
    """Lazy-load the LLM once and reuse across calls."""
    global _LLM_SINGLETON
    if _LLM_SINGLETON is None:
        _LLM_SINGLETON = LocalDeepSeekLLM()
    return _LLM_SINGLETON


def run_got_on_problem(
    problem: str,
    num_initial: int = 3,
    num_keep: int = 2,
) -> list[ThoughtNode]:
    """
    Run the full Besta GoT pipeline on one problem.
    Returns the finished thought-node(s) from the terminal operation.
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

    # The terminal operation holds the finished thoughts
    terminal = g.operations[-1]
    final_thoughts = terminal.get_thoughts()

    nodes: list[ThoughtNode] = []
    for th in final_thoughts:
        state = th.state if hasattr(th, "state") else {}
        nodes.append(ThoughtNode(
            problem=problem,
            text=state.get("current", ""),
            score=getattr(th, "score", None),
            phase=state.get("phase", "improved"),
        ))
    return nodes


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
    nodes = run_got_on_problem(problem, num_initial=3, num_keep=2)
    print(f"\n[smoke] produced {len(nodes)} thought-node(s).\n")
    for i, node in enumerate(nodes):
        print(f"{'='*70}")
        print(f"ThoughtNode {i}   score={node.score}   phase={node.phase}")
        print(f"{'='*70}")
        print(node.text[:1500])
        print()


if __name__ == "__main__":
    try:
        _smoke_test()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
