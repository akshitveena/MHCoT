"""
main/encoder.py
===============

The frozen-backbone precompute. Runs DeepSeek-R1-Distill-Qwen-1.5B ONCE per
reasoning text and caches the hidden states to disk, so every downstream
experiment and training run reuses the cache — the 1.5B model never sits in
a training loop. This is what makes M3 training fast.

What it caches (per problem idx):
  * candidates : pooled hidden vector per candidate, shape (n_cand, D)
  * final      : pooled hidden vector of the final_node, shape (D,)

"Pooled" = mean over tokens of the last hidden layer → one ℝᴰ "reasoning
fingerprint" per text. Pooled vectors serve Experiment −1 (distances,
clustering) and Experiment 0 (pooled interference). Per-token sequences for
Option 1 are a separate, larger precompute added later (final_nodes only).

Run directly to precompute over the current cache:
    python main/encoder.py
Outputs: data/hidden_cache/gsm8k_test_pooled.pt
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_dataset, ProblemRecord  # sibling import (run from main/ or with path)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HIDDEN_DIR = _PROJECT_ROOT / "data" / "hidden_cache"
_HIDDEN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
MAX_LEN = 2048   # candidates are < 1024 tokens; 2048 is a safe ceiling


# ---------------------------------------------------------------------------
# Lazy frozen backbone
# ---------------------------------------------------------------------------
_model = None
_tok = None


def get_backbone():
    global _model, _tok
    if _model is None:
        print(f"[encoder] loading {MODEL_NAME} on {DEVICE} (frozen) ...")
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=DTYPE, output_hidden_states=True
        ).to(DEVICE)
        _model.eval()
        for p in _model.parameters():       # belt-and-suspenders: no grads
            p.requires_grad_(False)
        print(f"[encoder] ready. hidden_size = {_model.config.hidden_size}")
    return _model, _tok


def hidden_size() -> int:
    return get_backbone()[0].config.hidden_size


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_text(text: str, pool: str | None = "mean") -> torch.Tensor:
    """
    text -> last-layer hidden states.
      pool="mean": returns (D,) pooled vector (mean over tokens)
      pool=None:   returns (T, D) full per-token sequence
    Output is float32 on CPU (ready to cache / feed the complex head).
    """
    model, tok = get_backbone()
    if not text:
        D = model.config.hidden_size
        return torch.zeros(D) if pool == "mean" else torch.zeros(1, D)

    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=MAX_LEN).to(DEVICE)
    out = model(**ids)
    H = out.hidden_states[-1].squeeze(0)          # (T, D)
    H = H.float().cpu()
    if pool == "mean":
        return H.mean(dim=0)                       # (D,)
    return H                                       # (T, D)


# ---------------------------------------------------------------------------
# Precompute pooled vectors over a dataset
# ---------------------------------------------------------------------------
def precompute_pooled(
    records: list[ProblemRecord],
    out_path: str | Path | None = None,
) -> dict:
    """
    For each record, encode every candidate + the final_node to a pooled
    vector. Returns and saves a dict:
        { idx: {"candidates": Tensor(n_cand, D), "final": Tensor(D),
                "gold": str, "cand_correct": list[bool], "final_correct": bool} }
    Resumable: if out_path exists, already-done idx are skipped.
    """
    if out_path is None:
        out_path = _HIDDEN_DIR / "gsm8k_test_pooled.pt"
    out_path = Path(out_path)

    cache: dict = {}
    if out_path.exists():
        cache = torch.load(out_path)
        print(f"[encoder] resuming — {len(cache)} idx already cached")

    todo = [r for r in records if r.idx not in cache and not r.error and r.candidates]
    print(f"[encoder] to encode: {len(todo)} problems "
          f"({sum(r.n_candidates + 1 for r in todo)} texts)")

    t0 = time.time()
    for n, r in enumerate(todo):
        cand_vecs = torch.stack([encode_text(c.text, pool="mean")
                                 for c in r.candidates])           # (n_cand, D)
        final_vec = encode_text(r.final_text, pool="mean")          # (D,)
        cache[r.idx] = {
            "candidates": cand_vecs,
            "final": final_vec,
            "gold": r.gold,
            "cand_correct": [c.correct for c in r.candidates],
            "final_correct": r.final_correct,
        }
        if (n + 1) % 10 == 0 or n == 0:
            dt = time.time() - t0
            rate = (n + 1) / dt
            eta = (len(todo) - n - 1) / max(rate, 1e-6)
            print(f"  [{n+1}/{len(todo)}] {rate:.2f} prob/s  ETA {eta/60:.1f} min")
            torch.save(cache, out_path)   # periodic checkpoint

    torch.save(cache, out_path)
    print(f"[encoder] done. {len(cache)} problems cached → {out_path}")
    print(f"  elapsed {(time.time()-t0)/60:.1f} min")
    return cache


def load_pooled(path: str | Path | None = None) -> dict:
    if path is None:
        path = _HIDDEN_DIR / "gsm8k_test_pooled.pt"
    return torch.load(Path(path))


# ---------------------------------------------------------------------------
# Run: precompute over the current cache
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    records = load_dataset()
    print(f"[encoder] loaded {len(records)} records from GoT cache")
    cache = precompute_pooled(records)
    # quick sanity
    any_idx = next(iter(cache))
    e = cache[any_idx]
    print(f"\n[sanity] idx={any_idx}: candidates {tuple(e['candidates'].shape)}, "
          f"final {tuple(e['final'].shape)}, gold={e['gold']}, "
          f"cand_correct={e['cand_correct']}")
