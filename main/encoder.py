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


# Default file for the multi-pool cache (mean / lastk / last)
MULTIPOOL_PATH = _HIDDEN_DIR / "gsm8k_test_pools.pt"
LASTK = 16   # answer-region window: mean of the last K tokens


@torch.no_grad()
def encode_text_multi(text: str, lastk: int = LASTK) -> dict:
    """
    ONE forward pass → three pooled representations:
      "mean"  : mean over all tokens          (dilutes the answer — the old way)
      "lastk" : mean over the last `lastk` tokens (the ANSWER REGION — hypothesis)
      "last"  : the final token's hidden state (the conclusion)
    Each is (D,) float on CPU.
    """
    model, tok = get_backbone()
    if not text:
        D = model.config.hidden_size
        z = torch.zeros(D)
        return {"mean": z, "lastk": z, "last": z}
    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=MAX_LEN).to(DEVICE)
    out = model(**ids)
    H = out.hidden_states[-1].squeeze(0).float().cpu()   # (T, D)
    T = H.shape[0]
    k = min(lastk, T)
    return {
        "mean": H.mean(dim=0),
        "lastk": H[-k:].mean(dim=0),
        "last": H[-1],
    }


# ---------------------------------------------------------------------------
# Precompute MULTI-pool vectors over a dataset
# ---------------------------------------------------------------------------
def precompute_pooled(
    records: list[ProblemRecord],
    out_path: str | Path | None = None,
) -> dict:
    """
    For each record, encode every candidate + the final_node to THREE pooled
    vectors (mean / lastk / last) in one forward pass. Saves a dict:
        { idx: {
            "candidates": {"mean": (n,D), "lastk": (n,D), "last": (n,D)},
            "final":      {"mean": (D,),  "lastk": (D,),  "last": (D,)},
            "gold": str, "cand_correct": list[bool], "final_correct": bool
        } }
    Resumable: already-done idx are skipped.
    """
    if out_path is None:
        out_path = MULTIPOOL_PATH
    out_path = Path(out_path)

    cache: dict = {}
    if out_path.exists():
        cache = torch.load(out_path)
        print(f"[encoder] resuming — {len(cache)} idx already cached")

    todo = [r for r in records if r.idx not in cache and not r.error and r.candidates]
    print(f"[encoder] to encode: {len(todo)} problems "
          f"({sum(r.n_candidates + 1 for r in todo)} texts), pools=mean/lastk/last")

    def stack_pools(dicts: list[dict]) -> dict:
        return {p: torch.stack([d[p] for d in dicts]) for p in ("mean", "lastk", "last")}

    t0 = time.time()
    for n, r in enumerate(todo):
        cand_pools = [encode_text_multi(c.text) for c in r.candidates]
        final_pools = encode_text_multi(r.final_text)
        cache[r.idx] = {
            "candidates": stack_pools(cand_pools),       # each (n_cand, D)
            "final": final_pools,                        # each (D,)
            "gold": r.gold,
            "cand_correct": [c.correct for c in r.candidates],
            "final_correct": r.final_correct,
        }
        if (n + 1) % 10 == 0 or n == 0:
            dt = time.time() - t0
            rate = (n + 1) / dt
            eta = (len(todo) - n - 1) / max(rate, 1e-6)
            print(f"  [{n+1}/{len(todo)}] {rate:.2f} prob/s  ETA {eta/60:.1f} min")
            torch.save(cache, out_path)
    torch.save(cache, out_path)
    print(f"[encoder] done. {len(cache)} problems cached → {out_path}")
    print(f"  elapsed {(time.time()-t0)/60:.1f} min")
    return cache


SEQ_PATH = _HIDDEN_DIR / "gsm8k_test_seq.pt"


def precompute_sequences(
    records: list[ProblemRecord],
    out_path: str | Path | None = None,
) -> dict:
    """
    Cache the FULL per-token hidden-state sequence of each final_node — what
    Option 1 / the MHCoTEncoder consumes (per-token I(t) needs the sequence,
    not a pooled vector). Stored fp16 to halve disk.
        { idx: {"h": (T, D) fp16, "length": int, "label": int (final_correct),
                "gold": str} }
    Resumable. ~0.5-1.5 GB for GSM8K test (final_nodes only).
    """
    if out_path is None:
        out_path = SEQ_PATH
    out_path = Path(out_path)

    cache: dict = {}
    if out_path.exists():
        cache = torch.load(out_path)
        print(f"[encoder] resuming sequences — {len(cache)} idx cached")

    todo = [r for r in records if r.idx not in cache and not r.error and r.final_text]
    print(f"[encoder] to encode (final_node sequences): {len(todo)}")

    t0 = time.time()
    for n, r in enumerate(todo):
        H = encode_text(r.final_text, pool=None)         # (T, D) float
        cache[r.idx] = {
            "h": H.half(),                               # fp16 to save disk
            "length": int(H.shape[0]),
            "label": int(r.final_correct),
            "gold": r.gold,
        }
        if (n + 1) % 25 == 0 or n == 0:
            dt = time.time() - t0
            rate = (n + 1) / dt
            eta = (len(todo) - n - 1) / max(rate, 1e-6)
            print(f"  [{n+1}/{len(todo)}] {rate:.2f} prob/s  ETA {eta/60:.1f} min")
            torch.save(cache, out_path)
    torch.save(cache, out_path)
    print(f"[encoder] done. {len(cache)} sequences cached → {out_path}")
    print(f"  elapsed {(time.time()-t0)/60:.1f} min")
    return cache


def load_sequences(path: str | Path | None = None) -> dict:
    return torch.load(Path(path) if path else SEQ_PATH)


def load_pooled(path: str | Path | None = None, pool: str = "lastk") -> dict:
    """
    Load the multi-pool cache and SELECT one pooling, returning the flat
    structure the experiments expect:
        { idx: {"candidates": (n,D), "final": (D,), "gold",
                "cand_correct", "final_correct"} }
    `pool` ∈ {"mean", "lastk", "last"}.
    """
    if path is None:
        path = MULTIPOOL_PATH
    raw = torch.load(Path(path))
    assert pool in ("mean", "lastk", "last"), f"bad pool {pool}"
    out = {}
    for idx, e in raw.items():
        # support both the new nested format and (defensively) flat legacy
        cand = e["candidates"][pool] if isinstance(e["candidates"], dict) else e["candidates"]
        fin = e["final"][pool] if isinstance(e["final"], dict) else e["final"]
        out[idx] = {
            "candidates": cand, "final": fin, "gold": e["gold"],
            "cand_correct": e["cand_correct"], "final_correct": e["final_correct"],
        }
    return out


# ---------------------------------------------------------------------------
# Run: precompute over the current cache
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", action="store_true",
                    help="precompute per-token SEQUENCES of final_nodes (for training); "
                         "default is the multi-pool vectors (for the gate experiments)")
    args = ap.parse_args()

    records = load_dataset()
    print(f"[encoder] loaded {len(records)} records from GoT cache")

    if args.seq:
        cache = precompute_sequences(records)
        any_idx = next(iter(cache))
        e = cache[any_idx]
        print(f"\n[sanity] idx={any_idx}: h {tuple(e['h'].shape)} ({e['h'].dtype}), "
              f"length={e['length']}, label={e['label']}, gold={e['gold']}")
    else:
        cache = precompute_pooled(records)
        any_idx = next(iter(cache))
        e = cache[any_idx]
        print(f"\n[sanity] idx={any_idx}: "
              f"candidates.lastk {tuple(e['candidates']['lastk'].shape)}, "
              f"final.lastk {tuple(e['final']['lastk'].shape)}, gold={e['gold']}, "
              f"cand_correct={e['cand_correct']}")
