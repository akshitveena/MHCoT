"""
solver/state_encoder.py  —  STATE -> TOKENS
===========================================

Turns a symbolic search state (a multiset of Fractions + a target) into a set of
real token vectors (B, T, D) + a padding mask, ready to be lifted into complex
space by the model.

Design choices that matter:
  * A state is a SET, not a sequence -> we add NO positional encoding to the
    number tokens, and we pool order-invariantly. Permutation of the numbers
    must not change the output. (This is the key difference from the GSM8K
    sequence encoder, which is order-dependent.)
  * Numbers can be non-integer Fractions (e.g. 8/3 mid-search), possibly large
    or negative -> each value is featurized with sign / magnitude / integrality
    / ratio-to-target / sinusoidal features, then an MLP maps to D.
  * The target is encoded as its own token so the same model handles Game-24
    (target 24) and Countdown (arbitrary target).

API:
    enc = StateEncoder(d_model=64)
    H, mask = enc(states, targets)     # H: (B, T, D) real, mask: (B, T) bool (True=pad)

Run the self-test:
    python solver/state_encoder.py
"""

from __future__ import annotations

import math
from fractions import Fraction

import torch
import torch.nn as nn

# raw value features, before the MLP
_FREQS = (0.5, 1.0, 2.0, 4.0, 8.0)          # sinusoidal frequencies over value
_RAW_DIM = 5 + 2 * len(_FREQS)              # see _featurize


def _featurize(value: Fraction, target: Fraction) -> list:
    """Hand-built features for a single rational value (exact -> float here is
    fine; mid-search values are bounded)."""
    v = float(value)
    t = float(target) if target != 0 else 1.0
    feats = [
        v / (abs(t) + 1.0),                       # scale-normalized value
        math.copysign(1.0, v) if v != 0 else 0.0, # sign
        math.log1p(abs(v)),                       # compressed magnitude
        1.0 if value.denominator == 1 else 0.0,   # integrality
        float(value - math.floor(v)),             # fractional part
    ]
    for f in _FREQS:                              # periodic features (detect 24, 12, ...)
        feats.append(math.sin(f * v))
        feats.append(math.cos(f * v))
    return feats


class StateEncoder(nn.Module):
    def __init__(self, d_model: int = 64, hidden: int = 64):
        super().__init__()
        self.d_model = d_model
        # +1 input flag marks the target token vs a number token
        self.num_mlp = nn.Sequential(
            nn.Linear(_RAW_DIM + 1, hidden), nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, states, targets):
        """states: list[tuple[Fraction,...]]  targets: list[Fraction|int]
        Returns H (B, T, D) real and mask (B, T) bool with True at padding."""
        B = len(states)
        # T = max numbers across the batch + 1 target token
        max_nums = max(len(s) for s in states)
        T = max_nums + 1
        dev = self.num_mlp[0].weight.device
        raw = torch.zeros(B, T, _RAW_DIM + 1, device=dev)
        mask = torch.ones(B, T, dtype=torch.bool, device=dev)  # True = pad
        for b, (st, tg) in enumerate(zip(states, targets)):
            tg = Fraction(tg)
            # token 0 = target token (flag = 1.0)
            raw[b, 0, :_RAW_DIM] = torch.tensor(_featurize(tg, tg), device=dev)
            raw[b, 0, _RAW_DIM] = 1.0
            mask[b, 0] = False
            for k, x in enumerate(st):
                raw[b, 1 + k, :_RAW_DIM] = torch.tensor(_featurize(Fraction(x), tg),
                                                        device=dev)
                # flag stays 0.0 for number tokens
                mask[b, 1 + k] = False
        H = self.num_mlp(raw)
        H = H.masked_fill(mask.unsqueeze(-1), 0.0)
        return H, mask


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _test():
    from game24 import canon

    print("[test] shapes + padding mask ...")
    enc = StateEncoder(d_model=32)
    states = [canon([4, 6, 8, 2]), canon([12, 2]), canon([24])]
    targets = [24, 24, 24]
    H, mask = enc(states, targets)
    assert H.shape == (3, 5, 32), H.shape       # T = max(4)+1 = 5
    # row 0: target + 4 numbers -> 0 pads ; row 1: target + 2 -> 2 pads ; row 2: 3 pads
    assert mask[0].sum() == 0 and mask[1].sum() == 2 and mask[2].sum() == 3
    print(f"    ok — H {tuple(H.shape)}, pad counts {[int(m.sum()) for m in mask]}")

    print("[test] permutation invariance (a state is a SET) ...")
    a, _ = enc([canon([4, 6, 8, 2])], [24])
    b, _ = enc([canon([2, 8, 6, 4])], [24])
    # canon sorts, so token sets are identical -> encodings identical
    assert torch.allclose(a, b, atol=1e-6), "permuted state must encode identically"
    print("    ok — permuted inputs map to the same tokens")

    print("[test] distinct values -> distinct tokens ...")
    H1, _ = enc([canon([24])], [24])
    H2, _ = enc([canon([23])], [24])
    diff = (H1[:, 1] - H2[:, 1]).abs().mean().item()
    assert diff > 1e-4, "24 and 23 should encode differently"
    print(f"    ok — |enc(24) - enc(23)| = {diff:.4f}")

    print("\n[all state_encoder tests passed]")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _test()
