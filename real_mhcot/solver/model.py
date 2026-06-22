"""
solver/model.py  —  THE THREE HEURISTICS  (where MHCoT is tested)
=================================================================

Each model maps a search state to a scalar value V(state) ~= P(this state can
still reach the target). That value is the search heuristic plugged into
search.py. Three architectures, matched in parameter count, trained identically;
only the inductive bias differs:

  MHCoTSolver(n_chains=2)  the real thing. tokens -> ComplexLift -> N=2
                           phase-separated epsilon-helix chains -> SolitonCell
                           coupling -> INTERFERENCE  Psi = psi0 + psi1 -> value.
                           The two chains are competing hypotheses about which
                           branch reaches the target; the epsilon-helix stops
                           them collapsing to the same guess.

  MHCoTSolver(n_chains=1)  ablation: single complex chain, no interference.

  RealSolver               baseline: matched real Transformer encoder -> value.

The state is a SET (no positional encoding); pooling is masked-mean, so the
output is permutation-invariant in the numbers.

Reuses the verified primitives in main/ (ComplexLinear, ComplexLift, modReLU,
MagnitudeLN, ComplexAttention, SolitonCell, epsilon_helix_loss, phase gap).

Run the self-test:
    python solver/model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent / "main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from complex_ops import (ComplexLift, ComplexAttention, modReLU,        # noqa: E402
                         MagnitudeLN)
from soliton import SolitonCell, per_dim_phase_gap                       # noqa: E402
from state_encoder import StateEncoder                                   # noqa: E402


# ---------------------------------------------------------------------------
# A complex semantic block: attention + modReLU + magnitude LN (residual)
# ---------------------------------------------------------------------------
class ComplexBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = ComplexAttention(d_model, n_heads)
        self.act = modReLU(d_model)
        self.norm = MagnitudeLN(d_model)

    def forward(self, z, mask=None):
        z = z + self.attn(z, mask=mask)
        z = self.act(z)
        return self.norm(z)


# ---------------------------------------------------------------------------
# MHCoT solver: N complex chains, soliton coupling, interference -> value
# ---------------------------------------------------------------------------
class MHCoTSolver(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_chains=2, n_semantic=2,
                 n_coupling=2, eps_min=1.15, alpha=0.01, beta=0.05, dt=0.1):
        super().__init__()
        self.n_chains = n_chains
        self.d_model = d_model
        self.enc = StateEncoder(d_model=d_model)
        self.lift = ComplexLift(d_model)
        # per-chain initial phase rotation (chain 0 fixed at 0 -> identity)
        self.chain_phase = nn.Parameter(torch.randn(n_chains, d_model) * 0.1)
        self.semantic = nn.ModuleList(ComplexBlock(d_model, n_heads)
                                      for _ in range(n_semantic))
        self.coupling = nn.ModuleList(
            SolitonCell(alpha=alpha, beta=beta, eps_min=eps_min, dt=dt,
                        detach_gate=True)   # gradient-safe gate (atan2 stability)
            for _ in range(n_coupling))
        # value head on the interference magnitude |Psi|, masked-mean pooled
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, states, targets):
        H, mask = self.enc(states, targets)              # (B,T,D) real, (B,T) pad
        B, T, D = H.shape
        z = self.lift(H)                                  # (B,T,D) complex
        # build N chains by per-chain phase rotation
        phase = self.chain_phase.view(1, self.n_chains, 1, D)   # (1,N,1,D)
        zc = z.unsqueeze(1)                               # (B,1,T,D)
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        psi = zc * rot                                    # (B,N,T,D)

        attn_mask = mask.unsqueeze(1).expand(B, T, T)     # (B,T,T) True=pad key
        attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_chains, T, T)
        attn_mask = attn_mask.reshape(B * self.n_chains, T, T)

        # semantic processing per chain (fold N into batch)
        for blk in self.semantic:
            flat = psi.reshape(B * self.n_chains, T, D)
            flat = blk(flat, mask=attn_mask)
            psi = flat.reshape(B, self.n_chains, T, D)
        # cross-chain soliton coupling
        for cell in self.coupling:
            psi = cell(psi)

        # INTERFERENCE: superpose the chains
        Psi = psi.sum(dim=1)                              # (B,T,D) complex
        mag = Psi.abs()                                   # (B,T,D) real
        # masked-mean pool over tokens (permutation-invariant set pooling)
        valid = (~mask).float().unsqueeze(-1)             # (B,T,1)
        pooled = (mag * valid).sum(1) / valid.sum(1).clamp_min(1.0)   # (B,D)
        score = self.scorer(pooled).squeeze(-1)           # (B,)

        if self.n_chains >= 2:
            div = per_dim_phase_gap(psi, 0, 1)            # (B,T)
            div = (div * (~mask).float()).sum() / (~mask).float().sum().clamp_min(1.0)
        else:
            div = torch.zeros((), device=H.device)
        return {"score": score, "psi": psi, "Psi": Psi, "chain_divergence": div}


# ---------------------------------------------------------------------------
# Real baseline: matched-param Transformer encoder -> value
# ---------------------------------------------------------------------------
class RealSolver(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=4):
        super().__init__()
        self.enc = StateEncoder(d_model=d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers,
                                        enable_nested_tensor=False)  # MPS-safe
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, states, targets):
        H, mask = self.enc(states, targets)
        h = self.tf(H, src_key_padding_mask=mask)         # (B,T,D)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        score = self.scorer(pooled).squeeze(-1)
        return {"score": score, "chain_divergence": torch.zeros((), device=H.device)}


# ---------------------------------------------------------------------------
# A heuristic adapter: wrap a trained model as search.py's `state -> float`
# ---------------------------------------------------------------------------
@torch.no_grad()
def make_heuristic(model, device="cpu", batch=False):
    """Return a callable heuristic(state, target) -> float for search.py."""
    model.eval()

    def h(state, target):
        out = model([tuple(state)], [target])
        return torch.sigmoid(out["score"]).item()
    return h


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _count(m):
    return sum(p.numel() for p in m.parameters())


def _test():
    from game24 import canon
    dev = "cpu"
    states = [canon([4, 6, 8, 2]), canon([12, 2]), canon([24])]
    targets = [24, 24, 24]

    print("[test] forward shapes + finiteness ...")
    for name, m in (("N=2", MHCoTSolver(n_chains=2)),
                    ("N=1", MHCoTSolver(n_chains=1)),
                    ("real", RealSolver())):
        m = m.to(dev)
        out = m(states, targets)
        assert out["score"].shape == (3,), (name, out["score"].shape)
        assert torch.isfinite(out["score"]).all(), f"{name} non-finite"
        print(f"    {name:<5} score {tuple(out['score'].shape)} "
              f"finite, div={out['chain_divergence'].item():.3f}, "
              f"params={_count(m)/1e3:.1f}K")

    print("[test] gradients flow (incl. through interference) ...")
    m = MHCoTSolver(n_chains=2)
    out = m(states, targets)
    out["score"].sum().backward()
    assert m.lift.W_imag.grad is not None, "no grad to ComplexLift"
    assert m.scorer[0].weight.grad is not None
    print("    ok — grads reach ComplexLift and scorer")

    print("[test] permutation invariance of the value ...")
    m = MHCoTSolver(n_chains=2).eval()
    with torch.no_grad():
        v1 = m([canon([4, 6, 8, 2])], [24])["score"]
        v2 = m([canon([2, 8, 6, 4])], [24])["score"]
    assert torch.allclose(v1, v2, atol=1e-5), "value must be order-invariant"
    print(f"    ok — V(permuted) identical ({v1.item():.4f})")

    print("[test] heuristic adapter plugs into search ...")
    from search import best_first_search
    h = make_heuristic(MHCoTSolver(n_chains=2))
    r = best_first_search([4, 6, 8, 2], 24, h, node_budget=50)
    print(f"    ok — untrained N=2 search ran: {r}")

    print("\n[all model tests passed]")


if __name__ == "__main__":
    _test()
