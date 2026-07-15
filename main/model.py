"""
main/model.py
=============

The MHCoTEncoder — the complex-valued representation model (Option 1).
Assembles the primitives from complex_ops.py + soliton.py into the two-stage
architecture:

    h (B,T,D real, from frozen backbone)
      → ComplexLift            real → complex
      → ComplexPositionalEnc   install the time axis
      → phase-init N=2 chains  → (B, N, T, D)
      → Stage 1 (semantic):  [ComplexAttention + ComplexMLP] × n_semantic
                             per-chain, PHASE-EQUIVARIANT (chains stay related)
      → Stage 2 (coupling):  [ComplexAttention + SolitonCell] × n_coupling
                             the SolitonCell BREAKS symmetry → chains diverge
      → interference  I(t) = |Σ_chains ψ|² / D     (per-token uncertainty)
      → pooled |Ψ| → ScoringHead → correctness score

Outputs a dict with score, I(t), the chains ψ, and the superposition Ψ, plus
a `chain_divergence` monitor (mean per-dim phase gap) so we can SEE whether
the soliton is actually differentiating the chains during training.

Trains the ~35M-param head; the backbone is frozen and out of the loop (its
hidden states are precomputed and fed in as `h`).

Self-test (random input):
    python main/model.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from Projects.MHCoT.main.complex_ops import (
    ComplexLift,
    ComplexPositionalEncoding,
    ComplexAttention,
    ComplexLinear,
    modReLU,
    MagnitudeLN,
)
from Projects.MHCoT.main.soliton import SolitonCell, per_dim_phase_gap, epsilon_helix_loss


# ---------------------------------------------------------------------------
# Complex MLP block (ComplexLinear → modReLU → ComplexLinear)
# ---------------------------------------------------------------------------
class ComplexMLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = ComplexLinear(d_model, d_hidden)
        self.act = modReLU(d_hidden)
        self.fc2 = ComplexLinear(d_hidden, d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(z)))


# ---------------------------------------------------------------------------
# Stage 1 — semantic layer (per-chain, phase-equivariant)
# ---------------------------------------------------------------------------
class SemanticLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_hidden: int):
        super().__init__()
        self.ln1 = MagnitudeLN(d_model)
        self.attn = ComplexAttention(d_model, n_heads)
        self.ln2 = MagnitudeLN(d_model)
        self.mlp = ComplexMLP(d_model, d_hidden)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # z: (B*N, T, D) complex — chains processed independently here
        z = z + self.attn(self.ln1(z), mask=mask)
        z = z + self.mlp(self.ln2(z))
        return z


# ---------------------------------------------------------------------------
# Stage 2 — coupling layer (attention per chain + soliton across chains)
# ---------------------------------------------------------------------------
class CouplingLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, eps_min: float,
                 alpha: float, beta: float, dt: float):
        super().__init__()
        self.ln1 = MagnitudeLN(d_model)
        self.attn = ComplexAttention(d_model, n_heads)
        self.soliton = SolitonCell(alpha=alpha, beta=beta, eps_min=eps_min, dt=dt)

    def forward(self, psi: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # psi: (B, N, T, D) complex
        B, N, T, D = psi.shape
        z = psi.reshape(B * N, T, D)
        z = z + self.attn(self.ln1(z), mask=mask)   # per-chain attention
        psi = z.reshape(B, N, T, D)
        psi = self.soliton(psi)                     # cross-chain coupling (symmetry break)
        return psi


# ---------------------------------------------------------------------------
# The MHCoTEncoder
# ---------------------------------------------------------------------------
class MHCoTEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,                 # backbone hidden size (e.g. 1536)
        d_model: int = 128,        # small internal working dim (fights overfit)
        n_heads: int = 8,
        n_chains: int = 2,
        n_semantic: int = 2,
        n_coupling: int = 2,
        d_hidden: int | None = None,
        eps_min: float = 1.15,
        alpha: float = 0.01,
        beta: float = 0.05,
        dt: float = 0.1,
        answer_tail: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_chains = n_chains
        self.eps_min = eps_min
        self.answer_tail = answer_tail
        d_hidden = d_hidden or (2 * d_model)

        # real down-projection: backbone dim → small working dim
        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.lift = ComplexLift(d_model)
        self.posenc = ComplexPositionalEncoding(d_model)
        self.semantic = nn.ModuleList(
            [SemanticLayer(d_model, n_heads, d_hidden) for _ in range(n_semantic)]
        )
        self.coupling = nn.ModuleList(
            [CouplingLayer(d_model, n_heads, eps_min, alpha, beta, dt)
             for _ in range(n_coupling)]
        )
        # Scoring head reads the pooled superposition magnitude (real).
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def phase_init(self, z: torch.Tensor) -> torch.Tensor:
        """(B,T,D) complex → (B,N,T,D): chain k rotated by k·ε_min/2."""
        chains = []
        for k in range(self.n_chains):
            phi = k * self.eps_min / 2.0
            rot = torch.complex(
                torch.tensor(math.cos(phi), device=z.device),
                torch.tensor(math.sin(phi), device=z.device),
            )
            chains.append(z * rot)
        return torch.stack(chains, dim=1)        # (B, N, T, D)

    def _answer_region_mask(self, B: int, T: int,
                            lengths: torch.Tensor | None,
                            device) -> torch.Tensor:
        """
        (B, T) float indicator of the ANSWER REGION = last `answer_tail` REAL
        tokens per sample (the representation that carries the signal:
        gates 0.589→0.689 going mean→last-16). With `lengths`, padding is
        never included; without, uses the last-k of T.
        """
        k = min(self.answer_tail, T)
        idx = torch.arange(T, device=device).unsqueeze(0)            # (1,T)
        if lengths is None:
            region = (idx >= (T - k)).float().expand(B, T)
        else:
            lo = (lengths - k).clamp(min=0).unsqueeze(1)             # (B,1)
            hi = lengths.unsqueeze(1)                                # (B,1)
            region = ((idx >= lo) & (idx < hi)).float()             # (B,T)
        return region

    def forward(self, h: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None,
                lengths: torch.Tensor | None = None) -> dict:
        """
        h: (B, T, D) real hidden states from the frozen backbone.
        key_padding_mask: (B, T) bool, True = padding position (optional).
        lengths: (B,) real sequence lengths (optional; for answer-region pool).
        Returns dict(score, I_t, psi, Psi, chain_divergence).
        """
        h = self.in_proj(h)                       # (B,T,d_in) → (B,T,d_model) real
        z = self.lift(h)                          # (B,T,d_model) complex
        z = self.posenc(z)
        psi = self.phase_init(z)                  # (B,N,T,D)

        # build (B,T,T) attention mask from key padding, repeated per chain
        B, N, T, D = psi.shape
        attn_mask = None
        if key_padding_mask is not None:
            # (B,T,T): mask out padded KEY positions
            km = key_padding_mask.unsqueeze(1).expand(B, T, T)       # (B,T,T)
            attn_mask = km.repeat_interleave(N, dim=0)               # (B*N,T,T)

        # Stage 1 — semantic, per chain (shared weights)
        z = psi.reshape(B * N, T, D)
        for layer in self.semantic:
            z = layer(z, mask=attn_mask)
        psi = z.reshape(B, N, T, D)

        # Stage 2 — coupling (soliton breaks symmetry)
        for layer in self.coupling:
            psi = layer(psi, mask=attn_mask)

        # Interference — per-token uncertainty signal
        Psi = psi.sum(dim=1)                       # (B,T,D) superposition
        I_t = Psi.abs().pow(2).sum(dim=-1) / D     # (B,T)

        # Answer-region mask (last-k REAL tokens) — used for BOTH the score
        # pooling and the answer-region interference (so padding is excluded).
        region = self._answer_region_mask(B, T, lengths, Psi.device)   # (B,T)
        denom = region.sum(dim=1, keepdim=True).clamp(min=1.0)          # (B,1)
        pooled = (Psi.abs() * region.unsqueeze(-1)).sum(dim=1) / denom  # (B,D)
        I_answer = (I_t * region).sum(dim=1) / denom.squeeze(-1)        # (B,)
        score = self.scorer(pooled).squeeze(-1)    # (B,)

        # Monitor: are the chains actually diverging? (mean per-dim phase gap)
        with torch.no_grad():
            divergence = per_dim_phase_gap(psi).mean().item() if N >= 2 else 0.0

        return {
            "score": score,         # (B,) correctness logit
            "I_t": I_t,             # (B,T) per-token interference
            "I_answer": I_answer,   # (B,) answer-region mean interference
            "psi": psi,             # (B,N,T,D) chains
            "Psi": Psi,             # (B,T,D) superposition
            "chain_divergence": divergence,
        }

    def helix_loss(self, psi: torch.Tensor) -> torch.Tensor:
        return epsilon_helix_loss(psi, eps_min=self.eps_min)


# ---------------------------------------------------------------------------
# Self-test on random input
# ---------------------------------------------------------------------------
def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    dev = _device()
    print(f"[device] {dev}\n")

    D_IN, D, B, T = 1536, 128, 4, 20    # backbone dim 1536 → working dim 128
    model = MHCoTEncoder(d_in=D_IN, d_model=D, n_heads=8).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MHCoTEncoder, d_in={D_IN}, d_model={D}, "
          f"trainable params = {n_params/1e6:.2f}M")

    h = torch.randn(B, T, D_IN, device=dev, requires_grad=True)
    out = model(h)

    print(f"[test] forward ...")
    assert out["score"].shape == (B,), out["score"].shape
    assert out["I_t"].shape == (B, T), out["I_t"].shape
    assert out["psi"].shape == (B, 2, T, D), out["psi"].shape
    assert torch.isfinite(out["score"]).all()
    assert torch.isfinite(out["I_t"]).all()
    print(f"    ok — score {tuple(out['score'].shape)}, I_t {tuple(out['I_t'].shape)}, "
          f"psi {tuple(out['psi'].shape)}")

    print(f"[test] gradient flow (task + helix) ...")
    task = out["score"].sum()
    helix = model.helix_loss(out["psi"])
    (task + helix).backward()
    g = model.lift.W_imag.grad
    assert g is not None and torch.isfinite(g).all()
    print(f"    ok — grads reach ComplexLift.W_imag, helix_loss = {helix.item():.4f}")

    print(f"[test] symmetry-breaking (chains diverge through the soliton) ...")
    init_gap = model.eps_min / 2.0
    final_gap = out["chain_divergence"]
    print(f"    phase-init gap ≈ {init_gap:.3f}  →  final chain divergence = {final_gap:.3f}")
    print(f"    (final ≠ init means the soliton differentiated the chains)")

    iv = out["I_t"].std().item()
    print(f"    interference I(t) std across tokens = {iv:.4f}  (want > 0)")
    assert iv > 0

    print("\n[all model tests passed]")


if __name__ == "__main__":
    main()
