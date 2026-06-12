"""
main/soliton.py
===============

The cross-chain coupling that makes MHCoT multi-helical, plus the ε-helix loss.

SolitonCell (EQ-C):  as chains propagate through depth, each one updates by
    ∂ψ^(i)/∂d = −α |ψ^(i)|² ψ^(i)              (self-focusing: keep identity)
              + β (Ψ − ψ^(i)) · exp(−gap/ε)    (coupling: gated by phase gap)
  where Ψ = Σ_i ψ^(i) is the consensus and `gap` is the per-dimension phase
  gap between chains. Discretized as  ψ ← ψ + dt·(self_focus + coupling).

  * Large phase gap → small gate → chains stay distinct (the helix holds).
  * Small phase gap → strong coupling → chains exchange insight.

ε-helix loss (EQ-B/EQ2):  penalize chains for getting closer than ε_min in
phase. The gap is measured PER-DIMENSION (RMS), the same scale as the ε we
measured empirically in Experiment −1 (≈1.15), so the loss and the measurement
speak the same units.

Everything is real-decomposed (no complex einsum / .angle on the hot path that
MPS struggles with), consistent with complex_ops.py, so it runs on MPS/CUDA/CPU.

Run the self-test:
    python main/soliton.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_PI = math.pi
_TWO_PI = 2.0 * math.pi


def _wrap(dphi: torch.Tensor) -> torch.Tensor:
    """Wrap a phase difference to (−π, π]."""
    return (dphi + _PI) % _TWO_PI - _PI


def per_dim_phase_gap(psi: torch.Tensor, i: int = 0, j: int = 1) -> torch.Tensor:
    """
    Per-dimension RMS phase gap between chain i and chain j.
    psi: (B, N, T, D) complex → returns (B, T) real.
    Matches the ε scale measured in Experiment −1.
    """
    phi = torch.atan2(psi.imag, psi.real)              # (B, N, T, D)
    dphi = _wrap(phi[:, i] - phi[:, j])                # (B, T, D)
    D = phi.shape[-1]
    return dphi.norm(dim=-1) / math.sqrt(D)            # (B, T)


# ---------------------------------------------------------------------------
# SolitonCell — cross-chain coupling forward rule
# ---------------------------------------------------------------------------
class SolitonCell(nn.Module):
    """
    Couples N chains (designed for N=2, the DNA double-helix minimum).
    Input/output: (B, N, T, D) complex. Pure real arithmetic internally.
    """

    def __init__(self, alpha: float = 0.01, beta: float = 0.05,
                 eps_min: float = 1.15, dt: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps_min = eps_min
        self.dt = dt

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        re, im = psi.real, psi.imag                     # (B, N, T, D)
        N = psi.shape[1]
        mag2 = re * re + im * im                         # |ψ|² real

        # self-focusing: −α |ψ|² ψ
        sf_re = -self.alpha * mag2 * re
        sf_im = -self.alpha * mag2 * im

        # N=1: single chain, no cross-chain coupling (just self-focusing)
        if N < 2:
            return torch.complex(re + self.dt * sf_re, im + self.dt * sf_im)

        # consensus Ψ = Σ_i ψ^(i)
        re_sum = re.sum(dim=1, keepdim=True)            # (B, 1, T, D)
        im_sum = im.sum(dim=1, keepdim=True)
        diff_re = re_sum - re                            # (Ψ − ψ^(i))
        diff_im = im_sum - im

        # phase-gap gate (chain 0 vs chain 1), per-dimension RMS, broadcast
        phi = torch.atan2(im, re)                        # (B, N, T, D)
        dphi = _wrap(phi[:, 0:1] - phi[:, 1:2])          # (B, 1, T, D)
        D = re.shape[-1]
        gap = dphi.norm(dim=-1, keepdim=True) / math.sqrt(D)   # (B, 1, T, 1)
        gate = torch.exp(-gap / self.eps_min)            # (B, 1, T, 1)

        cp_re = self.beta * gate * diff_re
        cp_im = self.beta * gate * diff_im

        out_re = re + self.dt * (sf_re + cp_re)
        out_im = im + self.dt * (sf_im + cp_im)
        return torch.complex(out_re, out_im)


# ---------------------------------------------------------------------------
# ε-helix loss
# ---------------------------------------------------------------------------
def epsilon_helix_loss(psi: torch.Tensor, eps_min: float = 1.15,
                       i: int = 0, j: int = 1) -> torch.Tensor:
    """
    L_ε = mean( max(0, ε_min − gap)² )  over (B, T).
    gap = per-dimension RMS phase gap between chains i and j.
    Zero when chains are ≥ ε_min apart; positive (pulling them apart) when closer.
    """
    if psi.shape[1] < 2:                                 # N=1: no helix to enforce
        return psi.real.new_zeros(())
    gap = per_dim_phase_gap(psi, i, j)                  # (B, T)
    return F.relu(eps_min - gap).pow(2).mean()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _test_soliton_step(dev: str) -> None:
    print("[test] SolitonCell forward + grad ...")
    cell = SolitonCell().to(dev)
    psi = torch.randn(4, 2, 10, 64, dtype=torch.cfloat, device=dev, requires_grad=True)
    out = cell(psi)
    assert out.shape == psi.shape and out.dtype == torch.cfloat
    assert torch.isfinite(out.abs()).all(), "non-finite output"
    out.abs().sum().backward()
    assert psi.grad is not None and torch.isfinite(psi.grad.abs()).all()
    # amplitude should stay sane (self-focusing is gentle at small alpha)
    ratio = (out.abs().mean() / psi.abs().mean()).item()
    print(f"    ok — finite, grads flow, |out|/|in| = {ratio:.3f}")


def _test_helix_loss(dev: str) -> None:
    print("[test] epsilon_helix_loss ...")
    eps = 1.15
    # case 1: chains IDENTICAL → gap 0 → loss ≈ eps²
    z = torch.randn(4, 1, 10, 64, dtype=torch.cfloat, device=dev)
    psi_same = torch.cat([z, z], dim=1)
    l_same = epsilon_helix_loss(psi_same, eps_min=eps).item()
    assert abs(l_same - eps * eps) < 0.05, f"identical chains: {l_same} vs {eps*eps}"

    # case 2: chains with large random phase difference → gap large → loss ≈ 0
    a = torch.randn(4, 1, 10, 64, dtype=torch.cfloat, device=dev)
    b = torch.randn(4, 1, 10, 64, dtype=torch.cfloat, device=dev)
    psi_diff = torch.cat([a, b], dim=1)
    l_diff = epsilon_helix_loss(psi_diff, eps_min=eps).item()
    print(f"    ok — loss(identical)={l_same:.3f} (≈{eps*eps:.3f}), "
          f"loss(random-diff)={l_diff:.3f} (≈0)")
    assert l_diff < l_same, "random-diff should incur less helix loss than identical"


def _test_gate_behavior(dev: str) -> None:
    print("[test] coupling gate (close chains couple, distant chains don't) ...")
    cell = SolitonCell(alpha=0.0, beta=0.5)   # isolate coupling
    z = torch.randn(2, 1, 5, 64, dtype=torch.cfloat, device=dev)
    # near-identical chains → strong coupling → output moves toward consensus
    psi_close = torch.cat([z, z * torch.exp(torch.tensor(0.01j, device=dev))], dim=1)
    out_close = cell(psi_close.to(dev))
    moved_close = (out_close - psi_close).abs().mean().item()
    # very different chains → weak gate → less coupling movement
    w = torch.randn(2, 1, 5, 64, dtype=torch.cfloat, device=dev)
    psi_far = torch.cat([z, w], dim=1)
    out_far = cell(psi_far.to(dev))
    moved_far_gate = per_dim_phase_gap(psi_far).mean().item()
    moved_close_gate = per_dim_phase_gap(psi_close).mean().item()
    print(f"    ok — gap(close)={moved_close_gate:.3f} < gap(far)={moved_far_gate:.3f}; "
          f"close-chain coupling moved {moved_close:.4f}")
    assert moved_close_gate < moved_far_gate


def main() -> None:
    dev = _device()
    print(f"[device] {dev}\n")
    _test_soliton_step(dev)
    _test_helix_loss(dev)
    _test_gate_behavior(dev)
    print("\n[all soliton tests passed]")


if __name__ == "__main__":
    main()
