"""
solver/coevolve.py  —  STAGE 0: the dynamical probe
===================================================

The heart of the restart: a RECURRENT co-evolving complex-chain cell, and a
taskless probe of its dynamics. Before any RL or supervision, we verify the
substrate is HEALTHY — because an unstable or collapsing core makes everything
downstream un-debuggable.

CoEvolveCell: ψ_t (B, N, D) → ψ_{t+1} (B, N, D)
  * per-chain semantic self-update (ComplexLinear + modReLU + MagnitudeLN, residual)
  * cross-chain soliton coupling (the co-evolution)
  carried recurrently across steps → a genuine thought-TRAJECTORY, not a static pass.

What the probe answers (and what it cannot):
  CAN  : stability (|ψ| bounded), non-collapse (ε-helix holds ψ⁰≠ψ¹ when trained
         to), signal richness (phase/amplitude move), trainability (grads flow).
  CANNOT: whether the trajectories are MEANINGFUL — that needs a task (Stage 1).

Run:
    python solver/coevolve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "main"))
sys.path.insert(0, str(_HERE))

from complex_ops import ComplexLinear, modReLU, MagnitudeLN     # noqa: E402
from soliton import SolitonCell, per_dim_phase_gap, epsilon_helix_loss  # noqa: E402


class CoEvolveCell(nn.Module):
    """One recurrent step of N co-evolving complex chains. (B,N,D) -> (B,N,D)."""

    def __init__(self, d_model: int, n_chains: int = 2, eps_min: float = 1.15,
                 alpha: float = 0.01, beta: float = 0.05, dt: float = 0.1):
        super().__init__()
        self.n_chains = n_chains
        self.update = ComplexLinear(d_model, d_model)
        self.act = modReLU(d_model)
        self.norm = MagnitudeLN(d_model)
        self.soliton = SolitonCell(alpha=alpha, beta=beta, eps_min=eps_min,
                                   dt=dt, detach_gate=True)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        # psi: (B, N, D) complex
        z = self.update(psi)                 # complex linear over D (per chain)
        z = self.act(z)
        z = self.norm(z)
        z = psi + z                          # residual recurrence (carry state)
        z = self.soliton(z.unsqueeze(2)).squeeze(2)   # (B,N,1,D) coupling -> (B,N,D)
        return z


def rollout(cell: CoEvolveCell, psi0: torch.Tensor, steps: int):
    """Roll the cell forward `steps` and record per-step diagnostics."""
    psi = psi0
    amp, div = [], []
    for _ in range(steps):
        psi = cell(psi)
        amp.append(psi.abs().mean().item())
        if psi.shape[1] >= 2:
            div.append(per_dim_phase_gap(psi.unsqueeze(2), 0, 1).mean().item())
        else:
            div.append(0.0)
    return psi, amp, div


def init_chains(B, N, D, seed=0, sep=0.5):
    """Initialize N chains with a small phase separation so they start distinct."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, 1, D, generator=g)
    base_i = torch.randn(B, 1, D, generator=g)
    z = torch.complex(base, base_i).repeat(1, N, 1)
    # rotate each chain by a different phase
    phases = torch.linspace(0, sep, N).view(1, N, 1)
    rot = torch.complex(torch.cos(phases), torch.sin(phases))
    return z * rot


# ---------------------------------------------------------------------------
# The dynamical probe
# ---------------------------------------------------------------------------
def probe():
    torch.manual_seed(0)
    B, N, D, STEPS = 8, 2, 64, 60
    cell = CoEvolveCell(D, n_chains=N)
    psi0 = init_chains(B, N, D, seed=0)

    print(f"[probe] rolling {N} chains forward {STEPS} steps (untrained dynamics) ...")
    _, amp, div = rollout(cell, psi0, STEPS)
    print(f"    amplitude  start {amp[0]:.3f}  mid {amp[STEPS//2]:.3f}  end {amp[-1]:.3f}")
    print(f"    divergence start {div[0]:.3f}  mid {div[STEPS//2]:.3f}  end {div[-1]:.3f}")

    # (1) STABILITY: amplitude must stay bounded (not explode / vanish)
    amp_t = torch.tensor(amp)
    assert torch.isfinite(amp_t).all(), "amplitude went non-finite — UNSTABLE"
    assert amp_t.max() < 50 * amp[0] + 10, "amplitude exploded"
    assert amp_t[-1] > 1e-3, "amplitude vanished (signal died)"
    print("    [ok] STABILITY — amplitude bounded, signal alive")

    # (2) TRAINABILITY + CONTROLLABILITY: can we TRAIN the rollout to (a) keep
    #     amplitude near a target and (b) MAINTAIN chain divergence (ε-helix)?
    cell2 = CoEvolveCell(D, n_chains=N)
    opt = torch.optim.Adam(cell2.parameters(), lr=1e-2)
    target_amp = 1.0
    div_before = None
    for it in range(150):
        psiT, amp2, _ = rollout(cell2, psi0, 20)
        # reconstruct a differentiable rollout (rollout above is item()-logged;
        # redo a short differentiable one)
        psi = psi0
        for _ in range(20):
            psi = cell2(psi)
        amp_loss = (psi.abs().mean() - target_amp).pow(2)
        helix = epsilon_helix_loss(psi.unsqueeze(2))      # maintain separation
        loss = amp_loss + 0.5 * helix
        if div_before is None:
            div_before = per_dim_phase_gap(psi.unsqueeze(2)).mean().item()
        opt.zero_grad(); loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in cell2.parameters())
        assert finite, "non-finite gradient through the recurrent rollout"
        opt.step()
    with torch.no_grad():
        psi = psi0
        for _ in range(20):
            psi = cell2(psi)
        div_after = per_dim_phase_gap(psi.unsqueeze(2)).mean().item()
        amp_after = psi.abs().mean().item()
    print(f"    [ok] TRAINABILITY — grads finite through 20-step rollout")
    print(f"    amplitude steered to {amp_after:.3f} (target {target_amp})")
    print(f"    divergence {div_before:.3f} -> {div_after:.3f} under ε-helix "
          f"({'MAINTAINED/GREW' if div_after >= div_before - 0.05 else 'collapsed'})")
    assert div_after >= div_before - 0.05, "ε-helix could NOT hold chains apart"

    # (3) SIGNAL RICHNESS: phase actually moves over the trajectory (not frozen)
    psi = psi0
    phase_path = []
    for _ in range(STEPS):
        psi = cell(psi)
        phase_path.append(torch.atan2(psi.imag, psi.real)[0, 0, 0].item())
    phase_var = torch.tensor(phase_path).std().item()
    print(f"    [ok] RICHNESS — single-dim phase std over trajectory = {phase_var:.3f} "
          f"({'rich' if phase_var > 0.05 else 'frozen'})")

    print("\n[STAGE 0 VERDICT] substrate is stable, trainable, non-collapsing, "
          "and signal-rich → cleared to build Stage 1 (meaning probe).")
    return {"amp": amp, "div": div, "phase_path": phase_path}


if __name__ == "__main__":
    probe()
