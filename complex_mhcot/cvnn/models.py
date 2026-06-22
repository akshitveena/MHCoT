"""
cvnn/models.py  —  complex transformer + matched real baseline (L1)
===================================================================

A clean single complex transformer (our verified complex_ops) vs a real
transformer that receives the SAME information (real+imag stacked as 2 channels,
exactly as Eilers & Jiang 2023 fed their real baseline). Both classify the
phase-carried chirp signals from signals.py.

The real baseline SEES the phase (via the imag channel), so any complex
advantage is about inductive bias / generalization, not information access.
Note: complex layers carry ~2x params (real+imag weights); we report counts. A
complex model that overfits LESS despite MORE params is a conservative,
stronger result.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "main"))
from complex_ops import (ComplexLinear, ComplexAttention, modReLU,        # noqa: E402
                         MagnitudeLN, ComplexPositionalEncoding)
from soliton import SolitonCell, epsilon_helix_loss                       # noqa: E402


class ComplexBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = ComplexAttention(d, h)
        self.n1 = MagnitudeLN(d)
        self.ff1 = ComplexLinear(d, 2 * d)
        self.act = modReLU(2 * d)
        self.ff2 = ComplexLinear(2 * d, d)
        self.n2 = MagnitudeLN(d)

    def forward(self, z):
        z = self.n1(z + self.attn(z))
        z = self.n2(z + self.ff2(self.act(self.ff1(z))))
        return z


class ComplexSeqClassifier(nn.Module):
    def __init__(self, d=48, h=4, layers=2, n_classes=8, T=64):
        super().__init__()
        self.inp = ComplexLinear(1, d)
        self.pe = ComplexPositionalEncoding(d, max_len=T)
        self.blocks = nn.ModuleList(ComplexBlock(d, h) for _ in range(layers))
        self.head = nn.Linear(d, n_classes)

    def forward(self, Z):                       # Z: (B, T) complex
        z = self.inp(Z.unsqueeze(-1))           # (B, T, d) complex
        z = self.pe(z)
        for b in self.blocks:
            z = b(z)
        feat = z.abs().mean(1)                  # (B, d) real
        return self.head(feat)


def _sinusoidal(T, d):
    pe = torch.zeros(T, d)
    pos = torch.arange(T).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class RealSeqClassifier(nn.Module):
    def __init__(self, d=64, h=4, layers=2, n_classes=8, T=64):
        super().__init__()
        self.inp = nn.Linear(2, d)
        self.register_buffer("pe", _sinusoidal(T, d), persistent=False)
        layer = nn.TransformerEncoderLayer(d, h, 2 * d, batch_first=True,
                                           activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)

    def forward(self, Z):                       # Z: (B, T) complex
        x = torch.stack([Z.real, Z.imag], dim=-1)   # (B, T, 2) — real sees phase too
        h = self.inp(x) + self.pe.unsqueeze(0)
        h = self.tf(h)
        return self.head(h.mean(1))


class MultiHelicalSeqClassifier(nn.Module):
    """L3: N co-evolving phase-separated chains + soliton coupling + interference
    readout (Ψ = Σ ψⁱ). N=2 is the multi-helical model; N=1 reduces to a single
    complex chain. The actual MHCoT novelty, now on phase-carried (home-field) data."""

    def __init__(self, d=48, h=4, layers=2, n_chains=2, n_classes=8, T=64,
                 eps_min=1.15):
        super().__init__()
        self.n_chains = n_chains
        self.inp = ComplexLinear(1, d)
        self.pe = ComplexPositionalEncoding(d, max_len=T)
        self.chain_phase = nn.Parameter(torch.randn(n_chains, d) * 0.1)
        self.blocks = nn.ModuleList(ComplexBlock(d, h) for _ in range(layers))
        self.soliton = SolitonCell(eps_min=eps_min, detach_gate=True)
        self.head = nn.Linear(d, n_classes)

    def forward(self, Z, return_psi=False):
        z = self.pe(self.inp(Z.unsqueeze(-1)))          # (B,T,d) complex
        B, T, d = z.shape
        phase = self.chain_phase.view(1, self.n_chains, 1, d)
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        psi = z.unsqueeze(1) * rot                      # (B,N,T,d)
        for blk in self.blocks:
            flat = psi.reshape(B * self.n_chains, T, d)
            flat = blk(flat)
            psi = flat.reshape(B, self.n_chains, T, d)
        psi = self.soliton(psi)
        Psi = psi.sum(1)                                # (B,T,d) interference
        feat = Psi.abs().mean(1)
        out = self.head(feat)
        if return_psi:
            return out, psi
        return out


class MultiHelicalPerChain(nn.Module):
    """L4b: per-chain readout (NO summation). Each chain pools its own |ψ_i| and
    decodes its own class logits → (B, N, C). A soft-OR over chains (done in the
    experiment) lets each chain own a different component — the sharpest readout
    for the multi-helical claim. N=1 reduces to a single complex chain."""

    def __init__(self, d=48, h=4, layers=2, n_chains=2, n_classes=8, T=64,
                 eps_min=1.15):
        super().__init__()
        self.n_chains = n_chains
        self.inp = ComplexLinear(1, d)
        self.pe = ComplexPositionalEncoding(d, max_len=T)
        self.chain_phase = nn.Parameter(torch.randn(n_chains, d) * 0.1)
        self.blocks = nn.ModuleList(ComplexBlock(d, h) for _ in range(layers))
        self.soliton = SolitonCell(eps_min=eps_min, detach_gate=True)
        self.heads = nn.ModuleList(nn.Linear(d, n_classes) for _ in range(n_chains))

    def forward(self, Z, return_psi=False):
        z = self.pe(self.inp(Z.unsqueeze(-1)))
        B, T, d = z.shape
        phase = self.chain_phase.view(1, self.n_chains, 1, d)
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        psi = z.unsqueeze(1) * rot
        for blk in self.blocks:
            flat = psi.reshape(B * self.n_chains, T, d)
            flat = blk(flat)
            psi = flat.reshape(B, self.n_chains, T, d)
        psi = self.soliton(psi)
        feats = psi.abs().mean(2)                            # (B, N, d) per chain
        logits = torch.stack([self.heads[i](feats[:, i])
                              for i in range(self.n_chains)], 1)   # (B, N, C)
        return (logits, psi) if return_psi else logits


class RealPerHead(nn.Module):
    """Real baseline with n_heads independent output heads → (B, H, C); soft-OR
    combined in the experiment. n_heads=2 is the 'two outputs' control."""

    def __init__(self, d=64, h=4, layers=2, n_heads=2, n_classes=8, T=64):
        super().__init__()
        self.inp = nn.Linear(2, d)
        self.register_buffer("pe", _sinusoidal(T, d), persistent=False)
        layer = nn.TransformerEncoderLayer(d, h, 2 * d, batch_first=True,
                                           activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.heads = nn.ModuleList(nn.Linear(d, n_classes) for _ in range(n_heads))

    def forward(self, Z):
        x = torch.stack([Z.real, Z.imag], dim=-1)
        h = self.inp(x) + self.pe.unsqueeze(0)
        h = self.tf(h)
        pooled = h.mean(1)
        return torch.stack([head(pooled) for head in self.heads], 1)   # (B, H, C)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from signals import make_dataset
    Z, y = make_dataset(16, n_classes=8, T=64, seed=0)
    cm = ComplexSeqClassifier(); rm = RealSeqClassifier()
    print(f"complex out {tuple(cm(Z).shape)}  params={n_params(cm)/1e3:.1f}K")
    print(f"real    out {tuple(rm(Z).shape)}  params={n_params(rm)/1e3:.1f}K")
    cm(Z).sum().backward(); rm(Z).sum().backward()
    print("ok — both forward+backward")
