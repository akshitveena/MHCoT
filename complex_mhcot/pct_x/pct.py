"""
pct_x/pct.py  —  Phase-Coherent Transformer (PCT) core + the multi-chain axis
=============================================================================

Implements the key primitive from Hioki 2026 ("Complex-Valued Phase-Coherent
Transformer", arXiv:2605.10123): a token-NON-competing attention that replaces
softmax row-normalisation with a real-valued, element-independent SIGMOID gate on
the L2-normalised complex Q·K cosine. The four conditions PCT requires:
  C1 real-valued gate output, C2 bounded gate, C3 nonzero gradient, C4 element-
  independent gate — all satisfied by sigmoid.

Our research question (the orthogonal extension): our multi-helical / multi-chain
co-evolution failed, but ONLY under token-COMPETING (magnitude-softmax) attention,
which PCT argues is the wrong primitive for complex nets. Does multi-chain
co-evolution behave differently under PCT's phase-coherent attention? This file
provides both the single-chain PCT baseline and the multi-chain PCT variant so we
can test it head-to-head.

Run the self-test:
    python pct_x/pct.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "main"))
from complex_ops import (ComplexLinear, modReLU, MagnitudeLN,            # noqa: E402
                         ComplexPositionalEncoding)
from soliton import SolitonCell, epsilon_helix_loss                      # noqa: E402


class PCTAttention(nn.Module):
    """Phase-coherent, token-non-competing attention. NO softmax row-norm:
    weight_ij = sigmoid(alpha * cos(Q_i, K_j) + beta), an element-independent
    real gate on L2-normalised complex Q,K. Output = unnormalised gated sum of V."""

    def __init__(self, d_model: int, n_heads: int = 4, alpha_init: float = 4.0,
                 beta_init: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = ComplexLinear(d_model, d_model)
        self.k = ComplexLinear(d_model, d_model)
        self.v = ComplexLinear(d_model, d_model)
        self.out = ComplexLinear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))   # gate sharpness
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))     # gate bias

    def forward(self, z, key_padding_mask=None):
        B, T, D = z.shape
        H, dh = self.n_heads, self.d_head
        Q = self.q(z).view(B, T, H, dh).transpose(1, 2)              # (B,H,T,dh) cplx
        K = self.k(z).view(B, T, H, dh).transpose(1, 2)
        V = self.v(z).view(B, T, H, dh).transpose(1, 2)
        # L2-normalise complex Q,K over the head dim
        Qn = Q / Q.abs().pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        Kn = K / K.abs().pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        # cos = Re<Qn, Kn^H>  in [-1, 1]
        score = (torch.einsum("bhid,bhjd->bhij", Qn.real, Kn.real)
                 + torch.einsum("bhid,bhjd->bhij", Qn.imag, Kn.imag))
        gate = torch.sigmoid(self.alpha * score + self.beta)         # NO row-norm
        if key_padding_mask is not None:
            gate = gate.masked_fill(key_padding_mask[:, None, None, :], 0.0)
        out_re = torch.einsum("bhij,bhjd->bhid", gate, V.real)
        out_im = torch.einsum("bhij,bhjd->bhid", gate, V.imag)
        out = torch.complex(out_re, out_im).transpose(1, 2).reshape(B, T, D)
        return self.out(out)

    @torch.no_grad()
    def row_sums(self, z):
        """Diagnostic: attention-weight row sums (≠ 1 ⇒ token non-competition)."""
        B, T, D = z.shape; H, dh = self.n_heads, self.d_head
        Q = self.q(z).view(B, T, H, dh).transpose(1, 2)
        K = self.k(z).view(B, T, H, dh).transpose(1, 2)
        Qn = Q / Q.abs().pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        Kn = K / K.abs().pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        s = (torch.einsum("bhid,bhjd->bhij", Qn.real, Kn.real)
             + torch.einsum("bhid,bhjd->bhij", Qn.imag, Kn.imag))
        return torch.sigmoid(self.alpha * s + self.beta).sum(-1)


class PCTBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = PCTAttention(d, h)
        self.n1 = MagnitudeLN(d)
        self.ff1 = ComplexLinear(d, 2 * d)
        self.act = modReLU(2 * d)
        self.ff2 = ComplexLinear(2 * d, d)
        self.n2 = MagnitudeLN(d)

    def forward(self, z, mask=None):
        z = self.n1(z + self.attn(z, key_padding_mask=mask))
        z = self.n2(z + self.ff2(self.act(self.ff1(z))))
        return z


class PCTClassifier(nn.Module):
    """Single-chain PCT (the baseline) OR multi-chain PCT (n_chains>1, with soliton
    co-evolution + interference readout) — the head-to-head for our question."""

    def __init__(self, d=48, h=4, layers=2, n_chains=1, n_classes=8, T=64,
                 eps_min=1.15, in_ch=1, readout="crossterm"):
        super().__init__()
        self.n_chains = n_chains
        self.readout = readout          # 'mag'(old) | 'phase' | 'sum' | 'crossterm'
        self.inp = ComplexLinear(in_ch, d)
        self.pe = ComplexPositionalEncoding(d, max_len=T)
        self.blocks = nn.ModuleList(PCTBlock(d, h) for _ in range(layers))
        if n_chains > 1:
            self.chain_phase = nn.Parameter(torch.randn(n_chains, d) * 0.1)
            self.soliton = SolitonCell(eps_min=eps_min, detach_gate=True)
        # readout dim: phase-preserving uses [Re,Im]=2d per quantity; crossterm
        # adds the interference cross-term (the ONLY genuinely multi-chain signal).
        if n_chains == 1:
            feat_dim = 2 * d                          # [Re,Im] of pooled chain
        elif readout == "sum" or readout == "mag":
            feat_dim = 2 * d                          # [Re,Im] of pooled Ψ (≈ average)
        elif readout == "crossterm":
            feat_dim = 6 * d                          # chain0, chain1, cross  (each Re,Im)
        else:                                          # 'phase' = both chains, no cross
            feat_dim = 4 * d
        self.head = nn.Linear(feat_dim, n_classes)

    def _ri(self, p):                                  # complex (B,d) -> [Re,Im] (B,2d)
        return torch.cat([p.real, p.imag], dim=-1)

    def forward(self, Z, return_psi=False):
        if Z.dim() == 2:
            Z = Z.unsqueeze(-1)                        # (B,T) -> (B,T,1)
        z = self.pe(self.inp(Z))                       # (B,T,d) complex
        B, T, d = z.shape
        if self.n_chains == 1:
            for blk in self.blocks:
                z = blk(z)
            p = z.mean(1)                              # phase-preserving pool (keeps Im!)
            out = self.head(self._ri(p))
            return (out, None) if return_psi else out

        phase = self.chain_phase.view(1, self.n_chains, 1, d)
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        psi = z.unsqueeze(1) * rot                     # (B,N,T,d)
        for blk in self.blocks:
            flat = psi.reshape(B * self.n_chains, T, d)
            flat = blk(flat)
            psi = flat.reshape(B, self.n_chains, T, d)
        psi = self.soliton(psi)
        p0, p1 = psi[:, 0].mean(1), psi[:, 1].mean(1)  # (B,d) complex per chain

        if self.readout in ("sum", "mag"):
            feat = self._ri((psi[:, 0] + psi[:, 1]).mean(1))    # the OLD averaging readout
        elif self.readout == "phase":
            feat = torch.cat([self._ri(p0), self._ri(p1)], -1)  # both chains, no cross
        else:  # crossterm — exposes 2·Re(ψ0·conj ψ1), the genuine multi-chain signal
            psi0, psi1 = psi[:, 0], psi[:, 1]
            cross_re = (psi0.real * psi1.real + psi0.imag * psi1.imag).mean(1)
            cross_im = (psi0.imag * psi1.real - psi0.real * psi1.imag).mean(1)
            feat = torch.cat([self._ri(p0), self._ri(p1), cross_re, cross_im], -1)
        out = self.head(feat)
        return (out, psi) if return_psi else out


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d = 4, 64, 48
    Z = torch.randn(B, T, dtype=torch.cfloat)

    print("[test] PCTAttention — token NON-competition (row sums ≠ 1) ...")
    z = torch.randn(B, T, d, dtype=torch.cfloat)
    attn = PCTAttention(d, 4)
    rs = attn.row_sums(z)
    print(f"    row-sum range [{rs.min():.2f}, {rs.max():.2f}] (softmax would be 1.0)")
    assert not torch.allclose(rs, torch.ones_like(rs), atol=0.1), "should NOT sum to 1"
    out = attn(z); out.abs().sum().backward()
    assert attn.alpha.grad is not None and torch.isfinite(attn.alpha.grad)
    print("    ok — non-competing, gradients flow to gate params")

    print("[test] PCTClassifier single-chain vs multi-chain ...")
    for nc in (1, 2):
        m = PCTClassifier(n_chains=nc, T=T)
        out = m(Z)
        assert out.shape == (B, 8) and torch.isfinite(out).all()
        out.sum().backward()
        print(f"    n_chains={nc}: out {tuple(out.shape)} finite, "
              f"params={n_params(m)/1e3:.1f}K, grads ok")

    print("\n[all PCT tests passed]")
