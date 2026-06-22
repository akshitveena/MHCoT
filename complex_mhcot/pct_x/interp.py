"""
pct_x/interp.py  —  MHCoT-v2: genuine interpretations from coherently-pooled strands
====================================================================================

The architecture you actually described — replacing the phase-rotation fake-
diversity with the real thing:

  strands (complex tokens, each = amplitude+phase descriptors)
     │   PCT phase-coherent attention processes the strands
     ▼
  CoherentInterpretation:  K GENUINELY-DIFFERENT interpretations, each built by
     its OWN learnable query (a "kind of wire") that gathers strands and pools
     them COHERENTLY (complex weighted sum: phase-aligned strands reinforce,
     misaligned cancel — "many thin wires → one thick wire / main phase").
     │
     ▼
  deep interpretation-level network:  PCT blocks over the K interpretation-tokens
     │
     ▼
  co-evolution:  SolitonCell couples the K interpretations (learn from each other),
     ε-helix keeps them distinct.
     │
     ▼
  readout:  phase-preserving [Re,Im] per interpretation + pairwise INTERFERENCE
     cross-terms  2·Re(ψᵢ·conj ψⱼ)  — keeps the imaginary part AND exposes the
     genuinely multi-interpretation signal (no averaging, no |·| discard).

Every piece the imaginary part touches is preserved to the output. The K
interpretations are distinct by construction (separate queries), not rotations.

    python pct_x/interp.py        # self-test
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "main"))
from complex_ops import (ComplexLinear, MagnitudeLN,                     # noqa: E402
                         ComplexPositionalEncoding)
from soliton import SolitonCell, epsilon_helix_loss                      # noqa: E402
from pct import PCTBlock                                                 # noqa: E402


class CoherentInterpretation(nn.Module):
    """Complex strands (B,T,d) -> K genuinely-distinct interpretations (B,K,d).
    Each interpretation has its OWN learnable complex query; it gates strands by
    phase-coherent similarity and pools them COHERENTLY (complex sum)."""

    def __init__(self, d, K, gate="sigmoid"):
        super().__init__()
        self.K, self.gate = K, gate
        self.q_re = nn.Parameter(torch.randn(K, d) * 0.5)   # distinct per interp
        self.q_im = nn.Parameter(torch.randn(K, d) * 0.5)
        self.key = ComplexLinear(d, d)
        self.val = ComplexLinear(d, d)
        self.norm = MagnitudeLN(d)

    def forward(self, psi, mask=None):                       # psi (B,T,d) complex
        kk = self.key(psi); vv = self.val(psi)               # (B,T,d)
        q = torch.complex(self.q_re, self.q_im)              # (K,d)
        qn = q / q.abs().pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        kn = kk / kk.abs().pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        # phase-coherent strand→interpretation similarity: Re<q_k, kn_t>  (B,K,T)
        score = (torch.einsum("kd,btd->bkt", qn.real, kn.real)
                 + torch.einsum("kd,btd->bkt", qn.imag, kn.imag))
        if mask is not None:
            score = score.masked_fill(mask[:, None, :], -30.0)
        a = (torch.softmax(score, -1) if self.gate == "softmax"
             else torch.sigmoid(4.0 * score))                # non-competing coherent gate
        # coherent pooling: main_k = Σ_t a_{k,t} · vv_t  (complex → aligned reinforce)
        main_re = torch.einsum("bkt,btd->bkd", a, vv.real)
        main_im = torch.einsum("bkt,btd->bkd", a, vv.imag)
        return self.norm(torch.complex(main_re, main_im))    # (B,K,d) complex


class GenuineInterpClassifier(nn.Module):
    def __init__(self, d=48, h=4, strand_layers=2, K=2, interp_depth=1,
                 n_classes=8, T=64, in_ch=1, coevolve=True, eps_min=1.15,
                 gate="sigmoid"):
        super().__init__()
        self.K, self.coevolve = K, coevolve
        self.inp = ComplexLinear(in_ch, d)
        self.pe = ComplexPositionalEncoding(d, max_len=T)
        self.strand_blocks = nn.ModuleList(PCTBlock(d, h) for _ in range(strand_layers))
        self.interp = CoherentInterpretation(d, K, gate=gate)
        self.interp_blocks = nn.ModuleList(PCTBlock(d, h) for _ in range(interp_depth))
        if coevolve:
            self.soliton = SolitonCell(eps_min=eps_min, detach_gate=True)
        feat_dim = 2 * d * K + 2 * d * (K * (K - 1) // 2)     # per-interp + pairwise cross
        self.head = nn.Linear(feat_dim, n_classes)

    @staticmethod
    def _ri(p):
        return torch.cat([p.real, p.imag], -1)

    def forward(self, Z, return_psi=False):
        if Z.dim() == 2:
            Z = Z.unsqueeze(-1)
        z = self.pe(self.inp(Z))                              # (B,T,d) strands
        for blk in self.strand_blocks:
            z = blk(z)
        interp = self.interp(z)                               # (B,K,d) genuine interps
        for blk in self.interp_blocks:                        # deep interp-level network
            interp = blk(interp)
        if self.coevolve:
            interp = self.soliton(interp.unsqueeze(2)).squeeze(2)   # co-evolve
        # phase-preserving + pairwise interference readout
        feats = [self._ri(interp[:, k]) for k in range(self.K)]
        for i in range(self.K):
            for j in range(i + 1, self.K):
                a, b = interp[:, i], interp[:, j]
                feats.append(a.real * b.real + a.imag * b.imag)   # Re(a·conj b)
                feats.append(a.imag * b.real - a.real * b.imag)   # Im(a·conj b)
        out = self.head(torch.cat(feats, -1))
        return (out, interp.unsqueeze(2)) if return_psi else out  # (B,K,1,d) for helix


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 4, 64
    Z = torch.randn(B, T, dtype=torch.cfloat)

    print("[test] CoherentInterpretation — K interpretations are DISTINCT ...")
    psi = torch.randn(B, T, 48, dtype=torch.cfloat)
    ci = CoherentInterpretation(48, K=3)
    interp = ci(psi)                                          # (B,3,48)
    assert interp.shape == (B, 3, 48)
    # pairwise distinctness (NOT rotations of one another)
    d01 = (interp[:, 0] - interp[:, 1]).abs().mean().item()
    d02 = (interp[:, 0] - interp[:, 2]).abs().mean().item()
    print(f"    ok — interp distances |0-1|={d01:.3f} |0-2|={d02:.3f} (>0 ⇒ distinct)")
    assert d01 > 1e-3 and d02 > 1e-3

    print("[test] GenuineInterpClassifier forward + grad + helix ...")
    for K in (2, 3):
        m = GenuineInterpClassifier(K=K, T=T, coevolve=True)
        out, psi_out = m(Z, return_psi=True)
        assert out.shape == (B, 8) and torch.isfinite(out).all()
        loss = out.sum() + 0.05 * epsilon_helix_loss(psi_out)
        loss.backward()
        assert m.interp.q_re.grad is not None and torch.isfinite(m.interp.q_re.grad).all()
        print(f"    K={K}: out {tuple(out.shape)} finite, helix ok, "
              f"params={n_params(m)/1e3:.1f}K, grads reach interpretation queries")

    print("\n[all interp tests passed]")
