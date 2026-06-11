"""
main/complex_ops.py
===================

Foundation primitives for MHCoT's complex-valued transformer head.

This is the BAR — everything higher (SolitonCell, MHCoTHead, training)
depends on these primitives behaving correctly under PyTorch's complex
autograd. Every module here has a unit test at the bottom of the file.

Modules
-------
ComplexLinear     real-parametrized complex linear layer (Trabelsi 2018)
ComplexLift       bridge from real LLM hidden states → complex MHCoT input
modReLU           complex activation: gates magnitude, preserves phase
MagnitudeLN       layer norm on magnitude, phase preserved
ComplexAttention  bidirectional multi-head attention with magnitude-softmax

Run the tests (CUDA preferred; cfloat support on MPS is incomplete):
    python main/complex_ops.py
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ComplexLinear — DCN-style real-parametrized complex linear
# ---------------------------------------------------------------------------
# A complex matmul  y = W z  with W, z ∈ ℂ can be implemented as
#   y_re = W_re x_re − W_im x_im
#   y_im = W_re x_im + W_im x_re
# i.e. two real weight matrices, four real matmuls. This is the standard
# pattern from Deep Complex Networks (Trabelsi 2018). We use it (instead
# of nn.Linear with dtype=cfloat) because it gives us explicit control
# over initialization, makes gradients inspectable, and works identically
# across CPU/CUDA without surprises.
# ---------------------------------------------------------------------------
class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Approximate complex Glorot: Xavier on each real component
        # independently. Trabelsi's exact Rayleigh-magnitude scheme is more
        # rigorous; this works in practice and is simpler.
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: complex (..., in_features) → out: complex (..., out_features)
        x_re, x_im = x.real, x.imag
        out_re = F.linear(x_re, self.weight_real) - F.linear(x_im, self.weight_imag)
        out_im = F.linear(x_re, self.weight_imag) + F.linear(x_im, self.weight_real)
        if self.bias_real is not None:
            out_re = out_re + self.bias_real
            out_im = out_im + self.bias_imag
        return torch.complex(out_re, out_im)


# ---------------------------------------------------------------------------
# ComplexLift — the GoT → MHCoT interface (EQ-A from the spec)
# ---------------------------------------------------------------------------
# Bridges real-valued LLM hidden states into complex space.
#     z = (W_real @ h) + i · (W_imag @ h)
# Init policy:
#     W_real = I               at step 0, Re(z) = h (preserves LLM understanding)
#     W_imag = small gaussian  non-trivial phase from step 0 (ε-helix can engage)
# ---------------------------------------------------------------------------
class ComplexLift(nn.Module):
    def __init__(self, d_model: int, imag_init_scale: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.W_real = nn.Parameter(torch.eye(d_model))
        self.W_imag = nn.Parameter(
            torch.randn(d_model, d_model) * (imag_init_scale / math.sqrt(d_model))
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: real (..., d_model) → z: complex (..., d_model)
        re = F.linear(h, self.W_real)
        im = F.linear(h, self.W_imag)
        return torch.complex(re, im)


# ---------------------------------------------------------------------------
# ComplexPositionalEncoding — installs the "time axis" that makes the
# per-token complex vectors into a coherent signal across the sequence
# ---------------------------------------------------------------------------
# After ComplexLift, each token has a complex vector but there is no notion
# of "the signal evolves across the sequence" — position is only implicit
# (whatever attention infers). This module rotates each complex channel by an
# angle proportional to token position, turning the sequence into a genuine
# multi-frequency complex waveform.
#
#     z'_t[k] = z_t[k] · exp(i · t · ω_k),     ω_k = base^(−k / d_model)
#
#   * Low-k channels rotate fast (high frequency → local structure)
#   * High-k channels rotate slow (low frequency → long-range structure)
#   * Multiplicative (rotary-style), so AMPLITUDE is preserved exactly:
#     |z'_t[k]| = |z_t[k]|. Only the phase carries the position information.
#   * At position t=0 the rotation is identity (exp(0) = 1), so the first
#     token is unchanged.
#
# Implemented with real-only arithmetic (cos/sin buffers) so it runs on
# MPS / CUDA / CPU without complex-kernel surprises.
# ---------------------------------------------------------------------------
class ComplexPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        k = torch.arange(d_model, dtype=torch.float32)
        omega = base ** (-k / d_model)                  # (d_model,)
        t = torch.arange(max_len, dtype=torch.float32)  # (max_len,)
        theta = torch.outer(t, omega)                   # (max_len, d_model)
        # Precompute unit-modulus rotation factors as real cos/sin buffers
        self.register_buffer("cos", torch.cos(theta), persistent=False)
        self.register_buffer("sin", torch.sin(theta), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, D) complex → (B, T, D) complex
        T = z.shape[1]
        if T > self.max_len:
            raise ValueError(
                f"sequence length {T} exceeds max_len {self.max_len}; "
                f"increase max_len in ComplexPositionalEncoding"
            )
        cos = self.cos[:T].unsqueeze(0)   # (1, T, D)
        sin = self.sin[:T].unsqueeze(0)
        z_re, z_im = z.real, z.imag
        # complex multiply z * (cos + i sin), done in real arithmetic
        out_re = z_re * cos - z_im * sin
        out_im = z_re * sin + z_im * cos
        return torch.complex(out_re, out_im)


# ---------------------------------------------------------------------------
# modReLU — complex activation that gates magnitude and preserves phase
# ---------------------------------------------------------------------------
# Real ReLU doesn't generalize to complex (zero is a manifold, not a
# threshold). modReLU (Arjovsky 2016) is the standard complex activation:
#     out = ReLU(|z| + b) · (z / |z|)
# b is a learnable per-channel scalar. Negative b zeroes out small-magnitude
# inputs; positive b widens the dynamic range. Phase is preserved EXACTLY.
# ---------------------------------------------------------------------------
class modReLU(nn.Module):
    def __init__(self, d_model: int, bias_init: float = -0.1):
        super().__init__()
        # Negative init encourages initial sparsity (some channels gated off).
        self.bias = nn.Parameter(torch.full((d_model,), bias_init))

    def forward(self, z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mag = z.abs()
        phase = z / (mag + eps)
        gated = F.relu(mag + self.bias)
        return gated * phase


# ---------------------------------------------------------------------------
# MagnitudeLN — LayerNorm on |z|, phase preserved
# ---------------------------------------------------------------------------
# Standard LayerNorm doesn't make sense on complex tensors (which axis do you
# normalize?). The cleanest complex variant: normalize the magnitude (as LN
# does for real values), then re-multiply by the original unit phase.
# Optional γ, β scale and shift the magnitude. A negative scaled magnitude
# would flip the phase by π — algebraically valid, lets the network learn
# more general dynamics. We allow it.
# ---------------------------------------------------------------------------
class MagnitudeLN(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = z.abs()
        phase = z / (mag + self.eps)
        mu = mag.mean(dim=-1, keepdim=True)
        sigma = mag.std(dim=-1, keepdim=True, unbiased=False)
        mag_normed = (mag - mu) / (sigma + self.eps)
        mag_out = self.gamma * mag_normed + self.beta
        return mag_out * phase


# ---------------------------------------------------------------------------
# ComplexAttention — bidirectional multi-head attention, magnitude-softmax
# ---------------------------------------------------------------------------
# For complex Q, K, V:
#   raw   = Q · K^H / sqrt(d)        complex, (B, H, T, T)
#   score = |raw|                    real, non-negative
#   attn  = softmax(score, dim=-1)   real, sums to 1
#   out   = attn @ V                 complex (real-weighted sum of complex V)
#
# Why this formulation:
#   * keeps attention weights real and bounded → numerically stable
#   * V stays complex (carries both amplitude and phase information)
#   * uses the magnitude of the complex inner product — which is exactly
#     the "alignment" measure that scaled-dot-product attention captures
#     in real space, generalized
#   * bidirectional by default (no causal mask) — MHCoT is encoder-style
#
# To use a mask later (e.g., for padding), pass a bool tensor (B, T, T)
# where True = positions to MASK OUT.
# ---------------------------------------------------------------------------
class ComplexAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model {d_model} not divisible by n_heads {n_heads}"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = ComplexLinear(d_model, d_model)
        self.k_proj = ComplexLinear(d_model, d_model)
        self.v_proj = ComplexLinear(d_model, d_model)
        self.out_proj = ComplexLinear(d_model, d_model)

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # MPS-safe implementation: do ALL heavy math in real space. Complex
        # einsum and .conj() on MPS have patchy support; the real-decomposition
        # path below is mathematically identical, runs everywhere (CUDA / MPS /
        # CPU), and has essentially the same cost as the complex variant
        # (complex matmul on CUDA is internally four real matmuls anyway).
        B, T, D = z.shape
        Q = self.q_proj(z).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(z).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(z).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Q · K^H = (Q_re + i Q_im) · (K_re − i K_im)^T
        #        = (Q_re K_re^T + Q_im K_im^T) + i (Q_im K_re^T − Q_re K_im^T)
        Q_re, Q_im = Q.real, Q.imag
        K_re, K_im = K.real, K.imag
        raw_re = (
            torch.einsum("bhid,bhjd->bhij", Q_re, K_re)
            + torch.einsum("bhid,bhjd->bhij", Q_im, K_im)
        )
        raw_im = (
            torch.einsum("bhid,bhjd->bhij", Q_im, K_re)
            - torch.einsum("bhid,bhjd->bhij", Q_re, K_im)
        )
        # |raw|  →  scaled  →  softmax
        scores = torch.sqrt(raw_re * raw_re + raw_im * raw_im + 1e-12) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)   # real (B, H, T, T)

        # attn @ V per real/imag component
        V_re, V_im = V.real, V.imag
        out_re = torch.einsum("bhij,bhjd->bhid", attn, V_re)
        out_im = torch.einsum("bhij,bhjd->bhid", attn, V_im)
        out = torch.complex(out_re, out_im)                       # (B, H, T, d_h) cfloat
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _pick_device() -> str:
    # Priority: CUDA > MPS > CPU.
    # MPS works for everything in this file because each module avoids
    # complex-tensor heavy ops (the math is done in real space, complex
    # tensors are only constructed at module boundaries).
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _test_complex_linear(device: str) -> None:
    print("[test] ComplexLinear ...")
    layer = ComplexLinear(16, 32).to(device)
    x = torch.randn(4, 16, dtype=torch.cfloat, device=device, requires_grad=True)
    y = layer(x)
    assert y.shape == (4, 32), f"unexpected shape {y.shape}"
    assert y.dtype == torch.cfloat, f"unexpected dtype {y.dtype}"
    y.abs().sum().backward()
    assert layer.weight_real.grad is not None
    assert layer.weight_imag.grad is not None
    print(f"    ok — out {tuple(y.shape)} {y.dtype}, gradients flow")


def _test_complex_lift(device: str) -> None:
    print("[test] ComplexLift ...")
    lift = ComplexLift(16).to(device)
    h = torch.randn(4, 16, device=device)
    z = lift(h)
    assert z.shape == (4, 16) and z.dtype == torch.cfloat
    # At step 0, W_real = I so Re(z) ≈ h
    assert torch.allclose(z.real, h, atol=1e-5), "W_real should init as identity"
    # W_imag is small but non-zero → Im(z) non-trivial
    assert z.imag.abs().mean() > 0, "W_imag should be non-zero at init"
    print(f"    ok — Re(z) ≈ h, mean |Im(z)| = {z.imag.abs().mean().item():.4f}")


def _test_positional_encoding(device: str) -> None:
    print("[test] ComplexPositionalEncoding ...")
    pe = ComplexPositionalEncoding(d_model=16, max_len=64).to(device)
    z = torch.randn(2, 10, 16, dtype=torch.cfloat, device=device)
    out = pe(z)
    assert out.shape == z.shape and out.dtype == torch.cfloat
    # Amplitude preserved exactly (unit-modulus rotation)
    assert torch.allclose(out.abs(), z.abs(), atol=1e-4), "PE changed amplitude"
    # Position 0 unchanged (theta = 0 → rotation = identity)
    assert torch.allclose(out[:, 0], z[:, 0], atol=1e-5), "PE changed position 0"
    # Later positions DID rotate in phase
    assert not torch.allclose(out[:, 5], z[:, 5], atol=1e-3), (
        "PE did not change later positions"
    )
    # Phase difference between pos 0 and pos 5 should be non-trivial
    drift = (out[:, 5] / (out[:, 5].abs() + 1e-6)
             * (z[:, 5] / (z[:, 5].abs() + 1e-6)).conj()).angle().abs().mean()
    print(f"    ok — amplitude preserved, mean phase drift pos5 = {drift.item():.3f} rad")


def _test_modrelu(device: str) -> None:
    print("[test] modReLU ...")
    act = modReLU(16).to(device)
    z = torch.randn(4, 16, dtype=torch.cfloat, device=device)
    out = act(z)
    assert out.shape == z.shape and out.dtype == torch.cfloat
    # Phase preservation where output is non-zero
    mag_in = z.abs()
    mag_out = out.abs()
    nonzero = mag_out > 1e-5
    if nonzero.any():
        phase_in = (z / (mag_in + 1e-6))[nonzero]
        phase_out = (out / (mag_out + 1e-6))[nonzero]
        assert torch.allclose(phase_in, phase_out, atol=1e-4), (
            "modReLU did not preserve phase"
        )
        pct = nonzero.float().mean().item() * 100
        print(f"    ok — phase preserved on {pct:.1f}% of positions (rest gated to 0)")
    else:
        print("    ok — all positions gated to 0 at init (bias too negative)")


def _test_magnitude_ln(device: str) -> None:
    print("[test] MagnitudeLN ...")
    ln = MagnitudeLN(16).to(device)
    z = torch.randn(4, 16, dtype=torch.cfloat, device=device) * 5.0
    out = ln(z)
    assert out.shape == z.shape and out.dtype == torch.cfloat
    # Phase preserved up to ± sign (negative scaled magnitudes flip by π)
    phase_in = z / (z.abs() + 1e-6)
    phase_out = out / (out.abs() + 1e-6)
    aligned = (phase_in * phase_out.conj()).real  # ±1 if phase aligned/flipped
    assert (aligned.abs() > 0.99).all(), "MagnitudeLN broke phase direction"
    print(f"    ok — phase preserved (with allowed ±1 sign flip)")


def _test_complex_attention(device: str) -> None:
    print("[test] ComplexAttention ...")
    attn = ComplexAttention(d_model=64, n_heads=8).to(device)
    z = torch.randn(2, 10, 64, dtype=torch.cfloat, device=device, requires_grad=True)
    out = attn(z)
    assert out.shape == z.shape and out.dtype == torch.cfloat
    # Gradient flow through the whole stack
    out.abs().sum().backward()
    assert attn.q_proj.weight_real.grad is not None
    # Attention weights sum to 1 along the last axis
    Q = attn.q_proj(z).view(2, 10, 8, 8).transpose(1, 2)
    K = attn.k_proj(z).view(2, 10, 8, 8).transpose(1, 2)
    scores = (torch.einsum("bhid,bhjd->bhij", Q, K.conj())).abs() / math.sqrt(8)
    attn_w = F.softmax(scores, dim=-1)
    sums = attn_w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    print(f"    ok — out {tuple(out.shape)}, attn weights sum to 1, gradients flow")


def main() -> None:
    device = _pick_device()
    print(f"[device] {device}")
    if device == "cpu":
        print("[warn]  running on CPU — slow. Prefer CUDA (Colab) or MPS (Apple Silicon).")
    elif device == "mps":
        print("[info]  running on Apple Silicon MPS. Real-only complex math path.")
    print()
    _test_complex_linear(device)
    _test_complex_lift(device)
    _test_positional_encoding(device)
    _test_modrelu(device)
    _test_magnitude_ln(device)
    _test_complex_attention(device)
    print("\n[all tests passed]")


if __name__ == "__main__":
    main()
