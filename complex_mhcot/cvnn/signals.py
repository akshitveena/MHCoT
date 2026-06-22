"""
cvnn/signals.py  —  L0: a naturally complex-valued dataset (phase carries the label)
====================================================================================

The home field we never gave complex nets. Each signal is a complex chirp:

    z[t] = A · exp(i·2π·(f0·t + ½·k_c·t²)) + noise        t ∈ [0,1]

The CLASS c sets the chirp rate k_c (the phase curvature). Crucially:
  * the amplitude A and base frequency f0 are per-sample NUISANCES (class-independent),
  * |z[t]| ≈ A is therefore class-independent → **magnitude carries no label**,
  * the label lives entirely in the PHASE dynamics (instantaneous frequency slope).

So a magnitude-only (or magnitude-reliant) model is at chance; a phase-aware
(complex) model can solve it. This is the intrinsic-complex structure the CVNN
literature says complex nets exploit — and exactly what reasoning text lacked.

The train-set size is the OVERFITTING knob: small train → real models overfit
(the regime where the paper's complex advantage appeared).

Self-test proves L0: magnitude features ≈ chance, phase features separable.
    python cvnn/signals.py
"""

from __future__ import annotations

import numpy as np
import torch


def make_dataset(n, n_classes=8, T=64, noise=0.15, f0_range=5.0,
                 amp_range=(0.5, 1.5), k_range=20.0, seed=0):
    """Return Z (n, T) complex64, y (n,) long. Label = chirp-rate class; magnitude
    is class-independent so the signal is purely phase-carried."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, size=n)
    ks = np.linspace(-k_range, k_range, n_classes)        # per-class chirp rate
    t = np.linspace(0.0, 1.0, T, dtype=np.float64)
    Z = np.zeros((n, T), dtype=np.complex64)
    for i in range(n):
        f0 = rng.uniform(-f0_range, f0_range)             # nuisance base freq
        A = rng.uniform(*amp_range)                        # nuisance amplitude (scalar)
        phase = 2 * np.pi * (f0 * t + 0.5 * ks[y[i]] * t * t)
        sig = A * np.exp(1j * phase)
        sig = sig + noise * (rng.randn(T) + 1j * rng.randn(T))
        Z[i] = sig.astype(np.complex64)
    return torch.from_numpy(Z), torch.from_numpy(y).long()


def make_superposition(n, n_classes=8, T=64, k_present=2, noise=0.15,
                       f0_range=5.0, amp_range=(0.5, 1.5), k_range=20.0, seed=0):
    """Multi-hypothesis home field: each signal is a SUM of `k_present` distinct
    chirps. Label = multi-hot of which rate-classes are present. Recovering BOTH
    components requires holding multiple hypotheses — the structure a single chain
    can't represent but two co-evolving chains might (one per component).
    Returns Z (n,T) complex64, Y (n, n_classes) float multi-hot."""
    rng = np.random.RandomState(seed)
    ks = np.linspace(-k_range, k_range, n_classes)
    t = np.linspace(0.0, 1.0, T, dtype=np.float64)
    Z = np.zeros((n, T), dtype=np.complex64)
    Y = np.zeros((n, n_classes), dtype=np.float32)
    for i in range(n):
        present = rng.choice(n_classes, size=k_present, replace=False)
        sig = np.zeros(T, dtype=np.complex128)
        for c in present:
            f0 = rng.uniform(-f0_range, f0_range)
            A = rng.uniform(*amp_range)
            sig += A * np.exp(1j * 2 * np.pi * (f0 * t + 0.5 * ks[c] * t * t))
        sig += noise * (rng.randn(T) + 1j * rng.randn(T))
        Z[i] = sig.astype(np.complex64)
        Y[i, present] = 1.0
    return torch.from_numpy(Z), torch.from_numpy(Y)


def make_multitone(n, n_classes=8, T=128, k_present=2, noise=0.15,
                   amp_range=(0.5, 1.5), seed=0):
    """genuine_interp's home field: each signal is a SUM of k_present STEADY tones
    (constant frequencies, not chirps), drawn from n_classes distinct frequencies.
    The signal genuinely DECOMPOSES into components, so K interpretations each have
    a real component to gather/pool. Label = multi-hot of which tones are present.
    Returns Z (n,T) complex64, Y (n, n_classes) multi-hot."""
    rng = np.random.RandomState(seed)
    freqs = np.arange(1, n_classes + 1).astype(np.float64)   # distinct steady freqs
    t = np.linspace(0.0, 1.0, T, dtype=np.float64)
    Z = np.zeros((n, T), dtype=np.complex64)
    Y = np.zeros((n, n_classes), dtype=np.float32)
    for i in range(n):
        present = rng.choice(n_classes, size=k_present, replace=False)
        sig = np.zeros(T, dtype=np.complex128)
        for c in present:
            A = rng.uniform(*amp_range); phi = rng.uniform(0, 2 * np.pi)
            sig += A * np.exp(1j * (2 * np.pi * freqs[c] * t + phi))   # STEADY tone
        sig += noise * (rng.randn(T) + 1j * rng.randn(T))
        Z[i] = sig.astype(np.complex64); Y[i, present] = 1.0
    return torch.from_numpy(Z), torch.from_numpy(Y)


def make_ambiguous(n, n_classes=8, T=64, noise=0.15, f0_range=5.0,
                   amp_range=(0.5, 1.5), k_range=20.0, jitter=0.15, seed=0):
    """Ambiguity home field: each signal is a SINGLE chirp whose rate sits BETWEEN
    two adjacent class prototypes (near their midpoint, jittered) — so the signal
    is genuinely consistent with BOTH interpretations. Label = the ambiguity set
    {c1, c2}. The right behavior is to represent both alternatives; a model forced
    to collapse picks one and misses the other. Two phase-separated chains could
    each point at one interpretation — the purest anti-collapse test.
    Returns Z (n,T) complex64, Y (n, n_classes) float multi-hot (2 ones)."""
    rng = np.random.RandomState(seed)
    ks = np.linspace(-k_range, k_range, n_classes)
    t = np.linspace(0.0, 1.0, T, dtype=np.float64)
    Z = np.zeros((n, T), dtype=np.complex64)
    Y = np.zeros((n, n_classes), dtype=np.float32)
    for i in range(n):
        c1 = rng.randint(0, n_classes - 1)
        c2 = c1 + 1                                        # adjacent ambiguity pair
        w = 0.5 + jitter * rng.uniform(-1, 1)              # near midpoint
        k = (1 - w) * ks[c1] + w * ks[c2]                  # rate between prototypes
        f0 = rng.uniform(-f0_range, f0_range)
        A = rng.uniform(*amp_range)
        sig = A * np.exp(1j * 2 * np.pi * (f0 * t + 0.5 * k * t * t))
        sig = sig + noise * (rng.randn(T) + 1j * rng.randn(T))
        Z[i] = sig.astype(np.complex64)
        Y[i, c1] = 1.0
        Y[i, c2] = 1.0
    return torch.from_numpy(Z), torch.from_numpy(Y)


# ---------------------------------------------------------------------------
# Feature helpers (for the L0 sanity check only — models will see raw Z)
# ---------------------------------------------------------------------------
def magnitude_features(Z):
    """Class-INDEPENDENT by construction: stats of |z|."""
    mag = Z.abs().numpy()
    return np.stack([mag.mean(1), mag.std(1), mag.max(1), mag.min(1)], 1)


def phase_features(Z):
    """Instantaneous-frequency stats: inst_freq[t] = angle(z[t+1]·conj(z[t])).
    For a chirp this rises linearly with slope ∝ k_c → class-discriminative."""
    z = Z.numpy()
    inst = np.angle(z[:, 1:] * np.conj(z[:, :-1]))        # (n, T-1)
    tt = np.arange(inst.shape[1])
    # linear slope of inst freq per sample
    slope = np.polyfit(tt, inst.T, 1)[0]                  # (n,)
    return np.stack([inst.mean(1), inst.std(1), slope,
                     inst[:, :inst.shape[1] // 2].mean(1),
                     inst[:, inst.shape[1] // 2:].mean(1)], 1)


# ---------------------------------------------------------------------------
# L0 self-test
# ---------------------------------------------------------------------------
def _test():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n_classes = 8
    Ztr, ytr = make_dataset(2000, n_classes=n_classes, seed=0)
    Zte, yte = make_dataset(1000, n_classes=n_classes, seed=1)
    chance = 1.0 / n_classes
    print(f"[L0] {n_classes}-class chirp dataset; chance = {chance:.3f}")

    def acc(feat_fn, name):
        Xtr, Xte = feat_fn(Ztr), feat_fn(Zte)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
        clf.fit(sc.transform(Xtr), ytr.numpy())
        a = clf.score(sc.transform(Xte), yte.numpy())
        print(f"    {name:<22} test acc = {a:.3f}")
        return a

    a_mag = acc(magnitude_features, "magnitude features")
    a_phase = acc(phase_features, "phase (inst-freq) features")

    print()
    assert a_mag < chance + 0.05, \
        f"magnitude should be ~chance ({chance:.3f}), got {a_mag:.3f} — label leaked into |z|"
    assert a_phase > 0.5, \
        f"phase features should separate classes, got {a_phase:.3f}"
    print(f"[L0 PASS] magnitude ≈ chance ({a_mag:.3f}), phase separable ({a_phase:.3f}) "
          f"→ the label is genuinely PHASE-CARRIED. This is complex nets' home field.")


if __name__ == "__main__":
    _test()
