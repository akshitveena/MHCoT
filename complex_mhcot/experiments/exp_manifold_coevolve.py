"""
experiments/exp_manifold_coevolve.py  —  class-conditioned co-evolving manifolds
================================================================================

The user's idea, tested honestly: instead of two phase-rotated copies of one input
(which never carried distinct meaning, §5.9), give the two chains GENUINELY
different content —
    chain_pos = the sequence of CORRECT reasoning-candidate signals
    chain_neg = the sequence of INCORRECT candidate signals
— encode each class's representation space, then CO-EVOLVE the two manifolds
(soliton) and classify a held-out trace by which manifold it constructively
INTERFERES with.

Data: cached candidate signals (data/hidden_cache/gsm8k_test_pools.pt), answer-
region (lastk) pooling, with derived correctness labels. 1318 correct / 497
incorrect candidates over 605 problems. No LLM needed.

Three models, param-comparable, episodic prototype training, 3 seeds:
    complex_coevolve : lift -> ComplexAttention manifold -> soliton co-evolve ->
                       classify by interference |q+m_pos|^2 - |q+m_neg|^2   (the idea)
    real_proto       : real attention manifold -> classify by -||q-m||^2 diff (control)
    plain_mlp        : MLP on the candidate vector (no manifolds)           (floor)

Pre-committed bar: complex_coevolve > real_proto AND > plain_mlp on held-out AUC,
across seeds. Also reports whether co-evolution increases pos/neg manifold
separation (exploratory).

Run:
    python experiments/exp_manifold_coevolve.py --smoke
    python experiments/exp_manifold_coevolve.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent / "main"))

from complex_ops import ComplexLift, ComplexLinear, ComplexAttention, MagnitudeLN  # noqa: E402
from soliton import SolitonCell, per_dim_phase_gap                                  # noqa: E402

_POOLS = _ROOT / "data" / "hidden_cache" / "gsm8k_test_pools.pt"


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_candidates(pool="lastk"):
    """Return (X: (N,1536) float, y: (N,) {1=correct,0=incorrect})."""
    c = torch.load(_POOLS, weights_only=False)
    X, y = [], []
    for e in c.values():
        vecs = e["candidates"][pool] if isinstance(e["candidates"], dict) else e["candidates"]
        for i, corr in enumerate(e["cand_correct"]):
            X.append(vecs[i]); y.append(1 if corr else 0)
    return torch.stack(X).float(), torch.tensor(y)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ComplexCoevolve(nn.Module):
    def __init__(self, d_in=1536, d=128, n_heads=4, coevolve_steps=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.lift = ComplexLift(d)
        self.manifold_attn = ComplexAttention(d, n_heads)   # encode a class's seq
        self.norm = MagnitudeLN(d)
        self.soliton = SolitonCell(detach_gate=True)
        self.steps = coevolve_steps
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode_manifold(self, support):                     # support: (S, d_in) real
        z = self.lift(self.proj(support)).unsqueeze(0)      # (1,S,d) complex
        z = self.norm(z + self.manifold_attn(z))            # let class items interact
        return z.mean(1)                                     # (1,d) class manifold rep

    def coevolve(self, m_pos, m_neg):
        psi = torch.stack([m_pos, m_neg], dim=1).unsqueeze(2)   # (1,2,1,d)
        for _ in range(self.steps):
            psi = self.soliton(psi)
        return psi[:, 0, 0], psi[:, 1, 0]                       # (1,d), (1,d)

    def query_logit(self, q, m_pos, m_neg):                 # q: (Q, d_in)
        qz = self.lift(self.proj(q))                        # (Q,d) complex
        e_pos = (qz + m_pos).abs().pow(2).mean(-1)          # constructive interference
        e_neg = (qz + m_neg).abs().pow(2).mean(-1)
        return self.scale * (e_pos - e_neg) + self.bias     # (Q,)

    def forward(self, support_pos, support_neg, q, return_sep=False):
        m_pos = self.encode_manifold(support_pos)
        m_neg = self.encode_manifold(support_neg)
        sep_before = (m_pos - m_neg).abs().mean().item()
        m_pos, m_neg = self.coevolve(m_pos, m_neg)
        sep_after = (m_pos - m_neg).abs().mean().item()
        logit = self.query_logit(q, m_pos, m_neg)
        if return_sep:
            return logit, sep_before, sep_after
        return logit


class RealProto(nn.Module):
    def __init__(self, d_in=1536, d=128, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode_manifold(self, support):
        h = self.proj(support).unsqueeze(0)                 # (1,S,d)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)
        return h.mean(1)                                     # (1,d)

    def query_logit(self, q, m_pos, m_neg):
        qh = self.proj(q)
        e_pos = -((qh - m_pos) ** 2).mean(-1)               # negative sq distance
        e_neg = -((qh - m_neg) ** 2).mean(-1)
        return self.scale * (e_pos - e_neg) + self.bias

    def forward(self, support_pos, support_neg, q):
        m_pos = self.encode_manifold(support_pos)
        m_neg = self.encode_manifold(support_neg)
        return self.query_logit(q, m_pos, m_neg)


class PlainMLP(nn.Module):
    def __init__(self, d_in=1536, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, support_pos, support_neg, q):
        return self.net(q).squeeze(-1)


# ---------------------------------------------------------------------------
# Episodic training + eval
# ---------------------------------------------------------------------------
def sample(idx, k, rng):
    return idx[rng.randint(0, len(idx), size=k)]


def train(model, Xtr, ytr, dev, steps=1500, S=32, Q=64, lr=3e-4, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(ytr.numpy() == 1)[0]
    neg = np.where(ytr.numpy() == 0)[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for _ in range(steps):
        sp = Xtr[sample(pos, S, rng)].to(dev)
        sn = Xtr[sample(neg, S, rng)].to(dev)
        qi = np.concatenate([sample(pos, Q // 2, rng), sample(neg, Q // 2, rng)])
        q = Xtr[qi].to(dev); ql = torch.tensor(ytr.numpy()[qi]).float().to(dev)
        logit = model(sp, sn, q)
        loss = F.binary_cross_entropy_with_logits(logit, ql)
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            opt.zero_grad(); continue
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


@torch.no_grad()
def evaluate(model, Xtr, ytr, Xte, yte, dev, S=256):
    model.eval()
    pos = np.where(ytr.numpy() == 1)[0][:S]
    neg = np.where(ytr.numpy() == 0)[0][:S]
    sp = Xtr[pos].to(dev); sn = Xtr[neg].to(dev)
    q = Xte.to(dev)
    sep_b = sep_a = None
    if isinstance(model, ComplexCoevolve):
        logit, sep_b, sep_a = model(sp, sn, q, return_sep=True)
    else:
        logit = model(sp, sn, q)
    s = logit.float().cpu().numpy()
    auc = roc_auc_score(yte.numpy(), s)
    return auc, sep_b, sep_a


CONFIGS = {
    "complex_coevolve": lambda: ComplexCoevolve(),
    "real_proto":       lambda: RealProto(),
    "plain_mlp":        lambda: PlainMLP(),
}


def run_seed(seed, X, y, steps, dev):
    g = np.random.RandomState(seed)
    n = len(y); perm = g.permutation(n); cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    out = {}
    for name, build in CONFIGS.items():
        torch.manual_seed(seed); np.random.seed(seed)
        m = build().to(dev)
        train(m, Xtr, ytr, dev, steps=steps, seed=seed)
        auc, sb, sa = evaluate(m, Xtr, ytr, Xte, yte, dev)
        out[name] = auc
        extra = f"  (manifold sep {sb:.3f} -> {sa:.3f})" if sb is not None else ""
        print(f"    seed{seed} {name:<17} AUC={auc:.3f}{extra}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.steps = [0], 150

    dev = _device()
    print(f"[manifold] device={dev}  loading candidate signals ...")
    X, y = load_candidates("lastk")
    print(f"[manifold] {len(y)} candidates ({int(y.sum())} correct / "
          f"{int((1-y).sum())} incorrect), d_in={X.shape[1]}\n")

    per_seed = {}
    for s in args.seeds:
        print(f"--- seed {s} ---")
        per_seed[s] = run_seed(s, X, y, args.steps, dev)

    print("\n" + "=" * 60)
    print(f"AUC (mean ± std over seeds {args.seeds})")
    agg = {}
    for name in CONFIGS:
        v = np.array([per_seed[s][name] for s in args.seeds])
        agg[name] = (v.mean(), v.std())
        print(f"  {name:<17} {v.mean():.3f} ± {v.std():.3f}")
    print("-" * 60)
    cc, rp, mlp = agg["complex_coevolve"][0], agg["real_proto"][0], agg["plain_mlp"][0]
    if cc > rp and cc > mlp:
        print(f"  -> complex co-evolve ({cc:.3f}) beats real_proto ({rp:.3f}) AND "
              f"plain_mlp ({mlp:.3f}): the class-manifold interference idea WORKS.")
    else:
        print(f"  -> complex co-evolve ({cc:.3f}) does NOT beat both "
              f"(real {rp:.3f}, mlp {mlp:.3f}): ties/loses, consistent with prior.")
    print("=" * 60)


if __name__ == "__main__":
    main()
