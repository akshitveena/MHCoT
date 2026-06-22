"""
solver/stage1_meaning.py  —  STAGE 1: the meaning probe
=======================================================

Can co-evolving chains carry DISTINCT, USEFUL meaning? We give the signals real
content (a Game-24 state) and a multi-dimensional target (the FULL set of good
first-moves), read the chains out SEPARATELY (never summed), and measure whether
two chains cover MORE of the good-move set than one — the diversity the scalar
readout could never express.

The four-way honest comparison:
    N2            complex, 2 chains, ε-helix diversity (the MHCoT thesis)
    N1            complex, 1 chain (no multiplicity)
    real2_naive   real, 2 heads, NO diversity pressure
    real2_diverse real, 2 heads + decorrelation loss   <-- the STRONG control

Metric: coverage@K of the good-move set, via a round-robin union of the per-chain
(per-head) rankings of a state's successors. If N2 beats real2_diverse, the
ε-helix produces better diversity than optimizing for it directly.

Folds in the Stage-0 fix: output-normalize each co-evolution step (scale control).

Run:
    python solver/stage1_meaning.py --smoke
    python solver/stage1_meaning.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "main"))
sys.path.insert(0, str(_HERE))

from complex_ops import ComplexLift, ComplexAttention, modReLU, MagnitudeLN  # noqa: E402
from soliton import SolitonCell, epsilon_helix_loss                          # noqa: E402
from state_encoder import StateEncoder                                       # noqa: E402
from game24 import make_split                                                # noqa: E402
from search import label_states                                             # noqa: E402
from multianswer import build_multianswer_set, coverage_at_k, union_topk     # noqa: E402


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Complex chain model with PER-CHAIN readout (no sum)
# ---------------------------------------------------------------------------
class ChainModel(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_chains=2, n_layers=2,
                 eps_min=1.15):
        super().__init__()
        self.n_chains = n_chains
        self.enc = StateEncoder(d_model=d_model)
        self.lift = ComplexLift(d_model)
        self.chain_phase = nn.Parameter(torch.randn(n_chains, d_model) * 0.1)
        self.attn = nn.ModuleList(ComplexAttention(d_model, n_heads)
                                  for _ in range(n_layers))
        self.act = nn.ModuleList(modReLU(d_model) for _ in range(n_layers))
        self.norm = nn.ModuleList(MagnitudeLN(d_model) for _ in range(n_layers))
        self.soliton = SolitonCell(eps_min=eps_min, detach_gate=True)
        self.out_norm = MagnitudeLN(d_model)            # Stage-0 fix: scale control
        # one readout head PER chain (separate -> rankings can differ)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                          nn.Linear(d_model, 1)) for _ in range(n_chains))

    def forward(self, states, targets):
        H, mask = self.enc(states, targets)
        B, T, D = H.shape
        z = self.lift(H)
        phase = self.chain_phase.view(1, self.n_chains, 1, D)
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        psi = z.unsqueeze(1) * rot                       # (B,N,T,D)

        am = mask.unsqueeze(1).expand(B, T, T)
        am = am.unsqueeze(1).expand(B, self.n_chains, T, T).reshape(B * self.n_chains, T, T)
        for attn, act, norm in zip(self.attn, self.act, self.norm):
            flat = psi.reshape(B * self.n_chains, T, D)
            flat = flat + attn(flat, mask=am)
            flat = norm(act(flat))
            psi = flat.reshape(B, self.n_chains, T, D)
        psi = self.soliton(psi)
        psi = self.out_norm(psi)                          # scale-controlled output

        valid = (~mask).float().view(B, 1, T, 1)
        scores = []
        for i in range(self.n_chains):
            mag = psi[:, i].abs()                         # (B,T,D)
            pooled = (mag * valid[:, 0]).sum(1) / valid[:, 0].sum(1).clamp_min(1.0)
            scores.append(self.heads[i](pooled).squeeze(-1))
        return {"chain_scores": torch.stack(scores, 1), "psi": psi}   # (B,N)


# ---------------------------------------------------------------------------
# Real two-head baseline (naive or diversity-regularized)
# ---------------------------------------------------------------------------
class RealTwoHead(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, n_outputs=2):
        super().__init__()
        self.n_chains = n_outputs
        self.enc = StateEncoder(d_model=d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dim_feedforward=2 * d_model,
                                           batch_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers,
                                        enable_nested_tensor=False)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                          nn.Linear(d_model, 1)) for _ in range(n_outputs))

    def forward(self, states, targets):
        H, mask = self.enc(states, targets)
        h = self.tf(H, src_key_padding_mask=mask)
        valid = (~mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        scores = [head(pooled).squeeze(-1) for head in self.heads]
        return {"chain_scores": torch.stack(scores, 1), "psi": None}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def balanced_batches(states, targets, labels, bs, steps, rng):
    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]
    half = bs // 2
    for _ in range(steps):
        idx = [rng.choice(pos) for _ in range(half)] + \
              [rng.choice(neg) for _ in range(bs - half)]
        rng.shuffle(idx)
        yield ([states[i] for i in idx], [targets[i] for i in idx],
               torch.tensor([float(labels[i]) for i in idx]))


def decorrelation(scores):
    """Penalize correlation between the two heads' outputs (push diversity)."""
    s0, s1 = scores[:, 0], scores[:, 1]
    s0 = s0 - s0.mean(); s1 = s1 - s1.mean()
    denom = (s0.norm() * s1.norm()).clamp_min(1e-6)
    return ((s0 * s1).sum() / denom).pow(2)


def train(model, data, dev, steps, kind, seed, lam_eps=0.05, lam_div=0.3,
          warmup_frac=0.4, bs=64, lr=3e-4):
    states, targets, labels = data
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = int(warmup_frac * steps)
    model.train()
    for step, (bs_, bt, by) in enumerate(balanced_batches(states, targets, labels, bs, steps, rng)):
        by = by.to(dev)
        out = model(bs_, bt)
        sc = out["chain_scores"]                         # (B,N)
        # each chain/head predicts the label
        loss = sum(F.binary_cross_entropy_with_logits(sc[:, i], by)
                   for i in range(sc.shape[1])) / sc.shape[1]
        if kind == "complex_n2" and step >= warmup:
            loss = loss + lam_eps * epsilon_helix_loss(out["psi"])
        if kind == "real2_diverse":
            loss = loss + lam_div * decorrelation(sc)
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(); loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters())
        if not finite:
            opt.zero_grad(); continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


# ---------------------------------------------------------------------------
# Coverage evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def coverage(model, ma_set, dev, K=3):
    model.eval()
    covs = []
    for d in ma_set:
        succ = d["successors"]
        if not succ:
            continue
        out = model(succ, [d["target"]] * len(succ))
        sc = out["chain_scores"].cpu().numpy()           # (n_succ, N)
        rankings = []
        for i in range(sc.shape[1]):
            order = list(np.argsort(-sc[:, i]))
            rankings.append([succ[j] for j in order])
        proposals = union_topk(rankings, K)
        c = coverage_at_k(proposals, d["good"], K)
        if c is not None:
            covs.append(c)
    return float(np.mean(covs)) if covs else 0.0


CONFIGS = {
    "complex_n2":   lambda: ChainModel(n_chains=2),
    "complex_n1":   lambda: ChainModel(n_chains=1),
    "real2_naive":  lambda: RealTwoHead(n_outputs=2),
    "real2_diverse": lambda: RealTwoHead(n_outputs=2),
}


def run_seed(seed, n_train, n_test, steps, K, dev):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    train_pz, test_pz = make_split(n_train, n_test, k=4, target=24, seed=seed)
    data = ([s for s, _, _ in label_states(train_pz)],
            [t for _, t, _ in label_states(train_pz)],
            [l for _, _, l in label_states(train_pz)])
    ma = build_multianswer_set(test_pz)
    out = {}
    for name, build in CONFIGS.items():
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        m = build().to(dev)
        train(m, data, dev, steps, name, seed)
        cov = coverage(m, ma, dev, K=K)
        out[name] = cov
        print(f"    seed{seed} {name:<14} coverage@{K} = {cov:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.n_train, args.n_test, args.steps = [0], 40, 30, 80

    dev = _device()
    print(f"[stage1] device={dev} seeds={args.seeds} steps={args.steps} K={args.K}\n")
    t0 = time.time()
    per_seed = {}
    for s in args.seeds:
        print(f"--- seed {s} ---")
        per_seed[s] = run_seed(s, args.n_train, args.n_test, args.steps, args.K, dev)

    print("\n" + "=" * 60)
    print(f"COVERAGE@{args.K}  (mean ± std over seeds {args.seeds})")
    agg = {}
    for name in CONFIGS:
        vals = np.array([per_seed[s][name] for s in args.seeds])
        agg[name] = (float(vals.mean()), float(vals.std()))
        print(f"  {name:<14} {vals.mean():.3f} ± {vals.std():.3f}")
    print("-" * 60)
    n2 = agg["complex_n2"][0]
    print("Q (meaning): does N2 cover more good moves than the baselines?")
    print(f"   N2 {n2:.3f}  vs  N1 {agg['complex_n1'][0]:.3f}  "
          f"real2_naive {agg['real2_naive'][0]:.3f}  "
          f"real2_diverse {agg['real2_diverse'][0]:.3f}")
    if n2 > max(agg['complex_n1'][0], agg['real2_naive'][0], agg['real2_diverse'][0]):
        print("   -> N2 beats ALL incl. real2_diverse: ε-helix diversity is REAL. "
              "Stage 2 (RL) justified.")
    elif n2 > agg['real2_naive'][0] and n2 >= agg['complex_n1'][0]:
        print("   -> N2 helps vs naive baselines but ties real2_diverse: any diversity "
              "mechanism works, ε-helix not special. Honest partial.")
    else:
        print("   -> N2 does NOT carry distinct meaning better than baselines. "
              "The vision is failing the cheap gate.")
    print("=" * 60)
    print(f"[done] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
