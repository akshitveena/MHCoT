"""
solver/stage1_recurrent.py  —  the CORRECT co-evolving architecture, on a task
==============================================================================

The gap we found: every N1-vs-N2 task comparison used SHALLOW co-evolution (1–2
soliton coupling layers over network depth). The architecture we actually
theorized as correct — the recurrent CoEvolveCell (multi-step TEMPORAL
co-evolution, state carried across steps, output-norm fix; verified healthy but
TASKLESS in Stage 0) — was never put head-to-head on a task.

This is that test. Same multi-answer coverage task and metric as stage1_meaning,
but the N=2 core is the recurrent CoEvolveCell rolled K steps. Matched recurrent
N=1 (single complex chain) and real (2-head) baselines.

This is the LAST untested form of the architecture: after this there is no
"but we didn't test the real co-evolution" left.

Pre-committed bar: complex_n2_recurrent > complex_n1_recurrent AND > real baselines
on held-out coverage@K, across seeds. Otherwise the architecture — in its proper
recurrent form — is finished.

Run:
    python solver/stage1_recurrent.py --smoke
    python solver/stage1_recurrent.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "main"))
sys.path.insert(0, str(_HERE))

from complex_ops import ComplexLift, ComplexAttention, MagnitudeLN          # noqa: E402
from coevolve import CoEvolveCell                                           # noqa: E402
from state_encoder import StateEncoder                                      # noqa: E402
from game24 import make_split                                               # noqa: E402
from search import label_states                                            # noqa: E402
from multianswer import build_multianswer_set                              # noqa: E402
# reuse the exact task harness so the ONLY change is the co-evolution core
from stage1_meaning import train, coverage, _device                        # noqa: E402


class RecurrentChainModel(nn.Module):
    """N complex chains co-evolved by the RECURRENT CoEvolveCell over K steps."""

    def __init__(self, d_model=64, n_heads=4, n_chains=2, coevolve_steps=6,
                 eps_min=1.15):
        super().__init__()
        self.n_chains = n_chains
        self.steps = coevolve_steps
        self.enc = StateEncoder(d_model=d_model)
        self.lift = ComplexLift(d_model)
        self.token_attn = ComplexAttention(d_model, n_heads)
        self.token_norm = MagnitudeLN(d_model)
        self.chain_phase = nn.Parameter(torch.randn(n_chains, d_model) * 0.1)
        self.cell = CoEvolveCell(d_model, n_chains=n_chains, eps_min=eps_min)
        self.out_norm = MagnitudeLN(d_model)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                          nn.Linear(d_model, 1)) for _ in range(n_chains))

    def forward(self, states, targets):
        H, mask = self.enc(states, targets)                 # (B,T,D),(B,T)
        B, T, D = H.shape
        z = self.lift(H)                                     # (B,T,D) complex
        am = mask.unsqueeze(1).expand(B, T, T)
        z = self.token_norm(z + self.token_attn(z, mask=am))
        valid = (~mask).float().unsqueeze(-1)
        zp = (z * valid).sum(1) / valid.sum(1).clamp_min(1.0)   # (B,D) pooled
        phase = self.chain_phase.view(1, self.n_chains, D)
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        psi = zp.unsqueeze(1) * rot                          # (B,N,D)
        for _ in range(self.steps):                          # RECURRENT co-evolution
            psi = self.cell(psi)
        psi = self.out_norm(psi)                             # (B,N,D)
        scores = [self.heads[i](psi[:, i].abs()).squeeze(-1)
                  for i in range(self.n_chains)]
        return {"chain_scores": torch.stack(scores, 1),
                "psi": psi.unsqueeze(2)}                     # (B,N,1,D) for helix loss


class RealRecurrent(nn.Module):
    """Matched real baseline: token attention -> pool -> recurrent residual cell
    rolled K steps -> n_outputs heads."""

    def __init__(self, d_model=64, n_heads=4, coevolve_steps=6, n_outputs=2):
        super().__init__()
        self.n_chains = n_outputs
        self.steps = coevolve_steps
        self.enc = StateEncoder(d_model=d_model)
        self.token_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.token_norm = nn.LayerNorm(d_model)
        self.cell = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.cell_norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                          nn.Linear(d_model, 1)) for _ in range(n_outputs))

    def forward(self, states, targets):
        H, mask = self.enc(states, targets)
        a, _ = self.token_attn(H, H, H, key_padding_mask=mask)
        h = self.token_norm(H + a)
        valid = (~mask).float().unsqueeze(-1)
        hp = (h * valid).sum(1) / valid.sum(1).clamp_min(1.0)   # (B,D)
        for _ in range(self.steps):                              # recurrent
            hp = self.cell_norm(hp + self.cell(hp))
        scores = [head(hp).squeeze(-1) for head in self.heads]
        return {"chain_scores": torch.stack(scores, 1), "psi": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--coevolve_steps", type=int, default=6)
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.n_train, args.n_test, args.steps = [0], 40, 30, 80

    cs = args.coevolve_steps
    CONFIGS = {
        "complex_n2_recurrent": (lambda: RecurrentChainModel(n_chains=2, coevolve_steps=cs), "complex_n2"),
        "complex_n1_recurrent": (lambda: RecurrentChainModel(n_chains=1, coevolve_steps=cs), "complex_n1"),
        "real2_recurrent":      (lambda: RealRecurrent(n_outputs=2, coevolve_steps=cs), "real2_naive"),
        "real2_diverse":        (lambda: RealRecurrent(n_outputs=2, coevolve_steps=cs), "real2_diverse"),
    }

    dev = _device()
    print(f"[recurrent] device={dev} seeds={args.seeds} steps={args.steps} "
          f"coevolve_steps={cs} K={args.K}\n")
    per_seed = {}
    for s in args.seeds:
        print(f"--- seed {s} ---")
        torch.manual_seed(s); np.random.seed(s)
        train_pz, test_pz = make_split(args.n_train, args.n_test, k=4, target=24, seed=s)
        labeled = label_states(train_pz)
        data = ([x[0] for x in labeled], [x[1] for x in labeled], [x[2] for x in labeled])
        ma = build_multianswer_set(test_pz)
        out = {}
        for name, (build, kind) in CONFIGS.items():
            torch.manual_seed(s); np.random.seed(s)
            m = build().to(dev)
            train(m, data, dev, args.steps, kind, s)
            cov = coverage(m, ma, dev, K=args.K)
            out[name] = cov
            print(f"    seed{s} {name:<22} coverage@{args.K} = {cov:.3f}")
        per_seed[s] = out

    print("\n" + "=" * 64)
    print(f"COVERAGE@{args.K}  (mean ± std over seeds {args.seeds})  [RECURRENT co-evolution]")
    agg = {}
    for name in CONFIGS:
        v = np.array([per_seed[s][name] for s in args.seeds])
        agg[name] = (v.mean(), v.std())
        print(f"  {name:<22} {v.mean():.3f} ± {v.std():.3f}")
    print("-" * 64)
    n2 = agg["complex_n2_recurrent"][0]
    others = max(agg["complex_n1_recurrent"][0], agg["real2_recurrent"][0],
                 agg["real2_diverse"][0])
    if n2 > others:
        print(f"  -> recurrent N2 ({n2:.3f}) BEATS all baselines ({others:.3f}): the "
              f"correct co-evolution finally shows an effect. Investigate.")
    else:
        print(f"  -> recurrent N2 ({n2:.3f}) does NOT beat baselines ({others:.3f}): "
              f"even the proper recurrent co-evolution adds nothing. The architecture "
              f"is finished — no untested form remains.")
    print("=" * 64)


if __name__ == "__main__":
    main()
