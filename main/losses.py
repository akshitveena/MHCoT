"""
main/losses.py
==============

The training objective for Option 1 (EQ-E), combining:

  L_task  = BCEWithLogits(score, label)            supervised correctness
  L_ε     = helix loss (phase gap ≥ ε_min)         the novelty — drives chains apart
  L_wave  = interference calibration               high I on correct, low on wrong

Total:  L = L_task + λ_ε·L_ε + λ_wave·L_wave,  introduced STAGED so training
stabilizes (task first, then helix, then wave).

`compute_losses` takes the MHCoTEncoder output dict + labels + step + a config
and returns (total, components_dict) — components for logging (so we can watch
each term and the chain divergence separately).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from soliton import epsilon_helix_loss


@dataclass
class LossConfig:
    eps_min: float = 1.15
    lambda_eps: float = 0.10
    lambda_wave: float = 0.05
    # staged introduction (in optimizer steps)
    stage_eps_step: int = 500
    stage_wave_step: int = 1500
    answer_tail: int = 16     # tokens at the end treated as the answer region


def wave_loss(I_answer: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Reward high interference on correct samples, low on wrong ones.
    `I_answer` (B,) is the model's answer-region mean interference (padding-
    correct). We standardize it across the batch (zero-mean/unit-var) so the
    loss is scale-stable, then use it as a logit predicting correctness.
    """
    mu, sd = I_answer.mean(), I_answer.std() + 1e-6
    logit = (I_answer - mu) / sd
    return F.binary_cross_entropy_with_logits(logit, label.float())


def compute_losses(
    out: dict,
    label: torch.Tensor,
    step: int,
    cfg: LossConfig,
) -> tuple[torch.Tensor, dict]:
    """
    out:   MHCoTEncoder forward output (score, I_t, psi, ...)
    label: (B,) float/int correctness labels (1 = correct)
    step:  current optimizer step (for staged introduction)
    Returns (total_loss, components) where components are python floats.
    """
    label = label.float()
    score = out["score"]                                   # (B,)

    L_task = F.binary_cross_entropy_with_logits(score, label)
    total = L_task
    comp = {"L_task": L_task.item()}

    if step >= cfg.stage_eps_step:
        L_eps = epsilon_helix_loss(out["psi"], eps_min=cfg.eps_min)
        total = total + cfg.lambda_eps * L_eps
        comp["L_eps"] = L_eps.item()
    else:
        comp["L_eps"] = 0.0

    if step >= cfg.stage_wave_step:
        L_wave = wave_loss(out["I_answer"], label)
        total = total + cfg.lambda_wave * L_wave
        comp["L_wave"] = L_wave.item()
    else:
        comp["L_wave"] = 0.0

    comp["L_total"] = total.item()
    comp["chain_divergence"] = out.get("chain_divergence", 0.0)
    return total, comp


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import MHCoTEncoder

    dev = _device()
    print(f"[device] {dev}\n")
    D_IN, D, B, T = 1536, 128, 8, 24
    model = MHCoTEncoder(d_in=D_IN, d_model=D).to(dev)
    h = torch.randn(B, T, D_IN, device=dev)
    label = torch.randint(0, 2, (B,), device=dev)
    cfg = LossConfig(stage_eps_step=0, stage_wave_step=0)   # all active for the test

    out = model(h)
    total, comp = compute_losses(out, label, step=10_000, cfg=cfg)
    print("[test] all-stage loss ...")
    print(f"    {comp}")
    assert torch.isfinite(total)
    total.backward()
    assert model.lift.W_imag.grad is not None
    print("    ok — total finite, grads flow")

    print("[test] staged gating ...")
    out2 = model(h)
    _, c0 = compute_losses(out2, label, step=0, cfg=LossConfig())     # only task
    assert c0["L_eps"] == 0.0 and c0["L_wave"] == 0.0
    out3 = model(h)
    _, c1 = compute_losses(out3, label, step=1000, cfg=LossConfig())  # task+eps
    assert c1["L_eps"] != 0.0 and c1["L_wave"] == 0.0
    print(f"    ok — step0: task only | step1000: task+eps | (wave at 1500+)")

    print("\n[all loss tests passed]")


if __name__ == "__main__":
    main()
