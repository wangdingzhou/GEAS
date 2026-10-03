"""Stable-LoRA optimizer hook (paper-faithful port with explicit pairing).

Reference: "Stable-LoRA: Pushing the Limits of Out-of-Domain Generalisation"
       (ICLR 2026), arXiv:2411.02505; official repo, stable_lora.py

Core idea (paper §3):
  LoRA's two projection matrices A, B start with imbalanced norms (A Kaiming,
  B zero). During training this imbalance can persist and harm stability.
  Stable-LoRA adds a multiplicative shrinkage on the *larger-norm* matrix
  (typically the Kaiming-init one) so that its norm stays close (per-element)
  to the zero-init one.

  Per-step rule (released code line 226-243):
      numel_ratio = numel(B) / numel(A)        # B = zero, A = Kaiming
      if ||A||_F < numel_ratio * ||B||_F:
          keep_shrink = False                 # PERMANENT stop, never resume
      else:
          A.mul_(1 - shrink_ratio)             # shrink

Why explicit pairing (user principle "don't trust index-based binding"):
  Released code binds shrink target to "param group index 0" (the first group).
  This is fragile: it depends on the optimizer's group ordering and silently
  breaks if the user re-orders groups or mixes LoRA with other param families.
  We bind by **(shrink_param, paired_param) reference pairs** captured at
  module-construction time. Binding rule: Kaiming-init matrix = shrink target,
  zero-init matrix = paired reference.

Our SSD-LoRA convention (verified src/model.py):
  lora_sem_A = nn.Parameter(empty(d_out, r_sem))       # Kaiming-UP  -> shrink
  lora_sem_B = nn.Parameter(zeros(r_sem, d_in))        # zero-DOWN   -> paired
  lora_str_A = nn.Parameter(empty(d_out, r_str))       # Kaiming-UP  -> shrink
  lora_str_B = nn.Parameter(zeros(r_str, d_in))        # zero-DOWN   -> paired

  NOTE: our naming is REVERSED vs PEFT standard. In PEFT, lora_A is Kaiming-DOWN
  with shape (r, d_in) and lora_B is zero-UP with shape (d_out, r). In our code,
  lora_*_A is Kaiming-UP with shape (d_out, r) and lora_*_B is zero-DOWN with
  shape (r, d_in). The Kaiming property (non-zero init) is preserved, but the
  shape (and therefore numel) attached to "A" is different.

numel_ratio translation:
  Reference (PEFT): numel_ratio = numel(zero-UP) / numel(Kaiming-DOWN)
                            = (d_out * r) / (d_in * r) = d_out / d_in
  Our convention:   to preserve the d_out/d_in numeric value, we use
                    numel_ratio = numel(Kaiming-UP) / numel(zero-DOWN)
                                = (d_out * r) / (d_in * r) = d_out / d_in  ✓
  That is, in our code the ratio is shrink.numel / paired.numel (NOT the
  inverse). Both numerator and denominator use the matrix actually assigned
  to each role.

Usage (preferred — optimizer pre-hook, AMP-safe):
  hook = StableLoRAShrinkageHook(model, shrink_ratio=1e-3)
  optimizer.register_step_pre_hook(
      lambda opt, args, kwargs: hook.pre_step_hook()
  )
  # The pre-hook fires only when optimizer.step() actually executes,
  # so it correctly skips on AMP inf/NaN steps when GradScaler skips.

Usage (legacy — inline call, NOT AMP-safe):
  hook.pre_step_hook()  # caller must guard against AMP skip steps
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StableLoRAShrinkageHook:
    """Pre-optimizer-step shrinkage for SSD-LoRA Kaiming matrices.

    Collects (shrink_target, paired_param) pairs from all SSDLoRAModule
    instances in the model. On each pre-step, applies Stable-LoRA shrinkage
    rule (released-code simplified form, with permanent stop flag).

    Args:
        model: top-level model containing SSDLoRAModule instances.
        shrink_ratio: per-step multiplicative shrink ratio (default 1e-3,
            matches released code).
        verbose: if True, prints #pairs bound on init.
    """

    def __init__(
        self,
        model: nn.Module,
        shrink_ratio: float = 1e-3,
        verbose: bool = True,
    ):
        self.shrink_ratio = shrink_ratio
        self.pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        # Per-pair permanent stop flag (released code's keep_shrink=False).
        # Once True → never check or shrink this pair again.
        self.stopped: list[bool] = []

        for mod in model.modules():
            for shrink_name, paired_name in (
                ("lora_sem_A", "lora_sem_B"),
                ("lora_str_A", "lora_str_B"),
            ):
                shrink_param = getattr(mod, shrink_name, None)
                paired_param = getattr(mod, paired_name, None)
                if shrink_param is None or paired_param is None:
                    continue
                if not isinstance(shrink_param, nn.Parameter):
                    continue
                if not isinstance(paired_param, nn.Parameter):
                    continue
                self.pairs.append((shrink_param, paired_param))
                self.stopped.append(False)

        if verbose:
            print(f"Stable-LoRA: bound {len(self.pairs)} (shrink, paired) pairs")

    @torch.no_grad()
    def pre_step_hook(self) -> None:
        """Apply shrinkage before optimizer.step().

        Stable-LoRA rule (released-code translated to our convention):
          For each (shrink=K, paired=Z) pair where K=Kaiming, Z=zero-init:
            if permanently stopped: skip
            numel_ratio = numel(K) / numel(Z)        # = d_out / d_in
            if ||K||_F < numel_ratio * ||Z||_F:
                mark pair as permanently stopped (keep_shrink=False)
            else:
                K.mul_(1 - shrink_ratio)
        """
        for i, (k_param, z_param) in enumerate(self.pairs):
            if self.stopped[i]:
                continue
            # numel_ratio preserves the reference's d_out/d_in numeric value
            # (see module docstring "numel_ratio translation").
            numel_ratio = k_param.numel() / max(z_param.numel(), 1)
            k_norm = k_param.detach().norm().item()
            z_norm = z_param.detach().norm().item()
            if k_norm < numel_ratio * z_norm:
                # Kaiming matrix is now small enough relative to zero matrix
                # → permanently stop shrinking this pair.
                self.stopped[i] = True
                continue
            k_param.mul_(1.0 - self.shrink_ratio)


__all__ = ["StableLoRAShrinkageHook"]
