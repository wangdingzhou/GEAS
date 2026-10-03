"""Conv-LoRA MoE module (AutoGluon-code-faithful port).

Reference: autogluon/multimodal/models/adaptation_layers.py:606-827
Paper: Zhong et al. 2024, "Convolution Meets LoRA" (ICLR 2024)
       https://arxiv.org/abs/2401.17868

Used as the structural-path spatial processor in SSDLoRAModule
(replaces MultiScaleDWConv / single DWConv when
``structural_path_type == "conv_lora_moe"``).

Byte-faithful details verified from released code:
  - expert = nn.Conv2d(r, r, 3, padding=1, bias=True) + nn.GELU()  (NOT depthwise)
  - upsample_ratios = list(range(1, n+1))                          (consecutive 1..n)
  - F.interpolate(..., mode="bicubic")                             (both up and down)
  - residual add: lora_res = lora_res + temp_lora_res
  - MoEGate: noisy_gating=True, gap pool input, top-K=1, loss_coef=1e-2

Deviation from released code (documented):
  - We replace SparseDispatcher with direct gather/scatter for top-K=1.
    Released code: dispatcher.dispatch -> per-expert forward -> dispatcher.combine
    with combine doing ``exp -> mul gate -> index_add -> log``. For top-K=1 this
    is mathematically ``expert_output[b, chosen_expert]`` (gate is *not* applied
    because parent sets multiply_by_gates=False, see line 670 of reference).
    Our implementation produces the same tensor without the exp/log round-trip,
    avoiding numerical issues for zero conv outputs.
  - Released code returns ``(result, moe_loss)`` as a tuple; we instead store
    ``self.last_moe_loss`` so the parent module's forward contract (single-tensor
    return) is preserved. ``HALoRASeg.get_aux_losses()`` aggregates these.

Forward contract:
  Input  h2d: (B, r, H, W)
  Output h2d: (B, r, H, W)  -- residual-added input
  Side effect: self.last_moe_loss is set to a scalar tensor (load balancing loss)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


# ---------------------------------------------------------------------------
# MoE Gate (Shazeer-style noisy top-K gating, byte-faithful to AutoGluon)
# ---------------------------------------------------------------------------

class MoEGate(nn.Module):
    """Noisy top-K gating for MoE-Conv.

    Args:
        d: input channel dim (r in LoRA bottleneck).
        M: number of experts.
        K: top-K experts chosen per sample (paper uses K=1).
        noisy_gating: if True, add learned noise to logits during training.
    """

    def __init__(self, d: int, M: int = 4, K: int = 1, noisy_gating: bool = True):
        super().__init__()
        self.d = d
        self.M = M
        self.k = K
        self.noisy_gating = noisy_gating

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.M

    def forward(
        self,
        feats: torch.Tensor,
        loss_coef: float = 1e-2,
        noise_epsilon: float = 1e-2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """feats: (B, r, H, W) -> gates (B, M), loss (scalar).

        gates[b, e] is the softmax weight of expert e if e is in top-K else 0.
        """
        batch_size = feats.shape[0]
        feats_S = self.gap(feats).view(batch_size, -1)  # (B, r)

        clean_logits = feats_S @ self.w_gate  # (B, M)
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_S @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # top-K selection
        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)  # (B, M)

        # load balancing loss
        if self.noisy_gating and self.k < self.M and self.training:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss = loss * loss_coef

        return gates, loss

    def _gates_to_load(self, gates: torch.Tensor) -> torch.Tensor:
        return (gates > 0).sum(0).float()

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(
        self,
        clean_values: torch.Tensor,
        noisy_values: torch.Tensor,
        noise_stddev: torch.Tensor,
        noisy_top_values: torch.Tensor,
    ) -> torch.Tensor:
        """Probability that each value is in the top-K under noisy gating.

        Shape: (B, M). Same formula as Shazeer 2017 AutoGluon port.
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


# ---------------------------------------------------------------------------
# Conv-LoRA MoE spatial processor
# ---------------------------------------------------------------------------

class ConvLoRAMoE(nn.Module):
    """MoE-Conv over the LoRA bottleneck feature map.

    Input/output shape: (B, r, H, W) -> (B, r, H, W). Output is residual-added
    input (matches AutoGluon line 720: ``lora_res = lora_res + temp_lora_res``).

    Side effect: ``self.last_moe_loss`` is set on every forward (scalar tensor,
    load-balancing loss already scaled by loss_coef=1e-2). Caller is responsible
    for adding it to the main loss via ``HALoRASeg.get_aux_losses()``.

    Args:
        channels: r, the LoRA bottleneck dim.
        n_experts: number of MoE-Conv experts (AutoGluon default 8).
        top_k: number of experts chosen per sample (paper uses 1).
        upsample_ratios: per-expert upsample factors. If None, defaults to
            ``list(range(1, n_experts+1))`` (AutoGluon default).
        loss_coef: coefficient for MoE load balancing loss (1e-2 in AutoGluon).
    """

    def __init__(
        self,
        channels: int,
        n_experts: int = 8,
        top_k: int = 1,
        upsample_ratios: Optional[list] = None,
        loss_coef: float = 1e-2,
    ):
        super().__init__()
        assert top_k == 1, (
            "Direct-index routing (our simplification) only supports top_k=1. "
            "Use SparseDispatcher from AutoGluon if you need top_k>1."
        )
        self.channels = channels
        self.n_experts = n_experts
        self.top_k = top_k
        self.loss_coef = loss_coef
        self.upsample_ratios = (
            list(range(1, n_experts + 1)) if upsample_ratios is None else list(upsample_ratios)
        )
        assert len(self.upsample_ratios) == n_experts

        self.gate = MoEGate(d=channels, M=n_experts, K=top_k, noisy_gating=True)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True),
                nn.GELU(),
            )
            for _ in range(n_experts)
        ])
        for expert in self.experts:
            nn.init.zeros_(expert[0].bias)

        # Last MoE loss; refreshed on every forward. Initialised to a non-grad
        # zero scalar so get_aux_losses() works on a forward pass that doesn't
        # hit this module (e.g. before the first training step).
        # NOT a register_buffer — buffers can lose requires_grad on assignment
        # and break the autograd path. Plain attribute keeps the loss tensor in
        # the graph when set from gate()'s moe_loss output.
        self.last_moe_loss = torch.zeros(1, dtype=torch.float32)

    def forward(self, h2d: torch.Tensor) -> torch.Tensor:
        """h2d: (B, r, H, W) -> (B, r, H, W).

        Computes ``h2d + sum_b expert[chosen_expert_b](upsample->conv->downsample)(h2d[b])``.
        """
        B, r, H, W = h2d.shape

        gates, moe_loss = self.gate(h2d, loss_coef=self.loss_coef)
        self.last_moe_loss = moe_loss  # scalar

        # For top_k=1: chosen_expert[b] = argmax of gates[b].
        # AutoGluon's SparseDispatcher with multiply_by_gates=False computes
        # exactly expert_output[b, chosen_expert_b] for each b. We do the same
        # via gather/scatter (cleaner than exp/log round-trip in combine()).
        chosen_expert = gates.argmax(dim=1)  # (B,)

        # We process all samples per-expert to allow batched conv calls.
        # out is created with h2d's dtype (fp16 under AMP). Expert outputs may
        # upcast to fp32 (bicubic interpolate + Conv2d in some autocast paths),
        # so we cast y_e back to out.dtype before index_put to avoid
        # "Index put requires the source and destination dtypes match".
        out = torch.zeros_like(h2d)
        for e in range(self.n_experts):
            mask = chosen_expert == e
            if not mask.any():
                continue
            x_e = h2d[mask]  # (B_e, r, H, W)
            ratio = self.upsample_ratios[e]
            if ratio != 1:
                x_e = F.interpolate(x_e, scale_factor=ratio, mode="bicubic")
            y_e = self.experts[e](x_e)
            if ratio != 1:
                y_e = F.interpolate(y_e, size=(H, W), mode="bicubic")
            out[mask] = y_e.to(out.dtype)

        return h2d + out


__all__ = ["MoEGate", "ConvLoRAMoE"]
