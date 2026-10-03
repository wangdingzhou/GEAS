"""Route A — Prototype Contrastive Auxiliary Loss Module.

Components:
  - PrototypeBank: per-class K-prototype bank with Eq.10-style EMA update.
      Init modes:
        "g2_kmeans" — spherical K-means
        "class_mean" — per-class mean (K=1)
      Update rule: μ_c^{(t)} = ε^{L-l} · μ_c^{(t-1)} + (1-ε^{L-l}) · batch_mean_c^{(t)}
      For p2 (l=2, L=4 in MM-DINO): ε^{L-l} = ε^2 = 0.9801 at ε=0.99.
  - grqa_contrastive_loss: QPrompt-R1 GRQA loss (M1 primary) — group-relative
      advantage + KL clipping (identity-capable via clamp(advantage, ±kl_clip)).
  - plain_contrastive_loss: InfoNCE-style loss (Variant D ablation, no GRQA norm).
  - RouteAModule: training-loop wrapper. omega ramp-up (5 ep zero-init),
      prototype EMA update under no_grad, AMP-safe fp32 loss computation.

Variant matrix:
  Variant A: K=3, init=g2_kmeans, use_ema=True,  loss_mode=grqa  (PRIMARY)
  Variant B: K=1, init=class_mean, use_ema=True,  loss_mode=grqa  (K=1 flat baseline)
  Variant C: K=3, init=g2_kmeans, use_ema=False,  loss_mode=grqa  (static ablation)
  Variant D: K=3, init=g2_kmeans, use_ema=True,   loss_mode=plain (no GRQA)
  Variant E: omega=0 (sanity control; identity to baseline)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Constants
# ============================================================================

EPS = 1e-8


# ============================================================================
# P2HookContext — capture p2 (pre-out_conv PRN output @ H/4) via neck hook
# ============================================================================
#
# Why a hook (not return_features=True):
#   src/model.py explicitly returns logits_h4 (B, num_classes, H/4, W/4)
#   when return_features=True — NOT p2. The comment there:
#     "Return logits_h4 (has num_classes chans) as the 'feature' — P1 doesn't
#      use consistency loss so this is OK."
#   For Route A we need p2 (B, embed_dim, H/4, W/4), the pre-out_conv PRN output.
#   Same hook pattern as G2/G3 diagnostics (g2_v3_class_cost_map.py:308,
#   g3_h4_to_h_decomp.py:167) — register_forward_hook on model.decoder._decoder.neck,
#   capture out[0].
#
# No detach: grad must flow through p2 → decoder for Route A to do its job.
# Under no_grad contexts (e.g. collect_p2_features_for_init), captured tensor is
# already detached implicitly.

class P2HookContext:
    """Context manager that captures p2 from model.decoder._decoder.neck output.

    Usage:
        with P2HookContext(model) as p2hook:
            output = model(images)
        p2 = p2hook.captured  # (B, embed_dim, H/4, W/4)
    """

    def __init__(self, model: nn.Module):
        if not (hasattr(model, "decoder")
                and hasattr(model.decoder, "_decoder")
                and hasattr(model.decoder._decoder, "neck")):
            raise AttributeError(
                "Route A needs model.decoder._decoder.neck (MMDINODecoder neck). "
                "Got model without that submodule. Is this an MM-DINO build?"
            )
        self._neck = model.decoder._decoder.neck
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.captured: Optional[torch.Tensor] = None

    def __enter__(self) -> "P2HookContext":
        def _neck_hook(module, inp, out):
            p2 = out[0] if isinstance(out, (tuple, list)) else out
            # NO .detach() here: grad must flow through p2 in train path.
            # Under no_grad, captured tensor is implicitly detached by PyTorch.
            self.captured = p2

        self._handle = self._neck.register_forward_hook(_neck_hook)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        # Leave self.captured populated; caller consumes it after the context.


# ============================================================================
# Init: G2 K-means or class-mean prototype fitting
# ============================================================================

def fit_init_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    K: int,
    mode: str,
    seed: int = 42,
    n_init: int = 10,
) -> torch.Tensor:
    """Fit initial prototypes for PrototypeBank.initialize.

    Args:
        features: (N, C) — L2-normalizable cell features (will be normalized inside).
        labels:   (N,) long — class labels, already filtered to [0, num_classes).
        num_classes: e.g. 7 for LoveDA.
        K: prototypes per class.
        mode: "g2_kmeans" (spherical K-means) or "class_mean" (per-class mean).
        seed, n_init: K-means reproducibility.

    Returns:
        prototypes (num_classes, K, C) on same device/dtype as features.
    """
    device = features.device
    dtype = features.dtype
    C = features.shape[1]
    protos = torch.zeros(num_classes, K, C, device=device, dtype=dtype)

    if mode == "class_mean":
        for c in range(num_classes):
            mask = (labels == c)
            if mask.any():
                mean_c = features[mask].mean(dim=0)
                for k in range(K):
                    protos[c, k] = mean_c
        return protos

    if mode == "g2_kmeans":
        # Deferred import to avoid module-load circular dependency
        diagnostics_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "diagnostics"
        )
        if diagnostics_dir not in sys.path:
            sys.path.insert(0, diagnostics_dir)
        from g2_v3_class_cost_map import fit_spherical_kmeans_per_class

        return fit_spherical_kmeans_per_class(
            features, labels,
            K=K, n_init=n_init, seed=seed, num_classes=num_classes,
        )

    raise ValueError(f"Unknown init mode: {mode!r}; expected 'g2_kmeans' or 'class_mean'")


# ============================================================================
# PrototypeBank: EMA-updated per-class K-prototype bank (Eq.10-style)
# ============================================================================

class PrototypeBank(nn.Module):
    """Per-class K-prototype bank with Eq.10-style EMA update.

    μ_c^{(t)} = ε_eff · μ_c^{(t-1)} + (1-ε_eff) · batch_mean_c^{(t)}

    Per-level momentum: ε_eff = ε_base^{L-l} (paper-faithful). For p2 (l=2, L=4),
    ε_eff = ε_base^2. With ε_base=0.99, ε_eff ≈ 0.9801.

    For K>1: each batch cell is assigned to its nearest prototype within the
    cell's true class, and only that prototype is updated (per-class multi-modal
    EMA). For K=1: simple per-class EMA.
    """

    def __init__(
        self,
        num_classes: int,
        K: int,
        feat_dim: int,
        epsilon_base: float = 0.99,
        level_l: int = 2,        # p2 → l=2
        level_L: int = 4,        # MM-DINO total pyramid levels
    ):
        super().__init__()
        self.num_classes = num_classes
        self.K = K
        self.feat_dim = feat_dim
        self.epsilon_base = epsilon_base
        self.level_l = level_l
        self.level_L = level_L
        self.epsilon_eff = epsilon_base ** (level_L - level_l)

        self.register_buffer("prototypes", torch.zeros(num_classes, K, feat_dim))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("update_count", torch.zeros(num_classes, K))

    def initialize(self, init_prototypes: torch.Tensor) -> None:
        if init_prototypes.shape != self.prototypes.shape:
            raise ValueError(
                f"Init prototypes shape {tuple(init_prototypes.shape)} != "
                f"bank shape {tuple(self.prototypes.shape)}"
            )
        self.prototypes.copy_(init_prototypes.to(self.prototypes))
        self.initialized.fill_(True)

    @torch.no_grad()
    def update_ema(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Eq.10-style EMA update on incoming batch.

        Args:
            features: (N, C) cell features. Promoted to fp32 internally for stable
                      accumulation (per feedback-amp-autocast-dtype-pitfalls).
            labels:   (N,) long, already filtered to [0, num_classes).
        """
        if features.dim() != 2:
            raise ValueError(f"features must be (N, C); got {tuple(features.shape)}")
        feats = features.float()

        for c in range(self.num_classes):
            mask = (labels == c)
            if not mask.any():
                continue
            feats_c = feats[mask]

            if self.K == 1:
                batch_mean = feats_c.mean(dim=0)
                self.prototypes[c, 0] = (
                    self.epsilon_eff * self.prototypes[c, 0]
                    + (1.0 - self.epsilon_eff) * batch_mean
                )
                self.update_count[c, 0] += feats_c.shape[0]
            else:
                # Assign each cell to nearest prototype within class c
                protos_c = self.prototypes[c]
                feats_norm = F.normalize(feats_c, dim=-1)
                protos_norm = F.normalize(protos_c, dim=-1)
                sims = feats_norm @ protos_norm.T
                assign = sims.argmax(dim=-1)

                for k in range(self.K):
                    members = feats_c[assign == k]
                    if members.shape[0] == 0:
                        continue
                    batch_mean_k = members.mean(dim=0)
                    self.prototypes[c, k] = (
                        self.epsilon_eff * self.prototypes[c, k]
                        + (1.0 - self.epsilon_eff) * batch_mean_k
                    )
                    self.update_count[c, k] += members.shape[0]

    def get_prototypes_normalized(self) -> torch.Tensor:
        return F.normalize(self.prototypes.float(), dim=-1)


# ============================================================================
# Loss functions
# ============================================================================

def grqa_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes_norm: torch.Tensor,
    kl_clip: float = 4.0,
    temperature: float = 1.0,
    hcl_normalize: bool = False,
    return_per_sample: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """QPrompt-R1 GRQA loss: group-relative advantage with KL clipping.

    Per-cell:
      tau_pos = max_k sim(cell, proto[label, k])           # positive
      tau_neg = mean over other classes of max_k sim(...)  # negative mean
      advantage = tau_pos - tau_neg
      advantage_clipped = clamp(advantage, -kl_clip, kl_clip)
      loss_per_cell = softplus(-advantage_clipped)         # smooth hinge

    Identity-capable: kl_clip bounds the loss landscape; softplus(0)≈0.69
    (constant gradient), so when prototypes equalize the loss is bounded.

    Args:
        hcl_normalize: if True, weight per-sample loss by 1/|I'_c| (HCL-style).
            For ablation only (M1 binding: GRQA primary, HCL ablation).
    """
    if features.dim() != 2:
        raise ValueError(f"features must be (N, C); got {tuple(features.shape)}")

    N, C = features.shape
    device = features.device
    feats_norm = F.normalize(features.float(), dim=-1)

    Cls, K, _ = prototypes_norm.shape
    sims_all = feats_norm @ prototypes_norm.reshape(Cls * K, C).T  # (N, Cls*K)
    sims_per_class = sims_all.view(N, Cls, K).max(dim=-1).values    # (N, Cls)
    sims_per_class = sims_per_class / max(temperature, EPS)

    arange = torch.arange(N, device=device)
    tau_pos = sims_per_class[arange, labels]

    # tau_neg = mean similarity to non-target classes (per QPrompt-R1 GRQA).
    # masked_fill(mask_pos, 0.0) zeros out the target-class slot so it doesn't
    # contribute to the sum, then we divide by Cls-1 (number of true negatives).
    # The 0.0 here is NOT "negative classes have 0 similarity" — it's a sum-exclusion
    # mask. Same effect as (sims_per_class * ~mask_pos).sum() / (Cls-1).
    mask_pos = F.one_hot(labels, Cls).bool()
    sims_neg = sims_per_class.masked_fill(mask_pos, 0.0)
    tau_neg = sims_neg.sum(dim=-1) / max(Cls - 1, 1)

    advantage = tau_pos - tau_neg
    advantage_clipped = torch.clamp(advantage, -kl_clip, kl_clip)
    loss_per_sample = F.softplus(-advantage_clipped)

    if hcl_normalize:
        class_counts = torch.bincount(labels, minlength=Cls).float()
        class_counts = class_counts.clamp(min=1.0)
        weights = 1.0 / class_counts[labels]
        out_mean = (loss_per_sample * weights.detach()).mean()
    else:
        out_mean = loss_per_sample.mean()

    if return_per_sample:
        return out_mean, loss_per_sample.detach()
    return out_mean


def plain_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes_norm: torch.Tensor,
    temperature: float = 0.1,
    return_per_sample: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Plain InfoNCE-style loss (Variant D ablation — no GRQA normalization).

    Per-cell: cross_entropy(logits / T, labels), where
        logits[c] = max_k sim(cell, proto[c, k]).
    """
    if features.dim() != 2:
        raise ValueError(f"features must be (N, C); got {tuple(features.shape)}")

    N, C = features.shape
    Cls, K, _ = prototypes_norm.shape
    feats_norm = F.normalize(features.float(), dim=-1)

    sims_all = feats_norm @ prototypes_norm.reshape(Cls * K, C).T
    sims_per_class = sims_all.view(N, Cls, K).max(dim=-1).values
    logits = sims_per_class / max(temperature, EPS)

    if return_per_sample:
        log_softmax = F.log_softmax(logits, dim=-1)
        per_sample = -log_softmax[torch.arange(N, device=feats_norm.device), labels]
        return per_sample.mean(), per_sample.detach()
    return F.cross_entropy(logits, labels)


# ============================================================================
# GT downsampling (majority vote)
# ============================================================================

def downsample_gt_majority(
    gt_H: torch.Tensor,
    target_size_h: int,
    target_size_w: int,
    ignore_value: int = 255,
    num_classes: int = 7,
) -> torch.Tensor:
    """Majority-vote downsample GT to (B, target_h, target_w).

    Adapted from diagnostics/g2_v3_class_cost_map.py:520 (downsample_gt_majority_v2).
    Cells with no valid pixels → ignore_value.

    Args:
        gt_H: (B, H, W) long.
        target_size_h, target_size_w: output spatial size (e.g. H/4).
    """
    B, H, W = gt_H.shape
    if H % target_size_h != 0 or W % target_size_w != 0:
        raise ValueError(f"GT {H}x{W} not divisible by target {target_size_h}x{target_size_w}")
    kh = H // target_size_h
    kw = W // target_size_w

    gt = gt_H.view(B, target_size_h, kh, target_size_w, kw)
    gt = gt.permute(0, 1, 3, 2, 4).contiguous().view(B, target_size_h, target_size_w, kh * kw)

    valid = (gt != ignore_value) & (gt < num_classes)

    best_votes = torch.zeros(B, target_size_h, target_size_w,
                             dtype=torch.long, device=gt_H.device)
    out = torch.full((B, target_size_h, target_size_w), ignore_value,
                     dtype=torch.long, device=gt_H.device)

    for c in range(num_classes):
        votes = ((gt == c) & valid).sum(dim=-1)
        improve = votes > best_votes
        best_votes = torch.where(improve, votes, best_votes)
        out = torch.where(improve, torch.full_like(out, c), out)

    # Cells with zero valid votes → ignore
    out = torch.where(best_votes > 0, out, torch.full_like(out, ignore_value))
    return out


# ============================================================================
# RouteAConfig + RouteAModule
# ============================================================================

@dataclass
class RouteAConfig:
    """Route-A module settings (all toggleable from YAML)."""
    enabled: bool = False
    row: str = "A"                  # A/B/C/D/E (M5 ablation matrix)
    omega: float = 0.1              # loss weight (target)
    omega_ramp_epochs: int = 5      # zero-init ramp-up
    kl_clip: float = 4.0            # GRQA KL clip
    temperature: float = 1.0
    epsilon_base: float = 0.99      # base momentum
    K: int = 3                      # prototypes per class
    init_mode: str = "g2_kmeans"    # "g2_kmeans" (A/C/D) or "class_mean" (B)
    loss_mode: str = "grqa"         # "grqa" (A/B/C) or "plain" (D)
    use_ema: bool = True            # True (A/B/D) or False (C static)
    hcl_normalize: bool = False     # HCL-style (ablation only)
    num_classes: int = 7
    ignore_value: int = 255
    init_sample_n_tiles: int = 50   # how many train tiles for prototype init fit
    init_seed: int = 42
    # causal pre-test subset filter.
    #   "" / None  → full train set (default; main run uses this)
    #   "building_forest_20pct" → keep tiles where Building+Forest pixel ratio ≥ 0.20
    subset_filter: str = ""

    @classmethod
    def from_cfg(cls, cfg) -> "RouteAConfig":
        ra = cfg.get("route_a", {}) if hasattr(cfg, "get") else cfg
        kwargs = {k: ra.get(k, getattr(cls, k)) for k in [
            "enabled", "row", "omega", "omega_ramp_epochs", "kl_clip",
            "temperature", "epsilon_base", "K", "init_mode", "loss_mode",
            "use_ema", "hcl_normalize",
        ]}
        kwargs["num_classes"] = cfg.get("num_classes", 7)
        kwargs["ignore_value"] = ra.get("ignore_value", 255)
        kwargs["init_sample_n_tiles"] = ra.get("init_sample_n_tiles", 50)
        kwargs["init_seed"] = ra.get("init_seed", cfg.get("seed", 42))
        kwargs["subset_filter"] = ra.get("subset_filter", "")
        return cls(**kwargs)


class RouteAModule(nn.Module):
    """Training-loop wrapper for Route A prototype contrastive loss.

    Lifecycle (per train iteration):
        route_a_module.set_epoch(epoch)
        total_loss = ce_loss + dice_loss + route_a_module(p2, masks)

    The caller extracts p2 via P2HookContext(model) — see cross_dataset/train_cross.py
    route_a injection. Do NOT use model(images, return_features=True): that returns
    logits_h4 (B, num_classes, H/4, W/4), NOT p2 (src/model.py).

    AMP safety (per feedback-amp-autocast-dtype-pitfalls):
      - features cast to fp32 in loss + EMA paths
      - prototype buffers stay fp32
      - loss returns fp32 scalar (grad flows through features only; prototypes
        are updated under no_grad)
    """

    def __init__(self, cfg: RouteAConfig, feat_dim: int, device: torch.device):
        super().__init__()
        self.cfg = self._normalize_row(cfg)
        self.feat_dim = feat_dim
        self.device = device

        self.prototype_bank = PrototypeBank(
            num_classes=self.cfg.num_classes,
            K=self.cfg.K,
            feat_dim=feat_dim,
            epsilon_base=self.cfg.epsilon_base,
        )
        self.current_epoch = -1
        # Per-class contrastive loss accumulators (epoch scope). Reset on epoch
        # change. Used for causal pre-test report (user 2026-08-01):
        # Building vs Forest per-class curves diagnose Forest multi-modal failure
        # (Route E pseudo-subclass signal) more informatively than total loss.
        self._per_class_sum = torch.zeros(self.cfg.num_classes, dtype=torch.float64)
        self._per_class_count = torch.zeros(self.cfg.num_classes, dtype=torch.long)

    @staticmethod
    def _normalize_row(cfg: RouteAConfig) -> RouteAConfig:
        """Force config to match the row semantics."""
        if cfg.row == "A":
            cfg.K, cfg.init_mode, cfg.use_ema, cfg.loss_mode = 3, "g2_kmeans", True, "grqa"
        elif cfg.row == "B":
            cfg.K, cfg.init_mode, cfg.use_ema, cfg.loss_mode = 1, "class_mean", True, "grqa"
        elif cfg.row == "C":
            cfg.K, cfg.init_mode, cfg.use_ema, cfg.loss_mode = 3, "g2_kmeans", False, "grqa"
        elif cfg.row == "D":
            cfg.K, cfg.init_mode, cfg.use_ema, cfg.loss_mode = 3, "g2_kmeans", True, "plain"
        elif cfg.row == "E":
            cfg.omega = 0.0
        else:
            raise ValueError(f"Unknown row: {cfg.row!r}; expected A/B/C/D/E")
        return cfg

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.current_epoch:
            # New epoch → reset per-class accumulators
            self._per_class_sum.zero_()
            self._per_class_count.zero_()
        self.current_epoch = epoch

    @property
    def epoch_per_class_loss(self) -> dict[int, float]:
        """Mean contrastive loss per class over the current epoch (weighted by
        cell count). NaN for classes unseen this epoch."""
        out = {}
        for i in range(self.cfg.num_classes):
            n = int(self._per_class_count[i].item())
            out[i] = float(self._per_class_sum[i].item() / n) if n > 0 else float("nan")
        return out

    @property
    def omega_eff(self) -> float:
        if self.current_epoch < self.cfg.omega_ramp_epochs:
            ramp = max(0.0, self.current_epoch / max(1, self.cfg.omega_ramp_epochs))
            return self.cfg.omega * ramp
        return self.cfg.omega

    def initialize_prototypes(self, init_protos: torch.Tensor) -> None:
        """Call after build, before training."""
        self.prototype_bank.initialize(init_protos)

    def forward(
        self,
        p2_features: torch.Tensor,
        labels_h4: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss (already scaled by omega_eff).

        Args:
            p2_features: (B, C, H/4, W/4) cell features.
            labels_h4:   (B, H/4, W/4) long; OR (B, H, W) long (will be downsampled).

        Returns:
            scalar fp32 tensor. Zero when omega_eff == 0 (Variant E or ramp start).
        """
        # Variant E sanity (omega == 0): no loss, no accumulation, no prototypes touched.
        # Ramp epochs (omega > 0, omega_eff == 0): we still compute the loss +
        # accumulate per-class diagnostics; the returned scalar is 0 so no
        # backward contribution, but the diagnostic curves are populated.
        if self.cfg.omega == 0.0:
            return torch.tensor(0.0, device=p2_features.device, dtype=torch.float32)

        B, Cf, Hp, Wp = p2_features.shape
        # Downsample labels if shape mismatches p2 spatial
        if labels_h4.dim() == 3 and labels_h4.shape[1] != Hp:
            labels_h4 = downsample_gt_majority(
                labels_h4, Hp, Wp,
                ignore_value=self.cfg.ignore_value,
                num_classes=self.cfg.num_classes,
            )

        feat_flat = p2_features.permute(0, 2, 3, 1).reshape(-1, Cf).float()
        label_flat = labels_h4.reshape(-1).long()

        valid = (
            (label_flat != self.cfg.ignore_value)
            & (label_flat < self.cfg.num_classes)
            & (label_flat >= 0)
        )
        feat_flat = feat_flat[valid]
        label_flat = label_flat[valid]

        if feat_flat.shape[0] == 0:
            return torch.tensor(0.0, device=p2_features.device, dtype=torch.float32)

        # EMA update (no grad) — uses detached features
        if self.cfg.use_ema:
            with torch.no_grad():
                self.prototype_bank.update_ema(feat_flat.detach(), label_flat)

        protos_norm = self.prototype_bank.get_prototypes_normalized()

        if self.cfg.loss_mode == "grqa":
            loss, per_sample = grqa_contrastive_loss(
                feat_flat, label_flat, protos_norm,
                kl_clip=self.cfg.kl_clip,
                temperature=self.cfg.temperature,
                hcl_normalize=self.cfg.hcl_normalize,
                return_per_sample=True,
            )
        else:
            loss, per_sample = plain_contrastive_loss(
                feat_flat, label_flat, protos_norm,
                temperature=max(0.1, self.cfg.temperature),
                return_per_sample=True,
            )

        # Per-class accumulation (unweighted; weighted-mean forms already in `loss`).
        # Detached CPU-side binning keeps the hot path on GPU.
        with torch.no_grad():
            lbl_cpu = label_flat.detach().cpu()
            ps_cpu = per_sample.detach().cpu()
            for c in range(self.cfg.num_classes):
                mask = (lbl_cpu == c)
                n = int(mask.sum().item())
                if n > 0:
                    self._per_class_sum[c] += float(ps_cpu[mask].sum().item())
                    self._per_class_count[c] += n

        return self.omega_eff * loss

    def diagnostic_summary(self) -> dict:
        return {
            "row": self.cfg.row,
            "omega_target": self.cfg.omega,
            "omega_eff": self.omega_eff,
            "current_epoch": self.current_epoch,
            "K": self.cfg.K,
            "init_mode": self.cfg.init_mode,
            "use_ema": self.cfg.use_ema,
            "loss_mode": self.cfg.loss_mode,
            "hcl_normalize": self.cfg.hcl_normalize,
            "initialized": bool(self.prototype_bank.initialized.item()),
            "prototype_norm_mean": float(self.prototype_bank.prototypes.norm(dim=-1).mean().item()),
            "epsilon_eff": float(self.prototype_bank.epsilon_eff),
        }


# ============================================================================
# Prototype init helper — collect p2 features from train subset
# ============================================================================

@torch.no_grad()
def collect_p2_features_for_init(
    model: nn.Module,
    train_loader,
    num_classes: int,
    ignore_value: int,
    n_tiles: int,
    device: torch.device,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward `n_tiles` train tiles; collect (N_total, C) p2 features + (N_total,) labels.

    Used by PrototypeBank.initialize() to fit G2 K-means or class-mean prototypes.
    Cells with ignore or invalid label are dropped.

    Captures p2 via P2HookContext(model) (NOT return_features=True — that returns
    logits_h4 per src/model.py). Under no_grad, captured p2 is implicitly
    detached; we .clone() for safety before squeezing out of the hook context.

    Args:
        model: HALoRASeg (must have model.decoder._decoder.neck).
        train_loader: yields (image, mask) batches; we sample first n_tiles from it.
        n_tiles: cap on tiles forwarded.

    Returns:
        features (N, C) fp32 on CPU, labels (N,) long on CPU.
    """
    model.eval()
    feats_all = []
    labels_all = []
    seen = 0

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_loader.dataset), generator=g).tolist()[:n_tiles]

    for idx in indices:
        image, mask = train_loader.dataset[idx]
        image = image.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        with P2HookContext(model) as p2hook:
            _ = model(image)  # discard logits; we only need p2
        p2 = p2hook.captured
        if p2 is None:
            raise RuntimeError(
                "P2 hook did not capture — is model.decoder._decoder.neck the right submodule?"
            )
        B, C, Hp, Wp = p2.shape

        labels_h4 = downsample_gt_majority(
            mask, Hp, Wp, ignore_value=ignore_value, num_classes=num_classes
        )

        feat_flat = p2[0].permute(1, 2, 0).reshape(-1, C).float()
        label_flat = labels_h4[0].reshape(-1).long()

        valid = (label_flat != ignore_value) & (label_flat < num_classes) & (label_flat >= 0)
        feats_all.append(feat_flat[valid].cpu())
        labels_all.append(label_flat[valid].cpu())

        seen += 1
        if seen >= n_tiles:
            break

    features = torch.cat(feats_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    return features, labels


# ============================================================================
# causal pre-test subset filter
# ============================================================================

# Mapping from subset_filter name → (target_classes, min_ratio).
# LoveDA labels (canonical, src/dataset.py:21): 0=Background, 1=Building, 2=Road,
#                3=Water, 4=Barren, 5=Forest, 6=Agriculture, 7=Ignore.
SUBSET_FILTER_PRESETS = {
    "building_forest_20pct": ([1, 5], 0.20),  # Building + Forest ≥ 20% pixel ratio
}


def compute_subset_filter(
    dataset,
    preset: str,
    ignore_value: int = 255,
    cache_path: Optional[str] = None,
) -> list[int]:
    """return indices of dataset tiles passing a pixel-ratio preset.

    A tile passes if the pixel mass of target_classes (over valid pixels) ≥ min_ratio.
    Used by causal pre-test (Building+Forest ≥20%). Main run
 does NOT use this — it trains on full LoveDA train.

    Args:
        dataset: LoveDA train dataset; dataset[i] returns (image, mask) where mask
                 is (H, W) long with ignore_value for invalid pixels.
        preset: key into SUBSET_FILTER_PRESETS (e.g. "building_forest_20pct").
        cache_path: optional JSON file for caching the index list. Re-runs read
                    the cache instead of re-scanning.

    Returns:
        List of integer indices (0-based into `dataset`).
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    if preset not in SUBSET_FILTER_PRESETS:
        raise ValueError(
            f"Unknown subset_filter preset: {preset!r}. "
            f"Expected one of {list(SUBSET_FILTER_PRESETS.keys())}"
        )
    target_classes, min_ratio = SUBSET_FILTER_PRESETS[preset]

    kept: list[int] = []
    n_total = len(dataset)
    for i in range(n_total):
        try:
            _, mask = dataset[i]
        except Exception as e:
            print(f"[subset_filter] skip idx {i}: {e}")
            continue
        if mask.dim() == 3:
            mask = mask.squeeze(0) if mask.shape[0] == 1 else mask.argmax(dim=0)
        mask = mask.long()
        valid = (mask != ignore_value)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue
        n_target = sum(int(((mask == c) & valid).sum().item()) for c in target_classes)
        if n_target / n_valid >= min_ratio:
            kept.append(i)
        if (i + 1) % 200 == 0:
            print(f"[subset_filter] scanned {i+1}/{n_total}, kept {len(kept)}")

    print(f"[subset_filter] preset={preset} ({target_classes} ≥{min_ratio:.0%}): "
          f"kept {len(kept)}/{n_total} tiles ({len(kept)/max(1,n_total):.1%})")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(kept, f)
        print(f"[subset_filter] cached → {cache_path}")

    return kept
