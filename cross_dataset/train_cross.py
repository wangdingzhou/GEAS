"""Cross-dataset training script for GEAS.
Supports LoveDA and Potsdam datasets via config.dataset_type.
"""

import argparse
import contextlib
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.amp import GradScaler
import torch.amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# M2F support
from m2f_loss import HungarianMatcher, M2FSetCriterion, compute_m2f_loss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from dataset import LoveDADataset
from dataset_potsdam import PotsdamDataset
from dataset_vaihingen import VaihingenDataset
from model import count_trainable_parameters, get_model
from sf_adapter import set_sf_adapter_disable

# Class-balanced copy-paste (gated by cfg.dataset_type == "loveda_cbcp")
# Imported eagerly so import errors surface at startup, not mid-training.
from class_balanced_copy_paste import (
    ClassBalancedCopyPaste,
    CopyPasteConfig,
    InstancePool,
    InstancePoolConfig,
    LABEL_AGRICULTURE,
    LABEL_FOREST,
)
from loveda_cbcp_dataset import LoveDACBCPDataset

# Route A: prototype contrastive auxiliary loss.
# Gated by cfg.route_a.enabled. Variant A/B/C/D/E ablation matrix via cfg.route_a.row.
# Side-loss: p2 captured via P2HookContext (NOT return_features=True — that returns
# logits_h4 per src/model.py not p2).
from route_a_module import (
    P2HookContext,
    RouteAConfig,
    RouteAModule,
    collect_p2_features_for_init,
    compute_subset_filter,
    fit_init_prototypes,
)


def compute_consistency_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_feat,
    teacher_feat,
    kl_enabled: bool = True,
    affinity_enabled: bool = True,
    lambda_kl: float = 0.05,
    lambda_rel: float = 0.02,
    tau: float = 2.0,
    conf_thresh: float = 0.70,
    affinity_tokens: int = 64,
):
    """Semantic-preserving consistency loss.

    teacher_* must be detached (no_grad). student_* carry gradient.
    - KL: KL(softmax(s/tau) || softmax(t/tau)) over easy pixels, where
      easy = teacher confidence (argmax softmax prob) >= conf_thresh.
    - Affinity: MSE between student/teacher cosine-affinity matrices on K
      sampled easy tokens from the decoder fused feature (B, D, H/16, W/16).

    Returns (loss_kl, loss_aff, diag). Disabled terms are python 0.0.
    diag keys: kl_raw, easy_frac, aff_raw, aff_tokens.
    """
    diag = {}
    loss_kl = 0.0
    loss_aff = 0.0

    # fp32: softmax / KL / affinity are fp16-unstable under AMP.
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        s = student_logits.float()
        t = teacher_logits.float()
        # Teacher confidence at full resolution (tau=1) — drives the easy mask.
        teacher_conf_full = F.softmax(t, dim=1).max(dim=1)[0]  # (B, H, W)

        # --- KL on easy pixels ---
        if kl_enabled:
            log_s = F.log_softmax(s / tau, dim=1)
            log_t = F.log_softmax(t / tau, dim=1)
            p_s = log_s.exp()
            kl_per_pixel = (p_s * (log_s - log_t)).sum(dim=1)  # (B, H, W)
            easy_mask = (teacher_conf_full >= conf_thresh).float()
            n_easy = easy_mask.sum().clamp(min=1.0)
            kl_mean = (kl_per_pixel * easy_mask).sum() / n_easy
            loss_kl = lambda_kl * kl_mean
            diag["kl_raw"] = kl_mean.item()
            diag["easy_frac"] = easy_mask.mean().item()

        # --- Affinity MSE on sampled easy tokens ---
        if affinity_enabled and student_feat is not None and teacher_feat is not None:
            sf = student_feat.float()
            tf = teacher_feat.float()
            B, D, h, w = sf.shape
            N = h * w
            # Pool teacher confidence to token resolution (H/16) for token easy mask.
            easy_tok = F.adaptive_avg_pool2d(
                teacher_conf_full.unsqueeze(1), (h, w)
            ).squeeze(1)
            easy_tok_mask = easy_tok >= conf_thresh  # (B, h, w)

            sf_flat = sf.reshape(B, D, N)
            tf_flat = tf.reshape(B, D, N)

            aff_terms = []
            tok_count = 0
            for b in range(B):
                idx = easy_tok_mask[b].reshape(-1).nonzero(as_tuple=False).squeeze(-1)
                n_easy_b = idx.numel()
                if n_easy_b < 2:
                    continue
                K = min(affinity_tokens, n_easy_b)
                perm = torch.randperm(n_easy_b, device=sf.device)[:K]
                sel = idx[perm]
                sb = F.normalize(sf_flat[b][:, sel], dim=0)  # (D, K)
                tb = F.normalize(tf_flat[b][:, sel], dim=0)  # (D, K)
                A_s = sb.t() @ sb  # (K, K) cosine affinity
                A_t = tb.t() @ tb
                aff_terms.append(F.mse_loss(A_s, A_t))
                tok_count += K

            if aff_terms:
                aff_mean = sum(aff_terms) / len(aff_terms)
                loss_aff = lambda_rel * aff_mean
                diag["aff_raw"] = aff_mean.item()
                diag["aff_tokens"] = tok_count / max(B, 1)
            else:
                diag["aff_raw"] = 0.0
                diag["aff_tokens"] = 0.0

    return loss_kl, loss_aff, diag


@contextlib.contextmanager
def frozen_bn_running_stats(model):
    """Snapshot BatchNorm running stats on enter; restore on exit (even on error).

    torch.no_grad() does NOT stop BatchNorm from updating running_mean /
    running_var / num_batches_tracked while the model is in train mode. A
    teacher (adapter-disabled) forward would therefore double-update BN stats
    each step (teacher + student), so R1-full's BN trajectory would diverge
    from R1-noC's single-update trajectory — breaking the clean comparison.

    This context snapshots every BN buffer before the teacher forward and
    copies it back afterward. The teacher still normalizes with the current
    batch statistics (train-mode BN), so its output is the true "student minus
    adapter delta"; only the running-stat accumulation is frozen, leaving the
    student pass as the sole per-step BN update (matched to R1-noC).
    """
    snapshots = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            snapshots.append((
                module,
                module.running_mean.clone(),
                module.running_var.clone(),
                module.num_batches_tracked.clone(),
            ))
    try:
        yield
    finally:
        for module, rm, rv, nbt in snapshots:
            module.running_mean.copy_(rm)
            module.running_var.copy_(rv)
            module.num_batches_tracked.copy_(nbt)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def get_dataset(cfg, split: str):
    dataset_type = cfg.get("dataset_type", "loveda")
    transform = (split == "train")
    split_name = "Train" if split == "train" else "Val"
    image_size = cfg.get("image_size", 512)
    input_norm = cfg.get("input_norm", "imagenet")

    if dataset_type == "loveda_cbcp":
        # Class-balanced copy-paste (LoveDA-only).
        # Val/Test path: identical to baseline LoveDADataset (cbcp_transform=None).
        if split != "train":
            return LoveDADataset(
                root=cfg.data_root, split=split_name, transform=transform,
                image_size=image_size, input_norm=input_norm,
            )

        # Build instance pool once (startup cost ~30s, then in-memory cache).
        rural_root = os.path.join(cfg.data_root, "Train", "Rural")
        pool_cfg = InstancePoolConfig(
            rural_root=rural_root,
            source_classes=tuple(cfg.get("cbcp_source_classes", [LABEL_FOREST, LABEL_AGRICULTURE])),
            cc_min_px=cfg.get("cbcp_cc_min_px", 100),
            cc_max_tile_ratio=cfg.get("cbcp_cc_max_tile_ratio", 0.25),
            tile_size_norm=image_size,
            max_per_class=cfg.get("cbcp_max_pool_per_class", 200),
            seed=cfg.get("seed", 42),
        )
        pool = InstancePool(pool_cfg)
        pool.build()
        print(
            f"[cbcp] InstancePool built: {pool.stats.n_tiles_scanned} rural tiles scanned, "
            f"per-class={pool.stats.n_instances_per_class}, skipped_size={pool.stats.n_skipped_size}, "
            f"skipped_overflow={pool.stats.n_skipped_overflow}"
        )

        cp_cfg = CopyPasteConfig(
            source_classes=tuple(cfg.get("cbcp_source_classes", [LABEL_FOREST, LABEL_AGRICULTURE])),
            max_instances_per_tile=cfg.get("cbcp_max_instances_per_tile", 3),
            patch_size_min_ratio=cfg.get("cbcp_patch_size_min_ratio", 0.10),
            patch_size_max_ratio=cfg.get("cbcp_patch_size_max_ratio", 0.50),
            scale_min=cfg.get("cbcp_scale_min", 0.70),
            scale_max=cfg.get("cbcp_scale_max", 1.40),
            ratio_cap=cfg.get("cbcp_ratio_cap", 2.0),
            harmonize=cfg.get("cbcp_harmonize", False),
            decouple=cfg.get("cbcp_decouple", False),
            harmonize_alpha=cfg.get("cbcp_harmonize_alpha", 0.5),
            paste_prob=cfg.get("cbcp_paste_prob", 1.0),
            tile_size=image_size,
            seed=cfg.get("seed", 42) + 1,
        )
        transform_obj = ClassBalancedCopyPaste(pool=pool, cfg=cp_cfg)
        print(f"[cbcp] CopyPasteConfig: harmonize={cp_cfg.harmonize}, decouple={cp_cfg.decouple}")

        return LoveDACBCPDataset(
            root=cfg.data_root, split=split_name, transform=transform,
            image_size=image_size, input_norm=input_norm,
            cbcp_transform=transform_obj,
        )

    if dataset_type == "potsdam":
        # HARD REQUIREMENT (Issue 3, spec v3): split_preset must be explicit for Potsdam.
        # Prevents silent fallback to old adv_ms_eg split.
        split_preset = cfg.get("split_preset", None)
        if split_preset is None:
            raise ValueError(
                "Potsdam config MUST set split_preset explicitly. "
                "Valid values: 'adv_ms_eg' (old project split) or 'mm_dino' (MM-DINO 18/6 split). "
                "Refusing to train with implicit default to prevent split leakage."
            )
        # dual_label only applies to train split — val returns standard 2-tuple for
        # fair-eval compatibility.
        _dual_label = bool(cfg.get("dual_label", False)) and split == "train"
        return PotsdamDataset(
            samples_per_tile=cfg.get("samples_per_tile", 1),
            root=cfg.data_root, split=split, transform=transform,
            image_size=image_size, input_norm=input_norm,
            split_preset=split_preset,
            use_noBoundary_labels=bool(cfg.get("use_noBoundary_labels", False)),
            dual_label=_dual_label,
            photo_metric=bool(cfg.get("photo_metric", False)),
            pmd_brightness=int(cfg.get("pmd_brightness", 24)),
            pmd_contrast=tuple(cfg.get("pmd_contrast", [0.65, 1.35])),
            pmd_saturation=tuple(cfg.get("pmd_saturation", [0.65, 1.35])),
            pmd_hue=int(cfg.get("pmd_hue", 0)),
            cat_max_ratio=cfg.get("cat_max_ratio", None),
            scale_first=bool(cfg.get("scale_first", False)),
        )

    if dataset_type == "vaihingen":
        # Locked: Vaihingen split is MM-DINO 12/4 ONLY.
        # Refuse any other preset / implicit default to prevent split leakage.
        split_preset = cfg.get("split_preset", None)
        if split_preset is None:
            raise ValueError(
                "Vaihingen config MUST set split_preset explicitly. "
                "Locked value: 'mm_dino' (12 train / 4 test). "
                "Refusing to train with implicit default."
            )
        if split_preset != "mm_dino":
            raise ValueError(
                f"Vaihingen split_preset={split_preset!r} not allowed — "
                "Split is locked to 'mm_dino' (12/4)."
            )
        # dual_label only applies to train split — val returns standard 2-tuple
        # for fair-eval compatibility.
        _dual_label = bool(cfg.get("dual_label", False)) and split == "train"
        return VaihingenDataset(
            samples_per_tile=cfg.get("samples_per_tile", 1),
            root=cfg.data_root, split=split, transform=transform,
            image_size=image_size, input_norm=input_norm,
            split_preset=split_preset,
            use_noBoundary_labels=bool(cfg.get("use_noBoundary_labels", False)),
            dual_label=_dual_label,
            photo_metric=bool(cfg.get("photo_metric", False)),
            pmd_brightness=int(cfg.get("pmd_brightness", 24)),
            pmd_contrast=tuple(cfg.get("pmd_contrast", [0.65, 1.35])),
            pmd_saturation=tuple(cfg.get("pmd_saturation", [0.65, 1.35])),
            pmd_hue=int(cfg.get("pmd_hue", 0)),
            cat_max_ratio=cfg.get("cat_max_ratio", None),
            scale_first=bool(cfg.get("scale_first", False)),
        )
    else:
        return LoveDADataset(
            root=cfg.data_root, split=split_name, transform=transform,
            image_size=image_size, input_norm=input_norm,
            photo_metric=bool(cfg.get("photo_metric", False)),
            pmd_hue=int(cfg.get("pmd_hue", 0)),
            pmd_brightness=int(cfg.get("pmd_brightness", 10)),
            pmd_contrast=tuple(cfg.get("pmd_contrast", [0.8, 1.2])),
            pmd_saturation=tuple(cfg.get("pmd_saturation", [0.8, 1.2])),
            cat_max_ratio=cfg.get("cat_max_ratio", None),
            scale_first=bool(cfg.get("scale_first", False)),
        )


def get_dataset_class(cfg):
    dataset_type = cfg.get("dataset_type", "loveda")
    if dataset_type == "potsdam":
        return PotsdamDataset
    if dataset_type == "vaihingen":
        return VaihingenDataset
    if dataset_type == "loveda_cbcp":
        return LoveDACBCPDataset
    return LoveDADataset


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential moving average of model parameters.

    Maintains a shadow copy whose params follow:
        ema_p = decay * ema_p + (1 - decay) * model_p
    Only trainable parameters are tracked; buffers (e.g., BN running stats)
    are also copied for consistency at eval time.

    Validate / save using ema.module instead of model — typically +0.3~0.7%
    mIoU for segmentation, free except for one extra FP32 model in memory.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995, device=None):
        import copy
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.device = device

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
        if self.ignore_index >= 0:
            valid = (targets != self.ignore_index).unsqueeze(1).float()
        else:
            valid = torch.ones_like(targets, dtype=torch.float32).unsqueeze(1)
        probs = F.softmax(logits, dim=1)
        intersection = (probs * targets_one_hot * valid).sum(dim=(2, 3))
        union = (probs * valid).sum(dim=(2, 3)) + (targets_one_hot * valid).sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SoftCrossEntropyLoss(nn.Module):
    """Cross-entropy with label smoothing (MM-DINO spec, smooth_factor=0.05).

    Thin wrapper around F.cross_entropy(label_smoothing=...) — PyTorch native
    since 1.10. The smooth_factor is split uniformly across non-target classes.
    """
    def __init__(self, smooth_factor: float = 0.05, ignore_index: int = 255):
        super().__init__()
        self.smooth_factor = float(smooth_factor)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.smooth_factor,
        )


class ConfusionPairSoftCELoss(nn.Module):
    """Confusion-pair aware SoftCE (P1.5 v7.2 §P1.5).

    Reweights pixel-wise SoftCE on pixels where (pred, gt) ∈ pair_set × pair_set
    and pred != gt. Preserves label smoothing. Sum-normalized by total weight
    so gradient scale matches SoftCE on non-pair pixels.

    Formula:
        w_p = 1 + (λ_cp - 1) · pair_mask       (per-pixel weight)
        L_cp = (Σ w_p · CE_pixelwise) / (Σ w_p)

    Default λ_cp=1.0 reduces bit-exactly to SoftCE (clean ablation).

    Symmetric: pair (a, b) and (b, a) are equivalent — both directions weighted.
    """
    def __init__(
        self,
        pairs: list,
        lambda_cp: float = 2.0,
        smooth_factor: float = 0.05,
        ignore_index: int = 255,
    ):
        super().__init__()
        # Normalize pairs: each pair is a (a, b) tuple; store as frozenset for set lookup.
        self.pair_sets = [frozenset((int(a), int(b))) for a, b in pairs]
        self.lambda_cp = float(lambda_cp)
        self.smooth_factor = float(smooth_factor)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Pixel-wise SoftCE (label_smoothing applied per-pixel via reduction='none').
        per_pixel = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.smooth_factor,
            reduction="none",
        )
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            valid = targets != self.ignore_index
            # pair_mask: any pixel where pred & gt are both in the same pair AND pred != gt
            pair_mask = torch.zeros_like(targets, dtype=torch.bool)
            for ps in self.pair_sets:
                pred_in = sum(pred == c for c in ps).bool()
                tgt_in = sum(targets == c for c in ps).bool()
                pair_mask |= (pred_in & tgt_in & (pred != targets))
            pair_mask &= valid
            w = torch.ones_like(per_pixel)
            w[pair_mask] = self.lambda_cp
        return (w * per_pixel).sum() / w.sum().clamp_min(1.0)


class EdgePointCELoss(nn.Module):
    """Edge-point CE — minimal port of MSEONet EPFO core idea (P1.5 v7.2 §P1.5).

    Full EPFO (Huang TGRS 2025, `epfo_head.py`) is a 344-LOC mmseg cascade head
    requiring boundary supervision + point_sample ops. The core insight is:
        "Focus loss on the K pixels with smallest top2-top1 logit gap
         (most uncertain, typically boundary pixels)."

    This minimal port preserves that single idea as a side loss term. No arch
    change, no boundary supervision, no point_sample dependency.

    Formula:
        1. uncertainty[u, v] = |top2_logit - top1_logit|     (smaller = more uncertain)
        2. pick top-K smallest-uncertainty valid pixels per batch
        3. L_edge = mean(CE_pixelwise on those K pixels)

    Default K=4096 (~1 tile's worth of pixels at 512x512 resolution,
    ~0.06% of total 262144 pixels per tile). At K=4096 with weight 0.5,
    adds ~2% to total loss scale.
    """
    def __init__(
        self,
        k_pixels: int = 4096,
        weight: float = 0.5,
        smooth_factor: float = 0.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.k_pixels = int(k_pixels)
        self.weight = float(weight)
        self.smooth_factor = float(smooth_factor)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        # Uncertainty: small |top2 - top1| = uncertain pixel (likely boundary).
        with torch.no_grad():
            top2 = torch.topk(logits, k=2, dim=1)[0]  # (B, 2, H, W)
            uncertainty = (top2[:, 1] - top2[:, 0]).abs()  # (B, H, W)
            valid = (targets != self.ignore_index)
            # Exclude invalid pixels by setting their uncertainty to +inf.
            uncertainty = uncertainty.masked_fill(~valid, float("inf"))
            flat = uncertainty.view(B, -1)
            # Pick top-K smallest-uncertainty pixels (negate to use topk for "smallest").
            k = min(self.k_pixels, flat.size(1))
            _, idx = torch.topk(-flat, k=k, dim=1)  # (B, K)
            mask_flat = torch.zeros_like(flat).scatter_(1, idx, 1.0)
            mask = mask_flat.view(B, H, W)
        per_pixel = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.smooth_factor,
            reduction="none",
        )
        selected = (mask * per_pixel).sum() / mask.sum().clamp_min(1.0)
        return self.weight * selected




class WeightedSoftCrossEntropyLoss(nn.Module):
    """SoftCE + per-class weight.

    Thin wrapper over F.cross_entropy(weight=..., label_smoothing=...).
    PyTorch native since 1.10 — both options compose cleanly.

    Used by loss_type=class_weighted_ce_plus_dice. Default weights boost
    Forest (idx 5) and Agri (idx 6) on LoveDA to attack D0.0 rural→Building
    pathology.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        smooth_factor: float = 0.05,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.register_buffer("weight", weight.float())
        self.smooth_factor = float(smooth_factor)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.smooth_factor,
        )


class EdgeAwareSoftCELoss(nn.Module):
    """Edge-aware SoftCE.

    Pixel-wise SoftCE with λ_edge weight on GT-boundary pixels, gradient-scale-
    preserved form (matches ConfusionPairSoftCELoss convention):
        L = Σ(w * CE_pixel) / Σ(w),   w = λ_edge if pixel on boundary else 1.0

    Boundary extracted from GT mask via dilation-erosion (extract_boundary_mask_gpu).
    Single-delta vs the baseline soft_ce_plus_dice: only the CE part changes — Dice
    untouched. Targets D0.2 boundary bottleneck.
    """

    def __init__(
        self,
        lambda_edge: float = 2.0,
        boundary_width: int = 8,
        smooth_factor: float = 0.05,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.lambda_edge = float(lambda_edge)
        self.boundary_width = int(boundary_width)
        self.smooth_factor = float(smooth_factor)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_pixel = F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.smooth_factor,
            reduction="none",
        )
        with torch.no_grad():
            boundary = extract_boundary_mask_gpu(
                targets, num_classes=logits.shape[1],
                boundary_width=self.boundary_width,
            ).squeeze(1)  # (B, H, W)
            w = torch.where(boundary > 0, self.lambda_edge, 1.0)
            # Zero out weight on ignored pixels so they don't bias the normalizer.
            valid = (targets != self.ignore_index).float()
            w = w * valid
        return (w * per_pixel).sum() / w.sum().clamp_min(1.0)


class BoundaryAwareDiceLoss(nn.Module):
    """Boundary-aware Dice.

    Standard full-image Dice + λ_boundary × boundary-only Dice. The boundary
    term sharpens the gradient on boundary pixels (intersection/union computed
    only within boundary mask).

    Single-delta vs the baseline soft_ce_plus_dice: only the Dice part changes — CE
    untouched. Complementary to EdgeAwareSoftCELoss (CE attacks per-pixel
    boundary, Dice attacks IoU-level boundary)..
    """

    def __init__(
        self,
        lambda_boundary: float = 2.0,
        boundary_width: int = 8,
        ignore_index: int = 255,
        smooth: float = 0.05,
    ):
        super().__init__()
        self.lambda_boundary = float(lambda_boundary)
        self.boundary_width = int(boundary_width)
        self.ignore_index = ignore_index
        self.smooth = float(smooth)

    def _dice(
        self,
        probs: torch.Tensor,
        targets_one_hot: torch.Tensor,
        valid: torch.Tensor,
        boundary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # boundary_mask: optional (B, 1, H, W) multiplier on valid pixels.
        if boundary_mask is not None:
            valid = valid * boundary_mask
        inter = (probs * targets_one_hot * valid).sum(dim=(2, 3))
        union = (probs * valid).sum(dim=(2, 3)) + (targets_one_hot * valid).sum(dim=(2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        targets_clamped = targets.clamp(0, num_classes - 1)
        targets_one_hot = F.one_hot(
            targets_clamped, num_classes
        ).permute(0, 3, 1, 2).float()
        valid = (targets != self.ignore_index).unsqueeze(1).float()
        probs = F.softmax(logits, dim=1)

        dice_full = self._dice(probs, targets_one_hot, valid)
        with torch.no_grad():
            boundary = extract_boundary_mask_gpu(
                targets, num_classes=num_classes,
                boundary_width=self.boundary_width,
            )  # (B, 1, H, W)
        dice_bnd = self._dice(probs, targets_one_hot, valid, boundary_mask=boundary)
        return dice_full + self.lambda_boundary * dice_bnd




class OHEMCELoss(nn.Module):
    """Online Hard Example Mining for Cross-Entropy loss."""
    def __init__(self, ignore_index=255, ohem_ratio=0.25):
        super().__init__()
        self.ignore_index = ignore_index
        self.ohem_ratio = ohem_ratio

    def forward(self, logits, targets):
        per_pixel = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction="none")
        loss_flat = per_pixel.view(-1)
        nonzero = loss_flat[loss_flat > 0]
        if len(nonzero) == 0:
            return per_pixel.mean()
        num_hard = max(1, int(len(nonzero) * self.ohem_ratio))
        topk_val, _ = torch.topk(nonzero, num_hard)
        return topk_val.mean()


# ---------------------------------------------------------------------------
# Boundary Loss utilities
# ---------------------------------------------------------------------------


def extract_boundary_mask_gpu(masks_tensor, num_classes, boundary_width=3):
    """GPU-accelerated boundary extraction using vectorized max/min pooling."""
    # One-hot: (B, num_classes, H, W) -> dilation/erosion on all channels at once
    one_hot = F.one_hot(masks_tensor.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    k = boundary_width * 2 + 1
    pad = boundary_width
    dilated = F.max_pool2d(one_hot, kernel_size=k, stride=1, padding=pad)
    eroded = -F.max_pool2d(-one_hot, kernel_size=k, stride=1, padding=pad)
    boundary_gt = (dilated - eroded).max(dim=1, keepdim=True)[0].clamp(0, 1)
    return boundary_gt


class BoundaryLoss(nn.Module):
    """Boundary-aware loss: BCE on boundary pixels + Dice on boundary."""

    def __init__(self, boundary_width=3):
        super().__init__()
        self.boundary_width = boundary_width

    def forward(self, boundary_logits, boundary_gt):
        """
        Args:
            boundary_logits: (B, 1, H, W) raw logits for boundary prediction
            boundary_gt: (B, 1, H, W) float32 boundary ground truth
        """
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(boundary_logits, boundary_gt, reduction="none")
        # Dice loss
        pred_prob = torch.sigmoid(boundary_logits)
        intersection = (pred_prob * boundary_gt).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + boundary_gt.sum(dim=(2, 3))
        dice = (2 * intersection + 1.0) / (union + 1.0)
        dice_loss = 1.0 - dice.mean()

        return bce.mean() + dice_loss

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_confusion(preds: np.ndarray, targets: np.ndarray, num_classes: int, ignore_idx: int = -1) -> np.ndarray:
    if ignore_idx >= 0:
        valid = (targets >= 0) & (targets < num_classes) & (targets != ignore_idx)
    else:
        valid = (targets >= 0) & (targets < num_classes)
    confusion = np.bincount(
        num_classes * targets[valid].astype(int) + preds[valid].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return confusion


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _m2f_predict(model, images):
    """Get semantic logits from M2F model via predict()."""
    B, _, H, W = images.shape
    features_raw = model.backbone.get_intermediate_layers(
        images, n=model.layers_to_extract_0idx, reshape=False, norm=True
    )
    h_p, w_p = H // model.patch_size, W // model.patch_size
    feature_maps = []
    for feat in features_raw:
        feat = feat.reshape(B, h_p, w_p, -1).permute(0, 3, 1, 2).contiguous()
        feature_maps.append(feat)
    return model.decoder.predict(feature_maps, img_size=(H, W))


def collect_routed_diagnostics(model):
    """Run A-main: collect per-iter diagnostics for RoutedSSDLoRAModule instances.

    Returns dict with:
      - routed_n_modules: count of routed modules in model
      - router_s_<name>_mean: mean scale per expert (sem/spa/tex[/spe])
      - router_s_<name>_std:  std of scale across samples and modules
      - router_entropy_mean:  mean entropy of router logits across modules
                              (using softmax over the n_expert logits to get a
                              comparable "which-expert-dominates" signal even
                              though the model uses 2*sigmoid, not softmax)
      - gamma_spe_mean:       mean gamma_spe across modules (0 if no spectral)

    Returns empty dict if model has no routed modules.
    """
    try:
        from model import RoutedSSDLoRAModule
    except ImportError:
        return {}

    routed_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, RoutedSSDLoRAModule):
            routed_modules.append((name, mod))
    if not routed_modules:
        return {}

    diag = {"routed_n_modules": len(routed_modules)}

    # Aggregate router out_proj weights → entropy signal.
    # We don't run router forward here (no input at this point); instead, inspect
    # the learned weights to summarize how far the router has moved from init.
    # init: out_proj.weight = 0, bias = 0.
    out_norms = []
    bias_norms = []
    gamma_spe_vals = []
    for name, mod in routed_modules:
        w = mod.router.out_proj.weight.detach()  # (n_experts, hidden_dim)
        b = mod.router.out_proj.bias.detach()    # (n_experts,)
        # Per-expert norm (L2 across hidden_dim), then mean → "how much this expert's
        # router weights have grown from zero init".
        per_expert_norm = w.norm(dim=1)  # (n_experts,)
        out_norms.append(per_expert_norm)
        bias_norms.append(b)
        if mod.spectral_enabled and hasattr(mod, "gamma_spe"):
            gamma_spe_vals.append(mod.gamma_spe.detach().item())

    if out_norms:
        stacked = torch.stack(out_norms)  # (n_modules, n_experts)
        diag["router_w_norm_mean"] = stacked.mean().item()
        diag["router_w_norm_std"] = stacked.std().item()
        per_expert = stacked.mean(dim=0)  # (n_experts,)
        expert_names = ["sem", "spa", "tex"]
        if routed_modules[0][1].spectral_enabled:
            expert_names.append("spe")
        for i, ename in enumerate(expert_names):
            diag[f"router_w_{ename}_mean"] = per_expert[i].item()

    if bias_norms:
        bias_stacked = torch.stack(bias_norms)  # (n_modules, n_experts)
        diag["router_bias_norm_mean"] = bias_stacked.norm(dim=1).mean().item()

    if gamma_spe_vals:
        diag["gamma_spe_mean"] = sum(gamma_spe_vals) / len(gamma_spe_vals)

    return diag


def build_optimizer_with_groups(model, lr_groups_cfg, weight_decay):
    """Build AdamW with Run A-main 4-group LR (override 2026-06-17)
    + R1 SF-Adapter group (override 2026-06-20).

    Groups (per override §"Optimizer Groups"):
      - decoder_r0: existing MLPDecoder params (lr from cfg)
      - lora_core:  existing SSD-LoRA params — lora_sem_*/lora_str_*, alpha_sem,
                    alpha_str, spatial_conv, tcam (lr from cfg)
      - router_gate: new router params (router.*, gamma_spe) (lr from cfg)
      - expert_new:  new high-source spectral params (lora_spe_*, feat_spectral_gate)
                     (lr from cfg). Empty if spectral disabled.
      - sf_adapter:  R1 Late SF-Adapter params (sf_adapters.*) (lr from cfg).
                     Empty if SF adapter disabled. Default LR = decoder_r0 LR
                     if unspecified.

    Backbone base params are excluded by requires_grad=False before this call.
    """
    LORA_CORE_KEYWORDS = (
        "lora_sem_", "lora_str_",
        "alpha_sem", "alpha_str",
        "spatial_conv", "tcam",
    )
    EXPERT_NEW_KEYWORDS = (
        "lora_spe_", "feat_spectral_gate", "gamma_spe",
    )
    ROUTER_KEYWORDS = ("router.",)
    SF_ADAPTER_KEYWORDS = ("sf_adapters.",)

    groups = {
        "decoder_r0": [],
        "lora_core": [],
        "router_gate": [],
        "expert_new": [],
        "sf_adapter": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ROUTER_KEYWORDS):
            groups["router_gate"].append(param)
        elif any(k in name for k in EXPERT_NEW_KEYWORDS):
            groups["expert_new"].append(param)
        elif any(k in name for k in SF_ADAPTER_KEYWORDS):
            groups["sf_adapter"].append(param)
        elif any(k in name for k in LORA_CORE_KEYWORDS):
            groups["lora_core"].append(param)
        elif name.startswith("decoder."):
            groups["decoder_r0"].append(param)
        else:
            # Default: treat unknown trainable params as lora_core (safer than
            # decoder_r0; avoids accidentally freezing new modules out of the
            # high-LR groups). All currently-known trainable params fall into
            # one of the explicit branches above.
            groups["lora_core"].append(param)

    param_groups = [
        {"params": groups["decoder_r0"], "lr": lr_groups_cfg["decoder_r0"], "weight_decay": weight_decay},
        {"params": groups["lora_core"], "lr": lr_groups_cfg["lora_core"], "weight_decay": 0.0},
        {"params": groups["router_gate"], "lr": lr_groups_cfg["router_gate"], "weight_decay": 0.0},
    ]
    # expert_new is omitted entirely if empty (override: don't create dummy groups).
    if groups["expert_new"]:
        if "expert_new" not in lr_groups_cfg:
            raise ValueError(
                "expert_new params detected but lr_groups.expert_new not set in config. "
                "Either add spectral params and set expert_new LR, or disable spectral."
            )
        param_groups.append(
            {"params": groups["expert_new"], "lr": lr_groups_cfg["expert_new"], "weight_decay": 0.0}
        )
    # sf_adapter is omitted entirely if empty (no SF params in this run).
    if groups["sf_adapter"]:
        # Default: reuse decoder_r0 LR if not explicitly specified. This is
        # appropriate for a new module that needs to learn from scratch.
        sf_lr = lr_groups_cfg.get("sf_adapter", lr_groups_cfg["decoder_r0"])
        param_groups.append(
            {"params": groups["sf_adapter"], "lr": sf_lr, "weight_decay": 0.0}
        )

    for gname in ["decoder_r0", "lora_core", "router_gate", "expert_new", "sf_adapter"]:
        n_params = sum(p.numel() for p in groups[gname])
        lr_val = lr_groups_cfg.get(gname, "—")
        print(f"  LR group {gname}: {len(groups[gname])} tensors, {n_params/1e6:.2f}M params, lr={lr_val}")
    return torch.optim.AdamW(param_groups)


def train(cfg):
    dataset_cls = get_dataset_class(cfg)
    num_classes = cfg.get("num_classes", 7)
    decoder_type = cfg.get("decoder", {}).get("type", "mlp")

    # ---- Data ----
    train_dataset = get_dataset(cfg, "train")
    val_dataset = get_dataset(cfg, "val")

    # Route A subset filter: keep only tiles where target classes
    # meet a pixel-ratio threshold. Default is "" (no filter; full train). Only
    # set when running the causal pre-test; main run leaves it empty.
    ra_subset = ((cfg.get("route_a") or {}).get("subset_filter") or "").strip()
    if ra_subset:
        from torch.utils.data import Subset
        _cache = os.path.join(
            cfg.get("output_dir", "outputs"),
            f"subset_filter_{ra_subset}.json",
        )
        kept_idx = compute_subset_filter(
            train_dataset, preset=ra_subset,
            ignore_value=cfg.get("ignore_index", 255),
            cache_path=_cache,
        )
        train_dataset = Subset(train_dataset, kept_idx)
        print(f"[route_a] subset filter applied: {len(train_dataset)} tiles "
              f"(preset={ra_subset!r})")

    nw = cfg.get("num_workers", 4)
    bs = cfg.get("batch_size", 4)
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=True, drop_last=True, persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True,
    )

    # ---- Model ----
    model = get_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    # ---- Warm-start from R0 (Run A-main: config-driven, before optimizer/EMA) ----
    # Loads backbone + LoRA/TCAM from R0 80.95% checkpoint. Decoder is loaded
    # only if cfg.warm_start_load_decoder=true (default false for SCHRR; Run
    # A-main sets it true because we keep R0 MLPDecoder structure).
    # Must be BEFORE optimizer/EMA creation so EMA shadow = warm-started model.
    warm_start_path = cfg.get("warm_start", None)
    warm_start_cfg = cfg.get("warm_start_config", {}) or {}
    warm_start_load_decoder = bool(warm_start_cfg.get("load_decoder", True))
    warm_start_prefer_ema = bool(warm_start_cfg.get("prefer_ema", True))
    warm_start_skip_prefixes = tuple(warm_start_cfg.get("skip_prefixes", []))

    if warm_start_path and os.path.isfile(warm_start_path):
        ckpt = torch.load(warm_start_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and warm_start_prefer_ema and "ema" in ckpt:
            pretrained_state = ckpt["ema"]
            warm_start_source = "ema"
        elif isinstance(ckpt, dict) and "model" in ckpt:
            pretrained_state = ckpt["model"]
            warm_start_source = "model"
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            pretrained_state = ckpt["state_dict"]
            warm_start_source = "state_dict"
        else:
            pretrained_state = ckpt
            warm_start_source = "raw"
        model_state = model.state_dict()

        # Load strategy (Run A-main):
        #   - Always load backbone.* (DINOv3 + internal SSD-LoRA / TCAM)
        #   - Load decoder.* ONLY if warm_start_load_decoder=true
        #     (Run A-main keeps the R0 MLPDecoder fusion/head structure, so
        #     decoder.* keys should match exactly)
        #   - Always skip any prefix in warm_start_skip_prefixes (manual override)
        ALLOWED_PREFIXES = ("backbone.",)
        if warm_start_load_decoder:
            ALLOWED_PREFIXES_WITH_DEC = ("backbone.", "decoder.")
        else:
            ALLOWED_PREFIXES_WITH_DEC = ("backbone.",)

        matched = {}
        skipped_decoder = []
        skipped_other = []
        skipped_by_user = []
        for k, v in pretrained_state.items():
            if warm_start_skip_prefixes and any(k.startswith(p) for p in warm_start_skip_prefixes):
                skipped_by_user.append(k)
                continue
            if not warm_start_load_decoder and k.startswith("decoder."):
                skipped_decoder.append(k)
                continue
            if not any(k.startswith(p) for p in ALLOWED_PREFIXES_WITH_DEC):
                skipped_other.append(k)
                continue
            if k in model_state and v.shape == model_state[k].shape:
                matched[k] = v
            else:
                skipped_other.append(f"{k} (shape mismatch or not in model)")
        model_state.update(matched)
        model.load_state_dict(model_state)
        missing = [k for k in model_state if k not in matched]
        print(f"Warm-started {len(matched)} keys from {warm_start_path} ({warm_start_source})")
        print(f"  load_decoder={warm_start_load_decoder}, skip_prefixes={list(warm_start_skip_prefixes)}")
        print(f"  Skipped decoder (load_decoder=false): {len(skipped_decoder)} keys")
        print(f"  Skipped by user (skip_prefixes): {len(skipped_by_user)} keys")
        print(f"  Skipped other (no match / shape mismatch): {len(skipped_other)} keys")
        print(f"  Missing in new model (random init): {len(missing)} keys")
        # Show a few missing key categories for debug.
        if missing:
            missing_categories = set()
            for k in missing[:50]:
                parts = k.split(".")
                if len(parts) >= 3 and parts[0] == "backbone" and parts[1] == "blocks":
                    # e.g. backbone.blocks.0.attn.qkv.router.out_proj.weight
                    missing_categories.add(".".join(parts[:5]))
                else:
                    missing_categories.add(".".join(parts[:3]))
            print(f"  Missing key category samples: {sorted(missing_categories)[:8]}")
    elif warm_start_path:
        print(f"WARNING: warm_start path not found: {warm_start_path}")

    # ---- Backbone base freeze (Run A-main: after warm_start, before optimizer) ----
    # Freeze base DINO weights, keep all PEFT-style params trainable:
    #   - lora_sem_*/lora_str_* (LoRA A/B matrices)
    #   - alpha_sem/alpha_str (LoRA alphas)
    #   - spatial_conv (SSD-LoRA dwconv/pw)
    #   - tcam (texture channel attention)
    #   - router.* (ExpertRouter in RoutedSSDLoRAModule)
    #   - gamma_spe (high-source spectral gate scalar)
    #   - lora_spe_*/feat_spectral_gate (high-source spectral, optional)
    lr_groups_cfg = cfg.get("lr_groups", None)
    if lr_groups_cfg:
        PEFT_KEYWORDS = (
            "lora_sem_", "lora_str_", "lora_spe_",
            "alpha_sem", "alpha_str",
            "spatial_conv", "tcam",
            "router.", "gamma_spe",
            "feat_spectral_gate",
        )
        n_frozen = 0
        n_peft_kept = 0
        for name, param in model.named_parameters():
            if name.startswith("backbone.") and not any(k in name for k in PEFT_KEYWORDS):
                param.requires_grad = False
                n_frozen += 1
            elif name.startswith("backbone.") and any(k in name for k in PEFT_KEYWORDS):
                n_peft_kept += 1
        print(f"Backbone freeze: {n_frozen} base params frozen, {n_peft_kept} PEFT params kept trainable")

    # ---- Optimizer ----
    if lr_groups_cfg:
        # v2: 4-group LR (decoder/hr_encoder/router_gate=1.5e-4, lora_tcam=3.0e-5)
        optimizer = build_optimizer_with_groups(model, lr_groups_cfg, cfg.get("weight_decay", 0.05))
    else:
        # Legacy: single-LR or LoRA+ (for non-v2 configs)
        lora_cfg = cfg.get("lora", {})
        lora_plus_lambda = lora_cfg.get("lora_plus_lambda", None)

        if lora_plus_lambda and lora_plus_lambda > 1:
            # LoRA+: separate A and B with different learning rates (Hayou 2024)
            lora_A_params = []
            lora_B_params = []
            alpha_params = []
            lora_other_params = []  # spatial_conv etc.
            decoder_params = []
            fad_v2_params = []  # FAD-v2 (Haar wavelet block on c2)
            epfo_params = []   # EPFO (sparse boundary head)
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "lora_sem_A" in name or "lora_str_A" in name:
                    lora_A_params.append(param)
                elif "lora_sem_B" in name or "lora_str_B" in name:
                    lora_B_params.append(param)
                elif "alpha_sem" in name or "alpha_str" in name:
                    alpha_params.append(param)
                elif "spatial_conv" in name:
                    lora_other_params.append(param)
                elif "fad_v2" in name:
                    # FAD-v2 zero-init Conv1x1 + per-channel gamma gate.
                    # wd=0 to protect gamma gate from L2 decay; same LR as decoder.
                    fad_v2_params.append(param)
                elif "epfo" in name:
                    # EPFO MLP + fc_seg. Normal init (std=0.01).
                    # wd=0 to protect point-prediction head from L2 shrinkage.
                    epfo_params.append(param)
                else:
                    decoder_params.append(param)
            lr = cfg.get("lr", 1.5e-4)
            wd = cfg.get("weight_decay", 0.05)
            param_groups = [
                {"params": lora_A_params, "lr": lr, "weight_decay": 0.0},
                {"params": lora_B_params, "lr": lr * lora_plus_lambda, "weight_decay": 0.0},
                {"params": alpha_params, "lr": lr, "weight_decay": 0.0},
                {"params": lora_other_params, "lr": lr, "weight_decay": 0.0},
                {"params": decoder_params, "lr": lr, "weight_decay": wd},
            ]
            if fad_v2_params:
                param_groups.append(
                    {"params": fad_v2_params, "lr": lr, "weight_decay": 0.0}
                )
                n_fad_v2 = sum(p.numel() for p in fad_v2_params)
                print(f"FAD-v2 optimizer group: {len(fad_v2_params)} tensors, "
                      f"{n_fad_v2:,} params, lr={lr}, wd=0.0")
            if epfo_params:
                param_groups.append(
                    {"params": epfo_params, "lr": lr, "weight_decay": 0.0}
                )
                n_epfo = sum(p.numel() for p in epfo_params)
                print(f"EPFO optimizer group: {len(epfo_params)} tensors, "
                      f"{n_epfo:,} params, lr={lr}, wd=0.0")
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
            print(f"LoRA+ enabled: λ={lora_plus_lambda}, A_lr={lr}, B_lr={lr * lora_plus_lambda}")
        else:
            lora_params = []
            decoder_params = []
            fad_v2_params = []
            epfo_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "fad_v2" in name:
                    fad_v2_params.append(param)
                elif "epfo" in name:
                    epfo_params.append(param)
                elif any(k in name for k in ("lora", "ssd", "alpha_sem", "alpha_str", "spatial_conv")):
                    lora_params.append(param)
                else:
                    decoder_params.append(param)

            lr = cfg.get("lr", 1.5e-4)
            wd = cfg.get("weight_decay", 0.05)
            param_groups = [
                {"params": lora_params, "weight_decay": 0.0},
                {"params": decoder_params, "weight_decay": wd},
            ]
            if fad_v2_params:
                param_groups.append(
                    {"params": fad_v2_params, "lr": lr, "weight_decay": 0.0}
                )
                n_fad_v2 = sum(p.numel() for p in fad_v2_params)
                print(f"FAD-v2 optimizer group: {len(fad_v2_params)} tensors, "
                      f"{n_fad_v2:,} params, lr={lr}, wd=0.0")
            if epfo_params:
                param_groups.append(
                    {"params": epfo_params, "lr": lr, "weight_decay": 0.0}
                )
                n_epfo = sum(p.numel() for p in epfo_params)
                print(f"EPFO optimizer group: {len(epfo_params)} tensors, "
                      f"{n_epfo:,} params, lr={lr}, wd=0.0")
            optimizer = torch.optim.AdamW(param_groups, lr=lr)

    # ---- Loss ----
    cw = cfg.get("class_weights", None)
    if cw is not None:
        cw = torch.tensor(cw, dtype=torch.float32).to(device)
        print(f"Class weights: {cw.tolist()}")
    # Dataset-specific ignore_index: LoveDA decoded to 0-6+255 at load (2026-08-21 fix), so always 255
    ignore_index = 255  # Potsdam has no ignore

    # loss_type selector (P1: soft_ce_plus_dice per MM-DINO spec;
    # P1.5: cp_soft_ce_plus_dice = Confusion-pair SoftCE + optional Edge-point CE)
    loss_type = cfg.get("loss_type", "ce_plus_dice")
    ohem_ratio = cfg.get("ohem_ratio", None)
    edge_loss_fn = None  # P1.5: optional side loss, default None
    if loss_type == "soft_ce_plus_dice":
        # MM-DINO spec: label smoothing on CE
        soft_smooth = float(cfg.get("soft_ce_smooth", 0.05))
        ce_loss_fn = SoftCrossEntropyLoss(smooth_factor=soft_smooth, ignore_index=ignore_index)
        dice_smooth = float(cfg.get("dice_smooth", 0.05))
        dice_loss_fn = DiceLoss(ignore_index=ignore_index, smooth=dice_smooth)
        print(f"Loss: SoftCE(smooth={soft_smooth}) + Dice(smooth={dice_smooth})")
    elif loss_type == "cp_soft_ce_plus_dice":
        # P1.5 (v7.2 plan): Confusion-pair SoftCE + Dice + optional Edge-point CE.
        # Required cfg: confusion_pairs (list of [a, b] class index pairs).
        # Defaults to λ_cp=2.0 per spec; edge CE optional (weight=0 disables).
        cp_pairs_cfg = cfg.get("confusion_pairs", [[0, 5]])  # default: Impervious↔Clutter
        cp_pairs = [tuple(p) for p in cp_pairs_cfg]
        lambda_cp = float(cfg.get("lambda_cp", 2.0))
        soft_smooth = float(cfg.get("soft_ce_smooth", 0.05))
        ce_loss_fn = ConfusionPairSoftCELoss(
            pairs=cp_pairs, lambda_cp=lambda_cp,
            smooth_factor=soft_smooth, ignore_index=ignore_index,
        )
        dice_smooth = float(cfg.get("dice_smooth", 0.05))
        dice_loss_fn = DiceLoss(ignore_index=ignore_index, smooth=dice_smooth)
        # Optional Edge-point CE side loss (minimal EPFO port)
        edge_weight = float(cfg.get("edge_weight", 0.0))
        if edge_weight > 0:
            edge_k = int(cfg.get("edge_k_pixels", 4096))
            edge_loss_fn = EdgePointCELoss(
                k_pixels=edge_k, weight=edge_weight,
                smooth_factor=0.0, ignore_index=ignore_index,
            )
            print(f"Loss: ConfusionPairSoftCE(pairs={cp_pairs}, λ_cp={lambda_cp}, smooth={soft_smooth}) "
                  f"+ Dice(smooth={dice_smooth}) + EdgePointCE(K={edge_k}, w={edge_weight})")
        else:
            print(f"Loss: ConfusionPairSoftCE(pairs={cp_pairs}, λ_cp={lambda_cp}, smooth={soft_smooth}) "
                  f"+ Dice(smooth={dice_smooth}) [edge CE disabled]")
    elif loss_type == "edge_aware_ce_plus_dice":
        # EdgeCE: boundary-weighted SoftCE (λ_edge on GT-boundary pixels)
        # + standard Dice. Zero-param loss, attacks D0.2 -26pt boundary gap.
        soft_smooth = float(cfg.get("soft_ce_smooth", 0.05))
        lambda_edge = float(cfg.get("lambda_edge", 2.0))
        boundary_width = int(cfg.get("boundary_width", 8))
        dice_smooth = float(cfg.get("dice_smooth", 0.05))
        ce_loss_fn = EdgeAwareSoftCELoss(
            lambda_edge=lambda_edge, boundary_width=boundary_width,
            smooth_factor=soft_smooth, ignore_index=ignore_index,
        )
        dice_loss_fn = DiceLoss(ignore_index=ignore_index, smooth=dice_smooth)
        print(f"Loss: EdgeAwareSoftCE(λ_edge={lambda_edge}, bw={boundary_width}, "
              f"smooth={soft_smooth}) + Dice(smooth={dice_smooth})")
    elif loss_type == "soft_ce_plus_boundary_dice":
        # BoundaryDice: standard SoftCE + full Dice + λ_boundary ×
        # boundary-only Dice. Zero-param loss, attacks D0.2 boundary bottleneck.
        soft_smooth = float(cfg.get("soft_ce_smooth", 0.05))
        lambda_boundary = float(cfg.get("lambda_boundary", 2.0))
        boundary_width = int(cfg.get("boundary_width", 8))
        dice_smooth = float(cfg.get("dice_smooth", 0.05))
        ce_loss_fn = SoftCrossEntropyLoss(smooth_factor=soft_smooth, ignore_index=ignore_index)
        dice_loss_fn = BoundaryAwareDiceLoss(
            lambda_boundary=lambda_boundary, boundary_width=boundary_width,
            ignore_index=ignore_index, smooth=dice_smooth,
        )
        print(f"Loss: SoftCE(smooth={soft_smooth}) + BoundaryAwareDice("
              f"λ_b={lambda_boundary}, bw={boundary_width}, smooth={dice_smooth})")
    elif loss_type == "class_weighted_ce_plus_dice":
        # ClassWeight: per-class weighted SoftCE + standard Dice.
        # Zero-param loss, attacks D0.0 rural→Building absorption (Forest/Agri).
        # Default weights boost LoveDA classes 5 (Forest) and 6 (Agriculture).
        soft_smooth = float(cfg.get("soft_ce_smooth", 0.05))
        cw_cfg = cfg.get("class_weights_loveda", [1.0, 1.0, 1.0, 1.0  , 1.0, 2.0, 2.0])
        cw_tensor = torch.tensor(cw_cfg, dtype=torch.float32).to(device)
        dice_smooth = float(cfg.get("dice_smooth", 0.05))
        ce_loss_fn = WeightedSoftCrossEntropyLoss(
            weight=cw_tensor, smooth_factor=soft_smooth, ignore_index=ignore_index,
        )
        dice_loss_fn = DiceLoss(ignore_index=ignore_index, smooth=dice_smooth)
        print(f"Loss: WeightedSoftCE(weights={cw_cfg}, smooth={soft_smooth}) "
              f"+ Dice(smooth={dice_smooth})")
    elif loss_type == "class_weighted_ce_plus_boundary_dice":
        # stack the winning variants — per-class weighted SoftCE (ClassWeight)
        # + BoundaryAwareDice (full Dice + λ_boundary × boundary Dice).
        # Zero-param loss. Cumulative attack on D0.0 rural→Building absorption
        # (Forest/Agri 2× boost) AND D0.2 boundary bottleneck (bDice).
        # Per D0.2 post-Wave-D: BDice uniform gain + F→B -6.57%,
        # CW F→B -3.40% — mechanisms complementary, expect cumulative delta.
        soft_smooth = float(cfg.get("soft_ce_smooth", 0.05))
        cw_cfg = cfg.get("class_weights_loveda", [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
        cw_tensor = torch.tensor(cw_cfg, dtype=torch.float32).to(device)
        lambda_boundary = float(cfg.get("lambda_boundary", 2.0))
        boundary_width = int(cfg.get("boundary_width", 8))
        dice_smooth = float(cfg.get("dice_smooth", 0.05))
        ce_loss_fn = WeightedSoftCrossEntropyLoss(
            weight=cw_tensor, smooth_factor=soft_smooth, ignore_index=ignore_index,
        )
        dice_loss_fn = BoundaryAwareDiceLoss(
            lambda_boundary=lambda_boundary, boundary_width=boundary_width,
            ignore_index=ignore_index, smooth=dice_smooth,
        )
        print(f"Loss: WeightedSoftCE(weights={cw_cfg}, smooth={soft_smooth}) "
              f"+ BoundaryAwareDice(λ_b={lambda_boundary}, bw={boundary_width}, "
              f"smooth={dice_smooth})")
    elif loss_type == "soft_ce_plus_bdice_v2":
        # BoundaryDice-v2. Three modes via boundary_weight_type:
        #   - "gate_closed"   : λ≡0, w≡1 — sanity + matched control. Reduces to baseline.
        #   - "per_cc_entropy" : Per-CC equal-budget entropy weighting (FLEX-Seg UBE).
        #   - "ring_difference": 3-granularity ring-difference static weights.
        # Loss formula (additive):
        #   L = [SoftCE_full + Dice_full] + λ(t) · (1/N_band) · Σ_band w(x,y) · SoftCE(x,y)
        # λ(t) closed form: λ(t) = min(1, t / (0.3 · T_total)).
        from bdice_v2 import BDiceV2Loss
        bwt = str(cfg.get("boundary_weight_type", "gate_closed"))
        soft_smooth = float(cfg.get("soft_ce_smooth", 0.05))
        dice_smooth = float(cfg.get("dice_smooth", 0.05))
        # Hyperparameters:
        # NOTE: total_steps computed inline here (epochs/grad_accum_steps are read from cfg;
        # the local re-reads at line ~1354 compute the same value for the scheduler).
        _bdice_v2_epochs = int(cfg.get("epochs", 50))
        _bdice_v2_grad_accum = int(cfg.get("grad_accum_steps", 1))
        _bdice_v2_steps_per_epoch = max(1, len(train_loader) // _bdice_v2_grad_accum)
        _bdice_v2_total_steps = _bdice_v2_steps_per_epoch * _bdice_v2_epochs
        bdice_v2_kwargs = dict(
            num_classes=int(cfg.get("num_classes", 7)),
            boundary_weight_type=bwt,
            smooth_factor=soft_smooth,
            dice_smooth=dice_smooth,
            ignore_index=ignore_index,
            alpha=float(cfg.get("bdice_v2_alpha", 3.0)),
            a_min=int(cfg.get("bdice_v2_a_min", 25)),
            ema_beta=float(cfg.get("bdice_v2_ema_beta", 0.9)),
            ema_warmup_steps=int(cfg.get("bdice_v2_ema_warmup_steps", 100)),
            eps=float(cfg.get("bdice_v2_eps", 1e-6)),
            k_main=int(cfg.get("bdice_v2_k_main", 3)),
            k_thin=int(cfg.get("bdice_v2_k_thin", 2)),
            k_medium=int(cfg.get("bdice_v2_k_medium", 5)),
            k_thick=int(cfg.get("bdice_v2_k_thick", 10)),
            w_thin=float(cfg.get("bdice_v2_w_thin", 3.0)),
            w_medium_thin=float(cfg.get("bdice_v2_w_medium_thin", 2.0)),
            w_thick_medium=float(cfg.get("bdice_v2_w_thick_medium", 1.0)),
            ramp_frac=float(cfg.get("bdice_v2_ramp_frac", 0.3)),
            total_steps=_bdice_v2_total_steps,
            dual_label=bool(cfg.get("dual_label", False)),
            band_class_weights=cfg.get("bdice_v2_band_class_weights", None),
        )
        bdice_v2_loss_fn = BDiceV2Loss(**bdice_v2_kwargs)
        # Stub ce_loss_fn / dice_loss_fn for compat with logging path. Actual loss is
        # computed inside the BDiceV2Loss forward (L_main + λ·L_boundary coupled).
        ce_loss_fn = None
        dice_loss_fn = None
        print(f"Loss: BDiceV2(type={bwt}, α={bdice_v2_kwargs['alpha']}, "
              f"A_min={bdice_v2_kwargs['a_min']}, K_main={bdice_v2_kwargs['k_main']}, "
              f"ramp_frac={bdice_v2_kwargs['ramp_frac']}, "
              f"total_steps={bdice_v2_kwargs['total_steps']}, "
              f"dual_label={bdice_v2_kwargs['dual_label']})")
    elif ohem_ratio is not None and ohem_ratio > 0:
        if cw is not None:
            print("WARNING: OHEM and class_weights both set; class_weights ignored.")
        ce_loss_fn = OHEMCELoss(ignore_index=ignore_index, ohem_ratio=ohem_ratio)
        dice_loss_fn = DiceLoss(ignore_index=ignore_index)
        print(f"OHEM enabled: ratio={ohem_ratio}")
    else:
        ce_loss_fn = nn.CrossEntropyLoss(weight=cw, ignore_index=ignore_index)
        dice_loss_fn = DiceLoss(ignore_index=ignore_index)
        print(f"Loss: CE + Dice (default)")
    dice_weight = cfg.get("dice_weight", 1.0)

    # ---- EPFO point-CE side-loss weight ----
    # Only active when decoder.epfo.enabled is true. Default 0.5.
    # total_loss = seg_loss + lambda_point * point_ce(point_logits, point_gt)
    lambda_point = float(cfg.get("lambda_point", 0.5))
    _epfo_module = (
        getattr(getattr(model, "decoder", None), "epfo", None)
        if hasattr(model, "decoder") else None
    )
    if _epfo_module is not None:
        print(f"EPFO point-CE: enabled, lambda_point={lambda_point}")

    # ---- Semantic-preserving consistency ----
    # Teacher = model forward with SF adapter disabled (base path), no_grad.
    # Only active when sf_adapter is enabled (consistency targets the SF delta).
    _sf_on = bool(cfg.get("sf_adapter", {}).get("enabled", False))
    cons_cfg = cfg.get("consistency", {}) or {}
    cons_enabled = bool(cons_cfg.get("enabled", False)) and _sf_on
    cons_kl_enabled = bool(cons_cfg.get("kl_enabled", True))
    cons_aff_enabled = bool(cons_cfg.get("affinity_enabled", True))
    cons_lambda_kl = float(cons_cfg.get("lambda_kl", 0.05))
    cons_lambda_rel = float(cons_cfg.get("lambda_rel", 0.02))
    cons_tau = float(cons_cfg.get("tau", 2.0))
    cons_conf = float(cons_cfg.get("conf_thresh", 0.70))
    cons_k = int(cons_cfg.get("affinity_tokens", 64))
    cons_need_feat = cons_enabled and cons_aff_enabled
    if cons_enabled:
        print(
            f"Consistency enabled (teacher=adapter-disabled base path): "
            f"kl={cons_kl_enabled}(lambda={cons_lambda_kl}) "
            f"aff={cons_aff_enabled}(lambda={cons_lambda_rel}) "
            f"tau={cons_tau} conf>={cons_conf} K={cons_k}"
        )

    # ---- M2F Loss (for mask2former decoder) ----
    is_m2f = (decoder_type == "mask2former")
    m2f_criterion = None
    if is_m2f:
        m2f_cfg = cfg.get("decoder", {}).get("mask2former", {})
        matcher = HungarianMatcher(
            cost_class=m2f_cfg.get("cost_class", 2.0),
            cost_mask=m2f_cfg.get("cost_mask", 5.0),
            cost_dice=m2f_cfg.get("cost_dice", 5.0),
            num_points=m2f_cfg.get("num_points", 12544),
        )
        m2f_criterion = M2FSetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_ce=m2f_cfg.get("weight_ce", 2.0),
            weight_mask=m2f_cfg.get("weight_mask", 5.0),
            weight_dice=m2f_cfg.get("weight_dice", 5.0),
            eos_coef=m2f_cfg.get("eos_coef", 0.1),
            num_points=m2f_cfg.get("num_points", 12544),
        ).to(device)
        print("M2F set prediction loss enabled")

    # ---- Boundary Loss ----
    boundary_weight = cfg.get("boundary_weight", 0.0)
    boundary_loss_fn = None
    boundary_head = None
    if boundary_weight > 0:
        boundary_loss_fn = BoundaryLoss(boundary_width=cfg.get("boundary_width", 3))
        boundary_head = nn.Sequential(
            nn.Conv2d(num_classes, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        ).to(device)
        print(f"Boundary loss enabled: weight={boundary_weight}, width={cfg.get('boundary_width', 3)}")

    # ---- Scheduler ----
    epochs = cfg.get("epochs", 50)
    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    grad_clip_norm = cfg.get("grad_clip_norm", 1.0)
    # Treat 0 / None / <= 0 as "no clipping" — passing max_norm=0 to
    # clip_grad_norm_ would ZERO all gradients (scale = 0/total_norm = 0).
    if grad_clip_norm is None or grad_clip_norm <= 0:
        grad_clip_norm = None  # disables clip below
        print("[Grad clip] disabled (grad_clip_norm<=0 in config)")
    else:
        print(f"[Grad clip] max_norm={grad_clip_norm}")
    warmup = cfg.get("warmup_iters", 1000)
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = steps_per_epoch * epochs

    scheduler_type = cfg.get("scheduler", "lambda")
    if scheduler_type == "cosine_anneal":
        # True CosineAnnealingLR (MM-DINO spec) — T_max=total_steps, eta_min from cfg.
        # Warmup is folded in via SequentialLR + LinearLR warmup prefix.
        eta_min = float(cfg.get("eta_min", 1.0e-7))
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=max(1, warmup),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup),
            eta_min=eta_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[max(1, warmup)],
        )
        print(f"Scheduler: cosine_anneal (T_max={total_steps - warmup}, eta_min={eta_min}, warmup={warmup})")
    else:
        # Legacy: custom LambdaLR with polynomial decay
        def _lr_lambda(step: int) -> float:
            if step >= total_steps:
                return 0.0
            warmup_factor = min(1.0, step / max(1, warmup))
            decay_factor = max(0.0, (1.0 - step / total_steps)) ** 0.9
            return warmup_factor * decay_factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        print(f"Scheduler: lambda (poly decay 0.9, warmup={warmup}, total_steps={total_steps})")

    # ---- AMP ----
    use_amp = cfg.get("fp16", True)
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- EMA ----
    ema_cfg = cfg.get("ema", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_decay = ema_cfg.get("decay", 0.9995)
    ema_start_step = ema_cfg.get("start_step", 0)
    ema = ModelEMA(model, decay=ema_decay, device=device) if ema_enabled else None
    if ema_enabled:
        print(f"EMA enabled: decay={ema_decay}, start_step={ema_start_step}")

    # ---- Route A: prototype contrastive auxiliary loss ----
    # Gated by cfg.route_a.enabled. Side-loss on p2 @ H/4 via return_features=True.
    route_a_cfg = RouteAConfig.from_cfg(cfg)
    route_a_module = None
    if route_a_cfg.enabled:
        # Determine feat_dim from a one-shot forward (decoder embed_dim)
        route_a_module = RouteAModule(route_a_cfg, feat_dim=cfg.get("decoder", {}).get("embed_dim", 256), device=device)
        route_a_module.to(device)
        print(f"[route_a] Module built: row={route_a_cfg.row}, K={route_a_cfg.K}, "
              f"omega={route_a_cfg.omega}, init_mode={route_a_cfg.init_mode}, "
              f"use_ema={route_a_cfg.use_ema}, loss_mode={route_a_cfg.loss_mode}")

        # Collect p2 features for prototype init (G2 K-means or class-mean)
        # Uses warm-init-loaded model + first N train tiles.
        print(f"[route_a] Collecting p2 features for prototype init "
              f"(n_tiles={route_a_cfg.init_sample_n_tiles}, seed={route_a_cfg.init_seed})")
        feats_init, labels_init = collect_p2_features_for_init(
            model=model,
            train_loader=train_loader,
            num_classes=route_a_cfg.num_classes,
            ignore_value=route_a_cfg.ignore_value,
            n_tiles=route_a_cfg.init_sample_n_tiles,
            device=device,
            seed=route_a_cfg.init_seed,
        )
        print(f"[route_a] Init features: {feats_init.shape}, labels: {labels_init.shape}, "
              f"per-class counts={[int((labels_init == c).sum()) for c in range(route_a_cfg.num_classes)]}")
        init_protos = fit_init_prototypes(
            features=feats_init.to(device),
            labels=labels_init.to(device),
            num_classes=route_a_cfg.num_classes,
            K=route_a_cfg.K,
            mode=route_a_cfg.init_mode,
            seed=route_a_cfg.init_seed,
        )
        route_a_module.initialize_prototypes(init_protos)
        print(f"[route_a] Prototype bank initialized "
              f"({route_a_cfg.num_classes}×{route_a_cfg.K}×{feats_init.shape[1]})")
        # Restore train mode (collect_p2_features_for_init sets eval)
        model.train()

    # ---- Stable-LoRA hook (Phase 1) ----
    # Pre-optimizer-step shrinkage of SSD-LoRA Kaiming-init matrices (lora_*_A)
    # toward the zero-init matrices (lora_*_B), per Stable-LoRA (ICLR 2026).
    # Bound by explicit (shrink, paired) pairs at construction time; not index-based.
    stable_lora_cfg = cfg.get("stable_lora", {}) or {}
    stable_lora_enabled = bool(stable_lora_cfg.get("enabled", False))
    stable_lora_hook = None
    if stable_lora_enabled:
        from stable_lora_hook import StableLoRAShrinkageHook
        stable_lora_hook = StableLoRAShrinkageHook(
            model,
            shrink_ratio=float(stable_lora_cfg.get("shrink_ratio", 1e-3)),
            verbose=True,
        )
        # AMP-safe placement: register shrinkage as an optimizer pre-hook so it
        # fires only when scaler.step(optimizer) actually invokes
        # optimizer.step(). On an AMP step where GradScaler detects inf/NaN,
        # scaler.step() returns without calling optimizer.step(), so the hook
        # (and thus the shrinkage) is correctly skipped. Inline placement
        # before scaler.step() would shrink on skipped steps — a real bug.
        def _stable_lora_pre_step_hook(_optimizer, _args, _kwargs):
            stable_lora_hook.pre_step_hook()
        optimizer.register_step_pre_hook(_stable_lora_pre_step_hook)
        print(f"Stable-LoRA: registered optimizer pre-step hook (shrink_ratio={float(stable_lora_cfg.get('shrink_ratio', 1e-3))})")

    # ---- Aux-loss coefficient (Phase 1) ----
    # Multiplier for the model's aggregated aux losses (e.g. Conv-LoRA MoE
    # balancing loss). Set aux_loss_lambda_moe > 0 in config to apply.
    aux_lambda_moe = float(cfg.get("aux_loss_lambda_moe", 0.0))

    # ---- TensorBoard ----
    log_dir = cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    output_dir = cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting training: {epochs} epochs, {len(train_loader)} batches/epoch, grad_accum={grad_accum_steps}")
    print(f"Effective steps/epoch: {steps_per_epoch}, total_steps: {total_steps}")
    print(f"AMP: {use_amp}, Warmup: {warmup} iters, Decoder: {decoder_type}")
    print(f"Batch size: {bs}, Effective batch: {bs * grad_accum_steps}")

    best_miou = 0.0
    global_step = 0
    start_epoch = 0
    optimizer.zero_grad(set_to_none=True)

    # ---- Resume from checkpoint ----
    resume_path = cfg.get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("best_miou", 0.0)
        if boundary_head is not None and "boundary_head" in ckpt:
            boundary_head.load_state_dict(ckpt["boundary_head"])
        if ema is not None and "ema" in ckpt:
            ema.module.load_state_dict(ckpt["ema"])
            print("Resumed EMA shadow weights from checkpoint")
        # Advance scheduler to correct step
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()
        print(f"Resumed from {resume_path}, epoch {start_epoch}, best_miou {best_miou:.4f}")

    for epoch in range(start_epoch, epochs):
        # v2: router temperature annealing (τ: 3.0→1.0 over anneal_epochs, then 1.0)
        router_cfg = cfg.get("decoder", {}).get("sc_cmrd_lar", {}).get("router", {})
        temp_cfg = router_cfg.get("temperature", {})
        if temp_cfg.get("enabled", False) and hasattr(model, "decoder") and hasattr(model.decoder, "set_router_temperature"):
            temp_start = temp_cfg.get("start", 3.0)
            temp_end = temp_cfg.get("end", 1.0)
            anneal_epochs = temp_cfg.get("anneal_epochs", 10)
            if epoch < anneal_epochs:
                tau = temp_start + (temp_end - temp_start) * (epoch / anneal_epochs)
            else:
                tau = temp_end
            model.decoder.set_router_temperature(tau)
            if ema is not None:
                ema.module.decoder.set_router_temperature(tau)
            print(f"Epoch {epoch}: router temperature τ={tau:.4f}")

        model.train()
        if route_a_module is not None:
            route_a_module.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dice = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Dual-label mode: batch = (images, noBoundary_masks, complete_masks).
            # Standard mode: batch = (images, masks).
            if len(batch) == 3:
                images, masks, complete_masks = batch
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                complete_masks = complete_masks.to(device, non_blocking=True)
            else:
                images, masks = batch
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                complete_masks = None

            with torch.amp.autocast("cuda", enabled=use_amp):
                # --- Consistency teacher forward: SF adapter disabled = base path ---
                teacher_logits = None
                teacher_feat = None
                if cons_enabled:
                    set_sf_adapter_disable(model, True)
                    try:
                        with torch.no_grad(), frozen_bn_running_stats(model):
                            if cons_need_feat:
                                teacher_logits, teacher_feat = model(
                                    images, return_features=True
                                )
                                teacher_feat = teacher_feat.detach()
                            else:
                                teacher_logits = model(images)
                            teacher_logits = teacher_logits.detach()
                    finally:
                        # Always restore adapter flag, even on exception.
                        set_sf_adapter_disable(model, False)

                # --- Student forward (adapter enabled = full path) ---
                # Route A needs p2 (embed_dim channels, pre-out_conv PRN output @ H/4).
                # model(images, return_features=True) returns logits_h4 (num_classes
                # channels), NOT p2 (src/model.py). So when route_a is on,
                # we hook model.decoder._decoder.neck to grab p2 (G2/G3 pattern).
                # cons_need_feat path keeps return_features=True for logits_h4 (cons
                # affinity was designed for that — see compute_consistency_loss).
                route_a_need_feat = route_a_module is not None
                student_feat_p2 = None
                if route_a_need_feat:
                    with P2HookContext(model) as _p2hook:
                        if cons_need_feat:
                            output, student_feat = model(images, return_features=True)
                        else:
                            output = model(images)
                            student_feat = None
                    student_feat_p2 = _p2hook.captured
                elif cons_need_feat:
                    output, student_feat = model(images, return_features=True)
                else:
                    output = model(images)
                    student_feat = None

                # Per-step diagnostics (populated for SC-CMRD-LAR; empty for others)
                step_diag = {}

                if is_m2f and m2f_criterion is not None:
                    # Mask2Former: use set prediction loss
                    loss, m2f_loss_dict = compute_m2f_loss(m2f_criterion, output, masks, num_classes)
                    l_ce = m2f_loss_dict.get("loss_ce", torch.tensor(0.0))
                    l_dice = m2f_loss_dict.get("loss_dice", torch.tensor(0.0))
                else:
                    # Standard decoders (PFU, MLP, UPerNet, SC-CMRD-LAR)
                    logits = output
                    aux_outputs = None
                    if isinstance(logits, dict):
                        aux_outputs = logits
                        logits = logits["logits"]
                    # BoundaryDice-v2: BDiceV2Loss computes L_main + λ(t)·L_boundary
                    # in a single coupled forward (ce_loss_fn / dice_loss_fn are None).
                    if loss_type == "soft_ce_plus_bdice_v2":
                        loss, bdice_v2_diag = bdice_v2_loss_fn(
                            logits=logits,
                            targets_ce=masks,
                            global_step=global_step,
                            gt_complete=complete_masks,
                        )
                        # Logging: project back into l_ce / l_dice for backward-compat with
                        # the existing log line. Use the diag dict for richer fields.
                        l_ce = torch.tensor(bdice_v2_diag.get("l_softce_full", 0.0))
                        l_dice = torch.tensor(bdice_v2_diag.get("l_dice_full", 0.0))
                        step_diag.update({
                            k: float(v) for k, v in bdice_v2_diag.items()
                            if isinstance(v, (int, float))
                        })
                    else:
                        l_ce = ce_loss_fn(logits, masks)
                        l_dice = dice_loss_fn(logits, masks)
                        loss = l_ce + dice_weight * l_dice
                    # P1.5: optional Edge-point CE side loss (already weighted inside)
                    l_edge_val = 0.0
                    if edge_loss_fn is not None:
                        l_edge = edge_loss_fn(logits, masks)
                        loss = loss + l_edge
                        l_edge_val = float(l_edge.detach())

                    # EPFO point-CE side loss.
                    # EPFO forward (train mode) caches point_logits (B, C, K) +
                    # point_coords (B, K, 2). Sample GT at H/4 (the resolution at
                    # which EPFO selected points) and add point CE to total loss.
                    l_point_val = 0.0
                    if _epfo_module is not None and _epfo_module.point_logits is not None:
                        out_H, out_W = logits.shape[-2:]
                        H4, W4 = out_H // 4, out_W // 4
                        # Downsample GT masks to H/4 via nearest (matches EPFO
                        # boundary selection resolution).
                        masks_h4 = F.interpolate(
                            masks.unsqueeze(1).float(),
                            size=(H4, W4),
                            mode="nearest",
                        ).squeeze(1).long()
                        # point_gt: (B, K) long, sampled at cached point_coords.
                        point_gt = _epfo_module.get_point_labels(masks_h4)
                        # F.cross_entropy treats dim 1 as class dim, so pass
                        # point_logits (B, C, K) directly with target (B, K).
                        l_point = F.cross_entropy(
                            _epfo_module.point_logits,
                            point_gt,
                            ignore_index=ignore_index,
                        )
                        loss = loss + lambda_point * l_point
                        l_point_val = float(l_point.detach())

                    # --- Semantic-preserving consistency ---
                    # teacher_* detached (no_grad); student logits/feat carry grad.
                    if cons_enabled and teacher_logits is not None:
                        loss_kl, loss_aff, cons_diag = compute_consistency_loss(
                            student_logits=logits,
                            teacher_logits=teacher_logits,
                            student_feat=student_feat,
                            teacher_feat=teacher_feat,
                            kl_enabled=cons_kl_enabled,
                            affinity_enabled=cons_aff_enabled,
                            lambda_kl=cons_lambda_kl,
                            lambda_rel=cons_lambda_rel,
                            tau=cons_tau,
                            conf_thresh=cons_conf,
                            affinity_tokens=cons_k,
                        )
                        loss = loss + loss_kl + loss_aff
                        step_diag.update(cons_diag)

                    # Native-scale aux supervision (SC-CMRD-LAR v2)
                    # Label is nearest-downsampled to aux resolution (not bilinear-up
                    # of logits). Aux heads learn native-scale discrimination.
                    if aux_outputs is not None and "aux_logits8" in aux_outputs:
                        aux_weight = cfg.get("decoder", {}).get(
                            "sc_cmrd_lar", {}
                        ).get("aux_weight", 0.4)
                        for key in ("aux_logits8", "aux_logits4", "aux_logits16"):
                            aux_logits = aux_outputs[key]
                            aux_h, aux_w = aux_logits.shape[-2:]
                            label_down = torch.nn.functional.interpolate(
                                masks.float().unsqueeze(1),  # (B,1,H,W) float
                                size=(aux_h, aux_w),
                                mode="nearest",
                            ).squeeze(1).long()  # (B,aux_h,aux_w) int
                            l_aux_ce = ce_loss_fn(aux_logits, label_down)
                            l_aux_dice = dice_loss_fn(aux_logits, label_down)
                            loss = loss + aux_weight * (l_aux_ce + dice_weight * l_aux_dice)
                            scale_tag = key.replace("aux_logits", "")
                            step_diag[f"aux_ce_{scale_tag}"] = l_aux_ce.item()
                            step_diag[f"aux_dice_{scale_tag}"] = l_aux_dice.item()

                    # NOTE: AffinityPreservationLoss + SCCMRDLarDecoder route removed in
                    # cleanup. The block previously here lazy-imported
                    # AffinityPreservationLoss from src.model which no longer exists.

                    # Router entropy regularization (v2: anti-collapse, from step 0)
                    # H = 0.5 * (H8 + H4), loss -= beta * H (maximize entropy)
                    router_cfg = cfg.get("decoder", {}).get("sc_cmrd_lar", {}).get("router", {})
                    ent_cfg = router_cfg.get("entropy_reg", {})
                    if ent_cfg.get("enabled", False) and aux_outputs is not None and "basis_probs_for_loss" in aux_outputs:
                        beta_start = ent_cfg.get("beta_start", 0.02)
                        beta_end = ent_cfg.get("beta_end", 0.005)
                        decay_epochs = ent_cfg.get("decay_epochs", 20)
                        # Linear decay beta_start -> beta_end over decay_epochs, then beta_end
                        if epoch < decay_epochs:
                            beta = beta_start + (beta_end - beta_start) * (epoch / decay_epochs)
                        else:
                            beta = beta_end
                        bp8 = aux_outputs["basis_probs_for_loss"].get("schrr_8")
                        bp4 = aux_outputs["basis_probs_for_loss"].get("schrr_4")
                        if bp8 is not None and bp4 is not None:
                            # fp32 + clamp_min: AMP softmax can produce exact 0 in fp16,
                            # causing NaN in 0*log(0). Cast to fp32 and clamp >= 1e-6.
                            bp8f = bp8.float().clamp_min(1e-6)
                            bp4f = bp4.float().clamp_min(1e-6)
                            H8 = -(bp8f * torch.log(bp8f)).sum(dim=1).mean()
                            H4 = -(bp4f * torch.log(bp4f)).sum(dim=1).mean()
                            H = 0.5 * (H8 + H4)  # average over scales, don't double beta
                            loss = loss - beta * H  # maximize entropy
                            step_diag["ent_beta"] = beta
                            step_diag["ent_H"] = H.item()

                    # Extract per-block model diagnostics (gate, coeff, lambda,
                    # basis usage, router entropy) from aux_outputs["diag"].
                    if aux_outputs is not None and "diag" in aux_outputs:
                        model_diag = aux_outputs["diag"]
                        for scale_key, scale_tag in [("schrr_8", "8"), ("schrr_4", "4")]:
                            d = model_diag.get(scale_key, {})
                            step_diag[f"gate_{scale_tag}"] = d.get(
                                "gate_mean", torch.tensor(float("nan"))
                            ).item()
                            step_diag[f"csem_{scale_tag}"] = d.get(
                                "coeff_sem_mean", torch.tensor(float("nan"))
                            ).item()
                            step_diag[f"cstr_{scale_tag}"] = d.get(
                                "coeff_str_mean", torch.tensor(float("nan"))
                            ).item()
                            step_diag[f"lam_{scale_tag}"] = d.get(
                                "lambda", torch.tensor(float("nan"))
                            ).item()
                            bp = d.get("basis_probs")
                            if bp is not None:
                                usage = bp.mean(dim=(0, 2, 3)).tolist()
                                for b_idx, u in enumerate(usage):
                                    step_diag[f"b{b_idx}_{scale_tag}"] = u
                                # fp32 + clamp_min: same numerical safety as entropy reg above.
                                bpf = bp.float().clamp_min(1e-6)
                                ent = -(
                                    bpf * torch.log(bpf)
                                ).sum(dim=1).mean().item()
                                step_diag[f"ent_{scale_tag}"] = ent

                    # Boundary loss (gradients flow back to main model)
                    if boundary_loss_fn is not None:
                        boundary_logits = boundary_head(logits)
                        boundary_gt = extract_boundary_mask_gpu(masks, num_classes,
                            boundary_width=cfg.get("boundary_width", 3))
                        l_boundary = boundary_loss_fn(boundary_logits, boundary_gt)
                        loss = loss + boundary_weight * l_boundary

                # ---- Phase 1 aux losses (Conv-LoRA MoE balancing, etc.) ----
                # Aggregate from model.get_aux_losses() and apply per-loss lambdas.
                if aux_lambda_moe > 0 and hasattr(model, "get_aux_losses"):
                    aux_losses = model.get_aux_losses()
                    moe_loss = aux_losses.get("moe_loss", None)
                    if moe_loss is not None and moe_loss.requires_grad:
                        loss = loss + aux_lambda_moe * moe_loss

                # --- Route A: prototype contrastive side loss ---
                # student_feat_p2 = p2 @ H/4 (PRN output, embed_dim channels) from
                # P2HookContext. Module handles: prototype EMA update (no_grad) +
                # GRQA/plain loss (grad flows through features only). Returns 0
                # when omega_eff == 0.
                if route_a_module is not None and student_feat_p2 is not None:
                    l_route_a = route_a_module(student_feat_p2, masks)
                    if l_route_a.requires_grad:
                        loss = loss + l_route_a
                        step_diag["route_a_loss"] = float(l_route_a.detach())

                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if grad_clip_norm is not None:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                else:
                    # No clipping — but still compute total norm for logging.
                    # max_norm=inf returns the norm without modifying grads.
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                if batch_idx % 50 == 0:
                    print(f"  [grad_norm: {total_norm:.4f}]")
                # Stable-LoRA pre-step shrinkage is registered as an
                # optimizer.register_step_pre_hook below — it fires only when
                # scaler.step(optimizer) actually invokes optimizer.step(),
                # so it correctly skips on AMP inf/NaN steps where GradScaler
                # bypasses the optimizer.
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                # EMA update runs from step 0 so it is warmed up by the time
                # eval switches to ema.module at ema_start_step. Gating both
                # update and eval usage on ema_start_step leaves EMA frozen at
                # initial weights during [0, start_step), causing a cold-start
                # cliff (val mIoU crashes to ~0.07) the moment eval switches.
                if ema is not None:
                    ema.update(model)

            epoch_loss += loss.item() * grad_accum_steps
            epoch_ce += l_ce.item()
            epoch_dice += l_dice.item()

            if batch_idx % 50 == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                bnd_str = f" Bnd: {l_boundary.item():.4f}" if boundary_loss_fn is not None else ""
                edge_str = f" Edge: {l_edge_val:.4f}" if edge_loss_fn is not None else ""
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item() * grad_accum_steps:.4f} (CE: {l_ce.item():.4f} Dice: {l_dice.item():.4f}{bnd_str}{edge_str}) LR: {lr_now:.6f}")
                # Compact SC-CMRD-LAR diagnostics: per-scale gate/coeff/lambda/basis/aux/aff
                if step_diag:
                    parts = []
                    for scale_tag in ("8", "4"):
                        if f"gate_{scale_tag}" in step_diag:
                            parts.append(
                                f"S{scale_tag}[g={step_diag[f'gate_{scale_tag}']:.3f},"
                                f"cs={step_diag[f'csem_{scale_tag}']:.3f},"
                                f"ct={step_diag[f'cstr_{scale_tag}']:.3f},"
                                f"l={step_diag[f'lam_{scale_tag}']:.3f},"
                                f"b=({step_diag.get(f'b0_{scale_tag}', 0):.2f},"
                                f"{step_diag.get(f'b1_{scale_tag}', 0):.2f},"
                                f"{step_diag.get(f'b2_{scale_tag}', 0):.2f}),"
                                f"e={step_diag.get(f'ent_{scale_tag}', 0):.3f},"
                                f"ax={step_diag.get(f'aux_ce_{scale_tag}', 0):.3f}]"
                            )
                    if "16" in str(step_diag.get("aux_ce_16", "")) or "aux_ce_16" in step_diag:
                        parts.append(f"Ax16={step_diag.get('aux_ce_16', 0):.3f}")
                    if "aff_w" in step_diag:
                        parts.append(f"Aff[w={step_diag['aff_w']:.4f},l={step_diag['aff_l']:.4f}]")
                    if "ent_beta" in step_diag:
                        parts.append(f"Ent[β={step_diag['ent_beta']:.4f},H={step_diag['ent_H']:.3f}]")
                    print(f"  DIAG: " + " ".join(parts))

                # BoundaryDice-v2 diagnostics (monitoring.json source).
                # Read raw bdice_v2_diag (not step_diag — line 1673 filters out the
                # boundary_weight_type string). Gated by loss_type so non-bdice runs skip.
                if loss_type == "soft_ce_plus_bdice_v2":
                    bparts = [f"T={bdice_v2_diag['boundary_weight_type']}"]
                    bparts.append(f"λ={bdice_v2_diag['lambda_t']:.3f}")
                    bparts.append(f"Lb={bdice_v2_diag['l_boundary']:.4f}")
                    bparts.append(f"nb={bdice_v2_diag['n_band']}")
                    bparts.append(f"nbv={bdice_v2_diag['n_band_valid']}")
                    # EMA + per-CC stats (only present for per_cc_entropy mode).
                    if "ema_mu_H" in bdice_v2_diag:
                        ema_mu = bdice_v2_diag["ema_mu_H"]
                        bparts.append(f"EMA[μ={ema_mu:.3f},σ={bdice_v2_diag['ema_sigma_H']:.3f},s={bdice_v2_diag['ema_step']}]")
                        if bdice_v2_diag.get("ema_warmup", 0):
                            bparts.append("WARMUP")
                    if "cc_count" in bdice_v2_diag:
                        bparts.append(f"CC[n={bdice_v2_diag['cc_count']},xb={bdice_v2_diag['cc_cross_boundary_count']}({bdice_v2_diag['cc_cross_boundary_frac']*100:.0f}%)]")
                    if "cc_area_min" in bdice_v2_diag and bdice_v2_diag["cc_count"] > 0:
                        bparts.append(
                            f"area[{bdice_v2_diag['cc_area_min']},"
                            f"μ={bdice_v2_diag['cc_area_mean']:.0f},"
                            f"{bdice_v2_diag['cc_area_max']},"
                            f"<Amin={bdice_v2_diag['cc_area_below_amin_count']}]"
                        )
                    if "cc_w_min" in bdice_v2_diag and bdice_v2_diag["cc_count"] > 0:
                        bparts.append(
                            f"w[{bdice_v2_diag['cc_w_min']:.4f},"
                            f"μ={bdice_v2_diag['cc_w_mean']:.4f},"
                            f"{bdice_v2_diag['cc_w_max']:.4f}]"
                        )
                        # dispatch-collapse monitor: >50% CCs w<0.1 → flag.
                        # NOTE: w<0.1 trigger is area-dominated (w_cc ≤ 4/a_eff for α=3);
                        # cc_budget_* separates entropy-driven from area-driven collapse.
                        bparts.append(f"w<0.1={bdice_v2_diag['cc_w_lt_01_count']}({bdice_v2_diag['cc_w_lt_01_frac']*100:.0f}%)")
                        bparts.append(
                            f"b[{bdice_v2_diag['cc_budget_min']:.2f},"
                            f"μ={bdice_v2_diag['cc_budget_mean']:.2f},"
                            f"{bdice_v2_diag['cc_budget_max']:.2f},"
                            f"H={bdice_v2_diag['cc_budget_entropy_norm']:.2f}]"
                        )
                    # ring stats (only present for ring_difference mode).
                    if "ring1_pix" in bdice_v2_diag:
                        bparts.append(
                            f"ring[r1={bdice_v2_diag['ring1_pix']}/"
                            f"r2={bdice_v2_diag['ring2_pix']}/"
                            f"r3={bdice_v2_diag['ring3_pix']}]"
                        )
                        bparts.append(
                            f"ringV[r1={bdice_v2_diag['ring1_pix_valid']}/"
                            f"r2={bdice_v2_diag['ring2_pix_valid']}/"
                            f"r3={bdice_v2_diag['ring3_pix_valid']}]"
                        )
                    print(f"  BDICE: " + " ".join(bparts))

                # Run A-main: routed SF-SSD-LoRA diagnostics
                routed_diag = collect_routed_diagnostics(model)
                if routed_diag:
                    rparts = [f"n={routed_diag['routed_n_modules']}"]
                    for ename in ("sem", "spa", "tex", "spe"):
                        key = f"router_w_{ename}_mean"
                        if key in routed_diag:
                            rparts.append(f"{ename}={routed_diag[key]:.3f}")
                    if "router_w_norm_std" in routed_diag:
                        rparts.append(f"std={routed_diag['router_w_norm_std']:.3f}")
                    if "gamma_spe_mean" in routed_diag:
                        rparts.append(f"γ_spe={routed_diag['gamma_spe_mean']:.4f}")
                    print(f"  ROUTED: " + " ".join(rparts))
                    step_diag.update({f"routed_{k}": v for k, v in routed_diag.items()})

                # Consistency diagnostics
                if cons_enabled:
                    cparts = []
                    if "kl_raw" in step_diag:
                        cparts.append(f"KL={step_diag['kl_raw']:.4f}")
                    if "aff_raw" in step_diag:
                        cparts.append(f"Aff={step_diag['aff_raw']:.4f}")
                    if "easy_frac" in step_diag:
                        cparts.append(f"easy={step_diag['easy_frac']:.2f}")
                    if "aff_tokens" in step_diag:
                        cparts.append(f"tok={step_diag['aff_tokens']:.0f}")
                    if cparts:
                        print(f"  CONS: " + " ".join(cparts))

            # TensorBoard per-step diagnostics (SC-CMRD-LAR only)
            if step_diag and writer is not None:
                for k, v in step_diag.items():
                    if isinstance(v, float) and (v != v):  # NaN check
                        continue
                    writer.add_scalar(f"diag/{k}", v, global_step)

        dt = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        # Owner directive 2026-09-11: validate every 2 epochs during the
        # baseline-reproduction wave (last epoch always validated). Training-time
        # router temperature is re-established at the start of every epoch, so
        # skipping the val block leaks no state.
        do_val = ((epoch + 1) % 2 == 0) or ((epoch + 1) == epochs)
        miou = None
        if do_val:
            # Validate using EMA model when available (typically +0.3~0.7% mIoU)
            # v2: reset router temperature to 1.0 for validation (both model and EMA)
            if hasattr(model, "decoder") and hasattr(model.decoder, "set_router_temperature"):
                model.decoder.set_router_temperature(1.0)
                if ema is not None:
                    ema.module.decoder.set_router_temperature(1.0)
            eval_model = ema.module if ema is not None and global_step >= ema_start_step else model
            eval_model.eval()
            confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        output = eval_model(images)
                        if is_m2f:
                            logits = _m2f_predict(eval_model, images)
                        else:
                            logits = output
                            if isinstance(logits, dict):
                                logits = logits["logits"]
                    preds = logits.argmax(dim=1).cpu().numpy()
                    targets = masks.numpy()
                    confusion += compute_confusion(preds, targets, num_classes)

            iou_per_class = []
            for c in range(num_classes):
                tp = confusion[c, c]
                fp = confusion[:, c].sum() - tp
                fn = confusion[c, :].sum() - tp
                if tp + fp + fn > 0:
                    iou_per_class.append(tp / (tp + fp + fn))
                else:
                    iou_per_class.append(0.0)
            miou = np.mean(iou_per_class)

        if miou is not None:
            print(f"\nEpoch [{epoch+1}/{epochs}] ({dt:.0f}s) Loss: {avg_loss:.4f} | Val mIoU: {miou:.4f}")
            for i, iou in enumerate(iou_per_class):
                name = dataset_cls.CLASS_NAMES[i] if i < len(dataset_cls.CLASS_NAMES) else f"Class{i}"
                print(f"  {name}: {iou:.4f}")
        else:
            print(f"\nEpoch [{epoch+1}/{epochs}] ({dt:.0f}s) Loss: {avg_loss:.4f} | Val skipped (every-2 schedule)")

        # External pathway γ monitor.
        # Per-epoch log of γ stats. diagnostics/external_pathway.py --mode monitor parses this.
        # Abort signal (WPD poison): γ_mean < 1e-3 at epoch >= 30 → identity death.
        # external_gamma lives on model.decoder (MMDINODecoderFull outer), NOT on
        # model.decoder._decoder (inner MMDINODecoder with frm/neck/out_conv).
        _ext_outer = getattr(model, "decoder", None)
        if (_ext_outer is not None
                and getattr(_ext_outer, "external_gamma", None) is not None):
            _g = _ext_outer.external_gamma.gamma.detach()
            print(f"[ExternalPathway] epoch={epoch+1} "
                  f"gamma_mean={_g.mean().item():.6e} "
                  f"gamma_min={_g.min().item():.6e} "
                  f"gamma_max={_g.max().item():.6e} "
                  f"gamma_std={_g.std().item():.6e}")

        # Route A per-class contrastive loss: Building vs Forest per-class curves
        # diagnose Forest multi-modal failure (Route E pseudo-subclass signal)
        # more informatively than total loss. Cached as epoch-mean per-class
        # on RouteAModule.
        if route_a_module is not None:
            pcl = route_a_module.epoch_per_class_loss
            parts = []
            for i in range(len(dataset_cls.CLASS_NAMES)):
                v = pcl.get(i, float("nan"))
                nm = dataset_cls.CLASS_NAMES[i]
                parts.append(f"{nm}={v:.4f}")
            print(f"[RouteA] epoch={epoch+1} per_class_contrastive_loss: "
                  + " ".join(parts))

        if miou is not None:
            writer.add_scalar("val/mIoU", miou, epoch)
        writer.add_scalar("train/loss", avg_loss, epoch)

        # Save best (gated by best_save_min_epoch: skip best-updates during the
        # EMA warmup epochs to avoid fake-peak best checkpoints, cf. the
        # s47 best=4.19 artifact 2026-08-27; default 1 = previous behavior)
        best_save_min_epoch = int(cfg.get("best_save_min_epoch", 1))
        if miou is not None and miou > best_miou and (epoch + 1) >= best_save_min_epoch:
            best_miou = miou
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
            }
            if ema is not None:
                ckpt["ema"] = ema.module.state_dict()
            if boundary_head is not None:
                ckpt["boundary_head"] = boundary_head.state_dict()
            torch.save(ckpt, os.path.join(output_dir, "best_model.pth"))
            print(f"  ** New best mIoU: {best_miou:.4f}")

        # Save latest
        ckpt_latest = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
        }
        if ema is not None:
            ckpt_latest["ema"] = ema.module.state_dict()
        if boundary_head is not None:
            ckpt_latest["boundary_head"] = boundary_head.state_dict()
        torch.save(ckpt_latest, os.path.join(output_dir, "latest_model.pth"))

    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    # HARD LOCK (Issue 4, spec v3): refuse to train if config is marked eval_only.
    if cfg.get('eval_only', False):
        raise RuntimeError(
            "Config has eval_only=true. This is an EVAL-ONLY config (e.g. leaky audit). "
            "Use cross_dataset/eval_potsdam.py instead of train_cross.py. Refusing to train."
        )

    train(cfg)
