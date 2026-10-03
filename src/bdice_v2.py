"""BoundaryDice-v2 loss.

Implements additive boundary-CE loss on a K=3 trimap:
    L_total = L_main + lambda(t) * L_boundary
    L_main     = SoftCE_full + Dice_full
    L_boundary = (1/N_band) * Sum_{x,y in band} w(x,y) * SoftCE(x,y)

Three weighting modes (selected via `boundary_weight_type` config):
    - "gate_closed": w ≡ 1, lambda ≡ 0 → reduces bit-exactly to the baseline.
      Used as sanity check + matched control.
    - "per_cc_entropy": per-CC equal-budget entropy weighting
      w_CC = (1 + alpha * sigmoid((H_CC - mu_H) / (sigma_H + eps))) / max(area_CC, A_min)
      with EMA stats (beta=0.9) and FULL no_grad on the weight path.
    - "ring_difference": 3-granularity ring-difference static weights.
      thin K=2 w=3 / medium-thin (medium minus thin) w=2 / thick-medium (thick minus medium) w=1.

Trimap convention:
    - K=3 dilate-erode iterations of 3x3 kernel → ~6-7px band
    - PER-CLASS binary dilate/erode (NOT on the multi-class label tensor — 255/Clutter
      would leak through dilation). 255/Clutter pixels auto-excluded by per-class binarization.
    - 8-connectivity CC labeling within the band, per class.
    - Border zero-padding (no reflected boundary).

Potsdam dual-label:
    - Trimap construction uses COMPLETE labels (6-class with Clutter) → Clutter boundaries
      participate in band geometry.
    - CE target in band = noBoundary labels (5-class eval, Clutter=255, original boundary=255).
    - Original-boundary pixels (noBoundary=255) that fall inside the K=3 trimap band are
      EXCLUDED from N_band and from L_boundary (else L_boundary is artificially shrunk).

lambda(t) schedule:
    lambda(t) = min(1, t / (0.3 * T_total))
    One-time ramp over first 30% of total training steps; constant lambda=1 thereafter.
    Per-epoch reset is FORBIDDEN — that would re-trigger ramp every epoch.

References:
    - FLEX-Seg UBE Eq.7: w = 1 + alpha * sigmoid((H - mu)/(sigma+eps)), weight on L_CE only.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Trimap extraction
# ---------------------------------------------------------------------------


def _per_class_boundary_band(gt: torch.Tensor, num_classes: int, k_iter: int) -> torch.Tensor:
    """Compute per-class binary dilate-erode boundary band, unioned across classes.

    Args:
        gt: (B, H, W) long tensor. May contain 255/Clutter (auto-excluded by per-class bin).
        num_classes: number of VALID classes (excluding ignore). For LoveDA=7, Potsdam=6.
        k_iter: number of 3x3 dilate-erode iterations. K=3 → ~6-7px band.

    Returns:
        band: (B, H, W) float tensor in {0.0, 1.0}. 1 = boundary band pixel.

    Per spec: per-class binary dilate/erode (NOT multi-class). Pixel is in band iff
    it lies on any class's boundary. 255/Clutter pixels never participate (each class-c
    binary mask is `(gt == c)`, so 255 is excluded by construction).
    """
    B, H, W = gt.shape
    device = gt.device
    # Pad=1 for 3x3 kernel, stride=1 → same-size output. k_iter iterations = k_iter * 2 px band.
    pad = 1
    band_total = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    for c in range(num_classes):
        bin_c = (gt == c).float().unsqueeze(1)  # (B, 1, H, W)
        for _ in range(k_iter):
            dilated = F.max_pool2d(bin_c, kernel_size=3, stride=1, padding=pad)
            eroded = -F.max_pool2d(-bin_c, kernel_size=3, stride=1, padding=pad)
            band_diff = (dilated - eroded).clamp(0, 1)  # (B, 1, H, W) in {0,1}
            bin_c = dilated  # grow the seed for next iteration (cumulative dilation)
            band_total = band_total + band_diff.squeeze(1).clamp(0, 1)
    return (band_total > 0).float()


def _ring_mask(gt: torch.Tensor, num_classes: int, k_iter: int) -> torch.Tensor:
    """Compute the boundary band at a given K (cumulative dilate-erode difference).

    Returns (B, H, W) float in {0, 1}: 1 where pixel is in the K-iteration band.
    """
    B, H, W = gt.shape
    device = gt.device
    pad = 1
    band_total = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    for c in range(num_classes):
        bin_c = (gt == c).float().unsqueeze(1)
        for _ in range(k_iter):
            dilated = F.max_pool2d(bin_c, kernel_size=3, stride=1, padding=pad)
            eroded = -F.max_pool2d(-bin_c, kernel_size=3, stride=1, padding=pad)
            band_total = band_total + (dilated - eroded).clamp(0, 1).squeeze(1)
            bin_c = dilated
    return (band_total > 0).float()


# ---------------------------------------------------------------------------
# 8-connectivity CC labeling
# ---------------------------------------------------------------------------


def _label_cc_8conn(bin_mask: torch.Tensor) -> torch.Tensor:
    """8-connectivity CC labeling on a binary mask (per batch sample).

    Args:
        bin_mask: (B, H, W) in {0, 1}.

    Returns:
        labels: (B, H, W) long. 0 = background (not in mask). 1..N = CC ids per sample.

    Uses scipy.ndimage.label on CPU (O(N) algorithm) within a no_grad context — typical
    boundary CC counts are small, and the CPU transfer overhead is negligible for 512x512.
    The 8-conn structure is `np.ones((3,3))`.

    Note: Cross-crop CCs are split into separate CCs (accepted behavior).
    """
    B, H, W = bin_mask.shape
    device = bin_mask.device
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity
    out = np.zeros((B, H, W), dtype=np.int64)
    bin_np = bin_mask.detach().cpu().numpy().astype(bool)
    for b in range(B):
        labeled, _ = scipy.ndimage.label(bin_np[b], structure=structure)
        out[b] = labeled
    return torch.from_numpy(out).to(device)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class BDiceV2Loss(nn.Module):
    """BoundaryDice-v2 loss .

    Three modes via `boundary_weight_type`:
        - "gate_closed"   : sanity / matched control. lambda ≡ 0, w ≡ 1.
        - "per_cc_entropy" : Per-CC equal-budget entropy weighting.
        - "ring_difference": 3-granularity ring-difference static weights.

    Args:
        num_classes: training num_classes (LoveDA=7, Potsdam=6 — Clutter included).
        boundary_weight_type: one of {"gate_closed", "per_cc_entropy", "ring_difference"}.
        smooth_factor: label smoothing for SoftCE (default 0.05, MM-DINO spec).
        dice_smooth: Dice smooth term (default 0.05).
        ignore_index: ignore_index for CE (default 255).
        # per_cc_entropy params:
        alpha: entropy weight scale (default 3.0, FLEX-Seg Eq.7).
        a_min: CC area floor (default 25, project choice; sensitivity contingency 16).
        ema_beta: EMA decay for entropy stats (default 0.9).
        ema_warmup_steps: first N steps use raw batch stats, no EMA (default 100, R2 mitigation).
        eps: numerical stability (default 1e-6).
        k_main: K iterations for main trimap (default 3).
        # ring_difference params:
        k_thin: K for thin ring (default 2 → ~4px band).
        k_medium: K for medium ring (default 5 → ~10px band).
        k_thick: K for thick ring (default 10 → ~20px band).
        w_thin/w_medium_thin/w_thick_medium: ring weights (default 3/2/1).
        # lambda(t) ramp:
        ramp_frac: fraction of total steps for ramp (default 0.3).
        total_steps: total training step count for the dataset (LoveDA ~31525, Potsdam ~57000).
            Used in lambda(t) = min(1, t / (ramp_frac * total_steps)).
        dual_label: True if dataset returns (noBoundary, complete) label pair (Potsdam dual-label mode).
            When True, the forward expects `gt_complete` kwarg for trimap construction.
    """

    def __init__(
        self,
        num_classes: int,
        boundary_weight_type: str = "gate_closed",
        *,
        smooth_factor: float = 0.05,
        dice_smooth: float = 0.05,
        ignore_index: int = 255,
        # per_cc_entropy
        alpha: float = 3.0,
        a_min: int = 25,
        ema_beta: float = 0.9,
        ema_warmup_steps: int = 100,
        eps: float = 1e-6,
        k_main: int = 3,
        # ring_difference
        k_thin: int = 2,
        k_medium: int = 5,
        k_thick: int = 10,
        w_thin: float = 3.0,
        w_medium_thin: float = 2.0,
        w_thick_medium: float = 1.0,
        # lambda ramp
        ramp_frac: float = 0.3,
        total_steps: int = 1,
        # Dual-label
        dual_label: bool = False,
        # Wave-2 class-protection (ring_difference only)
        band_class_weights: Optional[dict] = None,
    ):
        super().__init__()
        if boundary_weight_type not in {"gate_closed", "per_cc_entropy", "ring_difference"}:
            raise ValueError(f"Unknown boundary_weight_type={boundary_weight_type!r}")
        self.num_classes = int(num_classes)
        self.boundary_weight_type = boundary_weight_type
        self.smooth_factor = float(smooth_factor)
        self.dice_smooth = float(dice_smooth)
        self.ignore_index = int(ignore_index)

        # per_cc_entropy
        self.alpha = float(alpha)
        self.a_min = int(a_min)
        self.ema_beta = float(ema_beta)
        self.ema_warmup_steps = int(ema_warmup_steps)
        self.eps = float(eps)
        self.k_main = int(k_main)

        # ring_difference
        self.k_thin = int(k_thin)
        self.k_medium = int(k_medium)
        self.k_thick = int(k_thick)
        self.w_thin = float(w_thin)
        self.w_medium_thin = float(w_medium_thin)
        self.w_thick_medium = float(w_thick_medium)

        # lambda ramp
        self.ramp_frac = float(ramp_frac)
        self.total_steps = int(total_steps)

        self.dual_label = bool(dual_label)

        # Wave-2 class-protection: per-class multiplier on boundary weights (ring_difference only).
        # band_class_weights: dict {class_idx: multiplier}, e.g. {2: 1.5} boosts LowVeg CE in band.
        cw = torch.ones(self.num_classes)
        if band_class_weights:
            for cls_idx, mult in band_class_weights.items():
                cw[int(cls_idx)] = float(mult)
        self.register_buffer("band_cw", cw)
        self.band_class_weights = {int(k): float(v) for k, v in (band_class_weights or {}).items()}

        # EMA buffers for entropy stats. Initialized to NaN; first batch sets them.
        self.register_buffer("ema_mu_H", torch.full((1,), float("nan")))
        self.register_buffer("ema_sigma_H", torch.full((1,), float("nan")))
        self.register_buffer("ema_step", torch.zeros(1, dtype=torch.long))

    # -- helpers --

    def _lambda_at(self, global_step: int) -> float:
        """Closed-form lambda(t) = min(1, t / (0.3 * T_total)). gate_closed forces lambda=0."""
        if self.boundary_weight_type == "gate_closed":
            return 0.0
        denom = max(1, int(self.ramp_frac * self.total_steps))
        return min(1.0, float(global_step) / float(denom))

    def _soft_ce_per_pixel(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Smoothed SoftCE per pixel (reduction='none'). Same口径 as L_main SoftCE."""
        return F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.smooth_factor,
            reduction="none",
        )

    def _soft_ce_full(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Smoothed SoftCE mean over valid pixels (matches L_main SoftCE in soft_ce_plus_dice)."""
        return F.cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.smooth_factor,
        )

    def _dice_full(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Standard Dice on full image (matches DiceLoss in soft_ce_plus_dice)."""
        num_classes = logits.shape[1]
        targets_clamped = targets.clamp(0, num_classes - 1)
        targets_one_hot = F.one_hot(targets_clamped, num_classes).permute(0, 3, 1, 2).float()
        valid = (targets != self.ignore_index).unsqueeze(1).float()
        probs = F.softmax(logits, dim=1)
        inter = (probs * targets_one_hot * valid).sum(dim=(2, 3))
        union = (probs * valid).sum(dim=(2, 3)) + (targets_one_hot * valid).sum(dim=(2, 3))
        dice = (2 * inter + self.dice_smooth) / (union + self.dice_smooth)
        return 1.0 - dice.mean()

    # -- weight paths --

    @torch.no_grad()
    def _per_cc_weights(
        self, logits: torch.Tensor, band: torch.Tensor, targets_ce: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Per-CC equal-budget entropy weighting.

        Args:
            logits: (B, C, H, W)
            band: (B, H, W) in {0, 1} — main K=3 band, already restricted to valid CE pixels.
            targets_ce: (B, H, W) long — the CE target (noBoundary for Potsdam dual-label).

        Returns:
            w_pixel: (B, H, W) per-pixel weight inside band (0 outside).
            stats: dict of per-CC statistics for monitoring.json (dispatch-collapse
                + EMA stats). Always returned (zeroed if no CCs found).

        Per spec: ENTIRE weight path under no_grad
        (softmax → H_xy → H_CC → z → sigmoid → w_CC).

        Implementation: single CC-labeling pass collects (b, c, cc_mask, area, H_cc) tuples;
        EMA stats resolved from H_cc list; second fill pass uses cached tensors (no
        re-labeling). Cross-crop-boundary CC detection checks if cc_mask
        touches any of the 4 crop edges (rows 0/H-1 or cols 0/W-1) within its batch sample.
        """
        B, C, H, W = logits.shape
        device = logits.device
        # Per-pixel softmax entropy (on GPU for numerical stability).
        probs = F.softmax(logits, dim=1)
        H_xy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)  # (B, H, W)
        # Single bulk CPU transfer of H_xy for per-CC metadata extraction
        # (avoids 7 GPU syncs per CC: area.sum, H.mean, nonzero, 4× edge .item).
        H_xy_np = H_xy.detach().cpu().numpy()  # (B, H, W)

        w_pixel = torch.zeros((B, H, W), dtype=torch.float32, device=device)

        # Single CC-labeling pass: collect metadata for all CCs across all (b, c).
        # cc_meta tuple: (b_idx, cc_mask_gpu, area, H_cc, touches_edge)
        cc_meta: list[tuple[int, torch.Tensor, float, float, bool]] = []

        for c in range(self.num_classes):
            class_band = band * (targets_ce == c).float()
            if class_band.sum() == 0:
                continue
            labels = _label_cc_8conn(class_band)  # GPU tensor; CPU labeling inside
            labels_np = labels.detach().cpu().numpy()  # (B, H, W) — one transfer per (b,c)
            for b in range(B):
                lbl_b_np = labels_np[b]
                unique_ids = np.unique(lbl_b_np)
                unique_ids = unique_ids[unique_ids != 0]
                if unique_ids.size == 0:
                    continue
                # find_objects returns list indexed by CC id - 1; O(N) C call.
                # Each entry is (yslice, xslice) — bounding box of the CC.
                objects = scipy.ndimage.find_objects(lbl_b_np)
                H_xy_b_np = H_xy_np[b]
                lbl_b_gpu = labels[b]  # GPU tensor for cc_mask construction below
                for cc_id in unique_ids.tolist():
                    obj = objects[cc_id - 1]
                    if obj is None:
                        continue
                    ys, xs = obj
                    # Restrict to the bbox, then mask by CC id (bbox may include other CCs).
                    sub_lbl = lbl_b_np[ys, xs]
                    sub_H = H_xy_b_np[ys, xs]
                    cc_pixels_mask = (sub_lbl == cc_id)
                    area = int(np.count_nonzero(cc_pixels_mask))
                    if area == 0:
                        continue
                    H_cc = float(sub_H[cc_pixels_mask].mean())
                    # Cross-crop-boundary detection via bbox:
                    # CC touches any of the 4 crop edges of this batch sample.
                    # Note: edge-touch overcounts CCs at tile borders;
                    # accepted proxy — natural 8-conn crop-level behavior.
                    touches_edge = bool(
                        ys.start == 0 or ys.stop == H
                        or xs.start == 0 or xs.stop == W
                    )
                    cc_mask = (lbl_b_gpu == cc_id)  # GPU tensor for weight fill
                    cc_meta.append((b, cc_mask, float(area), H_cc, touches_edge))

        # Update EMA stats from batch H_cc list.
        H_CC_list = [h for (_, _, _, h, _) in cc_meta]
        if H_CC_list:
            mu_batch = sum(H_CC_list) / len(H_CC_list)
            var_batch = sum((h - mu_batch) ** 2 for h in H_CC_list) / max(1, len(H_CC_list))
            sigma_batch = math.sqrt(var_batch)
            step = int(self.ema_step.item())
            if step < self.ema_warmup_steps:
                mu_H = mu_batch
                sigma_H = sigma_batch
            elif math.isnan(self.ema_mu_H.item()):
                mu_H = mu_batch
                sigma_H = sigma_batch
            else:
                beta = self.ema_beta
                mu_H = beta * self.ema_mu_H.item() + (1 - beta) * mu_batch
                sigma_H = beta * self.ema_sigma_H.item() + (1 - beta) * sigma_batch
            self.ema_mu_H.fill_(mu_H)
            self.ema_sigma_H.fill_(sigma_H)
            self.ema_step.add_(1)

        # Build per-CC stats dict for monitoring.json.
        # Zeroed defaults — populated below if cc_meta is non-empty.
        stats: dict = {
            "cc_count": len(cc_meta),
            "cc_area_min": 0,
            "cc_area_mean": 0.0,
            "cc_area_max": 0,
            "cc_area_below_amin_count": 0,
            "cc_H_batch_mean": float(mu_batch) if H_CC_list else 0.0,
            "cc_H_batch_std": float(sigma_batch) if H_CC_list else 0.0,
            "cc_w_min": 0.0,
            "cc_w_mean": 0.0,
            "cc_w_max": 0.0,
            # dispatch_collapse monitor:
            # >50% CCs with w_CC < 0.1 → flag for diagnosis.
            "cc_w_lt_01_count": 0,
            "cc_w_lt_01_frac": 0.0,
            # Cross-crop-boundary CC ratio:
            # >30% → wave-2 contingency = switch to image-level CC.
            "cc_cross_boundary_count": int(sum(1 for (_, _, _, _, e) in cc_meta if e)),
            "cc_cross_boundary_frac": 0.0,
            # EMA state (for producer to detect EMA drift):
            "ema_mu_H": float(self.ema_mu_H.item()),
            "ema_sigma_H": float(self.ema_sigma_H.item()),
            "ema_step": int(self.ema_step.item()),
            "ema_warmup": int(int(self.ema_step.item()) < self.ema_warmup_steps),
        }
        if cc_meta:
            areas = [a for (_, _, a, _, _) in cc_meta]
            stats["cc_area_min"] = int(min(areas))
            stats["cc_area_mean"] = float(sum(areas) / len(areas))
            stats["cc_area_max"] = int(max(areas))
            stats["cc_area_below_amin_count"] = int(sum(1 for a in areas if a < self.a_min))
            stats["cc_cross_boundary_frac"] = float(stats["cc_cross_boundary_count"] / len(cc_meta))

        # If no CCs found (e.g., very early all-ignore batch), uniform weight inside band.
        if not H_CC_list or math.isnan(self.ema_mu_H.item()):
            return band.float(), stats

        mu_H = self.ema_mu_H.item()
        sigma_H = self.ema_sigma_H.item()

        # Fill weights from cached cc_meta + collect per-CC weight/budget samples for stats.
        # Budget b_cc = w_cc * a_eff ∈ [1, 1+α] isolates the entropy-driven part of the weight
        # from the area part (H(w_cc) is dominated by 1/area).
        w_vals: list[float] = []
        budget_vals: list[float] = []
        for (b, cc_mask, area, H_cc, _touches_edge) in cc_meta:
            a_eff = max(area, self.a_min)
            z = (H_cc - mu_H) / (sigma_H + self.eps)
            sig = 1.0 / (1.0 + math.exp(-z))
            w_cc_val = (1.0 + self.alpha * sig) / float(a_eff)
            w_pixel[b][cc_mask] = w_cc_val
            w_vals.append(w_cc_val)
            budget_vals.append(1.0 + self.alpha * sig)  # ∈ [1, 1+α]

        if w_vals:
            stats["cc_w_min"] = float(min(w_vals))
            stats["cc_w_mean"] = float(sum(w_vals) / len(w_vals))
            stats["cc_w_max"] = float(max(w_vals))
            # dispatch_collapse: fraction of CCs with w_CC < 0.1.
            # NOTE: this trigger is area-dominated — w_cc ≤ 4/a_eff for α=3,
            # so any CC ≥ 40 px trips it regardless of entropy. The producer should consult
            # cc_budget_* (entropy-driven component, ∈ [1, 1+α]) to separate area-driven
            # from entropy-driven collapse.
            stats["cc_w_lt_01_count"] = int(sum(1 for w in w_vals if w < 0.1))
            stats["cc_w_lt_01_frac"] = float(stats["cc_w_lt_01_count"] / len(w_vals))
            # Budget stats — entropy-driven component only.
            stats["cc_budget_min"] = float(min(budget_vals))
            stats["cc_budget_mean"] = float(sum(budget_vals) / len(budget_vals))
            stats["cc_budget_max"] = float(max(budget_vals))
            # Normalized Shannon entropy of budget b_cc ∈ [1, 1+α]:
            # H(b_cc) / log(min(bins, N)) ∈ [0, 1] — high = balanced dispatch, low = collapse.
            # Normalization ceiling = log(bins)/log(min(bins,N)); using log(min(bins,N)) makes
            # the metric comparable across steps as N varies.
            n = len(budget_vals)
            bins = 10
            if n <= 1:
                stats["cc_budget_entropy_norm"] = 0.0
            else:
                b_min = min(budget_vals)
                b_max = max(budget_vals)
                if b_max - b_min <= 1e-9:
                    # All-equal budgets → no dispatch differentiation → entropy = 0 (collapsed).
                    stats["cc_budget_entropy_norm"] = 0.0
                else:
                    hist = [0] * bins
                    for bv in budget_vals:
                        idx = min(bins - 1, int((bv - b_min) / (b_max - b_min) * bins))
                        hist[idx] += 1
                    ent = 0.0
                    for h in hist:
                        if h > 0:
                            p = h / n
                            ent -= p * math.log(p)
                    stats["cc_budget_entropy_norm"] = float(ent / math.log(min(bins, n)))
        return w_pixel, stats

    @torch.no_grad()
    def _ring_difference_weights(
        self, gt_for_trimap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """3-granularity ring-difference static weights.

        Args:
            gt_for_trimap: (B, H, W) long. complete labels for Potsdam (Clutter participates),
                           or single label for LoveDA.

        Returns:
            w_pixel: (B, H, W) per-pixel weight (0 interior, 3/2/1 in rings).
            ring1: (B, H, W) thin ring mask (disjoint, in {0,1}).
            ring2: (B, H, W) medium-thin ring mask.
            ring3: (B, H, W) thick-medium ring mask.

        Rings (disjoint, each pixel in exactly ONE ring):
            Ring 1 (thin):           in thin band
            Ring 2 (medium-thin):     in medium band NOT in thin band
            Ring 3 (thick-medium):    in thick band NOT in medium band
        """
        thin = _ring_mask(gt_for_trimap, self.num_classes, self.k_thin)
        medium = _ring_mask(gt_for_trimap, self.num_classes, self.k_medium)
        thick = _ring_mask(gt_for_trimap, self.num_classes, self.k_thick)
        ring1 = thin
        ring2 = medium * (1.0 - thin)
        ring3 = thick * (1.0 - medium)
        w_pixel = (
            self.w_thin * ring1
            + self.w_medium_thin * ring2
            + self.w_thick_medium * ring3
        )
        return w_pixel, ring1, ring2, ring3

    # -- forward --

    def forward(
        self,
        logits: torch.Tensor,
        targets_ce: torch.Tensor,
        global_step: int = 0,
        *,
        gt_complete: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute L_total = L_main + lambda(t) * L_boundary.

        Args:
            logits: (B, C, H, W).
            targets_ce: (B, H, W) long. CE target. For Potsdam dual-label, this is noBoundary
                (Clutter=255, original boundary=255). For LoveDA, the standard 7-class label.
            global_step: current training step (for lambda ramp + EMA warmup).
            gt_complete: (B, H, W) long, optional. Required for Potsdam dual-label (trimap
                constructed from complete labels). If None, trimap uses targets_ce.

        Returns:
            loss: scalar tensor.
            diag: dict of diagnostic values (l_main, l_boundary, lambda_t, n_band, etc.).
        """
        # Move EMA buffers to logits device if needed (buffer created on CPU by default).
        if self.ema_mu_H.device != logits.device:
            self.ema_mu_H = self.ema_mu_H.to(logits.device)
            self.ema_sigma_H = self.ema_sigma_H.to(logits.device)
            self.ema_step = self.ema_step.to(logits.device)

        # 1. L_main = SoftCE_full + Dice_full (always computed on full image, targets_ce).
        l_softce_full = self._soft_ce_full(logits, targets_ce)
        l_dice_full = self._dice_full(logits, targets_ce)
        l_main = l_softce_full + l_dice_full

        # gate_closed short-circuit: lambda ≡ 0, no boundary loss computed.
        lam = self._lambda_at(global_step)
        diag = {
            "l_main": float(l_main.detach().item()),
            "l_softce_full": float(l_softce_full.detach().item()),
            "l_dice_full": float(l_dice_full.detach().item()),
            "lambda_t": float(lam),
            "l_boundary": 0.0,
            "n_band": 0,
            "n_band_valid": 0,
            "boundary_weight_type": self.boundary_weight_type,
        }
        if self.boundary_weight_type == "gate_closed" or lam == 0.0:
            return l_main, diag

        # 2. Construct trimap from complete labels (if dual-label) or CE labels.
        gt_for_trimap = gt_complete if (self.dual_label and gt_complete is not None) else targets_ce

        # 3. Compute per-pixel weights.
        if self.boundary_weight_type == "per_cc_entropy":
            # per_cc_entropy: trimap K=3 main band; weights via per-CC entropy.
            band = _per_class_boundary_band(gt_for_trimap, self.num_classes, self.k_main)
            # Potsdam dual-label: exclude noBoundary=255 pixels from band.
            # (Original boundary pixels were erased to 255 in noBoundary TIF; they fall inside
            # the K=3 trimap band by construction. Excluding them keeps N_band semantics clean.)
            valid_ce = (targets_ce != self.ignore_index).float()
            band_valid = band * valid_ce
            w_pixel, cc_stats = self._per_cc_weights(logits, band_valid, targets_ce)
            diag.update(cc_stats)  # monitoring.json source
            n_band = int(band.sum().item())
            n_band_valid = int(band_valid.sum().item())
        elif self.boundary_weight_type == "ring_difference":
            # ring_difference: 3 rings; weights 3/2/1. Ring construction uses gt_for_trimap (complete).
            # Single mask-compute pass returns w_pixel + the 3 disjoint ring masks.
            w_pixel, ring1, ring2, ring3 = self._ring_difference_weights(gt_for_trimap)
            # Restrict to valid CE pixels (Potsdam dual-label exclusion).
            valid_ce = (targets_ce != self.ignore_index).float()
            w_pixel = w_pixel * valid_ce
            # Wave-2 class-protection: boost boundary weight for protected classes
            # (e.g. {2: 1.5} → LowVeg CE in band ×1.5, counters LowVeg→Tree drift).
            if self.band_class_weights:
                t_clamped = targets_ce.clamp(0, self.num_classes - 1)
                mult = self.band_cw.to(w_pixel.device)[t_clamped]
                w_pixel = w_pixel * mult
                diag["band_boosted_pix"] = int((mult > 1.0).sum().item())
            n_band = int((w_pixel > 0).sum().item())
            n_band_valid = n_band
            # Ring pixel counts — raw (from complete labels) + valid (post-noBoundary=255 mask).
            # Note: derived from the same masks returned by _ring_difference_weights
            # to avoid a second _ring_mask pass (k=2+5+10 × num_classes per step).
            ring1_v = (ring1 * valid_ce).sum().item()
            ring2_v = (ring2 * valid_ce).sum().item()
            ring3_v = (ring3 * valid_ce).sum().item()
            diag["ring1_pix"] = int(ring1.sum().item())
            diag["ring2_pix"] = int(ring2.sum().item())
            diag["ring3_pix"] = int(ring3.sum().item())
            diag["ring1_pix_valid"] = int(ring1_v)
            diag["ring2_pix_valid"] = int(ring2_v)
            diag["ring3_pix_valid"] = int(ring3_v)
        else:
            raise RuntimeError(f"Unreachable: boundary_weight_type={self.boundary_weight_type!r}")

        # 4. L_boundary = mean over batch of [(1/N_band_b) * Sum_band_b w(x,y) * SoftCE(x,y)]
        # Per spec: per-image normalization, then batch-mean. A batch-global
        # normalization would bias toward images with larger bands.
        if n_band_valid == 0:
            # Edge case: empty band (e.g., all-ignore batch). L_boundary = 0.
            l_boundary = torch.zeros((), device=logits.device, dtype=l_main.dtype)
        else:
            ce_per_pixel = self._soft_ce_per_pixel(logits, targets_ce)  # (B, H, W)
            weighted = w_pixel * ce_per_pixel  # (B, H, W)
            # Per-image: numerator sum, denominator count, then mean over batch.
            B = logits.shape[0]
            per_image_loss = torch.zeros(B, device=logits.device, dtype=l_main.dtype)
            for b in range(B):
                n_b = float(w_pixel[b].gt(0).sum().item())  # band pixels with positive weight
                if n_b > 0:
                    per_image_loss[b] = weighted[b].sum() / n_b
            l_boundary = per_image_loss.mean()

        diag["l_boundary"] = float(l_boundary.detach().item())
        diag["n_band"] = n_band
        diag["n_band_valid"] = n_band_valid

        loss = l_main + lam * l_boundary
        diag["loss_total"] = float(loss.detach().item())
        return loss, diag


__all__ = ["BDiceV2Loss"]
