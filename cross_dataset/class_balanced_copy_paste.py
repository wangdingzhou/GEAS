"""Class-balanced copy-paste augmentation for LoveDA Forest/Agriculture.

Components:
  - InstancePool: precomputed cache of Forest+Agri instance patches from rural tiles.
  - ClassBalancedCopyPaste: training-time transform.
      Variant A: plain paste with geometric sanity checks.
      Variant B: +harmonization (alpha-blend 0.5).
      Variant C: +layout-class decoupling (target context not class-bound).

LoveDA label encoding (must match src/dataset.py):
  0=Background, 1=Building, 2=Road, 3=Water, 4=Barren,
  5=Forest, 6=Agriculture, 255=Ignore (decoded via raw 0=void,1-7->0-6; 2026-08-21 fix).
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage


# LoveDA label constants (must match src/dataset.py:13-23)
LABEL_BACKGROUND = 0
LABEL_BUILDING = 1
LABEL_ROAD = 2
LABEL_WATER = 3
LABEL_BARREN = 4
LABEL_FOREST = 5
LABEL_AGRICULTURE = 6

CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]


# ---------------------------------------------------------------------------
# InstancePool: precomputed cache of source-class patches from rural tiles
# ---------------------------------------------------------------------------

@dataclass
class InstancePoolConfig:
    """Source class extraction from rural LoveDA Train."""

    rural_root: str                                 # /.../LoveDA/Train/Rural
    source_classes: tuple[int, ...] = (LABEL_FOREST, LABEL_AGRICULTURE)
    cc_min_px: int = 100                            # drop tiny noise specks
    cc_max_tile_ratio: float = 0.25                 # drop "superblobs"
    tile_size_norm: int = 512                       # cap cached patch max-dim
    max_per_class: int = 200                        # memory cap per source class
    seed: int = 42


@dataclass
class PoolStats:
    """Build-time diagnostics (logged once at startup)."""

    n_tiles_scanned: int = 0
    n_instances_per_class: dict[int, int] = field(default_factory=dict)
    n_skipped_size: int = 0           # dropped < cc_min_px
    n_skipped_overflow: int = 0       # dropped > cc_max_tile_ratio
    skipped_after_cap: int = 0        # dropped due to max_per_class cap


class InstancePool:
    """Pre-computed pool of source-class instance patches (image + mask).

    Built ONCE at dataset init by scanning rural LoveDA Train tiles. Each
    patch is a connected-component region of a source class label, size-
    filtered, and resized so max-dim <= tile_size_norm. In-memory
    cache (no disk IO after build).
    """

    def __init__(self, cfg: InstancePoolConfig):
        self.cfg = cfg
        self._patches: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            cls: [] for cls in cfg.source_classes
        }
        self.stats = PoolStats()
        self._rng = random.Random(cfg.seed)
        self._built = False

    def build(self) -> None:
        """Scan rural tiles, extract CC instances per source class, cache."""
        if self._built:
            return

        images_dir = os.path.join(self.cfg.rural_root, "images_png")
        masks_dir = os.path.join(self.cfg.rural_root, "masks_png")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Rural images dir not found: {images_dir}")

        fnames = sorted(f for f in os.listdir(images_dir) if f.endswith(".png"))

        for fname in fnames:
            self.stats.n_tiles_scanned += 1
            # 2026-08-21 fix: raw LoveDA 0=void,1-7=Background..Agriculture -> 0-6+255
            _m = cv2.imread(os.path.join(masks_dir, fname), cv2.IMREAD_GRAYSCALE)
            if _m is None:
                continue
            mask = np.where(_m == 0, 255, _m - 1).astype(np.uint8)
            image = cv2.imread(os.path.join(images_dir, fname))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for cls in self.cfg.source_classes:
                if len(self._patches[cls]) >= self.cfg.max_per_class:
                    self.stats.skipped_after_cap += 1
                    continue
                self._extract_instances_for_class(image, mask, cls)

        # Final stats init
        for cls in self.cfg.source_classes:
            self.stats.n_instances_per_class.setdefault(cls, len(self._patches[cls]))

        self._built = True

    def _extract_instances_for_class(
        self, image: np.ndarray, mask: np.ndarray, cls: int
    ) -> None:
        """Extract CC instances of one class from one tile."""
        binary = (mask == cls).astype(np.uint8)
        if binary.sum() == 0:
            return

        labeled, n_cc = ndimage.label(binary)
        if n_cc == 0:
            return

        tile_area = binary.shape[0] * binary.shape[1]
        max_area = self.cfg.cc_max_tile_ratio * tile_area

        for cc_id in range(1, n_cc + 1):
            cc_mask = (labeled == cc_id)
            cc_area = int(cc_mask.sum())

            if cc_area < self.cfg.cc_min_px:
                self.stats.n_skipped_size += 1
                continue
            if cc_area > max_area:
                self.stats.n_skipped_overflow += 1
                continue

            ys, xs = np.where(cc_mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1

            patch_img = image[y0:y1, x0:x1].copy()
            patch_mask = (cc_mask[y0:y1, x0:x1].astype(np.uint8)) * cls

            h, w = patch_img.shape[:2]
            if max(h, w) > self.cfg.tile_size_norm:
                scale = self.cfg.tile_size_norm / max(h, w)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                patch_img = cv2.resize(patch_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                patch_mask = cv2.resize(patch_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            self._patches[cls].append((patch_img, patch_mask))

    def sample(
        self, cls: int, rng: Optional[random.Random] = None
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Sample one instance patch of class `cls`. Returns None if pool empty."""
        r = rng or self._rng
        pool = self._patches.get(cls, [])
        if not pool:
            return None
        return r.choice(pool)

    def pool_size(self, cls: int) -> int:
        return len(self._patches.get(cls, []))


# ---------------------------------------------------------------------------
# ClassBalancedCopyPaste transform
# ---------------------------------------------------------------------------

@dataclass
class CopyPasteConfig:
    """Copy-paste transform."""

    source_classes: tuple[int, ...] = (LABEL_FOREST, LABEL_AGRICULTURE)
    max_instances_per_tile: int = 3
    patch_size_min_ratio: float = 0.10          # of tile_size
    patch_size_max_ratio: float = 0.50
    scale_min: float = 0.70                     # GSD prior
    scale_max: float = 1.40
    ratio_cap: float = 2.0                      # post-aug / original
    harmonize: bool = False                     # Variant B
    decouple: bool = False                      # Variant C (no-op for now; caller-gated)
    harmonize_alpha: float = 0.5
    paste_prob: float = 1.0                     # P(attempt paste) per tile (spec default)
    position_max_attempts: int = 5              # fallback
    position_invalid_overlap: float = 0.10      # reject if invalid-region > 10% of patch
    tile_size: int = 512
    ratio_window: int = 100                     # sliding window for ratio monitor
    ratio_log_every: int = 50                   # log to ratio_log every N tiles
    seed: int = 42


class ClassBalancedCopyPaste:
    """Spec-compliant class-balanced copy-paste transform.

    Lifecycle: __call__(image, mask) per train tile. Caller responsibility:
    only invoke on urban-context tiles. Rural tiles pass
    through unchanged (wrapper handles this via scene flag).

    Single-factor decomposition:
      Variant A: cfg.harmonize=False, cfg.decouple=False
      Variant B: cfg.harmonize=True,  cfg.decouple=False  (alpha-blend image)
      Variant C: cfg.harmonize=True,  cfg.decouple=True   (caller uses random urban context)
      Variant D: transform=None (no-aug control, sanity)
    """

    def __init__(self, pool: InstancePool, cfg: Optional[CopyPasteConfig] = None):
        self.pool = pool
        self.cfg = cfg or CopyPasteConfig()
        self._rng = random.Random(self.cfg.seed)
        # Per-class pixel ratio tracking (sliding window over recent tiles)
        self._orig_history: dict[int, list[int]] = {cls: [] for cls in self.cfg.source_classes}
        self._post_history: dict[int, list[int]] = {cls: [] for cls in self.cfg.source_classes}
        self._tiles_seen = 0
        # Periodic ratio log (for startup verification + per-epoch monitor)
        self.ratio_log: list[dict] = []

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply copy-paste. Returns augmented (image, mask).

        Mask label encoding: decoded 0-6 valid, 255=ignore (2026-08-21 fix) — same as LoveDADataset
        pre-tensor). After this transform, the dataset wrapper maps 7->255.
        """
        cfg = self.cfg
        tile_size = cfg.tile_size
        H, W = image.shape[:2]

        orig_counts = {cls: int((mask == cls).sum()) for cls in cfg.source_classes}

        do_paste = self._rng.random() < cfg.paste_prob
        if not do_paste:
            self._record_tile(orig_counts, orig_counts)
            return image, mask

        image_aug = image.copy()
        mask_aug = mask.copy()
        n_pasted = 0

        for cls in cfg.source_classes:
            if n_pasted >= cfg.max_instances_per_tile:
                break
            # window-level ratio check
            if self._class_ratio(cls) > cfg.ratio_cap:
                continue

            attempts_for_cls = 0
            while (
                n_pasted < cfg.max_instances_per_tile
                and attempts_for_cls < cfg.max_instances_per_tile
            ):
                attempts_for_cls += 1
                patch = self.pool.sample(cls, rng=self._rng)
                if patch is None:
                    break
                patch_img, patch_mask = patch

                # scale: Beta(2,2) clipped to [scale_min, scale_max]
                scale_factor = self._sample_scale()
                new_w = max(1, int(round(patch_img.shape[1] * scale_factor)))
                new_h = max(1, int(round(patch_img.shape[0] * scale_factor)))

                # patch-size bounds
                size_min = int(tile_size * cfg.patch_size_min_ratio)
                size_max = int(tile_size * cfg.patch_size_max_ratio)
                new_w = int(np.clip(new_w, size_min, size_max))
                new_h = int(np.clip(new_h, size_min, size_max))

                scaled_img = cv2.resize(patch_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scaled_mask = cv2.resize(patch_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # position mask
                pos = self._sample_position(mask_aug, cls, new_h, new_w)
                if pos is None:
                    continue

                top, left = pos
                bot = min(top + new_h, H)
                rgt = min(left + new_w, W)
                actual_h = bot - top
                actual_w = rgt - left

                paste_img = scaled_img[:actual_h, :actual_w]
                paste_mask = scaled_mask[:actual_h, :actual_w]

                # per-paste ratio check
                added = int((paste_mask == cls).sum())
                if self._would_exceed_ratio(cls, added):
                    continue

                self._paste_one_tile(
                    image_aug, mask_aug, paste_img, paste_mask, top, left, bot, rgt, cls
                )
                n_pasted += 1

        post_counts = {cls: int((mask_aug == cls).sum()) for cls in cfg.source_classes}
        self._record_tile(orig_counts, post_counts)
        return image_aug, mask_aug

    # ----------------------- helpers ----------------------------------------

    def _paste_one_tile(
        self,
        image_aug: np.ndarray,
        mask_aug: np.ndarray,
        paste_img: np.ndarray,
        paste_mask: np.ndarray,
        top: int, left: int, bot: int, rgt: int,
        cls: int,
    ) -> None:
        """Apply one paste to image_aug / mask_aug in-place."""
        cfg = self.cfg

        if cfg.harmonize:
            # Variant B: alpha-blend image patch
            target_slice = image_aug[top:bot, left:rgt].astype(np.float32)
            blend = target_slice * (1.0 - cfg.harmonize_alpha) + paste_img.astype(np.float32) * cfg.harmonize_alpha
            image_aug[top:bot, left:rgt] = np.clip(blend, 0, 255).astype(np.uint8)
        else:
            # Variant A: hard paste image
            image_aug[top:bot, left:rgt] = paste_img

        # Mask always hard-paste (where patch has source class)
        patch_class_pixels = (paste_mask == cls)
        target_slice = mask_aug[top:bot, left:rgt]
        target_slice[patch_class_pixels] = cls
        mask_aug[top:bot, left:rgt] = target_slice

    def _sample_scale(self) -> float:
        """Beta(2,2) clipped to [scale_min, scale_max]."""
        b = self._rng.betavariate(2.0, 2.0)
        return self.cfg.scale_min + b * (self.cfg.scale_max - self.cfg.scale_min)

    def _sample_position(
        self, target_mask: np.ndarray, cls: int, patch_h: int, patch_w: int
    ) -> Optional[tuple[int, int]]:
        """Position mask.

        Forest (cls=5): not in Water regions (cls=3).
        Agriculture (cls=6): not in Building regions (cls=1).
        Other classes: no constraint (fallthrough).

        Reject if patch region overlaps invalid pixels > position_invalid_overlap.
        """
        H, W = target_mask.shape
        if patch_h >= H or patch_w >= W:
            return None

        if cls == LABEL_FOREST:
            invalid = (target_mask == LABEL_WATER)
        elif cls == LABEL_AGRICULTURE:
            invalid = (target_mask == LABEL_BUILDING)
        else:
            invalid = None

        for _ in range(self.cfg.position_max_attempts):
            top = self._rng.randint(0, H - patch_h)
            left = self._rng.randint(0, W - patch_w)
            if invalid is None:
                return (top, left)
            region = invalid[top:top + patch_h, left:left + patch_w]
            if region.sum() < self.cfg.position_invalid_overlap * region.size:
                return (top, left)
        return None

    def _class_ratio(self, cls: int) -> float:
        """Sliding-window ratio = post / orig."""
        o = sum(self._orig_history[cls])
        p = sum(self._post_history[cls])
        if o == 0:
            return 1.0
        return p / o

    def _would_exceed_ratio(self, cls: int, added: int) -> bool:
        o = sum(self._orig_history[cls])
        p = sum(self._post_history[cls]) + added
        if o == 0:
            # No original pixels in window; allow small addition (avoid hard-block)
            return p > 10000
        return (p / o) > self.cfg.ratio_cap

    def _record_tile(
        self,
        orig_counts: dict[int, int],
        post_counts: dict[int, int],
    ) -> None:
        for cls in self.cfg.source_classes:
            self._orig_history[cls].append(orig_counts[cls])
            self._post_history[cls].append(post_counts[cls])
            if len(self._orig_history[cls]) > self.cfg.ratio_window:
                self._orig_history[cls].pop(0)
                self._post_history[cls].pop(0)

        self._tiles_seen += 1
        if self._tiles_seen % self.cfg.ratio_log_every == 0:
            self.ratio_log.append({
                "tiles_seen": self._tiles_seen,
                **{f"ratio_{CLASS_NAMES[cls]}": round(self._class_ratio(cls), 4) for cls in self.cfg.source_classes},
            })

    # ----------------------- diagnostics ------------------------------------

    def summary(self) -> dict:
        """Return current state for startup verification / logging."""
        return {
            "tiles_seen": self._tiles_seen,
            "ratios": {CLASS_NAMES[cls]: round(self._class_ratio(cls), 4) for cls in self.cfg.source_classes},
            "harmonize": self.cfg.harmonize,
            "decouple": self.cfg.decouple,
            "pool_sizes": {CLASS_NAMES[cls]: self.pool.pool_size(cls) for cls in self.cfg.source_classes},
        }
