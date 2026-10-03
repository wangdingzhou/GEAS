"""Training-time data augmentation utilities.

For P1 (MM-DINO recipe retrain) and beyond. Not used by P0.5 eval pilots.

Standard mmseg PhotometricDistortion translation. Note: MM-DINO's recipe
DISABLES ColorJitter (and rotation/grayscale), so using PhotoMetricDistortion
in P1 deviates from the MM-DINO byte-port. Use only as an A/B knob.
"""

from __future__ import annotations
import random
import numpy as np
import cv2


class PhotoMetricDistortion:
    """Photometric metric distortion (mmseg standard).

    Pipeline (applied in random order with 50% probability each):
      1. Brightness: add uniform delta in [-brightness_delta, +brightness_delta]
      2. Contrast: multiply by uniform factor in contrast_range
      3. Saturation: HSV-S channel multiply (mode 1)
      4. Hue: HSV-H channel shift
      5. Contrast (mode 0 — opposite order)
      6. Brightness (re-applied)

    Args:
        brightness_delta: int, max brightness delta in pixel units (0-255).
            Default 32 (mmseg standard).
        contrast_range: tuple, contrast multiplicative range.
            Default (0.5, 1.5).
        saturation_range: tuple, saturation multiplicative range.
            Default (0.5, 1.5).
        hue_delta: int, max hue delta in degrees (0-360).
            Default 18 (mmseg standard).

    Reference: mmseg.datasets.transforms.PhotoMetricDistortion
    """

    def __init__(self,
                 brightness_delta: int = 32,
                 contrast_range: tuple = (0.5, 1.5),
                 saturation_range: tuple = (0.5, 1.5),
                 hue_delta: int = 18):
        self.brightness_delta = brightness_delta / 255.0
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta / 360.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply photometric distortion.

        Args:
            image: (H, W, 3) uint8 or float32 RGB image.

        Returns:
            Distorted image, same dtype as input. Float32 internally to avoid
            precision loss in additive brightness step.
        """
        was_uint8 = image.dtype == np.uint8
        img = image.astype(np.float32) / (255.0 if was_uint8 else 1.0)

        # 1. Random brightness (additive delta)
        if random.randint(0, 1):
            img = img + random.uniform(-self.brightness_delta, self.brightness_delta)

        # 2. Random contrast mode (random 0 or 1, applied before or after HSV)
        mode = random.randint(0, 1)
        if mode == 1:
            img = img * random.uniform(*self.contrast_range)

        # 3-4. Convert to HSV for saturation + hue
        # cv2 expects float32 in [0, 1] for HSV
        img_clamped = np.clip(img, 0.0, 1.0).astype(np.float32)
        hsv = cv2.cvtColor(img_clamped, cv2.COLOR_RGB2HSV)

        # Saturation
        if random.randint(0, 1):
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(*self.saturation_range),
                                  0.0, 1.0)

        # Hue (circular shift)
        if random.randint(0, 1):
            hue_shift = random.uniform(-self.hue_delta, self.hue_delta)
            hsv[..., 0] = hsv[..., 0] + hue_shift
            hsv[..., 0][hsv[..., 0] > 1.0] -= 1.0
            hsv[..., 0][hsv[..., 0] < 0.0] += 1.0

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # 5. Contrast (mode 0 — applied after HSV)
        if mode == 0:
            img = img * random.uniform(*self.contrast_range)

        # 6. Brightness (re-applied per mmseg)
        if random.randint(0, 1):
            img = img + random.uniform(-self.brightness_delta, self.brightness_delta)

        if was_uint8:
            return np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img


# ---------------------------------------------------------------------------
# Recipe-v1 shared helpers (2026-08-16; used by LoveDA/Potsdam/Vaihingen datasets)
# ---------------------------------------------------------------------------

def sample_crop_with_ratio(h, w, sz, mask=None, ignore_idx=-1, num_classes=6,
                           cat_max_ratio=None, max_attempts=10):
    """Random (top, left) with cat_max_ratio rejection (GeoSA-style, mmsem
    10-attempt cap then accept). Rejects crops whose max VALID class ratio
    exceeds cat_max_ratio; ignore label excluded from the ratio."""
    import random as _random
    top, left = 0, 0
    for _ in range(max_attempts):
        top = _random.randint(0, max(0, h - sz))
        left = _random.randint(0, max(0, w - sz))
        if mask is None or cat_max_ratio is None:
            return top, left
        c = mask[top:top + sz, left:left + sz]
        valid = (c >= 0) & (c != ignore_idx)
        n = int(valid.sum())
        if n == 0:
            return top, left
        counts = np.bincount(c[valid].astype(np.int64), minlength=num_classes)
        if counts.max() / n <= cat_max_ratio:
            return top, left
    return top, left


def scale_first_resize(image, masks, scale_min=0.5, scale_max=2.0):
    """Recipe-v1 crop order: resize the FULL source by a random scale BEFORE
    cropping (changes field-of-view). Returns (image, *masks) resized."""
    import random as _random
    scale = _random.uniform(scale_min, scale_max)
    h, w = image.shape[:2]
    nh, nw = int(h * scale), int(w * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = [image]
    for m in masks:
        out.append(cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST))
    return out


# ---------------------------------------------------------------------------
# Convenience: full P1 augmentation pipeline (RandomResize + Crop + Flip + PMD)
# ---------------------------------------------------------------------------

class P1TrainAugment:
    """P1 train-time augmentation pipeline.

    MM-DINO recipe (byte-port) + optional PhotoMetricDistortion.

    MM-DINO base (DO match this):
      - RandomResize(0.5, 2.0)
      - RandomCrop(512)
      - HFlip(0.5), VFlip(0.5)
      - NO rotation, NO grayscale (commented out in MM-DINO source)

    Optional add-on (A/B knob, OFF by default to match MM-DINO):
      - PhotoMetricDistortion (color jitter, mmseg standard)

    Labels: pass through (no geometric distortion for now; if RandomResize/Crop
    enabled, labels must be transformed in lock-step).
    """

    def __init__(self, crop_size: int = 512,
                 use_pmd: bool = False,
                 pmd_kwargs: dict | None = None):
        self.crop_size = crop_size
        self.use_pmd = use_pmd
        self.pmd = PhotoMetricDistortion(**(pmd_kwargs or {})) if use_pmd else None

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """Apply augmentation to (image, mask) pair.

        Args:
            image: (H, W, 3) RGB uint8.
            mask: (H, W) int32 label map.

        Returns:
            (image, mask) augmented. Geometric transforms (resize/crop/flip)
            applied to BOTH image and mask; color transform (PMD) applied to
            image only.

        Note: This is a minimal stub for P1 planning. Real P1 training will
        likely use albumentations or mmseg transforms for performance. This
        utility exists so the augmentation policy is reviewable as Python code.
        """
        if self.use_pmd:
            image = self.pmd(image)
        # Geometric aug TBD when P1 train script is written.
        return image, mask
