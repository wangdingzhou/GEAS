"""Vaihingen (ISPRS) dataset for remote sensing semantic segmentation.

Directory structure:
    root/
    ├── top/                         — 33 orthophoto tiles (9 cm GSD, RGB TIF)
    │   └── top_mosaic_09cm_area{N}.tif
    ├── gts_complete/                — 33 complete label TIFs (with boundary, color-coded)
    │   └── top_mosaic_09cm_area{N}.tif
    └── gts_eroded_for_participants/ — 16 noBoundary label TIFs (eroded, color-coded)
        └── top_mosaic_09cm_area{N}_noBoundary.tif

Classes (6, ISPRS palette — identical to Potsdam):
    0=Impervious, 1=Building, 2=LowVeg, 3=Tree, 4=Car, 5=Clutter

Locked values:
  - Split: MM-DINO 12/4 — train areas {1,3,7,11,13,17,23,26,28,32,34,37},
    test areas {5,15,21,30}. LOCKED; no other preset is offered.
  - Normalization: ImageNet (standard, NOT MM-DINO geo stats).
  - dual_label=True: complete labels for trimap, noBoundary for CE target/fair metrics.
  - samples_per_tile=100 (config): 12 tiles x 100 x 60 epochs / batch 8 = 9,000 steps.
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VaihingenDataset(Dataset):
    """Vaihingen dataset with 6 classes (including Clutter).

    Label color map (BGR as read by cv2) — same ISPRS palette as Potsdam:
        (255, 255, 255) -> 0  Impervious (White)
        (255, 0,   0  ) -> 1  Building (Blue in RGB)
        (255, 255, 0  ) -> 2  Low vegetation (Cyan in RGB)
        (0,   255, 0  ) -> 3  Tree (Green)
        (0,   255, 255) -> 4  Car (Yellow in RGB)
        (0,   0,   255) -> 5  Clutter (Red in RGB)
    Black (0,0,0) = undefined / eroded boundary -> 255 (ignore).
    """

    # BGR color -> class index (verified from actual label TIF files, 2026-08-07)
    COLOR_MAP = {
        (255, 255, 255): 0,   # Impervious (White)
        (255, 0,   0  ): 1,   # Building (Blue in RGB)
        (255, 255, 0  ): 2,   # Low vegetation (Cyan in RGB)
        (0,   255, 0  ): 3,   # Tree (Green)
        (0,   255, 255): 4,   # Car (Yellow in RGB)
        (0,   0,   255): 5,   # Clutter (Red in RGB)
    }

    CLASS_NAMES = ["Impervious", "Building", "LowVeg", "Tree", "Car", "Clutter"]
    NUM_CLASSES = 6
    IGNORE_INDEX = -1  # Not used — all 6 classes participate in loss

    _NORM_MAP = {
        "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "sat493m": ([0.430, 0.411, 0.296], [0.213, 0.156, 0.143]),
    }

    # MM-DINO released split (12 train / 4 test from the 16 annotated tiles).
    # LOCKED — do not add other presets (prevents split leakage).
    MM_DINO_TRAIN_AREAS = [1, 3, 7, 11, 13, 17, 23, 26, 28, 32, 34, 37]
    MM_DINO_TEST_AREAS = [5, 15, 21, 30]

    SPLIT_PRESETS = {
        "mm_dino": (MM_DINO_TRAIN_AREAS, MM_DINO_TEST_AREAS),
    }

    def __init__(self, root: str, split: str = "train", transform: bool = True,
                 image_size: int = 512, tile_ids: list = None,
                 samples_per_tile: int = 1, input_norm: str = "imagenet",
                 split_preset: str = "mm_dino",
                 use_noBoundary_labels: bool = False,
                 dual_label: bool = False,
                 photo_metric: bool = False, pmd_brightness: int = 24,
                 pmd_contrast: tuple = (0.65, 1.35), pmd_saturation: tuple = (0.65, 1.35),
                 pmd_hue: int = 0, cat_max_ratio: float | None = None,
                 scale_first: bool = False):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.samples_per_tile = samples_per_tile if (split == "train" and transform) else 1
        # noBoundary labels: official eroded GT subdir.
        self.use_noBoundary_labels = bool(use_noBoundary_labels)
        # Recipe-v1 knobs (2026-08-16 port from LoveDA; defaults OFF = legacy).
        self.cat_max_ratio = cat_max_ratio
        self.scale_first = scale_first
        self.pmd = None
        if photo_metric:
            from transforms import PhotoMetricDistortion
            self.pmd = PhotoMetricDistortion(
                brightness_delta=pmd_brightness, contrast_range=pmd_contrast,
                saturation_range=pmd_saturation, hue_delta=pmd_hue)
            print(f"[recipe:vaihingen] PMD(b={pmd_brightness},c={pmd_contrast},"
                  f"s={pmd_saturation},hue={pmd_hue}) cat_max={cat_max_ratio} "
                  f"scale_first={scale_first}")
        # BoundaryDice-v2 dual-label mode: load BOTH complete (with Clutter) and
        # noBoundary labels. Returns (image, noBoundary_mask, complete_mask) per item.
        # Trimap constructed from complete; CE target = noBoundary.
        self.dual_label = bool(dual_label)
        if self.dual_label and not self.use_noBoundary_labels:
            raise ValueError(
                "dual_label=True requires use_noBoundary_labels=True "
                "(noBoundary is the CE target; complete is the trimap source)."
            )
        if input_norm not in self._NORM_MAP:
            raise ValueError(f"Unknown input_norm={input_norm}. Choose from {list(self._NORM_MAP)}")
        self.input_norm = input_norm

        # Determine tile split (only the locked MM-DINO 12/4 preset is offered).
        if tile_ids is not None:
            self.tile_ids = tile_ids
            preset_used = "custom"
        elif split_preset not in self.SPLIT_PRESETS:
            raise ValueError(f"Unknown split_preset={split_preset}. Valid: {list(self.SPLIT_PRESETS)}")
        else:
            train_areas, test_areas = self.SPLIT_PRESETS[split_preset]
            self.tile_ids = train_areas if split == "train" else test_areas
            preset_used = split_preset
        print(f"[Vaihingen split] preset={preset_used}, split={split}, n_tiles={len(self.tile_ids)}")
        print(f"[Vaihingen split] areas={self.tile_ids}")

        img_dir = os.path.join(root, "top")
        mask_dir = os.path.join(root, "gts_eroded_for_participants")
        complete_dir = os.path.join(root, "gts_complete")

        self.images: list[str] = []
        self.masks: list[str] = []
        self.complete_masks: list[str] = []  # dual_label mode only

        for aid in self.tile_ids:
            img_name = f"top_mosaic_09cm_area{aid}.tif"
            img_path = os.path.join(img_dir, img_name)
            mask_name = f"top_mosaic_09cm_area{aid}_noBoundary.tif"
            mask_path = os.path.join(mask_dir, mask_name)
            complete_path = None
            if self.dual_label:
                complete_name = f"top_mosaic_09cm_area{aid}.tif"
                complete_path = os.path.join(complete_dir, complete_name)

            if os.path.exists(img_path) and os.path.exists(mask_path):
                if self.dual_label and (complete_path is None or not os.path.exists(complete_path)):
                    print(f"WARNING: area {aid} dual_label=True but complete label missing "
                          f"(path={complete_path}); skipping tile")
                    continue
                self.images.append(img_path)
                self.masks.append(mask_path)
                if self.dual_label:
                    self.complete_masks.append(complete_path)
            else:
                print(f"WARNING: area {aid} not found (img={os.path.exists(img_path)}, "
                      f"mask={os.path.exists(mask_path)})")

        self.num_tiles = len(self.images)
        print(f"Vaihingen {split}: {self.num_tiles} tiles x {self.samples_per_tile} samples = {len(self)} total"
              + (f" [dual_label=True]" if self.dual_label else ""))

        mean, std = self._NORM_MAP[self.input_norm]
        self.normalize = transforms.Normalize(mean=mean, std=std)
        print(f"Vaihingen input_norm: {self.input_norm}")

        # Pre-compute nearest-color lookup (int32 to avoid overflow: 255^2=65025 > 32767)
        self._map_colors = np.array(list(self.COLOR_MAP.keys()), dtype=np.int32)
        self._map_indices = np.array(list(self.COLOR_MAP.values()), dtype=np.uint8)

        # Cache all tiles in memory (16 max tiles, irregular ~2-4k px each).
        self._image_cache: dict[int, np.ndarray] = {}
        self._mask_cache: dict[int, np.ndarray] = {}
        self._complete_cache: dict[int, np.ndarray] = {}  # dual_label mode only
        print(f"Caching {self.num_tiles} tiles in memory...")
        for i in range(self.num_tiles):
            img = cv2.imread(self.images[i])
            self._image_cache[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask_bgr = cv2.imread(self.masks[i])
            self._mask_cache[i] = self._color_to_label(mask_bgr)
            if self.dual_label:
                complete_bgr = cv2.imread(self.complete_masks[i])
                self._complete_cache[i] = self._color_to_label(complete_bgr)
            print(f"  [{i+1}/{self.num_tiles}] cached {os.path.basename(self.images[i])}")
        total_mb = (
            sum(v.nbytes for v in self._image_cache.values())
            + sum(v.nbytes for v in self._mask_cache.values())
            + sum(v.nbytes for v in self._complete_cache.values())
        )
        print(f"Cache done: {total_mb / 1024**3:.2f} GB")

    def _color_to_label(self, mask_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR label image to class index map using nearest-color matching.

        noBoundary labels contain black [0,0,0] (eroded boundary region) — mapped to
        255 (ignore_index), handled by ignoring them in loss (ignore_index=255) and
        in confusion matrix (targets < num_classes). Complete labels use only the
        6 class colors; the black branch is a no-op for them.
        """
        h, w = mask_bgr.shape[:2]
        pixels = mask_bgr.reshape(-1, 3).astype(np.int32)
        # Black mask: pixels where all channels are 0 (eroded boundary in noBoundary)
        is_black = (pixels.sum(axis=1) == 0)
        dists = ((pixels[:, None, :] - self._map_colors[None, :, :]) ** 2).sum(axis=2)
        nearest = dists.argmin(axis=1)
        labels = self._map_indices[nearest].astype(np.int32).copy()
        # Override: black -> 255 (ignore). Avoids argmin tie-breaking to Building.
        labels[is_black] = 255
        return labels.reshape(h, w).astype(np.int64)

    def __len__(self) -> int:
        return self.num_tiles * self.samples_per_tile

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tile_idx = idx % self.num_tiles

        image = self._image_cache[tile_idx].copy()
        mask = self._mask_cache[tile_idx].copy()
        complete = self._complete_cache[tile_idx].copy() if self.dual_label else None

        if self.transform and self.split == "train":
            if self.dual_label:
                image, mask, complete = self._train_augment_dual(image, mask, complete)
            else:
                image, mask = self._train_augment(image, mask)
        else:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            if self.dual_label:
                complete = cv2.resize(complete, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        mask = torch.from_numpy(mask).long()
        if self.dual_label:
            complete = torch.from_numpy(complete).long()
            return image, mask, complete
        return image, mask

    def _train_augment_dual(
        self, image: np.ndarray, mask: np.ndarray, complete: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Same augmentations as _train_augment, applied identically to image + noBoundary
        + complete label. Identical crop/rotation/flip/scale parameters ensure pixel-perfect
        alignment across the three tensors.
        """
        h, w = image.shape[:2]
        sz = self.image_size

        if self.scale_first:
            from transforms import scale_first_resize, sample_crop_with_ratio
            image, mask, complete = scale_first_resize(image, [mask, complete])
            h, w = image.shape[:2]
            if h >= sz and w >= sz:
                top, left = sample_crop_with_ratio(
                    h, w, sz, mask=mask, ignore_idx=self.IGNORE_INDEX,
                    num_classes=self.NUM_CLASSES, cat_max_ratio=self.cat_max_ratio)
            else:
                top, left = 0, 0
        else:
            # Random crop (Vaihingen tiles are irregular; small tiles handled below).
            if h >= sz and w >= sz:
                top = random.randint(0, h - sz)
                left = random.randint(0, w - sz)
            else:
                top, left = 0, 0
        image = image[top:top + sz, left:left + sz]
        mask = mask[top:top + sz, left:left + sz]
        complete = complete[top:top + sz, left:left + sz]

        # Random rotation (0/90/180/270)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k=k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
            complete = np.rot90(complete, k=k, axes=(0, 1)).copy()

        # Random flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            complete = np.flip(complete, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
            complete = np.flip(complete, axis=0).copy()

        # Random scale (legacy mid-pipeline; skipped under scale_first)
        if not self.scale_first and random.random() > 0.5:
            scale = random.uniform(0.5, 2.0)
            new_h, new_w = int(sz * scale), int(sz * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            complete = cv2.resize(complete, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if new_h >= sz and new_w >= sz:
                top = random.randint(0, new_h - sz)
                left = random.randint(0, new_w - sz)
            else:
                top, left = 0, 0
            image = image[top:top + sz, left:left + sz]
            mask = mask[top:top + sz, left:left + sz]
            complete = complete[top:top + sz, left:left + sz]

        image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
        complete = cv2.resize(complete, (sz, sz), interpolation=cv2.INTER_NEAREST)

        if self.pmd is not None:
            image = self.pmd(image)

        return image, mask, complete

    def _train_augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        sz = self.image_size

        if self.scale_first:
            from transforms import scale_first_resize, sample_crop_with_ratio
            image, mask = scale_first_resize(image, [mask])
            h, w = image.shape[:2]
            if h >= sz and w >= sz:
                top, left = sample_crop_with_ratio(
                    h, w, sz, mask=mask, ignore_idx=self.IGNORE_INDEX,
                    num_classes=self.NUM_CLASSES, cat_max_ratio=self.cat_max_ratio)
            else:
                top, left = 0, 0
        elif h >= sz and w >= sz:
            top = random.randint(0, h - sz)
            left = random.randint(0, w - sz)
        else:
            top, left = 0, 0
        image = image[top:top + sz, left:left + sz]
        mask = mask[top:top + sz, left:left + sz]

        # Random rotation (0/90/180/270) — essential for remote sensing
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k=k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        # Random flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random scale (legacy mid-pipeline; skipped under scale_first)
        if not self.scale_first and random.random() > 0.5:
            scale = random.uniform(0.5, 2.0)
            new_h, new_w = int(sz * scale), int(sz * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if new_h >= sz and new_w >= sz:
                top = random.randint(0, new_h - sz)
                left = random.randint(0, new_w - sz)
            else:
                top, left = 0, 0
            image = image[top:top + sz, left:left + sz]
            mask = mask[top:top + sz, left:left + sz]

        image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)

        if self.pmd is not None:
            image = self.pmd(image)

        return image, mask
