"""LoveDA dataset for remote sensing semantic segmentation."""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def decode_loveda_mask(mask: np.ndarray) -> np.ndarray:
    """Raw LoveDA mask -> mmseg/literature convention.

    Raw format (official): 0=void(no-data), 1=Background, 2=Building, 3=Road,
    4=Water, 5=Barren, 6=Forest, 7=Agriculture.
    Converted: 0=Background ... 6=Agriculture, 255=ignore (raw 0).
    uint8-safe: np.where form avoids the 255-1=254 wraparound of subtract-after-map.
    Single source of truth — import this everywhere LoveDA masks are decoded.
    """
    return np.where(mask == 0, 255, mask - 1).astype(np.uint8)


class LoveDADataset(Dataset):
    """LoveDA dataset with 7 classes + ignore index.

    Label encoding (mmseg/LoveDA official convention, post decode_loveda_mask):
        0=Background, 1=Building, 2=Road, 3=Water,
        4=Barren, 5=Forest, 6=Agriculture, 255=Ignore (raw 0 = no-data)
    """

    CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agriculture"]
    NUM_CLASSES = 7
    IGNORE_INDEX = 255

    _NORM_MAP = {
        "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "sat493m": ([0.430, 0.411, 0.296], [0.213, 0.156, 0.143]),
    }

    def __init__(self, root: str, split: str = "Train", transform: bool = True,
                 image_size: int = 512, input_norm: str = "imagenet",
                 photo_metric: bool = False,
                 pmd_brightness: int = 10,
                 pmd_contrast: tuple = (0.8, 1.2),
                 pmd_saturation: tuple = (0.8, 1.2),
                 pmd_hue: int = 0,
                 cat_max_ratio: float | None = None,
                 scale_first: bool = False):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_size = image_size
        if input_norm not in self._NORM_MAP:
            raise ValueError(f"Unknown input_norm={input_norm}. Choose from {list(self._NORM_MAP)}")
        self.input_norm = input_norm

        # Augmentation knobs:
        # mild photometric distortion (hue OFF — spectral-class hazard),
        # ignore-excluded class-ratio crop rejection, multi-scale-first crop.
        # All default OFF -> byte-identical legacy pipeline.
        self.cat_max_ratio = cat_max_ratio
        self.scale_first = scale_first
        self.pmd = None
        if photo_metric:
            from transforms import PhotoMetricDistortion
            self.pmd = PhotoMetricDistortion(
                brightness_delta=pmd_brightness,
                contrast_range=pmd_contrast,
                saturation_range=pmd_saturation,
                hue_delta=pmd_hue,
            )
            print(f"[recipe] PhotoMetricDistortion ON: brightness={pmd_brightness} "
                  f"contrast={pmd_contrast} saturation={pmd_saturation} hue={pmd_hue}")
        if cat_max_ratio is not None:
            print(f"[recipe] cat_max_ratio={cat_max_ratio} (ignore-excluded, 10-attempt cap)")
        if scale_first:
            print("[recipe] scale-first crop order (resize full image, then crop)")

        self.images: list[str] = []
        self.masks: list[str] = []

        for scene in ["Urban", "Rural"]:
            img_dir = os.path.join(root, split, scene, "images_png")
            mask_dir = os.path.join(root, split, scene, "masks_png")
            if not os.path.isdir(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith(".png"):
                    self.images.append(os.path.join(img_dir, fname))
                    self.masks.append(os.path.join(mask_dir, fname))

        print(f"LoveDA {split}: {len(self.images)} images")

        mean, std = self._NORM_MAP[self.input_norm]
        self.normalize = transforms.Normalize(mean=mean, std=std)
        print(f"LoveDA input_norm: {self.input_norm}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Decode to mmseg convention (0-6 + 255) BEFORE any augmentation so
        # _crop_with_cat_ratio and all downstream 255-ignore logic see correct labels.
        mask = decode_loveda_mask(cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE))

        if self.transform and self.split == "Train":
            image, mask = self._train_augment(image, mask)
        else:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image = self.normalize(image)
        mask = torch.from_numpy(mask).long()

        # Post 2026-08-21 fix: masks already carry 255 ignore from decode_loveda_mask;
        # this stays as a no-op safety net for any future decode path drift.
        mask[mask == self.IGNORE_INDEX] = 255

        return image, mask

    def _train_augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        sz = self.image_size

        if self.scale_first:
            # Recipe-v1 order (GeoSA): resize FULL image by 0.5-2.0, then crop.
            # Changes field-of-view (legacy order only re-scales a fixed crop).
            scale = random.uniform(0.5, 2.0)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if new_h < sz or new_w < sz:
                image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
            else:
                image, mask = self._crop_with_cat_ratio(image, mask, sz)
        else:
            # Legacy pipeline (byte-identical when recipe knobs are off).
            # Random crop
            if h >= sz and w >= sz:
                top = random.randint(0, h - sz)
                left = random.randint(0, w - sz)
                image = image[top : top + sz, left : left + sz]
                mask = mask[top : top + sz, left : left + sz]
            else:
                image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)

        # Random horizontal flip
        if random.random() > 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))

        # Random vertical flip
        if random.random() > 0.5:
            image = np.ascontiguousarray(np.flipud(image))
            mask = np.ascontiguousarray(np.flipud(mask))

        if not self.scale_first:
            # Random resize (scale 0.5–2.0) then crop back to sz
            # Use current image size (after crop) for correct scaling
            cur_h, cur_w = image.shape[:2]
            if random.random() > 0.5:
                scale = random.uniform(0.5, 2.0)
                new_h, new_w = int(cur_h * scale), int(cur_w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if new_h >= sz and new_w >= sz:
                    top = random.randint(0, new_h - sz)
                    left = random.randint(0, new_w - sz)
                    image = image[top : top + sz, left : left + sz]
                    mask = mask[top : top + sz, left : left + sz]
                else:
                    image = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(mask, (sz, sz), interpolation=cv2.INTER_NEAREST)

        # Photometric distortion LAST (mmseg order; image only).
        if self.pmd is not None:
            image = self.pmd(image)

        return image, mask

    def _crop_with_cat_ratio(self, image: np.ndarray, mask: np.ndarray,
                             sz: int) -> tuple[np.ndarray, np.ndarray]:
        """Random crop with cat_max_ratio rejection (ignore label excluded).

        mmsem-style: reject crops where any VALID class exceeds
        self.cat_max_ratio of valid pixels; up to 10 attempts, then accept
        (standard mmseg fallback — softens the sampler without data loss)."""
        h, w = image.shape[:2]
        for _ in range(10):
            top = random.randint(0, h - sz)
            left = random.randint(0, w - sz)
            c_img = image[top : top + sz, left : left + sz]
            c_msk = mask[top : top + sz, left : left + sz]
            if self.cat_max_ratio is None:
                return c_img, c_msk
            valid = (c_msk >= 0) & (c_msk != self.IGNORE_INDEX)
            n_valid = int(valid.sum())
            if n_valid == 0:
                return c_img, c_msk
            counts = np.bincount(
                c_msk[valid].astype(np.int64), minlength=self.NUM_CLASSES)
            if counts.max() / n_valid <= self.cat_max_ratio:
                return c_img, c_msk
        return c_img, c_msk
