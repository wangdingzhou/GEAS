"""LoveDA train wrapper for ClassBalancedCopyPaste.

LoveDACBCPDataset(LoveDADataset) — applies cbcp transform only on
urban-context tiles after baseline _train_augment. Rural tiles pass through
unchanged (source class already in correct context).

Val/Test path: identical to LoveDADataset (no augmentation, no cbcp).

Scene per tile resolved at __init__ by parsing self.images paths
(LoveDA directory structure: Train/{Urban,Rural}/images_png/).

Thread-safety: with multi-worker DataLoader, each worker process has its own
dataset copy; _current_idx is set per-worker (one __getitem__ at a time within
a worker process).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from dataset import LoveDADataset
from class_balanced_copy_paste import ClassBalancedCopyPaste


class LoveDACBCPDataset(LoveDADataset):
    """paste Forest/Agri onto urban tiles only."""

    def __init__(
        self,
        *args,
        cbcp_transform: Optional[ClassBalancedCopyPaste] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cbcp_transform = cbcp_transform
        self._is_urban: list[bool] = []
        for path in self.images:
            # path: /.../LoveDA/{Split}/{Urban,Rural}/images_png/NNNN.png
            scene = path.split(os.sep)[-3]
            self._is_urban.append(scene == "Urban")
        self._n_urban = sum(self._is_urban)
        self._current_idx: Optional[int] = None
        print(
            f"LoveDACBCPDataset: {self._n_urban}/{len(self._is_urban)} urban tiles "
            f"(cbcp applied on urban only)"
        )

    def __getitem__(self, idx: int):
        self._current_idx = idx
        return super().__getitem__(idx)

    def _train_augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Baseline LoveDA augment, then cbcp on urban tiles only."""
        image, mask = super()._train_augment(image, mask)
        if self.cbcp_transform is None or self._current_idx is None:
            return image, mask
        if self._is_urban[self._current_idx]:
            image, mask = self.cbcp_transform(image, mask)
        return image, mask

    def cbcp_summary(self) -> dict:
        """Forward transform summary for startup verification logging."""
        if self.cbcp_transform is None:
            return {"enabled": False}
        return {"enabled": True, **self.cbcp_transform.summary()}
