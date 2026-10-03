#!/usr/bin/env python3
"""Vaihingen (ISPRS): Sliding Window + TTA + MST Evaluation.

1:1 mirror of `eval_potsdam.py` with two improvements:
  (1) `--out_dir` CLI flag redirects confusion_matrix.csv output (the canonical
      eval_potsdam.py hardcodes `experiments/p1_full_mmdino_pots/eval/` — kept
      unchanged there per "don't edit canonical protocol file"; eval_vaihingen.py
      is new so we fix this from the start).
  (2) Uses VaihingenDataset (MM-DINO 12/4 split, test areas {5,15,21,30}) instead
      of PotsdamDataset.

Usage:
    python eval_vaihingen.py --checkpoint PATH --config PATH [--out_dir DIR]

Adapted from eval_potsdam.py per `feedback-reference-code-first` (don't reinvent).
"""
import sys
import os
import argparse
import json
import csv

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from omegaconf import OmegaConf

_CROSS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_CROSS_DIR)
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _CROSS_DIR)

from model import get_model  # noqa: F401  (imported for eval_potsdam side-effects)
from dataset_vaihingen import VaihingenDataset
from eval_potsdam import (
    load_model,
    sliding_window_inference,
    inference_with_tta,
    inference_with_mst,
    compute_confusion,
    _NORM_MAP,
)

CLASS_NAMES = VaihingenDataset.CLASS_NAMES  # 6: Impervious, Building, LowVeg, Tree, Car, Clutter
NUM_CLASSES = VaihingenDataset.NUM_CLASSES  # 6
IGNORE_IDX = -1
CLUTTER_IDX = 5  # excluded from mIoU-5 / mF1-5 (matches Potsdam convention)
FINAL_EVAL_STRIDE = 256
FINAL_EVAL_SCALES = "0.75,1.0,1.25,1.5"


def evaluate(model, data_root, tile_size, stride, device, tta_mode, scales,
             input_norm="imagenet", split_preset="mm_dino", out_dir=None, fusion="logit"):
    """Mirror eval_potsdam.evaluate() on Vaihingen MM-DINO test split.

    Returns (metrics_dict, confusion_matrix). Also writes confusion_matrix.csv
    to out_dir if provided.
    """
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    val_ds = VaihingenDataset(
        root=data_root,
        split="val",
        transform=False,
        image_size=6000,
        split_preset=split_preset,
        use_noBoundary_labels=True,
    )
    mean, std = _NORM_MAP.get(input_norm, _NORM_MAP["imagenet"])
    normalize = transforms.Normalize(mean=mean, std=std)
    scale_str = f", scales={[f'{s:.2f}' for s in scales]}" if scales else ""
    print(f"Eval: {len(val_ds)} tiles, tile={tile_size}, stride={stride}, "
          f"tta={tta_mode or 'none'}{scale_str}, norm={input_norm}")

    for i in range(len(val_ds)):
        img_path = val_ds.images[i]
        mask_path = val_ds.masks[i]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_bgr = cv2.imread(mask_path)
        mask = val_ds._color_to_label(mask_bgr)
        H, W = image.shape[:2]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = normalize(image_tensor)

        if scales and len(scales) > 1:
            logits = inference_with_mst(model, image_tensor, tile_size, stride, device, scales, tta_mode)
        elif tta_mode:
            logits = inference_with_tta(model, image_tensor, tile_size, stride, device, tta_mode)
        else:
            logits = sliding_window_inference(model, image_tensor, tile_size, stride, device, fusion=fusion)

        if logits.shape[1:] != (H, W):
            logits = F.interpolate(logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

        pred = logits.argmax(dim=0).numpy()
        confusion += compute_confusion(pred, mask, NUM_CLASSES, IGNORE_IDX)
        print(f"  [{i+1}/{len(val_ds)}] tile {os.path.basename(img_path)}")

    ious = []
    for cls in range(NUM_CLASSES):
        tp = confusion[cls, cls]
        union = confusion[cls].sum() + confusion[:, cls].sum() - tp
        ious.append(tp / union if union > 0 else 0.0)

    f1s = []
    for cls in range(NUM_CLASSES):
        tp = confusion[cls, cls]
        precision = tp / confusion[:, cls].sum() if confusion[:, cls].sum() > 0 else 0
        recall = tp / confusion[cls, :].sum() if confusion[cls, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(float(f1))

    total = confusion.sum()
    oa = float(confusion.diagonal().sum() / total) if total > 0 else 0.0

    ious_5 = ious[:CLUTTER_IDX] + ious[CLUTTER_IDX + 1:]
    f1s_5 = f1s[:CLUTTER_IDX] + f1s[CLUTTER_IDX + 1:]
    r = {
        "miou_5_class": float(np.mean(ious_5)),
        "miou_6_class": float(np.mean(ious)),
        "mf1_5_class": float(np.mean(f1s_5)),
        "mf1_6_class": float(np.mean(f1s)),
        "oa": float(oa),
        "per_class_iou": [float(x) for x in ious],
        "per_class_f1": [float(x) for x in f1s],
    }
    assert abs(np.mean(ious) - r["miou_6_class"]) < 1e-6
    assert abs(np.mean(ious_5) - r["miou_5_class"]) < 1e-6
    assert abs(np.mean(f1s) - r["mf1_6_class"]) < 1e-6
    assert abs(np.mean(f1s_5) - r["mf1_5_class"]) < 1e-6

    print(f"[metric_internal_check] 6-class mIoU verified: {r['miou_6_class']:.6f}")
    print(f"[metric_internal_check] 5-class mIoU verified: {r['miou_5_class']:.6f}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        cm_path = os.path.join(out_dir, "confusion_matrix.csv")
        with open(cm_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gt\\pred"] + CLASS_NAMES + ["row_total"])
            for row in range(NUM_CLASSES):
                w.writerow([CLASS_NAMES[row]] + [int(confusion[row, c]) for c in range(NUM_CLASSES)] + [int(confusion[row].sum())])
            w.writerow(["col_total"] + [int(confusion[:, c].sum()) for c in range(NUM_CLASSES)] + [int(total)])
        print(f"[confusion_matrix] saved to {cm_path}")

    return r, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=FINAL_EVAL_STRIDE)
    parser.add_argument("--tta", default="hflip+vflip", choices=["hflip", "vflip", "hflip+vflip", "none"])
    parser.add_argument("--scales", default=FINAL_EVAL_SCALES,
                        help="Multi-scale testing, e.g. 0.75,1.0,1.25; 'none' to disable")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_dir", default=None,
                        help="If set, confusion_matrix.csv + metrics.json saved here")
    parser.add_argument("--prefer_ema", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--fusion", default="logit", choices=["logit", "vote"],
                        help="Overlapping-tile aggregation: logit-avg (paper) or hard majority vote")
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.data_root is None:
        args.data_root = "./datasets/ISPRS_Vaihingen"
    if args.tta == "none":
        args.tta = ""
    scales = None
    if args.scales and args.scales.lower() != "none":
        scales = [float(s) for s in args.scales.split(",")]
        print(f"MST scales: {scales}")

    input_norm = cfg.get("input_norm", "imagenet")
    split_preset = cfg.get("split_preset", "mm_dino")
    print(f"Tile={args.tile_size}, Stride={args.stride}, TTA={args.tta or 'none'}, norm={input_norm}, split_preset={split_preset}")

    device = torch.device("cuda")
    model = load_model(args.checkpoint, args.config, device, prefer_ema=(args.prefer_ema == "true"))
    print("Model loaded")

    r, confusion = evaluate(
        model, args.data_root, args.tile_size, args.stride, device, args.tta, scales,
        input_norm=input_norm, split_preset=split_preset, out_dir=args.out_dir, fusion=args.fusion,
    )

    r["split_preset"] = split_preset
    r["checkpoint"] = os.path.basename(args.checkpoint)
    r["prefer_ema"] = (args.prefer_ema == "true")

    tags = ["Sliding Window"]
    if args.tta:
        tags.append(f"TTA({args.tta})")
    if scales:
        tags.append(f"MST({args.scales})")
    tag_str = " + ".join(tags)

    print(f"\n{tag_str}")
    print(f"  mIoU-5 (excl Clutter): {r['miou_5_class']:.2%}")
    print(f"  mIoU-6 (incl Clutter): {r['miou_6_class']:.2%}")
    print(f"  mF1-5  (excl Clutter): {r['mf1_5_class']:.2%}")
    print(f"  mF1-6  (incl Clutter): {r['mf1_6_class']:.2%}")
    print(f"  OA:                    {r['oa']:.2%}")
    print(f"  {'Class':<15} {'IoU':>8} {'F1':>8}")
    print(f"  {'-'*35}")
    for name, iou, f1 in zip(CLASS_NAMES, r["per_class_iou"], r["per_class_f1"]):
        print(f"  {name:<15} {iou:>8.2%} {f1:>8.2%}")

    print(f"\n[metrics_json] {json.dumps(r)}")

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        metrics_path = os.path.join(args.out_dir, "metrics.json")
        with open(metrics_path, "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"[metrics] saved to {metrics_path}")


if __name__ == "__main__":
    main()
