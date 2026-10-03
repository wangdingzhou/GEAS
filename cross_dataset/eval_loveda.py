#!/usr/bin/env python3
"""LoveDA: Sliding Window + TTA Evaluation.

Usage:
    python eval_loveda.py --checkpoint PATH --config PATH

Adds 3 features for producer parity with eval_potsdam.py / eval_vaihingen.py:
  (1) [metrics_json] stdout line (final line) — derived from same confusion
      matrix as the printed numbers, so byte-identical semantics.
  (2) confusion_matrix.csv output to --out_dir (if specified); header notes
      IGNORE_IDX=255 (same sentinel as Potsdam).
  (3) --prefer_ema flag (default true, override-only) + --out_dir flag.

"""

import sys, os, argparse, json, csv, torch, numpy as np, cv2
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import get_model
from dataset import LoveDADataset, decode_loveda_mask
from torchvision import transforms
from omegaconf import OmegaConf

CLASS_NAMES = LoveDADataset.CLASS_NAMES
NUM_CLASSES = LoveDADataset.NUM_CLASSES
# 2026-08-21 label-convention fix: raw masks decoded via decode_loveda_mask
# (0=void -> 255; raw 1-7=Background..Agriculture -> 0-6). mmseg/literature convention.
IGNORE_IDX = 255
FINAL_EVAL_STRIDE = 256
FINAL_EVAL_SCALES = "0.75,1.0,1.25,1.5"

_NORM_MAP = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "sat493m": ([0.430, 0.411, 0.296], [0.213, 0.156, 0.143]),
}


def _m2f_inference_tile(model, tile, device):
    B = tile.shape[0]
    features_raw = model.backbone.get_intermediate_layers(
        tile, n=model.layers_to_extract_0idx, reshape=False, norm=True
    )
    _, _, H, W = tile.shape
    h_p, w_p = H // model.patch_size, W // model.patch_size
    feature_maps = []
    for feat in features_raw:
        feat = feat.reshape(B, h_p, w_p, -1).permute(0, 3, 1, 2).contiguous()
        feature_maps.append(feat)
    return model.decoder.predict(feature_maps, img_size=(H, W))


def _inference_tile(model, tile, device):
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'predict'):
        return _m2f_inference_tile(model, tile, device)
    return model(tile)


def load_model(checkpoint_path, config_path, device, prefer_ema=True):
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    model = get_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded_from = "unknown"
    if isinstance(ckpt, dict):
        if prefer_ema and "ema" in ckpt:
            state = ckpt["ema"]
            loaded_from = "ema"
        elif "model" in ckpt:
            state = ckpt["model"]
            loaded_from = "model"
        else:
            state = ckpt
            loaded_from = "raw"
    else:
        state = ckpt
        loaded_from = "raw"
    print(f"Loaded weights from checkpoint[{loaded_from}] (prefer_ema={prefer_ema})")
    for k in list(state.keys()):
        if "pfeb" in k or "freq_plugin" in k or "sru" in k or "feat_plugin" in k:
            continue
        new_k = k
        if ".dw_conv." in k:
            new_k = k.replace(".dw_conv.", ".spatial_conv.0.")
        elif ".pw_conv." in k:
            new_k = k.replace(".pw_conv.", ".spatial_conv.1.")
        if new_k != k:
            state[new_k] = state.pop(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] strict=False missing={len(missing)} unexpected={len(unexpected)} "
              f"(alias _variant_out_conv keys are safe)")
    model = model.to(device)
    model.eval()
    return model


def sliding_window_inference(model, image_tensor, tile_size, stride, device, max_batch_size=16):
    C, H, W = image_tensor.shape
    if H <= tile_size and W <= tile_size:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            tile = image_tensor.unsqueeze(0).to(device)
            logits = _inference_tile(model, tile, device)
        return logits.squeeze(0).cpu().float()

    logits_sum = torch.zeros(NUM_CLASSES, H, W)
    count_map = torch.zeros(1, H, W)

    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        _, pH, pW = image_tensor.shape
    else:
        pH, pW = H, W

    ys = list(range(0, pH - tile_size + 1, stride))
    xs = list(range(0, pW - tile_size + 1, stride))
    if not ys or ys[-1] + tile_size < pH:
        ys.append(pH - tile_size)
    if not xs or xs[-1] + tile_size < pW:
        xs.append(pW - tile_size)

    # Collect all (y, x) coords and corresponding crops once, then forward in
    # batches of max_batch_size. Mathematically equivalent to the per-patch loop
    # under model.eval() + no_grad + autocast (LN/GN/Attention are per-sample;
    # eval-mode running stats don't depend on batch size). fp16 cuBLAS may pick
    # a different kernel for batch=1 vs batch=N, so bit-level differences vs the
    # legacy per-patch path can accumulate to ~1e-3 in logits — expect <0.05pt
    # mIoU drift. Validate against the baseline best (legacy 66.21%) before relying on it.
    coords = [(y, x) for y in ys for x in xs]
    crops = [image_tensor[:, y:y+tile_size, x:x+tile_size] for (y, x) in coords]

    for i in range(0, len(crops), max_batch_size):
        chunk_crops = crops[i:i+max_batch_size]
        chunk_coords = coords[i:i+max_batch_size]
        batch = torch.stack(chunk_crops).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            batch_logits = _inference_tile(model, batch, device)
        batch_logits = batch_logits.float().cpu()
        # Untile by coords — accumulation logic identical to legacy per-patch path.
        for j, (y, x) in enumerate(chunk_coords):
            crop_logits = batch_logits[j]
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            ch, cw = y_end - y, x_end - x
            # Crop (not interpolate) the valid region; reflect-padded rows/cols
            # at the bottom/right must be discarded, not squashed into the valid area.
            logits_sum[:, y:y_end, x:x_end] += crop_logits[:, :ch, :cw]
            count_map[:, y:y_end, x:x_end] += 1

    return logits_sum / count_map.clamp(min=1)


def inference_with_tta(model, image_tensor, tile_size, stride, device, tta_mode, max_batch_size=16):
    logits = sliding_window_inference(model, image_tensor, tile_size, stride, device, max_batch_size)
    count = 1

    if "hflip" in tta_mode:
        flipped = torch.flip(image_tensor, [-1])
        logits_flip = sliding_window_inference(model, flipped, tile_size, stride, device, max_batch_size)
        logits += torch.flip(logits_flip, [-1])
        count += 1

    if "vflip" in tta_mode:
        flipped = torch.flip(image_tensor, [-2])
        logits_flip = sliding_window_inference(model, flipped, tile_size, stride, device, max_batch_size)
        logits += torch.flip(logits_flip, [-2])
        count += 1

    return logits / count


def compute_confusion(pred, target, num_classes, ignore_idx=255):
    valid = (target >= 0) & (target < num_classes) & (target != ignore_idx)
    p = pred[valid].astype(np.int32)
    t = target[valid].astype(np.int32)
    idx = t * num_classes + p
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def inference_with_mst(model, image_tensor, tile_size, stride, device, scales, tta_mode, max_batch_size=16):
    C, H, W = image_tensor.shape
    logits_sum = None

    for si, scale in enumerate(scales):
        print(f"    [scale {si+1}/{len(scales)}] {scale:.2f}x")
        if scale != 1.0:
            new_H, new_W = int(H * scale), int(W * scale)
            scaled = F.interpolate(
                image_tensor.unsqueeze(0), size=(new_H, new_W),
                mode="bilinear", align_corners=False
            ).squeeze(0)
        else:
            scaled = image_tensor

        if tta_mode:
            logits = inference_with_tta(model, scaled, tile_size, stride, device, tta_mode, max_batch_size)
        else:
            logits = sliding_window_inference(model, scaled, tile_size, stride, device, max_batch_size)

        if logits.shape[1:] != (H, W):
            logits = F.interpolate(
                logits.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        if logits_sum is None:
            logits_sum = logits
        else:
            logits_sum += logits

    return logits_sum / len(scales)


def evaluate(model, data_root, tile_size, stride, device, tta_mode, scales=None, input_norm="imagenet", max_batch_size=16, metrics6_out=None):
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    # 2026-08-22: second accumulator for the 6-class no-Background convention
    # (DINO_Soars include_bg=False / ADVMSeg cluster). Faithful conversion: Bg
    # logit masked BEFORE argmax (runner-up re-decides), GT-Bg -> ignore. This is
    # variant (c) — NOT the row+col-drop re-score (b), which forgives pred-Bg FNs.
    confusion6 = np.zeros((6, 6), dtype=np.int64)

    val_ds = LoveDADataset(root=data_root, split="Val", transform=False)
    mean, std = _NORM_MAP.get(input_norm, _NORM_MAP["imagenet"])
    normalize = transforms.Normalize(mean=mean, std=std)
    scale_str = f", scales={[f'{s:.2f}' for s in scales]}" if scales else ""
    print(f"Eval: {len(val_ds)} tiles, tile={tile_size}, stride={stride}, tta={tta_mode}{scale_str}, norm={input_norm}, max_batch_size={max_batch_size}")

    for i in range(len(val_ds)):
        img_path = val_ds.images[i]
        mask_path = val_ds.masks[i]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 2026-08-21 fix: decode raw (0=void,1-7=Background..Agriculture) -> 0-6 + 255
        mask = decode_loveda_mask(mask).astype(np.int64)
        H, W = image.shape[:2]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = normalize(image_tensor)

        if scales and len(scales) > 1:
            logits = inference_with_mst(model, image_tensor, tile_size, stride, device, scales, tta_mode, max_batch_size)
        elif tta_mode:
            logits = inference_with_tta(model, image_tensor, tile_size, stride, device, tta_mode, max_batch_size)
        else:
            logits = sliding_window_inference(model, image_tensor, tile_size, stride, device, max_batch_size)

        if logits.shape[1:] != (H, W):
            logits = F.interpolate(logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

        pred = logits.argmax(dim=0).numpy()
        confusion += compute_confusion(pred, mask, NUM_CLASSES, IGNORE_IDX)

        pred6 = logits[1:].argmax(dim=0).numpy()
        target6 = np.where((mask == 0) | (mask == IGNORE_IDX), IGNORE_IDX, mask - 1)
        confusion6 += compute_confusion(pred6, target6, 6, IGNORE_IDX)

        print(f"  [{i+1}/{len(val_ds)}] {os.path.basename(img_path)}")

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

    if metrics6_out is not None:
        ious6 = []
        f1s6 = []
        for cls in range(6):
            tp = confusion6[cls, cls]
            union = confusion6[cls].sum() + confusion6[:, cls].sum() - tp
            ious6.append(tp / union if union > 0 else 0.0)
            precision = tp / confusion6[:, cls].sum() if confusion6[:, cls].sum() > 0 else 0
            recall = tp / confusion6[cls, :].sum() if confusion6[cls, :].sum() > 0 else 0
            f1s6.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
        total6 = confusion6.sum()
        metrics6_out.update({
            "miou": float(np.mean(ious6)),
            "per_class_iou": [float(x) for x in ious6],
            "per_class_f1": [float(x) for x in f1s6],
            "oa": float(confusion6.diagonal().sum() / total6) if total6 > 0 else 0.0,
            "confusion": confusion6.tolist(),
        })
        assert all(0.0 <= x <= 1.0 for x in ious6), f"6-class IoU out of range: {ious6}"

    # Patched 2026-08-07: return confusion matrix too (parity with eval_potsdam.py).
    # Existing tuple signature preserved — confusion appended as 6th element.
    return float(np.mean(ious)), ious, float(np.mean(f1s)), f1s, oa, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=FINAL_EVAL_STRIDE)
    parser.add_argument("--tta", default="hflip+vflip", choices=["hflip", "vflip", "hflip+vflip", "none"])
    parser.add_argument(
        "--scales",
        default=FINAL_EVAL_SCALES,
        help="Multi-scale testing, e.g. 0.75,1.0,1.25; use 'none' to disable",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--max_batch_size", type=int, default=16,
                        help="Sliding-window batch size. 1 = legacy per-patch, 16 = fast default.")
    # Patched 2026-08-07 (parity with eval_potsdam.py / eval_vaihingen.py):
    parser.add_argument("--prefer_ema", type=str, default="true", choices=["true", "false"],
                        help="If 'true' (default, matches pre-patch behavior), eval ckpt['ema']; "
                             "if 'false', eval ckpt['model'].")
    parser.add_argument("--out_dir", default=None,
                        help="If set, confusion_matrix.csv + metrics.json saved here.")
    args = parser.parse_args()

    cfg_for_path = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.data_root is None:
        args.data_root = cfg_for_path.get("data_root", "./datasets/LoveDA")

    if args.tta == "none":
        args.tta = ""

    scales = None
    if args.scales and args.scales.lower() != "none":
        scales = [float(s) for s in args.scales.split(",")]
        print(f"MST scales: {scales}")

    device = torch.device("cuda")
    print(f"Tile={args.tile_size}, Stride={args.stride}, TTA={args.tta or 'none'}")

    model = load_model(args.checkpoint, args.config, device, prefer_ema=(args.prefer_ema == "true"))
    print("Model loaded")

    cfg_dict = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    input_norm = cfg_dict.get("input_norm", "imagenet")

    m6 = {}
    miou, ious, mf1, f1s, oa, confusion = evaluate(
        model, args.data_root, args.tile_size, args.stride, device, args.tta, scales, input_norm, args.max_batch_size,
        metrics6_out=m6
    )

    tags = ["Sliding Window"]
    if args.tta:
        tags.append(f"TTA({args.tta})")
    if scales:
        tags.append(f"MST({args.scales})")
    tag_str = " + ".join(tags)

    # Printed block.
    print(f"\n{tag_str}")
    print(f"  mIoU: {miou:.2%}")
    print(f"  mF1:  {mf1:.2%}")
    print(f"  OA:   {oa:.2%}")
    print(f"  {'Class':<15} {'IoU':>8} {'F1':>8}")
    print(f"  {'-'*35}")
    for name, iou, f1 in zip(CLASS_NAMES, ious, f1s):
        print(f"  {name:<15} {iou:>8.2%} {f1:>8.2%}")

    # Build metrics dict from same confusion matrix + emit [metrics_json] stdout
    # line. Internal consistency assert guards against JSON vs print desync.
    r = {
        "miou_7_class": float(miou),
        "mf1_7_class": float(mf1),
        "oa": float(oa),
        "per_class_iou": [float(x) for x in ious],
        "per_class_f1": [float(x) for x in f1s],
        "ignore_idx": IGNORE_IDX,  # 255
        "num_classes": NUM_CLASSES,
    }
    assert abs(float(np.mean(ious)) - r["miou_7_class"]) < 1e-6
    assert abs(float(np.mean(f1s)) - r["mf1_7_class"]) < 1e-6
    # Sanity alarm: a 31%-share real class (Background,
    # class 0 post-fix) cannot score ~99. If it does, the old 7-ignore convention
    # resurrected somewhere (class 0 = void freebie).
    if ious[0] > 0.95:
        print(f"[ALERT] class-0 (Background) IoU {ious[0]*100:.2f} > 95 — VOID SIGNATURE. "
              f"Label decode is wrong somewhere upstream; DO NOT trust this eval.")
    print(f"[metric_internal_check] 7-class mIoU verified: {r['miou_7_class']:.6f}")

    r["checkpoint"] = os.path.basename(args.checkpoint)
    r["prefer_ema"] = (args.prefer_ema == "true")
    r["split"] = "Val"

    # 2026-08-22: 6-class no-Background convention (variant (c): Bg logit masked
    # pre-argmax, GT-Bg ignored). Comparable to native-6 rows (e.g. CAFe-DINO);
    # NOT the same protocol as row+col-drop re-scores of 7-class models.
    if m6:
        r["miou_6_class_nobg"] = m6["miou"]
        r["per_class_iou_6_class_nobg"] = m6["per_class_iou"]
        r["per_class_f1_6_class_nobg"] = m6["per_class_f1"]
        r["oa_6_class_nobg"] = m6["oa"]
        r["class_names_6"] = CLASS_NAMES[1:]
        r["confusion_6_class_nobg"] = m6["confusion"]
        print(f"\n  6-class (no-Bg, logit-masked) mIoU: {m6['miou']:.2%}")
        for name, iou, f1 in zip(CLASS_NAMES[1:], m6["per_class_iou"], m6["per_class_f1"]):
            print(f"  {name:<15} {iou:>8.2%} {f1:>8.2%}")

    print(f"\n[metrics_json] {json.dumps(r)}")

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        cm_path = os.path.join(args.out_dir, "confusion_matrix.csv")
        with open(cm_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gt\\pred"] + CLASS_NAMES + ["row_total"])
            for row in range(NUM_CLASSES):
                w.writerow([CLASS_NAMES[row]] + [int(confusion[row, c]) for c in range(NUM_CLASSES)] + [int(confusion[row].sum())])
            total = int(confusion.sum())
            w.writerow(["col_total"] + [int(confusion[:, c].sum()) for c in range(NUM_CLASSES)] + [total])
        if m6:
            cm6_path = os.path.join(args.out_dir, "confusion_matrix_6class.csv")
            with open(cm6_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["gt\\pred"] + CLASS_NAMES[1:] + ["row_total"])
                c6 = m6["confusion"]
                for row in range(6):
                    w.writerow([CLASS_NAMES[1 + row]] + [int(c6[row][c]) for c in range(6)] + [int(sum(c6[row]))])
        print(f"[confusion_matrix] saved to {cm_path}")
        metrics_path = os.path.join(args.out_dir, "metrics.json")
        with open(metrics_path, "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"[metrics] saved to {metrics_path}")


if __name__ == "__main__":
    main()
