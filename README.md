# GEAS

GEAS (Geospatial Efficient Adaptation with Slim decoder): accurate and efficient adaptation of a frozen DINOv3 ViT-L/16 backbone for high-resolution remote sensing semantic segmentation, evaluated on ISPRS Potsdam, ISPRS Vaihingen, LoveDA, and iSAID.

Paper: **GEAS: Accurate and Efficient Adaptation of a Frozen Foundation Model for Remote Sensing Semantic Segmentation** (under review at IEEE TGRS).

## Model weights

| Checkpoint | Dataset | Metrics (paper, no TTA) | Size | Download |
|---|---|---|---|---|
| `GEAS_Potsdam.pth` | ISPRS Potsdam (5 cls) | mIoU-5 88.16 / mF1-5 93.56 | 2.63 GB | Baidu Netdisk (link pending) |
| `GEAS_Vaihingen.pth` | ISPRS Vaihingen (5 cls) | mIoU-5 85.63 / mF1-5 92.09 | 2.63 GB | Baidu Netdisk (link pending) |
| `GEAS_LoveDA.pth` | LoveDA (7 cls) | mIoU-7 57.07 / mF1-7 71.73 | 2.45 GB | Baidu Netdisk (link pending) |
| `GEAS_iSAID.pth` | iSAID (16 cls) | mIoU-16 69.75 / mF1-16 81.08 / mIoU-15 67.79 | 2.63 GB | Baidu Netdisk (link pending) |

Metrics use sliding-window inference (512x512 tiles, stride 256) with no test-time augmentation or post-processing. Checkpoints contain the full training state (model + optimizer); load the model/EMA weights for evaluation. The frozen DINOv3 backbone is not duplicated in the optimizer state.

## Repository layout

```
src/            core model, datasets, training and evaluation
cross_dataset/  training entry and sliding-window / TTA evaluation
dinov3/         DINOv3 code (facebookresearch/dinov3, Apache-2.0)
```

## Setup

```bash
pip install -r requirements.txt
```

Download the public DINOv3 ViT-L/16 pretrained checkpoint and set `backbone.checkpoint` in your training configuration.

## Training

```bash
python cross_dataset/train_cross.py --config YOUR_CONFIG
```

## Evaluation

```bash
python cross_dataset/eval_potsdam.py   --checkpoint CKPT --config YOUR_CONFIG
python cross_dataset/eval_vaihingen.py --checkpoint CKPT --config YOUR_CONFIG
python cross_dataset/eval_loveda.py    --checkpoint CKPT --config YOUR_CONFIG
```
