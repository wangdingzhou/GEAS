# GEAS

Parameter-efficient adaptation of a frozen DINOv3 (ViT-L/16) backbone for remote-sensing semantic segmentation, evaluated on LoveDA, ISPRS Potsdam and ISPRS Vaihingen.

## Repository layout

```
src/            core model, datasets, training and evaluation
cross_dataset/  multi-dataset training + sliding-window / TTA evaluation
dinov3/         DINOv3 code (facebookresearch/dinov3, Apache-2.0)
```

## Setup

```bash
pip install -r requirements.txt
```

Download the DINOv3 ViT-L/16 (SAT-493M) pretrained checkpoint and point `backbone.checkpoint` in `src/config.yaml` to it.

## Data

LoveDA (defaults in `src/config.yaml`):

```
LoveDA/
├── Train/{urban,rural}/{images_png,masks_png}
└── Val/{urban,rural}/{images_png,masks_png}
```

Potsdam / Vaihingen: set `data_root` and `dataset_type` in the `cross_dataset` training config.

## Training

```bash
# single-dataset (LoveDA)
python src/train.py --config src/config.yaml

# multi-dataset entry (LoveDA / Potsdam / Vaihingen via config)
python cross_dataset/train_cross.py --config CONFIG
```

## Evaluation

```bash
python src/eval.py --config src/config.yaml --checkpoint CKPT

# sliding-window evaluation (tile 512 / stride 256, flip TTA)
python cross_dataset/eval_loveda.py    --checkpoint CKPT --config CONFIG
python cross_dataset/eval_potsdam.py   --checkpoint CKPT --config CONFIG
python cross_dataset/eval_vaihingen.py --checkpoint CKPT --config CONFIG
```

Model checkpoints and datasets are not included in this repository.
