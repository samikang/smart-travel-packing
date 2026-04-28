# Cloth Tool

A computer vision pipeline for automatically evaluating clothing attributes from photos — including type, season, material, fold state, weight, and storage size.

---

## What it does

Given a photo of a garment, the model predicts **9 attributes**:

| Attribute | Type | Classes / Range |
|---|---|---|
| `cloth_type` | classification | t_shirt, shirt, sweater, jacket, coat, down_jacket, pants, skirt, dress, vest |
| `season_group` | classification | summer, spring_autumn, winter |
| `material_group` | classification | cotton_like, knit, denim, wool_like, padded_down_like, leather_like, synthetic_sportswear, mixed_unknown |
| `fold_state` | classification | unfolded, folded, partially_folded, compressed |
| `weight_class` | classification | light, medium, heavy |
| `folded_size_class` | classification | xs, s, m, l |
| `pressed_size_class` | classification | pocket_size, compact, standard, bulky |
| `approx_weight_g` | regression | estimated weight in grams |
| `approx_volume_L` | regression | estimated folded volume in litres |

---

## Project structure

```
cloth-tool/
├── dataset/                    # Images + labeled CSV
│   ├── 1.png, 2.png ...
│   └── dataset1_clean.csv
├── schema/                     # Labeling reference
│   ├── labeling_schema.json    # Full attribute/class definitions
│   ├── labeling_guide.md       # Human labeling instructions
│   ├── attribute_classes.yaml  # Class lists for training
│   └── classes.yaml            # YOLO detector config (1 class: garment)
├── src/cloth_tool/
│   ├── dataset.py              # Data loading, label encoding, augmentation
│   ├── model.py                # EfficientNet-B0 backbone + 9 output heads
│   ├── train.py                # Training script (2-phase)
│   ├── predict.py              # Inference on image or folder
│   └── fix_dataset.py          # CSV schema validation + auto-fix utility
└── runs/
    └── exp2/                   # Latest trained model
        ├── best.pt
        ├── final.pt
        └── history.json
```

---

## Requirements

- Python 3.10+
- PyTorch 2.x + torchvision
- pandas, scikit-learn, Pillow, tqdm

---

## Train

```bash
cd /Users/zhangs/Desktop/cloth-tool

PYTHONPATH=src /opt/anaconda3/bin/python3 -m cloth_tool.train \
  --csv     dataset/dataset1_clean.csv \
  --imgdir  dataset \
  --outdir  runs/exp2
```

Optional arguments:

| Flag | Default | Description |
|---|---|---|
| `--epochs1` | 15 | Phase 1 epochs (frozen backbone) |
| `--epochs2` | 35 | Phase 2 epochs (full fine-tune) |
| `--batch` | 16 | Batch size |
| `--lr1` | 1e-3 | Phase 1 learning rate |
| `--lr2` | 3e-4 | Phase 2 learning rate |

Training uses a two-phase strategy:
- **Phase 1** — backbone frozen, only the 9 heads train. Fast initial convergence.
- **Phase 2** — full fine-tune at lower LR with cosine schedule. Deep adaptation.

Saves `best.pt` (best val accuracy), `final.pt` (last epoch), and `history.json` to `--outdir`.

---

## Predict

**Single image:**
```bash
cd /Users/zhangs/Desktop/cloth-tool

PYTHONPATH=src /opt/anaconda3/bin/python3 -m cloth_tool.predict \
  --model runs/exp2/best.pt \
  --input /path/to/photo.jpg
```

**Whole folder:**
```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 -m cloth_tool.predict \
  --model runs/exp2/best.pt \
  --input dataset/
```

**Save results to CSV:**
```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 -m cloth_tool.predict \
  --model runs/exp2/best.pt \
  --input dataset/ \
  --output results.csv
```

Example output:
```
──────────────────────────────────────────────────
Image : 1.png
──────────────────────────────────────────────────
  cloth_type             sweater                99%  ███████████████████
  season_group           spring_autumn          98%  ███████████████████
  material_group         knit                   99%  ███████████████████
  fold_state             unfolded               100% ████████████████████
  weight_class           medium                 100% ████████████████████
  folded_size_class      l                      92%  ██████████████████
  pressed_size_class     compact                100% ████████████████████
  approx_weight_g        344.0                  g
  approx_volume_L        2.3                    L
```

---

## Model performance (exp2, 115 images, 80/20 split)

| Attribute | Val Accuracy |
|---|---|
| fold_state | 100% |
| pressed_size_class | 100% |
| folded_size_class | 91% |
| season_group | 78% |
| weight_class | 78% |
| cloth_type | 61% |
| material_group | 57% |
| **Mean accuracy** | **83%** |
| approx_weight_g MAE | ~190 g |
| approx_volume_L MAE | ~1.9 L |

`cloth_type` and `material_group` are the weakest attributes — both have the most classes and fewest samples per class. Adding more labeled images will directly improve these.

---

## Fix / validate a dataset CSV

```bash
cd /Users/zhangs/Desktop/cloth-tool

PYTHONPATH=src /opt/anaconda3/bin/python3 -m cloth_tool.fix_dataset
```

Reads `dataset1_fixed.csv`, auto-remaps known schema mismatches, and writes:
- `dataset1_clean.csv` — all rows with fixes applied
- `dataset1_review.csv` — rows needing manual confirmation

---

## Labeling a new batch

1. Place new images in `dataset/`
2. Label attributes following `schema/labeling_guide.md`
3. Add rows to `dataset/dataset1_clean.csv` using the column format:
   ```
   image_path, cloth_type, season_group, material_group, fold_state,
   weight_class, approx_weight_g, folded_size_class, pressed_size_class,
   approx_volume_L, quality_flag, notes
   ```
4. Run `fix_dataset.py` to validate
5. Re-train with `train.py`
