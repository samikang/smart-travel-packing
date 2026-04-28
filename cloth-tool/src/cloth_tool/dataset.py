from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ATTRIBUTES = {
    "cloth_type":        ["t_shirt", "shirt", "sweater", "jacket", "coat",
                          "down_jacket", "pants", "skirt", "dress", "vest"],
    "season_group":      ["summer", "spring_autumn", "winter"],
    "material_group":    ["cotton_like", "knit", "denim", "wool_like",
                          "padded_down_like", "leather_like",
                          "synthetic_sportswear", "mixed_unknown"],
    "fold_state":        ["unfolded", "folded", "partially_folded", "compressed"],
    "weight_class":      ["light", "medium", "heavy"],
    "folded_size_class": ["xs", "s", "m", "l"],
    "pressed_size_class":["pocket_size", "compact", "standard", "bulky"],
}

LABEL2IDX = {
    attr: {lbl: i for i, lbl in enumerate(classes)}
    for attr, classes in ATTRIBUTES.items()
}

# Regression targets: (column_name, max_value used for [0,1] normalisation)
REGRESSION_ATTRS = {
    "approx_weight_g": 2000.0,
    "approx_volume_L": 20.0,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _resolve_image_path(img_dir: Path, filename: str) -> Path:
    """Try original filename, then swap extension to .png."""
    p = img_dir / filename
    if p.exists():
        return p
    stem = Path(filename).stem
    png = img_dir / f"{stem}.png"
    if png.exists():
        return png
    raise FileNotFoundError(f"Image not found: {filename} (also tried {png})")


def load_dataframe(csv_path: str, img_dir: str = "") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    valid_rows = []
    skipped_label = 0
    skipped_image = 0
    img_root = Path(img_dir) if img_dir else None

    for _, row in df.iterrows():
        # validate labels
        ok = True
        for attr, mapping in LABEL2IDX.items():
            if str(row.get(attr, "")).strip() not in mapping:
                ok = False
                break
        if not ok:
            skipped_label += 1
            continue

        # validate image exists
        if img_root is not None:
            try:
                _resolve_image_path(img_root, str(row["image_path"]))
            except FileNotFoundError:
                skipped_image += 1
                continue

        valid_rows.append(row)

    if skipped_label:
        print(f"[dataset] skipped {skipped_label} rows with invalid/missing labels")
    if skipped_image:
        print(f"[dataset] skipped {skipped_image} rows with missing image files")
    return pd.DataFrame(valid_rows).reset_index(drop=True)


def compute_class_weights(df: pd.DataFrame, device: torch.device) -> dict:
    """Inverse-frequency weights per attribute for weighted CrossEntropyLoss."""
    weights = {}
    for attr, classes in ATTRIBUTES.items():
        counts = df[attr].value_counts()
        w = torch.zeros(len(classes), dtype=torch.float32)
        for i, cls in enumerate(classes):
            n = counts.get(cls, 0)
            w[i] = 1.0 / max(n, 1)
        w = w / w.sum() * len(classes)
        weights[attr] = w.to(device)
    return weights


class ClothDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = get_transforms(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = _resolve_image_path(self.img_dir, str(row["image_path"]))
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        cls_labels = {
            attr: torch.tensor(LABEL2IDX[attr][str(row[attr]).strip()],
                               dtype=torch.long)
            for attr in ATTRIBUTES
        }
        reg_labels = {
            attr: torch.tensor(float(row[attr]) / max_val, dtype=torch.float32)
            for attr, max_val in REGRESSION_ATTRS.items()
        }
        return img, cls_labels, reg_labels
