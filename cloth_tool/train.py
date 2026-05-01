"""
Train cloth attribute classifier.

Usage:
  python -m cloth_tool.train \
    --csv    dataset/dataset1_clean.csv \
    --imgdir dataset \
    --outdir runs/exp1

Two-phase training:
  Phase 1  — backbone frozen,  heads only  (fast convergence)
  Phase 2  — full fine-tune,   lower LR    (adaptation)
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from cloth_tool.dataset import (ATTRIBUTES, REGRESSION_ATTRS, ClothDataset,
                                compute_class_weights, load_dataframe)
from cloth_tool.model import ClothClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default="dataset/dataset1_clean.csv")
    p.add_argument("--imgdir",  default="dataset")
    p.add_argument("--outdir",  default="runs/exp1")
    p.add_argument("--epochs1", type=int, default=15,
                   help="Phase 1 epochs (frozen backbone)")
    p.add_argument("--epochs2", type=int, default=35,
                   help="Phase 2 epochs (full fine-tune)")
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--lr1",     type=float, default=1e-3,
                   help="Phase 1 learning rate")
    p.add_argument("--lr2",     type=float, default=3e-4,
                   help="Phase 2 learning rate")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


REG_LOSS_WEIGHT = 2.0  # scale regression loss relative to classification


def run_epoch(model, loader, criterions, reg_criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss  = 0.0
    correct     = {attr: 0 for attr in ATTRIBUTES}
    reg_abs_err = {attr: 0.0 for attr in REGRESSION_ATTRS}
    total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, cls_labels, reg_labels in loader:
            imgs       = imgs.to(device)
            cls_labels = {k: v.to(device) for k, v in cls_labels.items()}
            reg_labels = {k: v.to(device) for k, v in reg_labels.items()}

            cls_out, reg_out = model(imgs)

            cls_loss = sum(criterions[attr](cls_out[attr], cls_labels[attr])
                           for attr in ATTRIBUTES)
            reg_loss = sum(reg_criterion(reg_out[attr], reg_labels[attr])
                           for attr in REGRESSION_ATTRS)
            loss = cls_loss + REG_LOSS_WEIGHT * reg_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = imgs.size(0)
            total_loss += loss.item() * n
            total      += n

            for attr in ATTRIBUTES:
                preds = cls_out[attr].argmax(dim=1)
                correct[attr] += (preds == cls_labels[attr]).sum().item()

            for attr, max_val in REGRESSION_ATTRS.items():
                mae = (reg_out[attr] - reg_labels[attr]).abs().sum().item() * max_val
                reg_abs_err[attr] += mae

    avg_loss = total_loss / total
    accs     = {attr: correct[attr] / total for attr in ATTRIBUTES}
    mean_acc = sum(accs.values()) / len(accs)
    mae      = {attr: reg_abs_err[attr] / total for attr in REGRESSION_ATTRS}
    return avg_loss, accs, mean_acc, mae


def log_epoch(phase: str, epoch: int, loss: float,
              accs: dict, mean_acc: float, mae: dict):
    acc_str = "  ".join(f"{a[:8]}: {v:.2%}" for a, v in accs.items())
    mae_str = "  ".join(f"{a}: {v:.1f}" for a, v in mae.items())
    print(f"[{phase}] ep{epoch:03d}  loss={loss:.4f}  mean={mean_acc:.2%}  |  {acc_str}  |  MAE: {mae_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    df = load_dataframe(args.csv, img_dir=args.imgdir)
    print(f"Loaded {len(df)} valid rows")

    # stratify on cloth_type for balanced split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=args.seed,
        stratify=df["cloth_type"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    train_ds = ClothDataset(train_df, args.imgdir, train=True)
    val_ds   = ClothDataset(val_df,   args.imgdir, train=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=pin)

    class_weights = compute_class_weights(train_df, device)
    criterions = {
        attr: nn.CrossEntropyLoss(weight=class_weights[attr])
        for attr in ATTRIBUTES
    }
    reg_criterion = nn.HuberLoss()

    model = ClothClassifier(dropout=0.4).to(device)

    best_acc  = 0.0
    best_path = out_dir / "best.pt"
    history   = []

    # -----------------------------------------------------------------------
    # Phase 1 — frozen backbone
    # -----------------------------------------------------------------------
    print(f"\n=== Phase 1: frozen backbone ({args.epochs1} epochs) ===")
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr1, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs1, eta_min=args.lr1 / 10
    )

    for ep in range(1, args.epochs1 + 1):
        tr_loss, tr_accs, tr_mean, tr_mae = run_epoch(
            model, train_loader, criterions, reg_criterion, optimizer, device, train=True)
        vl_loss, vl_accs, vl_mean, vl_mae = run_epoch(
            model, val_loader,   criterions, reg_criterion, optimizer, device, train=False)
        scheduler.step()

        log_epoch("train", ep, tr_loss, tr_accs, tr_mean, tr_mae)
        log_epoch("val  ", ep, vl_loss, vl_accs, vl_mean, vl_mae)

        history.append({"phase": 1, "epoch": ep,
                        "train_loss": tr_loss, "val_loss": vl_loss,
                        "train_acc": tr_mean, "val_acc": vl_mean,
                        "val_mae": vl_mae})

        if vl_mean > best_acc:
            best_acc = vl_mean
            torch.save({"epoch": ep, "phase": 1,
                        "model_state": model.state_dict(),
                        "val_acc": vl_mean, "val_accs": vl_accs},
                       best_path)
            print(f"  ↑ best saved  val_mean={best_acc:.2%}")

    # -----------------------------------------------------------------------
    # Phase 2 — full fine-tune
    # -----------------------------------------------------------------------
    print(f"\n=== Phase 2: full fine-tune ({args.epochs2} epochs) ===")
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs2, eta_min=args.lr2 / 20
    )

    for ep in range(1, args.epochs2 + 1):
        tr_loss, tr_accs, tr_mean, tr_mae = run_epoch(
            model, train_loader, criterions, reg_criterion, optimizer, device, train=True)
        vl_loss, vl_accs, vl_mean, vl_mae = run_epoch(
            model, val_loader,   criterions, reg_criterion, optimizer, device, train=False)
        scheduler.step()

        log_epoch("train", ep, tr_loss, tr_accs, tr_mean, tr_mae)
        log_epoch("val  ", ep, vl_loss, vl_accs, vl_mean, vl_mae)

        history.append({"phase": 2, "epoch": ep,
                        "train_loss": tr_loss, "val_loss": vl_loss,
                        "train_acc": tr_mean, "val_acc": vl_mean,
                        "val_mae": vl_mae})

        if vl_mean > best_acc:
            best_acc = vl_mean
            torch.save({"epoch": ep, "phase": 2,
                        "model_state": model.state_dict(),
                        "val_acc": vl_mean, "val_accs": vl_accs},
                       best_path)
            print(f"  ↑ best saved  val_mean={best_acc:.2%}")

    # -----------------------------------------------------------------------
    # Save final model + history
    # -----------------------------------------------------------------------
    final_path = out_dir / "final.pt"
    torch.save({"model_state": model.state_dict(),
                "val_acc": vl_mean, "val_accs": vl_accs}, final_path)

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val mean accuracy: {best_acc:.2%}")
    print(f"Best model : {best_path}")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
