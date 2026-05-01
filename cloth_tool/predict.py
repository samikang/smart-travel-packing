"""
Run cloth attribute prediction on one image or a folder of images.

Usage:
  # single image
  python -m cloth_tool.predict --model runs/exp1/best.pt --input dataset/1.png

  # folder of images
  python -m cloth_tool.predict --model runs/exp1/best.pt --input my_photos/

  # save results to CSV
  python -m cloth_tool.predict --model runs/exp1/best.pt --input my_photos/ --output results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

from cloth_tool.dataset import ATTRIBUTES, REGRESSION_ATTRS, get_transforms
from cloth_tool.model import ClothClassifier

IDX2LABEL = {
    attr: {i: lbl for i, lbl in enumerate(classes)}
    for attr, classes in ATTRIBUTES.items()
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model(checkpoint_path: str, device: torch.device) -> ClothClassifier:
    model = ClothClassifier().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_image(model: ClothClassifier,
                  img_path: Path,
                  device: torch.device,
                  transform) -> dict:
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_out, reg_out = model(tensor)

    result = {"image": img_path.name}

    for attr, head_logits in cls_out.items():
        probs    = torch.softmax(head_logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        result[attr]           = IDX2LABEL[attr][pred_idx]
        result[f"{attr}_conf"] = round(probs[pred_idx].item(), 3)

    for attr, max_val in REGRESSION_ATTRS.items():
        result[attr] = round(reg_out[attr].item() * max_val, 1)

    return result


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS)


def print_result(result: dict):
    print(f"\n{'─'*50}")
    print(f"Image : {result['image']}")
    print(f"{'─'*50}")
    for attr in ATTRIBUTES:
        label = result[attr]
        conf  = result[f"{attr}_conf"]
        bar   = "█" * int(conf * 20)
        print(f"  {attr:<22} {label:<22} {conf:.0%}  {bar}")
    print(f"  {'approx_weight_g':<22} {result['approx_weight_g']:<22.1f} g")
    print(f"  {'approx_volume_L':<22} {result['approx_volume_L']:<22.1f} L")


def save_csv(results: list[dict], output_path: Path):
    if not results:
        return
    fields = (["image"]
              + [f for attr in ATTRIBUTES for f in (attr, f"{attr}_conf")]
              + list(REGRESSION_ATTRS.keys()))
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="Path to best.pt or final.pt")
    p.add_argument("--input",  required=True, help="Image file or folder")
    p.add_argument("--output", default="",    help="Optional CSV output path")
    args = p.parse_args()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Device : {device}")
    print(f"Model  : {model_path}")

    model     = load_model(str(model_path), device)
    transform = get_transforms(train=False)
    images    = collect_images(input_path)

    if not images:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running on {len(images)} image(s)...")
    results = []
    for img_path in images:
        try:
            result = predict_image(model, img_path, device, transform)
            print_result(result)
            results.append(result)
        except Exception as e:
            print(f"  [skip] {img_path.name}: {e}")

    if args.output:
        save_csv(results, Path(args.output))


if __name__ == "__main__":
    main()
