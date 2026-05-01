"""
Image Recognition — Wardrobe Outfit Analyser
=============================================
Analyses wardrobe photos against the trip weather forecast and produces
packing advice. Supports three detection backends selectable via --vision:

  yolo    (default) YOLO11 local detection — offline, no API key required.
                    Tries yolo11n-cls.pt (classification) first, then
                    yolo11n.pt (object detection) as fallback.

  google            Google Cloud Vision API — online, needs GOOGLE_VISION_API_KEY
                    env var. Returns label annotations ranked by confidence.

  clip              Local CLIP (openai/clip-vit-base-patch32) — offline, no API
                    key. Compares image embedding against the 37 recommender item
                    names first; if best score < threshold, falls back to a
                    broader generic clothing vocabulary.

  both              Runs all three backends on every image and displays each
                    result in a side-by-side comparison table.

  cloth-tool        Trained EfficientNet-B0 multi-task classifier (cloth-tool
                    package). Predicts 7 attributes (cloth_type, season_group,
                    material_group, fold_state, weight_class, folded_size_class,
                    pressed_size_class) plus approx_weight_g and approx_volume_L
                    regression outputs — skips MiDaS/SAM entirely.
                    Model: cloth_tool/runs/exp2/best.pt

After detection, the garment label is passed to Gemini (optional, needs
GEMINI_API_KEY) for narrative packing advice. Falls back to rule-based
advice if Gemini is unavailable.

Environment variables
---------------------
    GOOGLE_VISION_API_KEY   — Google Cloud Vision API key  (vision=google/both)
    GEMINI_API_KEY          — Google Gemini API key        (narrative advice)

Model files (place in project root)
-------------------------------------
    yolo11n-cls.pt          — YOLO11 classification weights
    yolo11n.pt              — YOLO11 detection weights
    (CLIP weights auto-downloaded by HuggingFace on first run, ~350MB)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from models import DayRecommendation, TripContext
import torch
from typing import List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

_HERE         = Path(__file__).parent
YOLO_CLS_PATH = _HERE / "yolo11n-cls.pt"
# Prefer custom local YOLO weights if present (requested: yolo26x.pt).
# Falls back to the previous default if the file is missing.
#YOLO_DET_PATH = _HERE / "yolo26n.pt" if (_HERE / "yolo26x.pt").exists() else (_HERE / "yolo11n.pt")
# Use YOLO26n — lightweight (~9MB), fast on CPU, ideal for HF Spaces demo.
YOLO_DET_PATH = _HERE / "yolo26n.pt"

VALID_VISION_MODES = ("yolo", "google", "clip", "both", "cloth-tool")

# Raster files accepted when scanning a wardrobe folder (--images)
IMAGE_FILE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}

# CLIP: if best recommender-item similarity is below this, use generic vocab
CLIP_RECOMMENDER_THRESHOLD = 0.20

# ── cloth-tool integration ────────────────────────────────────────────────────

# CLOTH_TOOL_SRC   = _HERE / "cloth_tool" / "src"
CLOTH_TOOL_MODEL = _HERE / "cloth_tool" / "runs" / "exp2" / "best.pt"

_CLOTH_TOOL_MODEL_SINGLETON   = None
_CLOTH_TOOL_TRANSFORM_SINGLETON = None
_CLOTH_TOOL_DEVICE_SINGLETON    = None

# cloth_type labels (model uses underscores) → readable label
_CLOTH_TYPE_LABEL: Dict[str, str] = {
    "t_shirt":    "t-shirt",
    "shirt":      "shirt",
    "sweater":    "sweater",
    "jacket":     "jacket",
    "coat":       "coat",
    "down_jacket":"down jacket",
    "pants":      "trousers",
    "skirt":      "skirt",
    "dress":      "dress",
    "vest":       "vest",
}

# material_group → key in _FABRIC_AREAL_DENSITY
_CLOTH_MATERIAL_MAP: Dict[str, str] = {
    "cotton_like":          "cotton",
    "knit":                 "cotton/polyester",
    "denim":                "denim",
    "wool_like":            "fleece/wool",
    "padded_down_like":     "synthetic/down",
    "leather_like":         "leather/rubber",
    "synthetic_sportswear": "nylon/polyester",
    "mixed_unknown":        "unknown",
}

# weight_class → thickness string used in the rest of the system
_CLOTH_THICKNESS_MAP: Dict[str, str] = {
    "light":  "thin",
    "medium": "medium",
    "heavy":  "thick",
}

# ── 3D volume / weight estimation ─────────────────────────────────────────────

SAM_CHECKPOINT = _HERE / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# Fabric areal density (g per cm² of fabric surface)
# Used for weight = total_fabric_area × areal_density
_FABRIC_AREAL_DENSITY: Dict[str, float] = {
    "cotton":           0.025,   # jersey knit ~250 g/m²
    "cotton/polyester": 0.020,
    "polyester":        0.015,
    "denim":            0.075,   # heavyweight denim ~750 g/m²
    "cotton/denim":     0.065,
    "fleece/wool":      0.040,
    "wool blend":       0.045,
    "wool/acrylic":     0.035,
    "synthetic/down":   0.035,   # shell + down fill contribution
    "leather/rubber":   0.150,
    "leather/mesh":     0.080,
    "nylon":            0.012,
    "nylon/polyester":  0.018,
    "cotton/wool":      0.035,
    "wool/fleece":      0.038,
    "unknown":          0.025,
}

# Per-garment packing parameters — (fold_factor, pack_thickness_cm, fabric_area_factor)
#   fold_factor        : total_fabric_area / folded_footprint_area
#                        (accounts for how many times each type is typically folded)
#   pack_thickness_cm  : height of the folded garment stack when packed
#   fabric_area_factor : total_fabric_area / front_projected_area
#                        (front=1.0, front+back≈2.2, front+back+sleeves≈2.5-2.8)
_GARMENT_PACK: List[tuple] = [
    # keywords                                           fold  thick  area_f
    (("heavy winter coat", "overcoat", "trench coat"),   14,   7.0,   2.8),
    (("down jacket", "parka", "anorak"),                 20,   3.0,   2.5),
    (("jacket", "blazer", "windbreaker",
      "raincoat", "waterproof"),                         12,   4.0,   2.5),
    (("sweater", "sweatshirt", "hoodie",
      "cardigan", "pullover", "fleece"),                 12,   3.5,   2.5),
    (("suit",),                                          10,   5.0,   3.0),
    (("jeans", "denim"),                                 10,   4.5,   2.2),
    (("trousers", "pants", "chinos"),                    12,   3.0,   2.2),
    (("shorts", "miniskirt"),                            12,   1.2,   2.0),
    (("skirt",),                                         10,   1.5,   2.0),
    (("t-shirt", "tee", "polo", "jersey",
      "tank", "blouse"),                                 14,   1.5,   2.2),
    (("shirt",),                                         14,   1.8,   2.2),
    (("dress",),                                         12,   2.0,   2.5),
    (("boot",),                                           1,  14.0,   1.0),  # rigid
    (("shoe", "sneaker", "loafer", "runner"),              1,  11.0,   1.0),
    (("sock",),                                           3,   1.0,   2.0),
    (("scarf",),                                          5,   0.8,   2.0),
    (("glove",),                                          3,   1.5,   2.0),
    (("hat", "cap"),                                      2,   8.0,   1.5),
    (("umbrella",),                                       1,   5.0,   1.0),
    (("backpack",),                                       2,  20.0,   1.5),
]
_DEFAULT_PACK = (10, 3.0, 2.2)   # fold_factor, pack_thickness_cm, fabric_area_factor

# Lazy singletons for 3D models
_MIDAS_MODEL     = None
_MIDAS_PROCESSOR = None
_SAM_PREDICTOR   = None

# YOLO: clothing-related COCO / ImageNet class names
_YOLO_CLOTHING_CLASSES = {
    "tie", "handbag", "backpack", "umbrella", "suitcase",
    "jersey", "sweatshirt", "cardigan", "trench coat", "overcoat",
    "suit", "lab coat", "jean", "shorts", "miniskirt",
    "raincoat", "poncho", "hooded jacket", "anorak", "parka",
    "t-shirt", "turtleneck", "pullover", "polo shirt",
    "running shoe", "loafer", "boot",
}

# CLIP: broad clothing vocabulary used when recommender match is weak
_CLIP_GENERIC_VOCAB = [
    "t-shirt", "shirt", "blouse", "polo shirt", "tank top",
    "sweater", "hoodie", "sweatshirt", "fleece jacket", "cardigan",
    "jacket", "blazer", "suit jacket", "raincoat", "windbreaker",
    "winter coat", "down jacket", "parka", "trench coat",
    "trousers", "jeans", "shorts", "skirt", "dress",
    "formal wear", "business attire",
    "sneakers", "running shoes", "boots", "formal shoes", "sandals",
    "hat", "scarf", "gloves", "umbrella", "backpack",
]


# ── Backend 1: YOLOv8 ─────────────────────────────────────────────────────────

def _detect_yolo(image_path: Path) -> str:
    """
    Run YOLOv8 on a single image.
    Tries classification model first (clothing-aware ImageNet labels),
    then detection model (COCO labels), then filename fallback.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return _filename_fallback(image_path, note="ultralytics not installed")

    # ── Classification model ──────────────────────────────────────────────────
    if YOLO_CLS_PATH.exists():
        try:
            model  = YOLO(str(YOLO_CLS_PATH))
            # 1. Load the model
            #model = YOLO("openai/clip-vit-base-patch32") 
    
            # 2. FORCE IT TO CPU (Add this line right after loading)
            model.to('cpu') 
    
            result = model(str(image_path), verbose=False)[0]
            print("Enter -3")
            top1   = result.names[result.probs.top1]
            conf   = float(result.probs.top1conf)
            return f"{top1} (YOLOv8-cls, confidence {conf:.0%})"
        except Exception:
            pass

    # ── Detection model ───────────────────────────────────────────────────────
    print( str(YOLO_DET_PATH.exists()))
    if YOLO_DET_PATH.exists():
        try:
            model  = YOLO(str(YOLO_DET_PATH))
            result = model(str(image_path), verbose=False)[0]
            boxes  = result.boxes
            if boxes is not None and len(boxes):
                labels   = [result.names[int(c)] for c in boxes.cls]
                clothing = [l for l in labels if l.lower() in _YOLO_CLOTHING_CLASSES]
                chosen   = clothing[0] if clothing else labels[0]
                return f"{chosen} (yolo26x)"
        except Exception:
            pass

    return _filename_fallback(image_path, note="no YOLO model found")


# ── Backend 2: Google Cloud Vision API ───────────────────────────────────────

def _detect_google_vision(image_path: Path) -> str:
    """
    Call Google Cloud Vision API label + object detection.
    Filters results for clothing-related labels.
    Needs GOOGLE_VISION_API_KEY environment variable.
    """
    api_key = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()
    if not api_key:
        return "(Google Vision skipped — GOOGLE_VISION_API_KEY not set)"

    try:
        import base64
        import json
        import urllib.request
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = json.dumps({
            "requests": [{
                "image": {"content": image_b64},
                "features": [
                    {"type": "LABEL_DETECTION",     "maxResults": 10},
                    {"type": "OBJECT_LOCALIZATION",  "maxResults": 5},
                ],
            }]
        }).encode("utf-8")

        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        response = data.get("responses", [{}])[0]
        labels   = response.get("labelAnnotations", [])

        # Prefer clothing-related labels; fall back to top generic labels
        clothing_keywords = (
            "clothing", "shirt", "jacket", "coat", "dress", "trousers",
            "shorts", "shoe", "boot", "hat", "scarf", "glove", "bag",
            "wear", "garment", "fashion", "outfit", "sleeve", "collar",
            "sweater", "hoodie", "uniform", "suit", "jeans", "sock",
        )
        clothing_labels = [
            l for l in labels
            if any(k in l["description"].lower() for k in clothing_keywords)
        ]
        top_labels = clothing_labels[:3] if clothing_labels else labels[:3]

        if not top_labels:
            return "(Google Vision: no labels returned)"

        label_str = ", ".join(
            f"{l['description']} ({l['score']:.0%})" for l in top_labels
        )
        return f"{label_str} (Google Vision)"

    except urllib.error.HTTPError as e:
        # This will tell you EXACTLY why Google rejected it (e.g., "API Key not found")
        error_details = e.read().decode("utf-8")
        return f"(Google Vision error: {e.code} - {error_details})"
    except Exception as e:
        return f"(Google Vision error: {e})"

# ── Backend 3: CLIP ───────────────────────────────────────────────────────────
# Set this to prevent CLIP from hanging on certain operating systems
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Lazy-loaded CLIP Singletons ──────────────────────────────────────────────
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _get_clip_resources() -> Tuple[object, object, str]:
    """Ensures CLIP is loaded exactly once and returns (model, processor, device)."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor
        print(f"  [CLIP] Loading weights to {_CLIP_DEVICE} (first time only)...")
        
        model_name = "openai/clip-vit-base-patch32"
        # We set local_files_only=False to ensure it downloads if missing
        _CLIP_MODEL = CLIPModel.from_pretrained(model_name)
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        
        _CLIP_MODEL.to(_CLIP_DEVICE).eval()
        print("  [CLIP] Model ready.")
        
    return _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_DEVICE

# ── Backend 3: CLIP ───────────────────────────────────────────────────────────

def _detect_clip(image_path: Path,
                 recommender_items: Optional[List[str]] = None) -> str:
    """
    Use CLIP to identify garments using a two-pass semantic match.
    Pass 1: Check against the system's packing item vocabulary.
    Pass 2: Fallback to generic clothing categories.
    """
    try:
        from PIL import Image
        import torch
    except ImportError:
        return "(CLIP skipped — missing PIL or torch. Run: pip install torch Pillow)"

    try:
        # 1. Get the singleton model/processor
        model, processor, device = _get_clip_resources()

        # 2. Prepare Image
        image = Image.open(str(image_path)).convert("RGB")

        # ── Pass 1: Recommender Vocabulary Match ─────────────────────────────
        if recommender_items:
            prompts = [f"a photo of {item.lower()}" for item in recommender_items]
            
            inputs = processor(
                text=prompts, 
                images=image,
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # Get probabilities
                probs = outputs.logits_per_image.softmax(dim=1)[0]

            best_idx = int(probs.argmax())
            best_score = float(probs[best_idx])

            # If it's a strong match for an item we actually recommend, return it
            if best_score >= CLIP_RECOMMENDER_THRESHOLD:
                return (f"{recommender_items[best_idx]} "
                        f"(CLIP match, score {best_score:.2f})")

        # ── Pass 2: Generic Clothing Fallback ────────────────────────────────
        # If Pass 1 wasn't confident, check against the broader vocab
        generic_prompts = [f"a photo of {item}" for item in _CLIP_GENERIC_VOCAB]
        
        inputs = processor(
            text=generic_prompts, 
            images=image,
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        # Return the top 2 likely generic categories
        top_k = probs.topk(2)
        indices = top_k.indices.tolist()
        scores = top_k.values.tolist()
        
        label_str = ", ".join(
            f"{_CLIP_GENERIC_VOCAB[idx]} ({scores[i]:.2f})" 
            for i, idx in enumerate(indices)
        )
        return f"{label_str} (CLIP generic)"

    except Exception as e:
        return f"(CLIP error: {e})"


# ── Backend 4: cloth-tool (trained EfficientNet-B0 multi-task classifier) ─────

def _get_cloth_tool_model():
    """Lazy-load the cloth-tool model; adds cloth-tool/src to sys.path once."""
    global _CLOTH_TOOL_MODEL_SINGLETON, _CLOTH_TOOL_TRANSFORM_SINGLETON, _CLOTH_TOOL_DEVICE_SINGLETON
    if _CLOTH_TOOL_MODEL_SINGLETON is not None:
        return (_CLOTH_TOOL_MODEL_SINGLETON,
                _CLOTH_TOOL_TRANSFORM_SINGLETON,
                _CLOTH_TOOL_DEVICE_SINGLETON)

    # import sys
    # src = str(CLOTH_TOOL_SRC)
    # if src not in sys.path:
    #    sys.path.insert(0, src)

    from cloth_tool.model import ClothClassifier
    from cloth_tool.dataset import get_transforms

    _CLOTH_TOOL_DEVICE_SINGLETON = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"  [cloth-tool] Loading model on {_CLOTH_TOOL_DEVICE_SINGLETON} ...")
    _CLOTH_TOOL_MODEL_SINGLETON = ClothClassifier().to(_CLOTH_TOOL_DEVICE_SINGLETON)
    ckpt = torch.load(str(CLOTH_TOOL_MODEL), map_location=_CLOTH_TOOL_DEVICE_SINGLETON,
                      weights_only=True)
    _CLOTH_TOOL_MODEL_SINGLETON.load_state_dict(ckpt["model_state"])
    _CLOTH_TOOL_MODEL_SINGLETON.eval()
    _CLOTH_TOOL_TRANSFORM_SINGLETON = get_transforms(train=False)
    print("  [cloth-tool] Model ready.")
    return (_CLOTH_TOOL_MODEL_SINGLETON,
            _CLOTH_TOOL_TRANSFORM_SINGLETON,
            _CLOTH_TOOL_DEVICE_SINGLETON)


def _detect_cloth_tool(image_path: Path) -> dict:
    """
    Run the trained cloth-tool classifier on a single image.
    Returns a dict with cloth_type, season_group, material_group, fold_state,
    weight_class, folded_size_class, pressed_size_class (each with *_conf),
    plus approx_weight_g and approx_volume_L regression outputs.
    Returns an empty dict on error.
    """
    if not CLOTH_TOOL_MODEL.exists():
        print(f"  [cloth-tool] Model not found: {CLOTH_TOOL_MODEL}")
        return {}
    try:
        model, transform, device = _get_cloth_tool_model()   # adds cloth_tool/src to sys.path
        from cloth_tool.predict import predict_image
        return predict_image(model, image_path, device, transform)
    except Exception as e:
        print(f"  [cloth-tool] error: {e}")
        return {}


def _cloth_tool_readable_label(result: dict) -> str:
    """Convert cloth_type prediction to a human-readable label string."""
    raw_type = result.get("cloth_type", "")
    label    = _CLOTH_TYPE_LABEL.get(raw_type, raw_type.replace("_", " "))
    conf     = result.get("cloth_type_conf", 0.0)
    season   = result.get("season_group", "").replace("_", " ")
    return f"{label} ({season}, cloth-tool {conf:.0%})"


def _print_cloth_tool_result(path: Path, result: dict, label: str, advice: str) -> None:
    """Print the multi-attribute cloth-tool prediction in a rich table."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box as rbox
        console = Console()
        console.print(f"\n  [cyan]File      :[/] {path.name}")
        console.print(f"  [cyan]Backend   :[/] [bold]CLOTH-TOOL[/]")
        console.print(f"  [cyan]Detected  :[/] {label}")
        tbl = Table(box=rbox.SIMPLE, show_header=True, header_style="bold yellow",
                    title="  Attribute Predictions", title_style="bold cyan")
        tbl.add_column("Attribute",  style="cyan",  no_wrap=True, width=22)
        tbl.add_column("Value",      style="white", width=22)
        tbl.add_column("Conf",       style="green", justify="right", width=6)
        from cloth_tool.dataset import ATTRIBUTES
        for attr in ATTRIBUTES:
            val  = result.get(attr, "—")
            conf = result.get(f"{attr}_conf", 0.0)
            bar  = "█" * int(conf * 12)
            tbl.add_row(attr, val, f"{conf:.0%} {bar}")
        tbl.add_row("approx_weight_g", f"{result.get('approx_weight_g', 0):.0f} g",  "")
        tbl.add_row("approx_volume_L", f"{result.get('approx_volume_L', 0):.1f} L",  "")
        console.print(tbl)
        console.print(f"  [cyan]Advice    :[/] {advice}")
    except ImportError:
        print(f"\n  File    : {path.name}")
        print(f"  Backend : CLOTH-TOOL")
        print(f"  Detected: {label}")
        for k, v in result.items():
            if not k.endswith("_conf") and k != "image":
                print(f"  {k:<24}: {v}")
        print(f"  Advice  : {advice}")


# ── Gemini narrative advice ───────────────────────────────────────────────────

def _build_clothing_narrative(recommendations: List[DayRecommendation]) -> str:
    from display import _clothing_narrative
    return "\n".join(_clothing_narrative(rec) for rec in recommendations)


def _gemini_advice(garment_label: str, image_path: Path,
                   narrative: str, context: TripContext) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return ""
    try:
        from google import genai
        from PIL import Image as PILImage
        client   = genai.Client(api_key=api_key)
        img      = PILImage.open(str(image_path))
        prompt   = (
            f"I am travelling to {context.city}, {context.country} "
            f"for a {context.purpose or 'personal'} trip.\n\n"
            f"Day-by-day clothing forecast:\n{narrative}\n\n"
            f"The image shows: {garment_label}.\n\n"
            "Should I pack this item? Give a brief practical answer (2-4 sentences) "
            "mentioning the most relevant weather conditions."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=[prompt, img],
        )
        return response.text.strip()
    except Exception as e:
        return f"(Gemini unavailable: {e})"


def _rule_based_advice(garment_label: str,
                       recommendations: List[DayRecommendation]) -> str:
    
    label_lower  = garment_label.lower()
    all_clothing = {c.lower() for rec in recommendations for c in rec.clothing}
    rain_days    = sum(1 for rec in recommendations
                       if any("rain" in a.lower() or "shower" in a.lower()
                              for a in rec.alerts))

    if any(k in label_lower for k in ("jacket", "coat", "anorak", "parka",
                                       "raincoat", "poncho", "windbreaker")):
        if rain_days > 0:
            return (f"Rain expected on {rain_days} day(s) — "
                    "a waterproof outer layer is a good idea, pack it.")
        if any("jacket" in c or "coat" in c for c in all_clothing):
            return "An outer layer is on the packing list — this jacket looks suitable."
        return "Forecast is mild — a light jacket may only be needed for cool evenings."

    if any(k in label_lower for k in ("t-shirt", "jersey", "polo", "tee",
                                       "shirt", "blouse", "top")):
        if any("lightweight" in c or "t-shirt" in c or "short" in c
               for c in all_clothing):
            return "Light tops are recommended — this is a good fit, pack it."
        return "Forecast may call for warmer layers — consider as a base layer only."

    if any(k in label_lower for k in ("sweater", "sweatshirt", "fleece",
                                       "pullover", "cardigan", "hoodie")):
        if any("sweater" in c or "fleece" in c or "warm" in c for c in all_clothing):
            return "Warm mid-layers are on the packing list — this item fits well."
        return "Forecast looks mild-to-warm — pack only if expecting cold evenings."

    if any(k in label_lower for k in ("shoe", "boot", "loafer",
                                       "sneaker", "running shoe", "sandal")):
        if rain_days > 0:
            return "Wet conditions expected — waterproof or sturdy shoes advisable."
        return "Comfortable walking shoes are always useful for travel."

    return ("Check the day-by-day clothing table to see if this garment "
            "matches the recommended items for your trip.")


# ── Garment property estimation ──────────────────────────────────────────────

_GARMENT_RULES = [
    # (keywords,                                    material,          thickness, weight_g, volume_l)
    (("heavy winter coat", "down jacket", "parka", "overcoat", "trench coat"),
                                                    "synthetic/down",  "thick",   1500,     5.0),
    (("jacket", "blazer", "windbreaker", "anorak", "raincoat", "waterproof"),
                                                    "polyester",       "medium",   600,     3.0),
    (("sweater", "sweatshirt", "hoodie", "fleece", "cardigan", "pullover"),
                                                    "fleece/wool",     "medium",   550,     3.0),
    (("jeans", "denim"),                            "denim",           "medium",   700,     2.0),
    (("trousers", "pants", "chinos"),               "cotton",          "medium",   500,     2.0),
    (("shorts", "miniskirt", "skirt"),              "cotton/polyester","thin",     280,     1.0),
    (("t-shirt", "tee", "polo", "jersey", "tank", "blouse", "shirt"),
                                                    "cotton",          "thin",     200,     1.0),
    (("dress", "formal wear"),                      "cotton/polyester","thin",     400,     1.5),
    (("suit", "business attire"),                   "wool blend",      "medium",   900,     3.5),
    (("boot",),                                     "leather/rubber",  "thick",   1000,     3.5),
    (("shoe", "sneaker", "runner", "loafer"),        "leather/mesh",    "medium",   700,     2.5),
    (("sock",),                                     "cotton/wool",     "thin",      80,     0.2),
    (("scarf",),                                    "wool/acrylic",    "thin",     150,     0.5),
    (("glove",),                                    "wool/fleece",     "thin",     100,     0.3),
    (("hat", "cap"),                                "cotton/wool",     "thin",     120,     0.3),
    (("umbrella",),                                 "nylon",           "thin",     400,     1.0),
    (("backpack",),                                 "nylon/polyester", "medium",   800,     8.0),
]


def _rule_based_properties(label: str) -> dict:
    label_lower = label.lower()
    for keywords, material, thickness, weight_g, volume_l in _GARMENT_RULES:
        if any(k in label_lower for k in keywords):
            return {"material": material, "thickness": thickness,
                    "weight_g": weight_g, "volume_l": volume_l}
    return {"material": "unknown", "thickness": "medium", "weight_g": 400, "volume_l": 2.0}


# ── 3D model loaders ──────────────────────────────────────────────────────────

def _get_midas():
    """Lazy-load Intel DPT-Hybrid-MiDaS for monocular depth estimation."""
    global _MIDAS_MODEL, _MIDAS_PROCESSOR
    if _MIDAS_MODEL is not None:
        return _MIDAS_MODEL, _MIDAS_PROCESSOR
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    _mid = "Intel/dpt-hybrid-midas"
    print(f"  [MiDaS] Loading {_mid} (first time ~400 MB)...")
    _MIDAS_PROCESSOR = DPTImageProcessor.from_pretrained(_mid)
    _MIDAS_MODEL     = DPTForDepthEstimation.from_pretrained(_mid)
    _MIDAS_MODEL.eval()
    print("  [MiDaS] Ready.")
    return _MIDAS_MODEL, _MIDAS_PROCESSOR


def _get_sam():
    """Lazy-load SAM ViT-B. Requires sam_vit_b_01ec64.pth (run download_sam.py)."""
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is not None:
        return _SAM_PREDICTOR
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            "segment-anything not installed.\n"
            "Run: pip install segment-anything\n"
            "Then: python download_sam.py"
        )
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint missing: {SAM_CHECKPOINT}\nRun: python download_sam.py"
        )
    print("  [SAM] Loading ViT-B checkpoint (first time only)...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.eval()
    _SAM_PREDICTOR = SamPredictor(sam)
    print("  [SAM] Ready.")
    return _SAM_PREDICTOR


# ── 3D helper functions ───────────────────────────────────────────────────────

def _get_depth_map(image_path: Path):
    """Return a (H, W) numpy array of normalised depth (0=far, 1=near)."""
    import numpy as np
    import torch
    from PIL import Image as PILImage

    model, processor = _get_midas()
    image  = PILImage.open(str(image_path)).convert("RGB")
    orig_w, orig_h = image.size
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        depth = model(**inputs).predicted_depth        # (1, H', W')
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(orig_h, orig_w),
        mode="bicubic", align_corners=False,
    ).squeeze().numpy()
    dmin, dmax = depth.min(), depth.max()
    return (depth - dmin) / (dmax - dmin + 1e-6)      # 0–1


def _get_garment_bbox(image_path: Path) -> List[float]:
    """
    Return [x1, y1, x2, y2] of the dominant garment.
    Tries YOLO detection first; falls back to central 70% crop.
    """
    from PIL import Image as PILImage
    img  = PILImage.open(str(image_path))
    w, h = img.size
    if YOLO_DET_PATH.exists():
        try:
            from ultralytics import YOLO
            model  = YOLO(str(YOLO_DET_PATH))
            result = model(str(image_path), verbose=False)[0]
            boxes  = result.boxes
            if boxes is not None and len(boxes):
                areas  = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes.xyxy]
                best   = int(max(range(len(areas)), key=lambda i: areas[i]))
                return [float(v) for v in boxes.xyxy[best]]
        except Exception:
            pass
    px, py = int(w * 0.15), int(h * 0.15)
    return [px, py, w - px, h - py]


def _get_sam_mask(image_path: Path, bbox: List[float]):
    """Run SAM with a bounding-box prompt; return the best boolean mask (H×W)."""
    import numpy as np
    from PIL import Image as PILImage

    predictor = _get_sam()
    image     = np.array(PILImage.open(str(image_path)).convert("RGB"))
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=np.array(bbox, dtype=float), multimask_output=True,
    )
    return masks[int(scores.argmax())]   # bool (H, W)


def _get_pack_params(label: str) -> tuple:
    """Return (fold_factor, pack_thickness_cm, fabric_area_factor) for a garment label."""
    label_lower = label.lower()
    for keywords, fold_factor, pack_thickness_cm, fabric_area_factor in _GARMENT_PACK:
        if any(k in label_lower for k in keywords):
            return fold_factor, pack_thickness_cm, fabric_area_factor
    return _DEFAULT_PACK


def _calc_volume_weight(area_px: int, image_size: tuple,
                         label: str, material: str,
                         depth_map=None) -> dict:
    """
    Physically-grounded packed volume and weight.

    Physical model
    --------------
    front_area_cm²   = pixel_area × (cm/px)²
                       calibration: garment ≈ 80 cm tall, fills 70 % of image height

    total_fabric_cm² = front_area × fabric_area_factor
                       accounts for back panel, sleeves, etc. (per garment type)

    folded_footprint = total_fabric_cm² / fold_factor
                       fold_factor encodes how many times each type is folded

    packed_volume    = folded_footprint × pack_thickness_cm   → cm³ → L

    weight           = total_fabric_cm² × areal_density       → g
                       areal_density is g/cm² of fabric (material property)

    MiDaS depth corrections (optional)
    -----------------------------------
    worn_correction  : high depth std → garment on 3D body → projected area
                       overestimates true fabric area → scale down by up to 35 %
    bulkiness_boost  : for thick garments (coats) with high depth range →
                       slightly increase pack_thickness (capped at +30 %)
    """
    img_w, img_h = image_size
    cm_per_px    = 80.0 / (img_h * 0.70)          # 80 cm garment = 70 % of frame
    front_area   = area_px * (cm_per_px ** 2)

    fold_factor, pack_thickness_cm, fabric_area_factor = _get_pack_params(label)
    total_fabric  = front_area * fabric_area_factor

    # ── Depth-based corrections ────────────────────────────────────────────
    if depth_map is not None and depth_map.size > 0:
        # Worn-body correction: body curvature inflates apparent projected area
        # Flat garment: depth_std ≈ 0.05–0.10; worn on body: 0.15–0.25
        d_std = float(depth_map.std())
        worn_corr = max(0.65, 1.0 - max(0.0, (d_std - 0.10) / 0.15) * 0.35)
        total_fabric *= worn_corr

        # Bulkiness boost only for inherently thick garments (coats, down)
        if pack_thickness_cm >= 4.0:
            d_range   = float(depth_map.max() - depth_map.min())
            bulk_boost = 1.0 + max(0.0, (d_range - 0.20) / 0.20) * 0.25
            pack_thickness_cm = min(pack_thickness_cm * bulk_boost,
                                    pack_thickness_cm * 1.30)

    folded_footprint = total_fabric / fold_factor
    volume_cm3       = folded_footprint * pack_thickness_cm
    volume_l         = round(volume_cm3 / 1000.0, 2)

    areal_density    = _FABRIC_AREAL_DENSITY.get(material, 0.025)
    weight_g         = int(total_fabric * areal_density)

    return {"weight_g": weight_g, "volume_l": volume_l}


def _estimate_midas(image_path: Path, label: str,
                     material: str) -> Optional[dict]:
    """Volume/weight using MiDaS depth corrections + YOLO bbox area."""
    try:
        from PIL import Image as PILImage
        depth        = _get_depth_map(image_path)
        bbox         = _get_garment_bbox(image_path)
        x1,y1,x2,y2 = [int(v) for v in bbox]
        region_depth = depth[y1:y2, x1:x2]
        area_px      = (x2 - x1) * (y2 - y1)
        img          = PILImage.open(str(image_path))
        r = _calc_volume_weight(area_px, img.size, label, material,
                                depth_map=region_depth)
        r["note"] = "MiDaS depth"
        return r
    except Exception as e:
        print(f"  [MiDaS] skipped: {e}")
        return None


def _estimate_sam(image_path: Path, label: str,
                   material: str) -> Optional[dict]:
    """Volume/weight using SAM pixel-precise mask area (no depth correction)."""
    try:
        from PIL import Image as PILImage
        bbox    = _get_garment_bbox(image_path)
        mask    = _get_sam_mask(image_path, bbox)
        area_px = int(mask.sum())
        img     = PILImage.open(str(image_path))
        r = _calc_volume_weight(area_px, img.size, label, material, depth_map=None)
        r["note"] = "SAM mask area"
        return r
    except Exception as e:
        print(f"  [SAM] skipped: {e}")
        return None


def _estimate_midas_sam(image_path: Path, label: str,
                         material: str) -> Optional[dict]:
    """Volume/weight using SAM precise mask area + MiDaS depth corrections."""
    try:
        from PIL import Image as PILImage
        bbox          = _get_garment_bbox(image_path)
        depth         = _get_depth_map(image_path)
        mask          = _get_sam_mask(image_path, bbox)
        area_px       = int(mask.sum())
        depth_in_mask = depth[mask] if area_px > 0 else depth
        img           = PILImage.open(str(image_path))
        r = _calc_volume_weight(area_px, img.size, label, material,
                                depth_map=depth_in_mask)
        r["note"] = "MiDaS depth + SAM mask"
        return r
    except Exception as e:
        print(f"  [MiDaS+SAM] skipped: {e}")
        return None


def _estimate_garment_properties(label: str, image_path: Path,
                                  method: str = "midas_sam") -> dict:
    """
    Estimate material/thickness then compute weight/volume using the chosen method:
      midas_sam  — SAM mask area + MiDaS depth corrections  (default, best accuracy)
      midas      — MiDaS depth corrections over YOLO bbox
      sam        — SAM pixel-precise mask area, no depth
      rule_based — static lookup table, no models needed

    Returns material, thickness, weight_g, volume_l and an 'estimates' sub-dict.
    """
    # ── Step 1: classify material + thickness ─────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    mat_thick = None
    if api_key:
        try:
            import json as _json
            from google import genai
            from PIL import Image as PILImage
            client = genai.Client(api_key=api_key)
            img    = PILImage.open(str(image_path))
            prompt = (
                f"This garment has been identified as: {label}.\n"
                "Return ONLY a JSON object (no markdown) with:\n"
                '  "material": primary fabric\n'
                '  "thickness": one of "thin", "medium", "thick"\n'
            )
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=[prompt, img],
            )
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            mat_thick = _json.loads(text)
        except Exception:
            pass
    if mat_thick is None:
        base = _rule_based_properties(label)
        mat_thick = {"material": base["material"], "thickness": base["thickness"]}

    material      = mat_thick.get("material", "unknown")
    thickness_key = mat_thick.get("thickness", "medium")

    # ── Step 2: rule-based baseline (always computed, used as fallback) ─────
    base = _rule_based_properties(label)
    rule_est = {"weight_g": base["weight_g"], "volume_l": base["volume_l"],
                "note": "lookup table"}

    # ── Step 3: run only the requested method ─────────────────────────────
    from PIL import Image as _PILImage
    img_size = _PILImage.open(str(image_path)).size

    midas_est     = None
    sam_est       = None
    midas_sam_est = None

    if method in ("midas", "midas_sam"):
        bbox      = _get_garment_bbox(image_path)
        depth_map = None
        try:
            depth_map = _get_depth_map(image_path)
        except Exception as e:
            print(f"  [MiDaS] skipped: {e}")

    if method in ("sam", "midas_sam"):
        if method == "sam":
            bbox = _get_garment_bbox(image_path)
        sam_mask = None
        sam_area = None
        try:
            sam_mask = _get_sam_mask(image_path, bbox)
            sam_area = int(sam_mask.sum())
        except Exception as e:
            print(f"  [SAM] skipped: {e}")

    if method == "midas" and depth_map is not None:
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            area_px         = (x2 - x1) * (y2 - y1)
            r = _calc_volume_weight(area_px, img_size, label, material,
                                    depth_map=depth_map[y1:y2, x1:x2])
            r["note"] = "MiDaS depth"
            midas_est = r
        except Exception as e:
            print(f"  [MiDaS] calc skipped: {e}")

    elif method == "sam" and sam_mask is not None:
        try:
            r = _calc_volume_weight(sam_area, img_size, label, material,
                                    depth_map=None)
            r["note"] = "SAM mask area"
            sam_est = r
        except Exception as e:
            print(f"  [SAM] calc skipped: {e}")

    elif method == "midas_sam" and depth_map is not None and sam_mask is not None:
        try:
            depth_in_mask = depth_map[sam_mask] if sam_area > 0 else depth_map
            r = _calc_volume_weight(sam_area, img_size, label, material,
                                    depth_map=depth_in_mask)
            r["note"] = "MiDaS depth + SAM mask"
            midas_sam_est = r
        except Exception as e:
            print(f"  [MiDaS+SAM] calc skipped: {e}")

    # ── Step 4: use selected result, fall back to rule_based ──────────────
    best = midas_sam_est or midas_est or sam_est or rule_est

    return {
        "material":  material,
        "thickness": thickness_key,
        "weight_g":  best["weight_g"],
        "volume_l":  best["volume_l"],
        "estimates": {
            "rule_based": rule_est,
            "midas":      midas_est,
            "sam":        sam_est,
            "midas_sam":  midas_sam_est,
        },
    }


# ── Display helpers ───────────────────────────────────────────────────────────

def _filename_fallback(image_path: Path, note: str = "") -> str:
    stem = image_path.stem.replace("_", " ").replace("-", " ")
    return f"garment '{stem}'" + (f" ({note})" if note else "")


def _print_header(title: str) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        Console().print(Panel(f"[bold]{title}[/]",
                              title="[bold blue]👗 Image Recognition[/]",
                              border_style="blue"))
    except ImportError:
        print(f"\n{'─'*60}\n{title}\n{'─'*60}")


def _print_single_result(path: Path, backend: str,
                         label: str, advice: str) -> None:
    try:
        from rich.console import Console
        console = Console()
        console.print(f"\n  [cyan]File    :[/] {path.name}")
        console.print(f"  [cyan]Backend :[/] [bold]{backend.upper()}[/]")
        console.print(f"  [cyan]Detected:[/] {label}")
        console.print(f"  [cyan]Advice  :[/] {advice}")
    except ImportError:
        print(f"\n  File    : {path.name}")
        print(f"  Backend : {backend.upper()}")
        print(f"  Detected: {label}")
        print(f"  Advice  : {advice}")


def _print_both_results(path: Path,
                        labels: Dict[str, str],
                        advice: Dict[str, str]) -> None:
    """Side-by-side comparison table for --vision both."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
        console.print(f"\n  [bold cyan]{path.name}[/]")
        tbl = Table(box=box.ROUNDED, show_header=True,
                    header_style="bold magenta", expand=True)
        tbl.add_column("Backend",  style="bold cyan", no_wrap=True, width=10)
        tbl.add_column("Detected", style="white",     ratio=2)
        tbl.add_column("Advice",   style="white",     ratio=3)
        for backend in ("yolo", "google", "clip"):
            if backend in labels:
                tbl.add_row(backend.upper(),
                            labels[backend],
                            advice.get(backend, "—"))
        console.print(tbl)
    except ImportError:
        sep = "─" * 60
        print(f"\n  {path.name}\n  {sep}")
        for backend in ("yolo", "google", "clip"):
            if backend in labels:
                print(f"  [{backend.upper()}]")
                print(f"    Detected : {labels[backend]}")
                print(f"    Advice   : {advice.get(backend, '—')}")
        print(f"  {sep}")


def _print_footer(vision: str) -> None:
    hints = []
    if vision in ("google", "both") and not os.environ.get("GOOGLE_VISION_API_KEY"):
        hints.append("Set GOOGLE_VISION_API_KEY to enable Google Vision.")
    if not os.environ.get("GEMINI_API_KEY"):
        hints.append("Set GEMINI_API_KEY for AI-powered packing advice.")
    if hints:
        msg = "  ".join(hints)
        try:
            from rich.console import Console
            Console().print(f"[dim]{msg}[/]")
        except ImportError:
            print(msg)


def _print_weight_volume_comparison(estimates: dict, material: str, thickness: str) -> None:
    """Print a 4-row comparison table of weight/volume for all estimation methods."""
    rows = [
        ("midas_sam",  "MiDaS + SAM"),
        ("midas",      "MiDaS"),
        ("sam",        "SAM"),
        ("rule_based", "Rule-based"),
    ]
    best_key = next(
        (k for k in ("midas_sam", "midas", "sam", "rule_based") if estimates.get(k)),
        "rule_based",
    )
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box as rbox
        console = Console()
        tbl = Table(box=rbox.SIMPLE, show_header=True, header_style="bold yellow",
                    title=f"  Weight & Volume  [{material} / {thickness}]",
                    title_style="bold cyan")
        tbl.add_column("Method",  style="cyan",  no_wrap=True, width=18)
        tbl.add_column("Weight",  style="white", justify="right", width=10)
        tbl.add_column("Volume",  style="white", justify="right", width=10)
        tbl.add_column("Note",    style="dim")
        for key, name in rows:
            est = estimates.get(key)
            if est:
                hi = key == best_key
                tbl.add_row(
                    f"[bold green]{name}[/]" if hi else name,
                    f"[bold green]{est['weight_g']} g[/]" if hi else f"{est['weight_g']} g",
                    f"[bold green]{est['volume_l']:.2f} L[/]" if hi else f"{est['volume_l']:.2f} L",
                    est.get("note", ""),
                )
            else:
                tbl.add_row(name, "—", "—", "[dim]unavailable[/]")
        console.print(tbl)
    except ImportError:
        print(f"\n  Weight & Volume  [{material} / {thickness}]")
        print(f"  {'Method':<18} {'Weight':>9} {'Volume':>9}")
        print(f"  {'-'*40}")
        for key, name in rows:
            est = estimates.get(key)
            mark = " ←" if key == best_key else ""
            if est:
                print(f"  {name:<18} {est['weight_g']:>7} g {est['volume_l']:>7.2f} L{mark}")
            else:
                print(f"  {name:<18} {'—':>9} {'—':>9}")


# ── Public API ────────────────────────────────────────────────────────────────

def collect_image_paths_from_folder(folder: str) -> List[str]:
    """
    Return sorted absolute paths to image files in ``folder`` (non-recursive).

    Only regular files whose suffix is in IMAGE_FILE_SUFFIXES are included.
    If ``folder`` is missing or not a directory, returns an empty list.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        return []
    out: List[str] = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_FILE_SUFFIXES:
            out.append(str(p.resolve()))
    return sorted(out)


def analyse_outfits(image_paths: List[str],
                    recommendations: List[DayRecommendation],
                    context: TripContext,
                    vision: str = "yolo",
                    depth: str = "midas_sam") -> List[dict]:
    """
    Analyse each wardrobe photo and print packing advice to the terminal.

    Parameters
    ----------
    image_paths    : paths to wardrobe photos (jpg/png)
    recommendations: per-day clothing recommendations from recommender.py
    context        : TripContext (city, country, purpose)
    vision         : "yolo" | "google" | "clip" | "both" | "cloth-tool"  (default: "yolo")
    depth          : "midas_sam" | "midas" | "sam" | "rule_based"        (default: "midas_sam")

    Returns
    -------
    List of dicts, one per image:
        image_name, detected_label, material, thickness, weight_g, volume_l,
        advice  (and backends dict when vision="both")
    """
    if not image_paths:
        return []

    vision = vision.lower()
    if vision not in VALID_VISION_MODES:
        print(f"  [!] Unknown --vision mode '{vision}'. "
              f"Choose from: {VALID_VISION_MODES}")
        vision = "yolo"

    backends          = ["yolo", "google", "clip"] if vision == "both" else [vision]
    narrative         = _build_clothing_narrative(recommendations)
    recommender_items = _get_recommender_items()
    results: List[dict] = []

    _print_header(f"Wardrobe Analysis  —  backend: {vision.upper()}")
    for raw_path in image_paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"  [!] File not found: {path}")
            continue

        if vision == "cloth-tool":
            raw     = _detect_cloth_tool(path)
            label   = _cloth_tool_readable_label(raw) if raw else _filename_fallback(path)
            mat_key = raw.get("material_group", "mixed_unknown")
            material  = _CLOTH_MATERIAL_MAP.get(mat_key, "unknown")
            thickness = _CLOTH_THICKNESS_MAP.get(raw.get("weight_class", "medium"), "medium")
            weight_g  = int(raw.get("approx_weight_g", 400))
            volume_l  = round(raw.get("approx_volume_L", 2.0), 2)
            adv = _gemini_advice(label, path, narrative, context)
            if not adv:
                adv = _rule_based_advice(label, recommendations)
            _print_cloth_tool_result(path, raw, label, adv)
            results.append({
                "image_name":      path.name,
                "detected_label":  label,
                "material":        material,
                "thickness":       thickness,
                "weight_g":        weight_g,
                "volume_l":        volume_l,
                "cloth_tool_attrs":raw,
                "advice":          adv,
            })
            continue

        if vision == "both":
            labels, advice = {}, {}
            for backend in backends:
                lbl = _run_backend(backend, path, recommender_items)
                labels[backend] = lbl
                adv = _gemini_advice(lbl, path, narrative, context)
                advice[backend] = adv or _rule_based_advice(lbl, recommendations)
            _print_both_results(path, labels, advice)
            # Use yolo label (most specific) for property estimation; fall back to first available
            primary_label = labels.get("yolo") or next(iter(labels.values()), path.stem)
            props = _estimate_garment_properties(primary_label, path, depth)
            _print_weight_volume_comparison(props["estimates"], props["material"], props["thickness"])
            results.append({
                "image_name":              path.name,
                "backends":                labels,
                "detected_label":          primary_label,
                "material":                props["material"],
                "thickness":               props["thickness"],
                "weight_g":                props["weight_g"],
                "volume_l":                props["volume_l"],
                "weight_volume_estimates": props["estimates"],
                "advice":                  advice,
            })
        else:
            label = _run_backend(vision, path, recommender_items)
            adv   = _gemini_advice(label, path, narrative, context)
            if not adv:
                adv = _rule_based_advice(label, recommendations)
            _print_single_result(path, vision, label, adv)
            props = _estimate_garment_properties(label, path, depth)
            _print_weight_volume_comparison(props["estimates"], props["material"], props["thickness"])
            results.append({
                "image_name":              path.name,
                "detected_label":          label,
                "material":                props["material"],
                "thickness":               props["thickness"],
                "weight_g":                props["weight_g"],
                "volume_l":                props["volume_l"],
                "weight_volume_estimates": props["estimates"],
                "advice":                  adv,
            })

    _print_footer(vision)
    return results


def _run_backend(backend: str, image_path: Path,
                 recommender_items: List[str]) -> str:
    """Dispatch to the correct detection backend."""
    if backend == "yolo":
        return _detect_yolo(image_path)
    if backend == "google":
        return _detect_google_vision(image_path)
    if backend == "clip":
        return _detect_clip(image_path, recommender_items)
    if backend == "cloth-tool":
        raw = _detect_cloth_tool(image_path)
        return _cloth_tool_readable_label(raw) if raw else _filename_fallback(image_path)
    return _filename_fallback(image_path, note=f"unknown backend '{backend}'")


def _get_recommender_items() -> List[str]:
    """Return the 37-item vocabulary from recommender.py."""
    try:
        from recommender import ALL_ITEMS
        return list(ALL_ITEMS)
    except ImportError:
        return []
    
# ==================================================================================
# ADDITION by Kevin: Dynamic Dimension Estimation Algorithm
# Replaces hardcoded rule-based weight/volume with CV pixel-area heuristics.
# ==================================================================================
def estimate_dimensions_from_cv(image_path: Path, label: str, mask = None) -> dict:
    """
    Algorithmic estimation of volume and weight using CV segmentation pixel-area.
    Replaces the hardcoded _GARMENT_RULES dictionary for dynamic packing.
    """
    import numpy as np # Isolated import to avoid modifying top-of-file imports
    
    try:
        from PIL import Image
        img = Image.open(str(image_path))
        img_area = img.width * img.height
        
        # Density constants (kg per pixel^2 approximation for standard folded garments)
        category_density = {
            "outer": 4.5e-6,  # Heavy coats, thick jackets
            "mid": 2.8e-6,    # Sweaters, fleeces
            "base": 1.5e-6,   # T-shirts, trousers
            "feet": 3.5e-6,   # Shoes, boots
            "acc": 1.0e-6     # Scarves, umbrellas
        }
        
        # Determine category roughly from label
        cat = "base"
        if any(k in label.lower() for k in ("coat", "jacket", "windproof")): cat = "outer"
        elif any(k in label.lower() for k in ("sweater", "fleece", "hoodie")): cat = "mid"
        elif any(k in label.lower() for k in ("shoe", "boot", "sneaker")): cat = "feet"
        elif any(k in label.lower() for k in ("scarf", "hat", "umbrella")): cat = "acc"
        
        density = category_density.get(cat, 2.0e-6)
        
        # If YOLO26 segmentation mask is provided, use exact pixel area
        if mask is not None:
            # Convert torch tensor to numpy array if necessary
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            garment_pixels = np.count_nonzero(mask)
            coverage_ratio = garment_pixels / img_area
        else:
            # Heuristic: garment usually covers 30-60% of the image bounding box
            coverage_ratio = 0.45 
            
        effective_area = img_area * coverage_ratio
        estimated_weight_g = (effective_area * density) * 1000 # Convert to grams
        
        # Volume estimation (assuming standard folded thickness based on category)
        thickness_cm = {"base": 3, "mid": 5, "outer": 8, "feet": 10, "acc": 2}.get(cat, 4)
        folded_area_cm2 = (effective_area ** 0.5) * 0.8 * 0.0264 # Approx pixel to cm conversion
        volume_cm3 = folded_area_cm2 * thickness_cm
        volume_l = volume_cm3 / 1000
        
        # Cross-validate with Gemini if available (soft fallback)
        gemini_est = _estimate_garment_properties(label, image_path)
        if gemini_est.get("weight_g") and gemini_est.get("volume_l"):
            # Average the algorithmic calculation with LLM visual estimation
            final_weight = (estimated_weight_g + gemini_est["weight_g"]) / 2
            final_volume = (volume_l + gemini_est["volume_l"]) / 2
        else:
            final_weight = estimated_weight_g
            final_volume = volume_l
            
        return {
            "weight_g": int(max(50, min(3000, final_weight))), # Clamp to realistic bounds
            "volume_l": round(max(0.1, min(15.0, final_volume)), 1)
        }
        
    except Exception as e:
        print(f"[Dynamic Estimation Error] {e}")
        return _rule_based_properties(label) # Graceful fallback to original rules
    