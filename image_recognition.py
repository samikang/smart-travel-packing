"""
Image Recognition — Wardrobe Outfit Analyser
=============================================
Analyses wardrobe photos against the trip weather forecast and produces
packing advice. Supports three detection backends selectable via --vision:

  yolo    (default) YOLOv8 local detection — offline, no API key required.
                    Tries yolov8n-cls.pt (classification) first, then
                    yolov8n.pt (object detection) as fallback.

  google            Google Cloud Vision API — online, needs GOOGLE_VISION_API_KEY
                    env var. Returns label annotations ranked by confidence.

  clip              Local CLIP (openai/clip-vit-base-patch32) — offline, no API
                    key. Compares image embedding against the 37 recommender item
                    names first; if best score < threshold, falls back to a
                    broader generic clothing vocabulary.

  both              Runs all three backends on every image and displays each
                    result in a side-by-side comparison table.

After detection, the garment label is passed to Gemini (optional, needs
GEMINI_API_KEY) for narrative packing advice. Falls back to rule-based
advice if Gemini is unavailable.

Environment variables
---------------------
    GOOGLE_VISION_API_KEY   — Google Cloud Vision API key  (vision=google/both)
    GEMINI_API_KEY          — Google Gemini API key        (narrative advice)

Model files (place in project root)
-------------------------------------
    yolov8n-cls.pt          — YOLOv8 classification weights
    yolov8n.pt              — YOLOv8 detection weights
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
YOLO_CLS_PATH = _HERE / "yolov8n-cls.pt"
YOLO_DET_PATH = _HERE / "yolov8n.pt"

VALID_VISION_MODES = ("yolo", "google", "clip", "both")

# Raster files accepted when scanning a wardrobe folder (--images)
IMAGE_FILE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
}

# CLIP: if best recommender-item similarity is below this, use generic vocab
CLIP_RECOMMENDER_THRESHOLD = 0.20

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
                return f"{chosen} (YOLOv8-det)"
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


def _estimate_garment_properties(label: str, image_path: Path) -> dict:
    """
    Estimate material, thickness, weight_g, volume_l for a detected garment.
    Tries Gemini first (needs GEMINI_API_KEY); falls back to rule-based lookup.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        try:
            import json as _json
            from google import genai
            from PIL import Image as PILImage
            client = genai.Client(api_key=api_key)
            img    = PILImage.open(str(image_path))
            prompt = (
                f"This garment has been identified as: {label}.\n"
                "Return ONLY a JSON object (no markdown, no explanation) with:\n"
                '  "material": primary fabric (e.g. "cotton", "wool", "polyester", "denim", "down", "leather")\n'
                '  "thickness": one of "thin", "medium", "thick"\n'
                '  "weight_g": estimated packed weight in grams (integer)\n'
                '  "volume_l": estimated packed volume in litres (one decimal)\n'
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=[prompt, img],
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return _json.loads(text)
        except Exception:
            pass
    return _rule_based_properties(label)


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
                    vision: str = "yolo") -> List[dict]:
    """
    Analyse each wardrobe photo and print packing advice to the terminal.

    Parameters
    ----------
    image_paths    : paths to wardrobe photos (jpg/png)
    recommendations: per-day clothing recommendations from recommender.py
    context        : TripContext (city, country, purpose)
    vision         : "yolo" | "google" | "clip" | "both"  (default: "yolo")

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
            props = _estimate_garment_properties(primary_label, path)
            results.append({
                "image_name":     path.name,
                "backends":       labels,
                "detected_label": primary_label,
                "material":       props["material"],
                "thickness":      props["thickness"],
                "weight_g":       props["weight_g"],
                "volume_l":       props["volume_l"],
                "advice":         advice,
            })
        else:
            label = _run_backend(vision, path, recommender_items)
            adv   = _gemini_advice(label, path, narrative, context)
            if not adv:
                adv = _rule_based_advice(label, recommendations)
            _print_single_result(path, vision, label, adv)
            props = _estimate_garment_properties(label, path)
            results.append({
                "image_name":     path.name,
                "detected_label": label,
                "material":       props["material"],
                "thickness":      props["thickness"],
                "weight_g":       props["weight_g"],
                "volume_l":       props["volume_l"],
                "advice":         adv,
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