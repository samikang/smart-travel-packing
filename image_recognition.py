"""
Image Recognition — Wardrobe Outfit Analyser
=============================================
Analyses user-supplied wardrobe photos against the trip weather forecast and
produces packing advice: which outfits to bring and why.

Two-stage pipeline
------------------
1. **Local detection** (YOLOv8 object/classification model, offline, no API key)
   - Detects the garment category from the image (jacket, shirt, etc.)
   - Falls back to a simple heuristic colour/shape description if YOLO is
     unavailable (ultralytics not installed or model file missing).

2. **Narrative advice** (Google Gemini, optional)
   - Sends the detected garment label + full clothing narrative for the trip to
     Gemini 2.5 Flash and returns concise packing advice.
   - Skipped gracefully if the google-genai package is not installed or
     GEMINI_API_KEY is not set.

Usage (called from main.py)
---------------------------
    from image_recognition import analyse_outfits
    analyse_outfits(image_paths, recommendations, context)

    image_paths    : list of file paths (str or Path) to wardrobe photos
    recommendations: list[DayRecommendation] from recommender.py
    context        : TripContext

Environment
-----------
    GEMINI_API_KEY  — Google Gemini API key (optional; skips LLM step if absent)

Model files
-----------
Place YOLOv8 weights in the project root or pass the path explicitly:
    yolov8n.pt       — object detection  (detects 'person', 'tie', etc.)
    yolov8n-cls.pt   — classification    (ImageNet classes incl. clothing)
The module tries the classification model first, then falls back to detection.
"""

import os
import sys
from pathlib import Path
from typing import List

from models import DayRecommendation, TripContext

# Paths to YOLOv8 weight files (project root, same directory as this file)
_HERE         = Path(__file__).parent
YOLO_CLS_PATH = _HERE / "yolov8n-cls.pt"
YOLO_DET_PATH = _HERE / "yolov8n.pt"

# Clothing-related ImageNet / COCO class names we care about
_CLOTHING_CLASSES = {
    # COCO detection labels
    "tie", "handbag", "backpack", "umbrella", "suitcase",
    # ImageNet classification labels (partial — YOLO cls uses top-1 label)
    "jersey", "sweatshirt", "cardigan", "trench coat", "overcoat",
    "suit", "lab coat", "jean", "shorts", "miniskirt",
    "raincoat", "poncho", "hooded jacket", "anorak", "parka",
    "t-shirt", "turtleneck", "pullover", "polo shirt",
    "running shoe", "loafer", "boot",
}


# ── Stage 1: Local garment detection ─────────────────────────────────────────

def _detect_with_yolo(image_path: Path) -> str:
    """
    Run YOLOv8 on a single image and return a human-readable garment label.
    Tries classification model first (more clothing-aware), then detection.
    Returns a plain-English description string.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return _fallback_description(image_path)

    # ── Try classification model ──────────────────────────────────────────────
    if YOLO_CLS_PATH.exists():
        try:
            model  = YOLO(str(YOLO_CLS_PATH))
            result = model(str(image_path), verbose=False)[0]
            top1   = result.names[result.probs.top1]
            conf   = float(result.probs.top1conf)
            return f"{top1} (classification confidence {conf:.0%})"
        except Exception:
            pass

    # ── Fallback: detection model ─────────────────────────────────────────────
    if YOLO_DET_PATH.exists():
        try:
            model   = YOLO(str(YOLO_DET_PATH))
            result  = model(str(image_path), verbose=False)[0]
            boxes   = result.boxes
            if boxes is not None and len(boxes):
                labels = [result.names[int(c)] for c in boxes.cls]
                # Prefer clothing-related labels; fall back to highest-confidence box
                clothing = [l for l in labels if l.lower() in _CLOTHING_CLASSES]
                chosen   = clothing[0] if clothing else labels[0]
                return f"{chosen} (detection)"
        except Exception:
            pass

    return _fallback_description(image_path)


def _fallback_description(image_path: Path) -> str:
    """
    Minimal description when YOLO is unavailable.
    Returns the filename stem as a best-effort label.
    """
    stem = image_path.stem.replace("_", " ").replace("-", " ")
    return f"garment '{stem}' (no model — filename only)"


# ── Stage 2: Gemini narrative advice ─────────────────────────────────────────

def _build_clothing_narrative(recommendations: List[DayRecommendation]) -> str:
    """Summarise the trip's clothing needs as a plain-text paragraph."""
    from display import _clothing_narrative, _fmt_date
    lines = []
    for rec in recommendations:
        lines.append(_clothing_narrative(rec))
    return "\n".join(lines)


def _gemini_advice(garment_label: str, image_path: Path,
                   narrative: str, context: TripContext) -> str:
    """
    Ask Gemini whether this garment is suitable for the trip.
    Returns the model's text response, or an empty string on any failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return ""

    try:
        from google import genai
        import PIL.Image

        client = genai.Client(api_key=api_key)
        img    = PIL.Image.open(str(image_path))

        prompt = (
            f"I am travelling to {context.city}, {context.country} "
            f"for a {context.purpose or 'personal'} trip.\n\n"
            f"Here is the day-by-day weather and clothing forecast for my trip:\n"
            f"{narrative}\n\n"
            f"The image shows: {garment_label}.\n\n"
            "Based on the forecast, should I pack this item? "
            "Give a brief, practical answer (2-4 sentences): "
            "mention the most relevant weather conditions and whether "
            "this garment fits them."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img],
        )
        return response.text.strip()

    except Exception as e:
        return f"(Gemini unavailable: {e})"


# ── Public API ────────────────────────────────────────────────────────────────

def analyse_outfits(image_paths: List[str],
                    recommendations: List[DayRecommendation],
                    context: TripContext) -> None:
    """
    Analyse each wardrobe photo and print packing advice to the terminal.

    Parameters
    ----------
    image_paths    : paths to wardrobe photos (jpg/png)
    recommendations: per-day clothing recommendations from recommender.py
    context        : TripContext (city, country, purpose)
    """
    if not image_paths:
        return

    # Build clothing narrative once (shared across all images)
    narrative = _build_clothing_narrative(recommendations)

    _print_header("Wardrobe Photo Analysis")

    for raw_path in image_paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"  [!] File not found: {path}")
            continue

        print(f"\n  Analysing: {path.name}")

        # Stage 1 — local detection
        garment_label = _detect_with_yolo(path)
        print(f"  Detected : {garment_label}")

        # Stage 2 — Gemini narrative (optional)
        advice = _gemini_advice(garment_label, path, narrative, context)
        if advice:
            print(f"  Advice   : {advice}")
        else:
            # Minimal rule-based fallback when Gemini is not available
            advice = _rule_based_advice(garment_label, recommendations)
            print(f"  Advice   : {advice}")

    _print_footer()


def _rule_based_advice(garment_label: str, recommendations: List[DayRecommendation]) -> str:
    """
    Simple keyword-matching advice when no LLM is available.
    Checks whether the detected garment type matches the trip's clothing list.
    """
    label_lower = garment_label.lower()
    all_clothing = {c.lower() for rec in recommendations for c in rec.clothing}
    rain_days    = sum(1 for rec in recommendations
                       if any("rain" in a.lower() or "shower" in a.lower()
                              for a in rec.alerts))

    # Jacket / coat keywords
    if any(k in label_lower for k in ("jacket", "coat", "anorak", "parka", "raincoat", "poncho")):
        if rain_days > 0:
            return (f"Rain is expected on {rain_days} day(s). "
                    "A waterproof or windproof jacket is a good idea — pack it.")
        if any("jacket" in c or "coat" in c for c in all_clothing):
            return "An outer layer is recommended for this trip. This jacket looks suitable."
        return "The forecast is mild — a light jacket may suffice for cool evenings."

    # T-shirt / light top keywords
    if any(k in label_lower for k in ("t-shirt", "jersey", "polo", "tee", "shirt")):
        if any("lightweight" in c or "t-shirt" in c or "short" in c for c in all_clothing):
            return "Light tops are recommended for this trip. This shirt is a good fit — pack it."
        return "The forecast may call for warmer layers. Consider this as a base layer only."

    # Sweater / fleece keywords
    if any(k in label_lower for k in ("sweater", "sweatshirt", "fleece", "pullover", "cardigan")):
        if any("sweater" in c or "fleece" in c or "warm" in c for c in all_clothing):
            return "Warm mid-layers are on the packing list for this trip. This item fits well."
        return "Forecast looks mild-to-warm — this may be too heavy. Pack only if expecting cold evenings."

    # Shoes / boots keywords
    if any(k in label_lower for k in ("shoe", "boot", "loafer", "sneaker", "running shoe")):
        if rain_days > 0:
            return "Wet conditions expected — waterproof or comfortable walking shoes are advisable."
        return "Comfortable walking shoes are always useful for travel. This looks like a reasonable choice."

    # Generic fallback
    return ("Check the day-by-day clothing table above to see if this garment "
            "matches the recommended items for your trip.")


def _print_header(title: str) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        Console().print(Panel(f"[bold]{title}[/]",
                              title="[bold blue]👗 Image Recognition[/]",
                              border_style="blue"))
    except ImportError:
        sep = "-" * 60
        print(f"\n{sep}\n{title}\n{sep}")


def _print_footer() -> None:
    try:
        from rich.console import Console
        Console().print("[dim]Set GEMINI_API_KEY for richer AI-powered advice.[/]")
    except ImportError:
        print("\n(Set GEMINI_API_KEY for richer AI-powered advice.)")
