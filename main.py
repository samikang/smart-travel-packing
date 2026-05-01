#!/usr/bin/env python3
"""
Travel Weather Advisor
======================
Usage:
    # Live forecast (up to 16 days ahead) + save chart:
    python main.py --city "Singapore" --start 2026-03-15 --end 2026-03-20 --purpose business --chart

    # Custom historical prediction (any future date):
    python main.py --city "Tokyo" --start 2026-07-10 --end 2026-07-15 --purpose tourism --chart

    # Retrain the recommendation model:
    python main.py --retrain

    # Optional: analyse wardrobe photos (folder with PNG/JPEG, etc.):
    python main.py --city "Singapore" --start 2026-05-01 --end 2026-05-05 --images img/
"""
import os  # Must be first — env vars must be set before torch/transformers load
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import re
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path
from dataclasses import asdict

from dotenv import load_dotenv

from geocoder import get_location
from weather import get_forecast
from historical_forecast import get_historical_forecast
from recommender import (
    recommend_day,
    build_trip_packing_list,
    suitability_scores,
    calculate_needed_quantity,
)
from models import TripContext
from display import display, plot_forecast, save_gui_json
from image_recognition import analyse_outfits, collect_image_paths_from_folder
from packing_optimizer import (
    optimise_from_recommendations,
    optimise_items,
    optimise_dynamic_items,
    ITEM_WEIGHTS,
    ITEM_DIMS_CM,
    DEFAULT_DIMS,
    DEFAULT_WEIGHT,
)


VALID_PURPOSES = ("business", "tourism", "visiting")
MAX_FORECAST_DAYS = 16


def parse_args():
    """Parse command‑line arguments.

    Returns:
        An argparse.Namespace containing all user‑provided arguments.
    """
    parser = argparse.ArgumentParser(
        description="Travel Weather Advisor — clothing & packing recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--city",    help="Destination city, e.g. 'Tokyo' or 'Paris, France'")
    parser.add_argument("--start",   help="Travel start date (YYYY-MM-DD)")
    parser.add_argument("--end",     help="Travel end date (YYYY-MM-DD)")
    parser.add_argument("--purpose", choices=VALID_PURPOSES,
                        help="Trip purpose: business, tourism, or visiting")
    parser.add_argument("--years",   type=int, default=10, metavar="N",
                        help="[historical] years of archive data to use (default: 10)")
    parser.add_argument("--method",  default="theil_sen",
                        choices=["ewm_ols", "holt_des", "theil_sen", "gpr"],
                        help="prediction algorithm: theil_sen (default), ewm_ols, holt_des, gpr")
    parser.add_argument("--chart",   action="store_true",
                        help="Generate and save a matplotlib forecast chart (PNG)")
    parser.add_argument("--model",   default="knn",
                        choices=["lgbm", "random_forest", "knn", "rules"],
                        help="recommendation algorithm: knn (default), lgbm, random_forest, rules")
    parser.add_argument(
        "--images",
        metavar="DIR",
        default=None,
        help="Optional. Wardrobe photo folder (PNG/JPEG/etc.). Omit to skip image analysis; "
             "if given, analyse_outfits runs only when the folder has at least one image file",
    )
    parser.add_argument("--vision",  default="yolo",
                        choices=["yolo", "google", "clip", "both", "cloth_tool"],
                        help="Image recognition backend: yolo (default), google, clip, both, cloth_tool")
    parser.add_argument("--depth", default="midas_sam",
                        choices=["midas_sam", "midas", "sam", "rule_based"],
                        help="Weight/volume estimation method: midas_sam (default), midas, sam, rule_based")
    parser.add_argument("--optimize", action="store_true",
                        help="Run GA → Knapsack → 3D packing optimization")
    parser.add_argument("--opt-mode", default="balanced",
                        choices=["light", "balanced", "aggressive"],
                        help="Optimization aggressiveness: light, balanced (default), aggressive")
    parser.add_argument("--weight-limit", type=float, default=20.0, metavar="KG",
                        help="Baggage weight limit in kg for optimization (default: 20.0)")
    parser.add_argument("--luggage-dims", nargs=3, type=float, default=[70, 45, 30],
                        metavar=("L", "W", "H"),
                        help="Luggage inner dimensions in cm (default: 70 45 30)")
    parser.add_argument("--reserve-space", type=float, default=15.0, metavar="PCT",
                        help="Reserve luggage space percentage for souvenirs (default: 15)")
    parser.add_argument("--optimize-items", nargs="+", metavar="ITEM",
                        help="Standalone optimization: provide item names directly "
                             "(skips recommender; use with --weight-limit)")
    parser.add_argument("--json",    action="store_true",
                        help="Save forecasts, recommendations, and packing list as a JSON file")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain the recommendation model (respects --model)")
    parser.add_argument("--multi-bag", action="store_true",
                    help="Generate a carry‑on vs. check‑in split suggestion")
    return parser.parse_args()


def validate_dates_forecast(start_str, end_str):
    """Validate dates for the live forecast window (up to 16 days ahead).

    Args:
        start_str: Start date in YYYY-MM-DD format.
        end_str: End date in YYYY-MM-DD format.

    Returns:
        A tuple (start_iso, end_iso) of validated ISO date strings.
        The end date may be clipped to the maximum forecast limit.
    """
    try:
        start = date.fromisoformat(start_str)
    except ValueError:
        sys.exit(f"Error: Invalid start date '{start_str}'. Use YYYY-MM-DD.")
    try:
        end = date.fromisoformat(end_str)
    except ValueError:
        sys.exit(f"Error: Invalid end date '{end_str}'. Use YYYY-MM-DD.")
    if end < start:
        sys.exit("Error: End date must be on or after start date.")
    today = date.today()
    future_limit = today + timedelta(days=MAX_FORECAST_DAYS - 1)
    if start > future_limit:
        sys.exit(
            f"Error: '{start_str}' is beyond the {MAX_FORECAST_DAYS}-day forecast window "
            f"(max: {future_limit}).\n"
        )
    if end > future_limit:
        print(f"[Warning] End date clamped to forecast limit ({future_limit}).")
        end = future_limit
    return start.isoformat(), end.isoformat()


def validate_dates_historical(start_str, end_str, n_years):
    """Validate dates for historical prediction.

    Args:
        start_str: Start date in YYYY-MM-DD format.
        end_str: End date in YYYY-MM-DD format.
        n_years: Number of past years used (for context in error messages).

    Returns:
        A tuple (start_iso, end_iso) of validated ISO date strings.
    """
    try:
        start = date.fromisoformat(start_str)
        end   = date.fromisoformat(end_str)
    except ValueError as e:
        sys.exit(f"Error: {e}. Use YYYY-MM-DD format.")
    if end < start:
        sys.exit("Error: End date must be on or after start date.")
    if (end - start).days + 1 > 60:
        sys.exit("Error: Historical prediction supports up to 60 days per query.")
    if start.year < 1950:
        sys.exit("Error: Historical data only available from 1950 onwards.")
    return start.isoformat(), end.isoformat()


def build_photo_recommendations(
    wardrobe_items: list,
    suitability_scores: dict,
    trip_days: int,
    context: TripContext,
) -> dict:
    """Match detected wardrobe items to recommender items, linking specific photos.

    Groups detected labels from image recognition, filters out non‑clothing
    detections, and for each suitable recommender item determines how many
    of the user's photos match and how many more are needed.

    **New behaviour** – each photo can only be assigned to one recommender
    item (the one with the highest suitability score).  This avoids the same
    garment being recommended for multiple conflicting categories.

    Args:
        wardrobe_items: List of dicts from ``analyse_outfits`` with keys
            ``image_name``, ``detected_label``, ``weight_g``, ``volume_l``, etc.
        suitability_scores: Mapping from item name → comfort score (0‑1).
        trip_days: Trip duration in days.
        context: Trip metadata (purpose, city, country).

    Returns:
        A dict with keys:
          - ``recommended_items``: items found in wardrobe with photo references.
          - ``items_without_photos``: items needed but not in wardrobe.
          - ``total_items_to_pack``: count of items to pack from photos.
          - ``total_items_missing``: count of items missing from wardrobe.
    """
    from collections import defaultdict

    recommended_items = []
    items_without_photos = []
    photos_by_type = defaultdict(list)

    NON_CLOTHING_PATTERN = re.compile(r"person|no detection|unknown|car|dog|cat", re.IGNORECASE)

    # Group wardrobe photos by detected label, skipping non-clothing
    for item in wardrobe_items:
        label = item.get("detected_label", "Unknown")
        if NON_CLOTHING_PATTERN.search(label):
            continue
        photos_by_type[label].append(item)

    # Sort recommender items by score descending so that the most suitable
    # items get first pick of the available photos.
    sorted_scores = sorted(suitability_scores.items(), key=lambda x: x[1], reverse=True)

    # Track which photos have already been assigned to an item
    assigned_photos = set()

    for recommender_item, score in sorted_scores:
        if score < 0.3:
            continue

        quantity_needed = calculate_needed_quantity(recommender_item, trip_days, context.purpose)
        if quantity_needed == 0:
            continue

        # Find matching photos that are not yet assigned to another item
        matching_photos = []
        for detected_label, photos in photos_by_type.items():
            if _is_item_match(detected_label, recommender_item):
                for p in photos:
                    photo_id = p.get("image_name", "")
                    if photo_id not in assigned_photos:
                        matching_photos.append(p)

        if matching_photos:
            available_count = len(matching_photos)
            pack_count = min(available_count, quantity_needed)

            # The photos we actually pack (first `pack_count` matching items)
            packed_photos = matching_photos[:pack_count]

            # Build the photo details for the recommendation
            photos_available = [
                {
                    "image_name": p.get("image_name", f"photo_{i}.jpg"),
                    "detected_label": p.get("detected_label", "Unknown"),
                    "weight_g": p.get("weight_g", p.get("calculated_weight_g", 300)),
                    "volume_l": p.get("volume_l", p.get("calculated_volume_l", 2.0)),
                    "material": p.get("material", "unknown"),
                }
                for i, p in enumerate(packed_photos)
            ]
            photos_to_pack = [p.get("image_name", f"photo_{i}.jpg") for i, p in enumerate(packed_photos)]

            recommended_items.append({
                "item_name": recommender_item,
                "quantity_needed": quantity_needed,
                "photos_available": photos_available,
                "photos_to_pack": photos_to_pack,
                "available_count": available_count,
                "pack_count": pack_count,
                "missing_count": max(0, quantity_needed - available_count),
                "suitability_score": score,
            })

            # Mark the chosen photos as assigned so they can't be reused
            for p in packed_photos:
                assigned_photos.add(p.get("image_name", ""))
        else:
            # No unassigned photos match this item
            items_without_photos.append({
                "item_name": recommender_item,
                "quantity_needed": quantity_needed,
                "suitability_score": score,
                "note": "Not found in wardrobe — consider purchasing",
            })

    return {
        "recommended_items": recommended_items,
        "items_without_photos": items_without_photos,
        "total_items_to_pack": sum(r["pack_count"] for r in recommended_items),
        "total_items_missing": sum(r["missing_count"] for r in recommended_items),
    }


def _is_item_match(detected_label: str, recommender_item: str) -> bool:
    """Check if a YOLO‑detected label matches a recommender item name.

    Args:
        detected_label: Raw label from the detection backend.
        recommender_item: Item name from the 37‑item vocabulary.

    Returns:
        True if the detected label corresponds to the recommender item.
    """
    DEEPFASHION_MAP = {
        "short_sleeved_shirt": ["t-shirt", "short sleeve", "short_sleeved"],
        "long_sleeved_shirt": ["long-sleeve", "long_sleeved", "shirt"],
        "long_sleeved_outwear": ["coat", "winter coat", "heavy", "outwear"],
        "short_sleeved_outwear": ["jacket", "fleece", "light jacket"],
        "trousers": ["jeans", "trouser", "pant"],
        "shorts": ["short", "light trouser"],
        "dress": ["dress", "casual", "smart casual"],
        "vest": ["vest", "jacket", "light"],
        "skirt": ["skirt", "casual"],
        "vest_dress": ["dress", "casual"],
        "long_sleeved_dress": ["dress", "smart", "formal"],
    }

    detected_lower = detected_label.lower().replace("_", " ")
    recommender_lower = recommender_item.lower()

    for df_label, keywords in DEEPFASHION_MAP.items():
        if df_label in detected_lower:
            if any(kw in recommender_lower for kw in keywords):
                return True

    detected_words = set(detected_lower.split())
    recommender_words = set(recommender_lower.split())
    if detected_words & recommender_words:
        return True

    return False


def print_photo_recommendations(photo_recs: dict) -> None:
    """Print a user‑friendly summary of photo‑aware recommendations.

    Args:
        photo_recs: The dict returned by ``build_photo_recommendations``.
    """
    print(f"\n{'='*70}")
    print(f"📸 PHOTO-AWARE PACKING RECOMMENDATIONS")
    print(f"{'='*70}")

    if photo_recs.get("recommended_items"):
        print(f"\n✅ ITEMS TO PACK (with photos):")
        for item in photo_recs["recommended_items"]:
            name = item["item_name"]
            pack = item["pack_count"]
            need = item["quantity_needed"]
            score = item["suitability_score"]

            status = "✅" if pack >= need else "⚠️"
            print(f"\n  {status} {name}")
            print(f"     Needed: {need} | Pack: {pack} | Score: {score:.2f}")

            if item["photos_to_pack"]:
                print(f"     📷 Pack these: {', '.join(item['photos_to_pack'][:5])}")
                if len(item["photos_to_pack"]) > 5:
                    print(f"        ... and {len(item['photos_to_pack']) - 5} more")

            if item["missing_count"] > 0:
                print(f"     ⚠️  Missing: {item['missing_count']} more needed")

    if photo_recs.get("items_without_photos"):
        print(f"\n🛒 ITEMS TO ACQUIRE (not in wardrobe):")
        for item in photo_recs["items_without_photos"]:
            print(f"     • {item['item_name']}: Need {item['quantity_needed']} "
                  f"(score: {item['suitability_score']:.2f})")

    print(f"\n  Total items to pack: {photo_recs.get('total_items_to_pack', 0)}")
    print(f"  Total items missing: {photo_recs.get('total_items_missing', 0)}")


def main():
    """Run the full Travel Weather Advisor pipeline.

    Workflow:
      1. Parse command‑line arguments.
      2. Optionally retrain the recommender model.
      3. Validate dates and geocode the destination city.
      4. Fetch historical weather predictions.
      5. Generate daily clothing recommendations and a master packing list.
      6. (Optional) Analyse wardrobe photos with YOLO/Google Vision.
      7. (Optional) Run the GA → Knapsack → 3D packing optimization.
      8. Export JSON and/or a forecast chart if requested.
      9. Display the results in the terminal.
    """
    args = parse_args()
    load_dotenv()

    # ── Retrain-only mode ──────────────────────────────────────────────────────
    if args.retrain:
        from recommender import train_and_save, MODEL_PATHS
        if args.model == "rules":
            print("'rules' model type needs no training — nothing to do.")
            return
        MODEL_PATHS[args.model].unlink(missing_ok=True)
        train_and_save(verbose=True, model_type=args.model)
        return

    # ── Validate required args for normal run ──────────────────────────────────
    for flag, val in [("--city", args.city), ("--start", args.start),
                      ("--end", args.end), ("--purpose", args.purpose)]:
        if not val:
            sys.exit(f"Error: {flag} is required.")

    # ── Date validation ────────────────────────────────────────────────────
    start_date, end_date = validate_dates_historical(args.start, args.end, args.years)
    trip_days = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1
    print(f"  Trip duration: {trip_days} days")

    # ── Geocoding ──────────────────────────────────────────────────────────────
    print(f"Looking up location for '{args.city}'...")
    try:
        loc = get_location(args.city)
    except (ValueError, ConnectionError) as e:
        sys.exit(f"Error: {e}")
    print(f"Found: {loc['name']}, {loc['country']} ({loc['latitude']:.2f}, {loc['longitude']:.2f})")

    # ── Weather data ───────────────────────────────────────────────────────────
    try:
        print(f"Running historical prediction for {start_date} → {end_date}...")
        forecasts = get_historical_forecast(
            lat=loc["latitude"], lon=loc["longitude"], timezone=loc["timezone"],
            start_date=start_date, end_date=end_date, n_years=args.years,
            method=args.method,
        )

    except (ValueError, ConnectionError) as e:
        sys.exit(f"Error: {e}")

    # ── Recommendations (evaluates items against weather) ──────────────────
    context = TripContext(purpose=args.purpose, city=loc["name"], country=loc["country"])
    recommendations = [recommend_day(f, context, model_type=args.model, trip_days=trip_days) for f in forecasts]
    trip_packing = build_trip_packing_list(recommendations)

    # ── Build master packing list with trip-aware quantities ─────────
    master_quantities = {}
    for item in trip_packing.get("clothing", []):
        master_quantities[item] = calculate_needed_quantity(
            item, trip_days, args.purpose, args.opt_mode
        )
    for item in trip_packing.get("packing", []):
        master_quantities[item] = calculate_needed_quantity(
            item, trip_days, args.purpose, args.opt_mode
        )

    # ── Suitability scores (0-1 for each of the 37 items) ──────────────────
    suitability = {}
    if forecasts:
        suitability = suitability_scores(
            forecast=forecasts[0], context=context,
            model_type=args.model, trip_days=trip_days,
        )

    # ── Image recognition (optional — when --images is provided) ───────────
    wardrobe_items: list = []
    photo_recommendations = None
    outfit_paths = []  

    if args.images is not None and str(args.images).strip():
        outfit_paths = collect_image_paths_from_folder(args.images)
        if outfit_paths:
            print(f"\n📸 Analysing {len(outfit_paths)} wardrobe photos...")
            wardrobe_items = analyse_outfits(outfit_paths, recommendations, context, args.vision, args.depth)

            photo_recommendations = build_photo_recommendations(
                wardrobe_items=wardrobe_items,
                suitability_scores=suitability,
                trip_days=trip_days,
                context=context,
            )
            print_photo_recommendations(photo_recommendations)
    else:
        print(f"\n📋 RECOMMENDED ITEMS (quantities for {trip_days}-day {args.purpose} trip):")
        for item, score in sorted(suitability.items(), key=lambda x: x[1], reverse=True):
            if score >= 0.5:
                qty = calculate_needed_quantity(item, trip_days, args.purpose)
                print(f"     {qty}x {item} (suitability: {score:.2f})")

    # ═══════════════════════════════════════════════════════════════════════
    # OPTIMIZATION — Three paths depending on input
    # ═══════════════════════════════════════════════════════════════════════
    opt_result = None

    if getattr(args, "optimize_items", None):
        print(f"\nRunning standalone optimization on {len(args.optimize_items)} items...")
        opt_result = optimise_items(
            item_names=args.optimize_items, forecasts=forecasts, context=context,
            weight_limit_kg=args.weight_limit, optimization_mode=args.opt_mode,
            bin_dims=tuple(args.luggage_dims), reserve_space_percent=args.reserve_space,
            multi_bag=args.multi_bag,
        )
    
    elif args.optimize:
    # ═══════════════════════════════════════════════════════════════
    # OPTIMIZATION – with or without wardrobe photos
    # ═══════════════════════════════════════════════════════════════

        # Helper to get volume from static dimensions (used when no photos)
        def _get_static_vol(item_name):
            dims = ITEM_DIMS_CM.get(item_name, DEFAULT_DIMS)
            return (dims[0] * dims[1] * dims[2]) / 1000.0

        if wardrobe_items:
            # ── Photos available → pass every photo to the optimizer ──────────
            # Build photo_recommendations if not already done
            if photo_recommendations is None:
                photo_recommendations = build_photo_recommendations(
                    wardrobe_items=wardrobe_items,
                    suitability_scores=suitability,
                    trip_days=trip_days,
                    context=context,
                )

            # Additional recommended items (anything from the recommender that
            # has suitability ≥ 0.5 and isn't already covered by photos)
            all_recommended = list(trip_packing.get("clothing", [])) + list(trip_packing.get("packing", []))
            ml_extra = [item for item in all_recommended if suitability.get(item, 0) >= 0.5]

            opt_result = optimise_dynamic_items(
                wardrobe_items=wardrobe_items,
                ml_recommended_items=ml_extra,
                weight_limit_kg=args.weight_limit,
                forecasts=forecasts,
                context=context,
                suitability_scores=suitability,
                optimization_mode=args.opt_mode,
                bin_dims=tuple(args.luggage_dims),
                reserve_space_percent=args.reserve_space,
                trip_days=trip_days,
                multi_bag=args.multi_bag,
            )
        else:
            # ── No photos → build candidates from static packing list ────────
            master_items = list(trip_packing.get("clothing", [])) + list(trip_packing.get("packing", []))
            suitable_master_items = [item for item in master_items if suitability.get(item, 0) >= 0.5]

            # Create one candidate dict per needed copy, each with static weight/volume
            static_candidates = []
            for item_name in suitable_master_items:
                qty = calculate_needed_quantity(item_name, trip_days, args.purpose)
                for i in range(qty):
                    static_candidates.append({
                        "image_name":      f"{item_name}_copy_{i+1}",
                        "detected_label":  item_name,
                        "weight_g":        ITEM_WEIGHTS.get(item_name, DEFAULT_WEIGHT) * 1000,
                        "volume_l":        _get_static_vol(item_name),
                    })

            opt_result = optimise_dynamic_items(
                wardrobe_items=static_candidates,
                ml_recommended_items=suitable_master_items,
                weight_limit_kg=args.weight_limit,
                forecasts=forecasts,
                context=context,
                suitability_scores=suitability,
                optimization_mode=args.opt_mode,
                bin_dims=tuple(args.luggage_dims),
                reserve_space_percent=args.reserve_space,
                trip_days=trip_days,
                multi_bag=args.multi_bag,
            )

        # ── Copy selected wardrobe photos to PackPalOut ─────────────────
        if outfit_paths and opt_result is not None and opt_result.final_items:
            out_dir = Path("PackPalOut")
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir()
                
            name_to_path = {Path(p).name: p for p in outfit_paths}
                
            copied = 0
            for item_id in opt_result.final_items:
                src = name_to_path.get(item_id)
                if src:
                    shutil.copy2(src, out_dir / item_id)
                    copied += 1
            print(f"📁 Copied {str(copied)} selected garment photos to {out_dir}/")    

    # ── JSON export ────────────────────────────────────────────────────────
    if args.json:
        payload = {
            "context": asdict(context), "start_date": start_date,
            "end_date": end_date, "trip_days": trip_days,
            "method": args.method, "model": args.model,
            "forecasts": [asdict(f) for f in forecasts],
            "recommendations": [asdict(r) for r in recommendations],
            "trip_packing": trip_packing, "suitability_scores": suitability,
            "wardrobe_items": wardrobe_items,
            "photo_recommendations": photo_recommendations,
        }
        if opt_result is not None:
            payload["optimization"] = opt_result.to_dict()
        json_path = Path(__file__).parent / "output.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nJSON saved → {json_path}")

    # ── Display ────────────────────────────────────────────────────────────
    display(context, start_date, end_date, recommendations, trip_packing,
            n_years=args.years, optimization_result=opt_result,
            master_quantities=master_quantities)

    # ── Export GUI-friendly JSON ─────────────────────────────────────────
    if args.optimize and opt_result is not None:
        gui_path = save_gui_json(
            optimization_result=opt_result,
            trip_packing=trip_packing,
            photo_recommendations=photo_recommendations if photo_recommendations else None,
        )
        print(f"GUI-ready output saved → {gui_path}")

    # ── Chart ──────────────────────────────────────────────────────────────
    if args.chart:
        chart_path = plot_forecast(forecasts, context, start_date, end_date)
        print(f"\nChart saved → {chart_path}")


if __name__ == "__main__":
    main()