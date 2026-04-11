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

import argparse
import sys
from datetime import date, timedelta

from geocoder import get_location
from weather import get_forecast
from historical_forecast import get_historical_forecast
from recommender import recommend_day, build_trip_packing_list
from models import TripContext
from display import display, plot_forecast
import subprocess
import sys
import threading
import base64
import requests
from image_recognition import analyse_outfits, collect_image_paths_from_folder
from packing_optimizer import optimise_from_recommendations, optimise_items
import os


VALID_PURPOSES = ("business", "tourism", "visiting")
MAX_FORECAST_DAYS = 16


def parse_args():
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
                        choices=["yolo", "google", "clip", "both"],
                        help="Image recognition backend: yolo (default), google, clip, both")
    '''
    parser.add_argument("--optimize", action="store_true",
                        help="Run GA → Knapsack → Summary optimization pipeline")
    parser.add_argument("--weight-limit", type=float, default=20.0, metavar="KG",
                        help="Baggage weight limit in kg for optimization (default: 20.0)")
    parser.add_argument("--optimize-items", nargs="+", metavar="ITEM",
                        help="Standalone optimization: provide item names directly "
                             "(skips recommender; use with --weight-limit)")   
                             ''' 
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain the recommendation model (respects --model)")
    return parser.parse_args()


def validate_dates_forecast(start_str, end_str):
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


def main():
    args = parse_args()

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

    # ── Date validation ────────────────────────────────────────────────────────
    start_date, end_date = validate_dates_historical(args.start, args.end, args.years)

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

    # ── Recommendations ────────────────────────────────────────────────────────
    context = TripContext(purpose=args.purpose, city=loc["name"], country=loc["country"])
    recommendations = [recommend_day(f, context, model_type=args.model) for f in forecasts]
    trip_packing    = build_trip_packing_list(recommendations)


    # ── Image recognition (optional — only when --images is passed) ─────────────
    if args.images is not None and str(args.images).strip():
        outfit_paths = collect_image_paths_from_folder(args.images)
        if outfit_paths:
            analyse_outfits(outfit_paths, recommendations, context, args.vision)



    # ── Display ────────────────────────────────────────────────────────────────
    display(context, start_date, end_date, recommendations, trip_packing, n_years=args.years)

    # ── Chart ──────────────────────────────────────────────────────────────────
    if args.chart:
        chart_path = plot_forecast(forecasts, context, start_date, end_date)
        print(f"\nChart saved → {chart_path}")


if __name__ == "__main__":
    main()
