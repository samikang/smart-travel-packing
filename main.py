#!/usr/bin/env python3
"""
Travel Weather Advisor
======================
Usage:
    # Live forecast (up to 16 days ahead):
    python main.py --city "Singapore" --start 2026-03-15 --end 2026-03-20 --purpose business

    # Current-conditions prediction (anomaly + decay):
    python main.py --city "Tokyo" --start 2026-04-10 --end 2026-04-15 --method current

    # Long-range historical prediction (any future date):
    python main.py --city "Tokyo" --start 2026-07-10 --end 2026-07-15 --purpose tourism --method historical

    # Save a forecast chart:
    python main.py --city "Paris, France" --start 2026-05-01 --end 2026-05-07 --chart

    # Analyse wardrobe photos against the forecast:
    python main.py --city "Singapore" --start 2026-05-01 --end 2026-05-05 --images img/outfit1.jpg img/outfit2.jpg

    # Retrain the recommendation model:
    python main.py --retrain
"""

import argparse
import sys
from datetime import date, timedelta

from geocoder import get_location
from weather import get_forecast
from historical_forecast import get_historical_forecast
from current_forecast import get_current_forecast
from recommender import recommend_day, build_trip_packing_list
from models import TripContext
from display import display, plot_forecast
from image_recognition import analyse_outfits

VALID_PURPOSES   = ("business", "tourism", "visiting")
VALID_METHODS    = ("forecast", "current", "historical")
MAX_FORECAST_DAYS = 16


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Travel Weather Advisor — clothing & packing recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--city",    help="Destination city, e.g. 'Tokyo' or 'Paris, France'")
    parser.add_argument("--start",   help="Travel start date (YYYY-MM-DD)")
    parser.add_argument("--end",     help="Travel end date (YYYY-MM-DD)")
    parser.add_argument("--purpose", choices=VALID_PURPOSES, default="tourism",
                        help="Trip purpose: business, tourism, or visiting (default: tourism)")
    parser.add_argument("--method",  choices=VALID_METHODS, default="forecast",
                        help="Forecast method: forecast (default), current, or historical")
    parser.add_argument("--years",   type=int, default=10, metavar="N",
                        help="[historical] years of archive data to use (default: 10)")
    parser.add_argument("--chart",   action="store_true",
                        help="Generate and save a matplotlib forecast chart (PNG)")
    parser.add_argument("--model",   default="lgbm",
                        choices=["lgbm", "random_forest", "knn", "rules"],
                        help="Recommendation algorithm: lgbm (default), random_forest, knn, rules")
    parser.add_argument("--images",  nargs="+", metavar="PATH",
                        help="Wardrobe photo paths to analyse against the forecast")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain the recommendation model (respects --model)")
    return parser.parse_args()


# ── Date validation ───────────────────────────────────────────────────────────

def validate_dates(start_str: str, end_str: str, method: str, n_years: int):
    try:
        start = date.fromisoformat(start_str)
        end   = date.fromisoformat(end_str)
    except ValueError as e:
        sys.exit(f"Error: {e}. Use YYYY-MM-DD format.")

    if end < start:
        sys.exit("Error: End date must be on or after start date.")

    if method == "forecast":
        today = date.today()
        future_limit = today + timedelta(days=MAX_FORECAST_DAYS - 1)
        if start > future_limit:
            sys.exit(
                f"Error: '{start_str}' is beyond the {MAX_FORECAST_DAYS}-day forecast window "
                f"(max: {future_limit}). Use --method historical for long-range trips."
            )
        if end > future_limit:
            print(f"[Warning] End date clamped to forecast limit ({future_limit}).")
            end = future_limit

    if method == "historical":
        if (end - start).days + 1 > 60:
            sys.exit("Error: Historical prediction supports up to 60 days per query.")
        if start.year < 1950:
            sys.exit("Error: Historical data only available from 1950 onwards.")

    return start.isoformat(), end.isoformat()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Retrain-only mode ─────────────────────────────────────────────────────
    if args.retrain:
        from recommender import train_and_save, MODEL_PATHS
        if args.model == "rules":
            print("'rules' model type needs no training — nothing to do.")
            return
        MODEL_PATHS[args.model].unlink(missing_ok=True)
        train_and_save(verbose=True, model_type=args.model)
        return

    # ── Validate required args ────────────────────────────────────────────────
    for flag, val in [("--city", args.city), ("--start", args.start), ("--end", args.end)]:
        if not val:
            sys.exit(f"Error: {flag} is required.")

    start_date, end_date = validate_dates(args.start, args.end, args.method, args.years)

    # ── Geocoding ─────────────────────────────────────────────────────────────
    print(f"Looking up location for '{args.city}'...")
    try:
        loc = get_location(args.city)
    except (ValueError, ConnectionError) as e:
        sys.exit(f"Error: {e}")
    print(f"Found: {loc['name']}, {loc['country']} "
          f"({loc['latitude']:.2f}, {loc['longitude']:.2f})")

    # ── Weather data ──────────────────────────────────────────────────────────
    try:
        if args.method == "historical":
            print(f"Running historical prediction for {start_date} → {end_date}...")
            forecasts = get_historical_forecast(
                lat=loc["latitude"], lon=loc["longitude"], timezone=loc["timezone"],
                start_date=start_date, end_date=end_date, n_years=args.years,
            )
        elif args.method == "current":
            print(f"Running current-conditions prediction for {start_date} → {end_date}...")
            forecasts = get_current_forecast(
                lat=loc["latitude"], lon=loc["longitude"], timezone=loc["timezone"],
                start_date=start_date, end_date=end_date,
            )
        else:
            print(f"Fetching live forecast for {start_date} → {end_date}...")
            forecasts = get_forecast(
                lat=loc["latitude"], lon=loc["longitude"], timezone=loc["timezone"],
                start_date=start_date, end_date=end_date,
            )
    except (ValueError, ConnectionError) as e:
        sys.exit(f"Error: {e}")

    # ── Recommendations ───────────────────────────────────────────────────────
    context         = TripContext(city=loc["name"], country=loc["country"], purpose=args.purpose)
    recommendations = [recommend_day(f, context, model_type=args.model) for f in forecasts]
    trip_packing    = build_trip_packing_list(recommendations)

    # ── Display ───────────────────────────────────────────────────────────────
    display(
        forecasts=forecasts,
        context=context,
        start_date=start_date,
        end_date=end_date,
        method=args.method,
        n_years=args.years,
        recommendations=recommendations,
        trip_packing=trip_packing,
    )

    # ── Image recognition ─────────────────────────────────────────────────────
    if args.images:
        analyse_outfits(args.images, recommendations, context)

    # ── Chart ─────────────────────────────────────────────────────────────────
    if args.chart:
        chart_path = plot_forecast(forecasts, context, start_date, end_date,
                                   method=args.method, n_years=args.years)
        print(f"\nChart saved → {chart_path}")


if __name__ == "__main__":
    main()
