# Smart Travel Packing — Travel Weather Advisor

A CLI tool that recommends what to wear and pack for a trip based on weather forecasts or historical climate data. It uses a LightGBM classifier to generate per-day clothing and packing lists, and optionally analyses your wardrobe photos with Google Gemini.

## How it works

1. **Geocoding** — resolves the destination city to coordinates via Open-Meteo.
2. **Weather data** — two modes:
   - `forecast`: live 16-day forecast from Open-Meteo (default).
   - `historical`: fetches the same calendar window from the past N years of archive data, then predicts with exponentially weighted averaging + OLS linear trend.
3. **Recommendation** — a LightGBM multi-label classifier (37 items across clothing and packing) trained on 15,000 synthetic weather samples. Auto-trains on first run and is cached to `model/recommender.joblib`.
4. **Display** — rich terminal output with a per-day table, master packing list, and day-by-day alerts. Falls back to plain text if `rich` is not installed.
5. **Photo suggestions** *(optional)* — sends clothing narratives + images from `img/` to Google Gemini 2.5 Flash for luggage advice.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+

## Usage

```bash
# Live forecast (up to 16 days ahead)
python main.py --city "Singapore" --start 2026-03-15 --end 2026-03-20 --purpose business

# Historical climate prediction (any future date, no 16-day limit)
python main.py --city "Tokyo" --start 2026-07-10 --end 2026-07-15 --purpose tourism

# Save a matplotlib forecast chart (PNG)
python main.py --city "Paris, France" --start 2026-05-01 --end 2026-05-07 --purpose visiting --chart

# Use more years of archive data (default: 10)
python main.py --city "New York" --start 2026-08-01 --end 2026-08-05 --purpose tourism --years 15

# Force retrain the recommendation model
python main.py --retrain

# Run with trigger Optimization 
python main.py --city "Singapore" --start 2026-05-01 --end 2026-05-05 --purpose visiting --model knn --optimize
```

### Arguments

| Argument | Description |
|---|---|
| `--city` | Destination city, e.g. `"Tokyo"` or `"Paris, France"` |
| `--start` | Travel start date (`YYYY-MM-DD`) |
| `--end` | Travel end date (`YYYY-MM-DD`) |
| `--purpose` | `business`, `tourism`, or `visiting` |
| `--years` | Years of archive data for historical mode (default: 10) |
| `--chart` | Generate and save a matplotlib forecast chart (PNG) |
| `--retrain` | Retrain the LightGBM model and exit |

## Project structure

```
main.py               Entry point and argument parsing
geocoder.py           City name → lat/lon/timezone (Open-Meteo geocoding API)
weather.py            Live weather forecast (Open-Meteo forecast/archive API)
historical_forecast.py Historical climate prediction with parquet cache
recommender.py        LightGBM multi-label classifier (clothing + packing)
models.py             Data classes: DayForecast, TripContext, DayRecommendation
display.py            Terminal output (rich) and matplotlib chart generation
requirements.txt      Python dependencies
model/                Trained model cache (auto-generated)
cache/                Parquet cache of historical API responses
img/                  Wardrobe photos for Gemini photo suggestions
```

## Configuration

**Google Gemini API key** — required for the photo suggestion feature. Set your key in `display.py` (the `api_key` parameter in the `genai.Client` call).

## APIs used

- [Open-Meteo Geocoding API](https://open-meteo.com/en/docs/geocoding-api) — free, no key required
- [Open-Meteo Forecast API](https://open-meteo.com/en/docs) — free, no key required
- [Open-Meteo Historical Archive API](https://open-meteo.com/en/docs/historical-weather-api) — free, no key required
- [Google Gemini API](https://ai.google.dev/) — requires API key (for photo suggestions)
