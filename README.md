# Smart Travel Packing — Travel Weather Advisor

A CLI tool that forecasts weather for any city and date range, recommends what to wear and pack, and optionally analyses your wardrobe photos to advise which outfits to bring.

---

## Features

- Look up any city worldwide by name (automatic geocoding)
- **Three forecast methods**: live API, current-conditions prediction, or historical climate
- **LightGBM recommender** — per-day clothing & packing list (37 items, trained on 15 000 synthetic samples)
- **Image recognition** — analyse wardrobe photos with YOLOv8 + optional Google Gemini advice
- Per-day weather table: condition, temperature, precipitation, UV, wind, cloud cover
- Colour-coded terminal output (red/yellow highlights for high-risk values)
- **Matplotlib** 3-panel forecast chart saved as PNG
- **Parquet cache** — historical data cached on disk, repeated queries load instantly

---

## Requirements

- Python 3.9+
- Internet connection (for weather data)

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `rich` | Colour terminal output |
| `numpy` | Weighted mean, OLS trend, array operations |
| `pandas` | DataFrame storage for multi-year historical data |
| `lightgbm` / `scikit-learn` / `joblib` | Clothing & packing recommender |
| `matplotlib` | 3-panel forecast chart |
| `pyarrow` | Parquet cache for historical API responses |
| `ultralytics` | YOLOv8 local garment detection *(optional)* |
| `Pillow` | Image loading for wardrobe analysis *(optional)* |
| `google-genai` | Gemini AI narrative packing advice *(optional)* |

No API keys required for core weather features. All weather data is from [Open-Meteo](https://open-meteo.com/) (free, no key).

---

## Usage

```bash
python main.py --city CITY --start YYYY-MM-DD --end YYYY-MM-DD [options]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--city` | Yes | Destination city, e.g. `"Tokyo"` or `"Paris, France"` |
| `--start` | Yes | Travel start date (`YYYY-MM-DD`) |
| `--end` | Yes | Travel end date (`YYYY-MM-DD`) |
| `--purpose` | No | `business`, `tourism`, or `visiting` (default: `tourism`) |
| `--method` | No | `forecast` (default), `current`, or `historical` |
| `--years` | No | Years of archive data (default: `10`, historical mode only) |
| `--chart` | No | Save a 3-panel matplotlib forecast chart as PNG |
| `--model` | No | Recommender: `lgbm` (default), `random_forest`, `knn`, `rules` |
| `--images` | No | Paths to wardrobe photos to analyse |
| `--retrain` | No | Retrain the recommendation model and exit |

### Examples

```bash
# Live forecast + clothing recommendations
python main.py --city "Singapore" --start 2026-05-01 --end 2026-05-07 --purpose business

# Current-conditions prediction
python main.py --city "Tokyo" --start 2026-04-10 --end 2026-04-15 --method current

# Long-range historical prediction + chart
python main.py --city "Paris, France" --start 2026-08-01 --end 2026-08-07 --method historical --chart

# Analyse wardrobe photos against the forecast
python main.py --city "Tokyo" --start 2026-07-01 --end 2026-07-05 \
    --images img/outfit1.jpg img/outfit2.jpg

# Retrain the recommender model
python main.py --retrain
```

---

## Forecast Methods

### `forecast` (default)
Live NWP output from the **Open-Meteo forecast API**. Most accurate for near-future trips up to **16 days ahead**.

### `current`
**Anomaly-persistence prediction** — uses the last 14 days of observed weather to detect current anomalies, decays them toward the historical baseline using `exp(−days/τ)` with τ = 4 days. Good for 1–10 day near-future trips.

### `historical`
**Exponentially weighted climatology + OLS trend** from the same calendar window in past N years. No 16-day limit — suitable for any future date. Historical API responses are cached as Parquet files (`cache/`).

---

## Image Recognition (`--images`)

Two-stage pipeline:

1. **Local detection (YOLOv8, offline)** — detects the garment category from each photo using `yolov8n-cls.pt` (classification) or `yolov8n.pt` (detection). Place model files in the project root. Falls back to filename-based description if `ultralytics` is not installed.

2. **Gemini AI advice (optional)** — sends the detected garment label + full trip clothing narrative to Google Gemini 2.5 Flash for concise packing advice. Requires `GEMINI_API_KEY` environment variable. Falls back to rule-based advice if the key is absent.

```bash
# Set API key (optional — rule-based fallback used if not set)
export GEMINI_API_KEY="your-key-here"

python main.py --city "London" --start 2026-06-01 --end 2026-06-05 \
    --images wardrobe/raincoat.jpg wardrobe/tshirt.jpg wardrobe/suit.jpg
```

Place YOLOv8 model files in the project root:
- `yolov8n-cls.pt` — classification model (preferred for clothing)
- `yolov8n.pt` — detection model (fallback)

---

## Project Structure

```
main.py                Entry point and argument parsing
geocoder.py            City name → lat/lon/timezone (Open-Meteo geocoding API)
weather.py             Live weather forecast (Open-Meteo forecast API)
historical_forecast.py Historical climate prediction with Parquet cache
current_forecast.py    Current-conditions prediction (anomaly + decay)
recommender.py         LightGBM multi-label classifier (clothing + packing)
image_recognition.py   YOLOv8 garment detection + Gemini packing advice
models.py              Data classes: DayForecast, TripContext, DayRecommendation
display.py             Terminal output (rich) and matplotlib chart
requirements.txt       Python dependencies
model/                 Trained recommender model cache (auto-generated)
cache/                 Parquet cache of historical API responses
yolov8n-cls.pt         YOLOv8 classification weights (place in project root)
yolov8n.pt             YOLOv8 detection weights (place in project root)
```

---

## Data Sources

| Source | Usage | Cost |
|---|---|---|
| [Open-Meteo Forecast API](https://api.open-meteo.com) | Live 16-day forecast | Free, no key |
| [Open-Meteo Archive API](https://archive-api.open-meteo.com) | Historical weather (1940–present) | Free, no key |
| [Open-Meteo Geocoding API](https://geocoding-api.open-meteo.com) | City name → coordinates | Free, no key |
| [Google Gemini API](https://ai.google.dev/) | AI wardrobe advice | Free tier, key required |

---

## Limitations

- `forecast` is limited to **16 days ahead**
- `current` requires archive data up to ~2 days ago; very recent observations may occasionally be unavailable
- `historical` accuracy decreases for unusual weather years — outliers are smoothed by the weighted mean
- UV data in the historical archive is sparse before ~2019; a latitude/cloud proxy is used as fallback
- YOLOv8 classification uses ImageNet classes — clothing detection works best with clear, well-lit photos on plain backgrounds
