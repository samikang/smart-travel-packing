"""
Historical Climate-Based Weather Predictor
==========================================
Refactored to use:
  - pandas  : store multi-year observations as DataFrames
  - numpy   : weighted mean (np.average), OLS trend (np.polyfit), std
  - pyarrow : cache each year's API response as Parquet on disk;
              subsequent runs load from cache instead of re-fetching

Algorithm (unchanged):
  1. Fetch same calendar window from past N years (archive API / parquet cache)
  2. Exponentially weighted mean — recent years weighted higher (α = 0.85)
  3. OLS linear trend via np.polyfit — extrapolated to target year
  4. Adaptive blend: trust trend less when variability >> trend signal
  5. UV proxy when satellite records are sparse (< 3 valid years)
  6. Weather code: exponentially weighted mode (categorical)
  7. Physical constraints clamp all values to valid ranges
"""

import urllib.request
import urllib.parse
import json
from datetime import date, timedelta
from calendar import isleap
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401 — ensures parquet engine is available

from models import DayForecast

CACHE_DIR = Path(__file__).parent / "cache"
ALPHA = 0.85
TREND_WEIGHT = 0.25
MIN_OBS_FOR_TREND = 3

DAILY_VARS = [
    "uv_index_max",
    "cloud_cover_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "weather_code",
]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(lat: float, lon: float, year: int, mmdd_start: str, mmdd_end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{lat:.2f}_{lon:.2f}_{year}_{mmdd_start}_{mmdd_end}.parquet"


# ── Date helpers ──────────────────────────────────────────────────────────────

def _shift_year(d: date, target_year: int) -> date:
    if d.month == 2 and d.day == 29 and not isleap(target_year):
        return date(target_year, 2, 28)
    return date(target_year, d.month, d.day)


# ── API fetch ─────────────────────────────────────────────────────────────────

def _fetch_from_api(lat: float, lon: float, timezone: str,
                    start: str, end: str) -> pd.DataFrame:
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "timezone": timezone,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS),
    })
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read())
    daily = data.get("daily", {})
    if not daily.get("time"):
        return pd.DataFrame()
    return pd.DataFrame(daily).rename(columns={
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum":  "precipitation_mm",
        "wind_speed_10m_max": "wind_speed_max",
    })


def _fetch_year(lat: float, lon: float, timezone: str,
                hist_start: date, hist_end: date,
                target_start: date, target_end: date) -> pd.DataFrame:
    """Return DataFrame for one historical year, using parquet cache when available."""
    mmdd_start = target_start.strftime("%m%d")
    mmdd_end   = target_end.strftime("%m%d")
    cache_file = _cache_path(lat, lon, hist_start.year, mmdd_start, mmdd_end)

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    try:
        df = _fetch_from_api(lat, lon, timezone,
                             hist_start.isoformat(), hist_end.isoformat())
        if not df.empty:
            df.to_parquet(cache_file, index=False)
        return df
    except Exception:
        return pd.DataFrame()


# ── Data collection ───────────────────────────────────────────────────────────

def _collect(lat: float, lon: float, timezone: str,
             target_start: date, target_end: date,
             n_years: int) -> pd.DataFrame:
    """
    Fetch same calendar window from past n_years.
    Returns a DataFrame with columns:
      year, day_offset, temp_max, temp_min, precipitation_mm,
      wind_speed_max, uv_index_max, cloud_cover_mean, weather_code
    Rows are sorted by (day_offset, year desc).
    """
    target_year = target_start.year
    n_days = (target_end - target_start).days + 1
    frames = []

    for offset in range(1, n_years + 1):
        hist_year = target_year - offset
        if hist_year < 1940:
            break
        hist_start = _shift_year(target_start, hist_year)
        hist_end   = _shift_year(target_end,   hist_year)
        df = _fetch_year(lat, lon, timezone, hist_start, hist_end,
                         target_start, target_end)
        if df.empty or len(df) < n_days:
            continue
        df = df.head(n_days).copy()
        df["year"]       = hist_year
        df["day_offset"] = range(n_days)
        frames.append(df)

    if not frames:
        raise ValueError("Could not retrieve any historical data for this location.")

    all_df = pd.concat(frames, ignore_index=True)
    years_fetched = sorted(all_df["year"].unique())
    print(f"  [Historical] Used {len(years_fetched)} years of data "
          f"({years_fetched[0]}–{years_fetched[-1]})")
    return all_df


# ── Prediction helpers (numpy-based) ─────────────────────────────────────────

def _exp_weights(n: int) -> np.ndarray:
    """Normalised exponential decay; index 0 = most recent year."""
    w = ALPHA ** np.arange(n, dtype=float)
    return w / w.sum()


def _predict_continuous(years: np.ndarray, values: np.ndarray,
                        target_year: int) -> float:
    """
    Exponentially weighted mean + OLS linear trend, adaptively blended.
    years/values must be sorted most-recent-first.
    """
    mask = ~np.isnan(values)
    if not mask.any():
        return 0.0
    y, v = years[mask], values[mask]
    n = len(v)

    w = _exp_weights(n)
    mu_w    = np.average(v, weights=w)
    sigma_w = np.sqrt(np.average((v - mu_w) ** 2, weights=w))

    if n >= MIN_OBS_FOR_TREND:
        # np.polyfit returns [slope, intercept] for degree-1 fit
        slope, intercept = np.polyfit(y.astype(float), v, 1)
        std_v = np.std(v)
        slope = np.clip(slope, -2 * std_v, 2 * std_v)
        x_trend = intercept + slope * target_year

        trend_signal  = abs(slope * 5)
        trend_reliability = trend_signal / (trend_signal + sigma_w + 1e-9)
        eff_tw = TREND_WEIGHT * min(1.0, trend_reliability)
    else:
        x_trend = mu_w
        eff_tw  = 0.0

    return float((1.0 - eff_tw) * mu_w + eff_tw * x_trend)


def _predict_code(years: np.ndarray, codes: np.ndarray) -> int:
    """Exponentially weighted mode — calmer code wins ties."""
    mask = ~np.isnan(codes)
    if not mask.any():
        return 0
    y, c = years[mask], codes[mask].astype(int)
    w = _exp_weights(len(c))
    score: dict = {}
    for wi, ci in zip(w, c):
        score[ci] = score.get(ci, 0.0) + wi
    return min(score, key=lambda x: (-score[x], x))


def _uv_proxy(lat: float, cloud: float) -> float:
    base = max(0.0, 12.0 - abs(lat) * 0.15)
    return base * (1.0 - 0.7 * (cloud / 100.0))


# ── Public API ────────────────────────────────────────────────────────────────

def get_historical_forecast(lat: float, lon: float, timezone: str,
                             start_date: str, end_date: str,
                             n_years: int = 10) -> list:
    """
    Predict daily weather for a future date range from past climate data.
    Uses parquet cache to avoid re-fetching on repeated runs.
    Returns list[DayForecast].
    """
    target_start = date.fromisoformat(start_date)
    target_end   = date.fromisoformat(end_date)
    target_year  = target_start.year
    n_days       = (target_end - target_start).days + 1

    print(f"  [Historical] Fetching up to {n_years} years of archive data...")
    df = _collect(lat, lon, timezone, target_start, target_end, n_years)

    forecasts = []
    for day_i in range(n_days):
        day_df = (
            df[df["day_offset"] == day_i]
            .sort_values("year", ascending=False)
            .reset_index(drop=True)
        )
        years = day_df["year"].to_numpy(dtype=float)

        def col(name):
            return day_df[name].to_numpy(dtype=float) if name in day_df else np.array([np.nan])

        temp_max = _predict_continuous(years, col("temp_max"),          target_year)
        temp_min = _predict_continuous(years, col("temp_min"),          target_year)
        cloud    = _predict_continuous(years, col("cloud_cover_mean"),   target_year)
        precip   = _predict_continuous(years, col("precipitation_mm"),  target_year)
        wind     = _predict_continuous(years, col("wind_speed_max"),     target_year)
        code     = _predict_code(years,       col("weather_code"))

        uv_vals = col("uv_index_max")
        valid_uv = uv_vals[~np.isnan(uv_vals)]
        if len(valid_uv) >= MIN_OBS_FOR_TREND:
            uv = _predict_continuous(years, uv_vals, target_year)
        else:
            uv = _uv_proxy(lat, cloud)

        # Physical constraints
        temp_max = float(np.clip(temp_max, -80, 60))
        temp_min = float(np.clip(temp_min, -80, 60))
        if temp_max < temp_min + 0.5:
            temp_max = temp_min + 0.5
        cloud  = float(np.clip(cloud,  0, 100))
        precip = float(max(0.0, precip))
        wind   = float(max(0.0, wind))
        uv     = float(np.clip(uv, 0, 16))

        current_date = (target_start + timedelta(days=day_i)).isoformat()
        forecasts.append(DayForecast(
            date=current_date,
            uv_index_max=round(uv, 1),
            cloud_cover_mean=round(cloud, 1),
            temp_min=round(temp_min, 1),
            temp_max=round(temp_max, 1),
            precipitation_mm=round(precip, 1),
            wind_speed_max=round(wind, 1),
            weather_code=int(code),
        ))

    return forecasts
