"""
Current-Conditions Forecast — Anomaly Persistence with Exponential Decay
=========================================================================
Uses real recent observed weather (last 14 days) to predict near-future
weather without relying on Open-Meteo's forecast model.

Algorithm:  predicted[d] = clim_target[d]  +  anomaly × exp(−days_ahead / τ)

Where:
  clim_target[d]  = weighted historical mean for the target calendar date
                    (same month/day, past N years) — the "normal" for that day
  anomaly         = recent_mean − clim_current
                    how much recent weather deviates from its own "normal"
  τ (TAU)         = 4 days — anomalies decay to 1/e (~37%) after 4 days,
                    ~2% after 16 days — representing persistence fading over time

Additionally, a short-term linear trend from the last 7 days is extracted
with numpy (np.polyfit) and applied for days 1–3, tapering off after that.

Data fetched:
  1. Recent 14-day observed window    → archive API (2..15 days ago)
  2. Historical recent window (5yr)   → archive API, same 14-day window past years
  3. Historical target window (5yr)   → archive API, target dates past years
     (all cached as Parquet to avoid repeat fetches)

Returns list[DayForecast] — identical interface to weather.get_forecast().
"""

import urllib.request
import urllib.parse
import json
from datetime import date, timedelta
from calendar import isleap
from pathlib import Path

import numpy as np
import pandas as pd

from models import DayForecast

CACHE_DIR = Path(__file__).parent / "cache"
ALPHA     = 0.85          # exponential decay weight for historical years
TAU       = 4.0           # anomaly persistence e-folding time (days)
N_CLIM_YEARS  = 5         # years used for climatology baseline
RECENT_DAYS   = 14        # days of recent observed data to fetch
TREND_DAYS    = 7         # last N days used to compute short-term trend
TREND_HORIZON = 3         # short-term trend applied only to first N forecast days

DAILY_VARS = [
    "uv_index_max", "cloud_cover_mean",
    "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "wind_speed_10m_max",
    "weather_code",
]
RENAME = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "precipitation_sum":  "precipitation_mm",
    "wind_speed_10m_max": "wind_speed_max",
}


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_path(lat: float, lon: float, year: int,
                mmdd_start: str, mmdd_end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{lat:.2f}_{lon:.2f}_{year}_{mmdd_start}_{mmdd_end}.parquet"


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _fetch_archive(lat: float, lon: float, timezone: str,
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
    return pd.DataFrame(daily).rename(columns=RENAME)


def _fetch_with_cache(lat: float, lon: float, timezone: str,
                      start: date, end: date) -> pd.DataFrame:
    """Fetch archive window, using parquet cache when available."""
    mmdd_s = start.strftime("%m%d")
    mmdd_e = end.strftime("%m%d")
    cp = _cache_path(lat, lon, start.year, mmdd_s, mmdd_e)
    if cp.exists():
        return pd.read_parquet(cp)
    try:
        df = _fetch_archive(lat, lon, timezone, start.isoformat(), end.isoformat())
        if not df.empty:
            df.to_parquet(cp, index=False)
        return df
    except Exception:
        return pd.DataFrame()


def _shift_year(d: date, year: int) -> date:
    if d.month == 2 and d.day == 29 and not isleap(year):
        return date(year, 2, 28)
    return date(year, d.month, d.day)


# ── Step 1: Recent observed data ──────────────────────────────────────────────

def _fetch_recent(lat: float, lon: float, timezone: str) -> pd.DataFrame:
    """
    Fetch the last RECENT_DAYS of actually observed weather.
    Archive API has ~2-day lag so we fetch ending 2 days ago.
    """
    today      = date.today()
    obs_end    = today - timedelta(days=2)
    obs_start  = obs_end - timedelta(days=RECENT_DAYS - 1)
    df = _fetch_archive(lat, lon, timezone, obs_start.isoformat(), obs_end.isoformat())
    return df


# ── Step 2: Historical climatology windows ────────────────────────────────────

def _fetch_climatology(lat: float, lon: float, timezone: str,
                       ref_start: date, ref_end: date,
                       n_years: int) -> pd.DataFrame:
    """
    Fetch same calendar window from past n_years.
    Returns concatenated DataFrame with 'year' and 'day_offset' columns.
    """
    target_year = ref_start.year
    n_days = (ref_end - ref_start).days + 1
    frames = []
    for offset in range(1, n_years + 1):
        yr = target_year - offset
        if yr < 1940:
            break
        ys = _shift_year(ref_start, yr)
        ye = _shift_year(ref_end,   yr)
        df = _fetch_with_cache(lat, lon, timezone, ys, ye)
        if df.empty or len(df) < n_days:
            continue
        df = df.head(n_days).copy()
        df["year"]       = yr
        df["day_offset"] = range(n_days)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Step 3: numpy-based prediction helpers ────────────────────────────────────

def _exp_weights(n: int) -> np.ndarray:
    w = ALPHA ** np.arange(n, dtype=float)
    return w / w.sum()


def _clim_mean(clim_df: pd.DataFrame, day_offset: int, col: str) -> float:
    """Exponentially weighted mean for a specific day offset across years."""
    sub = (clim_df[clim_df["day_offset"] == day_offset]
           .sort_values("year", ascending=False)
           .reset_index(drop=True))
    vals = pd.to_numeric(sub[col], errors="coerce").to_numpy()
    mask = ~np.isnan(vals)
    if not mask.any():
        return 0.0
    v = vals[mask]
    w = _exp_weights(len(v))
    return float(np.average(v, weights=w))


def _recent_mean(recent_df: pd.DataFrame, col: str) -> float:
    """Mean of the last TREND_DAYS rows for a column."""
    if recent_df.empty or col not in recent_df.columns:
        return 0.0
    vals = pd.to_numeric(recent_df[col].tail(TREND_DAYS), errors="coerce").dropna()
    return float(vals.mean()) if len(vals) > 0 else 0.0


def _short_term_slope(recent_df: pd.DataFrame, col: str) -> float:
    """
    Linear trend (slope per day) from last TREND_DAYS of observations.
    Uses np.polyfit on the day index vs. values.
    Returns 0 if insufficient data.
    """
    if recent_df.empty or col not in recent_df.columns:
        return 0.0
    vals = pd.to_numeric(recent_df[col].tail(TREND_DAYS), errors="coerce").dropna()
    if len(vals) < 3:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    slope, _ = np.polyfit(x, vals.to_numpy(), 1)
    # Clip to prevent explosive extrapolation
    std = float(vals.std()) if len(vals) > 1 else 1.0
    return float(np.clip(slope, -std, std))


def _clim_mode(clim_df: pd.DataFrame, day_offset: int) -> int:
    """Exponentially weighted mode of weather_code for a given day offset."""
    sub = (clim_df[clim_df["day_offset"] == day_offset]
           .sort_values("year", ascending=False)
           .reset_index(drop=True))
    codes = pd.to_numeric(sub["weather_code"], errors="coerce").dropna().astype(int)
    if len(codes) == 0:
        return 0
    w = _exp_weights(len(codes))
    score: dict = {}
    for wi, c in zip(w, codes):
        score[c] = score.get(c, 0.0) + wi
    return min(score, key=lambda c: (-score[c], c))


# ── Main prediction function ──────────────────────────────────────────────────

def get_current_forecast(lat: float, lon: float, timezone: str,
                         start_date: str, end_date: str) -> list:
    """
    Predict near-future weather using current observed conditions.

    For each target day d and each weather variable v:

      prediction[d][v] = clim_target[d][v]                   (historical baseline)
                       + anomaly[v] × exp(−days_ahead / τ)   (recent anomaly, decaying)
                       + slope[v]  × d × taper(d)            (short-term trend, first 3 days)

    Where:
      clim_target[d][v]  = weighted historical mean for target calendar date
      anomaly[v]         = mean(recent 7 days) − mean(historical same-period)
      slope[v]           = linear trend in recent 7 days (np.polyfit)
      τ = 4 days         = anomaly e-folding time
      taper(d)           = max(0, 1 − d / TREND_HORIZON)  smoothly kills trend after day 3

    Returns list[DayForecast].
    """
    today        = date.today()
    target_start = date.fromisoformat(start_date)
    target_end   = date.fromisoformat(end_date)
    n_days       = (target_end - target_start).days + 1

    # ── Fetch recent observations ─────────────────────────────────────────────
    print("  [Current] Fetching recent observed conditions (last 14 days)...")
    recent_df = _fetch_recent(lat, lon, timezone)
    if recent_df.empty:
        raise ValueError("Could not fetch recent weather observations.")

    # ── Fetch historical climatology for the recent window ────────────────────
    # (to compute anomaly = recent - what's normal right now)
    obs_end   = today - timedelta(days=2)
    obs_start = obs_end - timedelta(days=RECENT_DAYS - 1)
    print(f"  [Current] Fetching historical climatology for current period "
          f"({obs_start.strftime('%b %d')}–{obs_end.strftime('%b %d')}, "
          f"past {N_CLIM_YEARS} years)...")
    clim_current_df = _fetch_climatology(lat, lon, timezone,
                                         obs_start, obs_end, N_CLIM_YEARS)

    # ── Fetch historical climatology for the target window ────────────────────
    print(f"  [Current] Fetching historical climatology for target dates "
          f"({start_date} → {end_date}, past {N_CLIM_YEARS} years)...")
    clim_target_df = _fetch_climatology(lat, lon, timezone,
                                        target_start, target_end, N_CLIM_YEARS)
    if clim_target_df.empty:
        raise ValueError("No historical data available for target dates.")

    years_used = sorted(clim_target_df["year"].unique())
    print(f"  [Current] Climatology from {len(years_used)} years "
          f"({years_used[0]}–{years_used[-1]})")

    # ── Compute per-variable anomalies ────────────────────────────────────────
    CONT_VARS = {
        "temp_max":        ("temp_max",        -80, 60),
        "temp_min":        ("temp_min",        -80, 60),
        "cloud_cover_mean":("cloud_cover_mean",  0, 100),
        "precipitation_mm":("precipitation_mm",  0, None),
        "wind_speed_max":  ("wind_speed_max",     0, None),
        "uv_index_max":    ("uv_index_max",       0, 16),
    }

    anomalies = {}
    slopes    = {}

    for col, (_, lo, hi) in CONT_VARS.items():
        if col not in recent_df.columns:
            anomalies[col] = 0.0
            slopes[col]    = 0.0
            continue

        recent_val = _recent_mean(recent_df, col)

        # Current-period climatological mean (using day_offset to align)
        n_recent = len(recent_df)
        clim_vals = []
        for di in range(n_recent):
            clim_vals.append(_clim_mean(clim_current_df, di, col))
        clim_current_mean = float(np.mean(clim_vals)) if clim_vals else recent_val

        anomalies[col] = recent_val - clim_current_mean
        slopes[col]    = _short_term_slope(recent_df, col)

    # ── Build per-day predictions ─────────────────────────────────────────────
    forecasts = []
    for day_i in range(n_days):
        current_date = (target_start + timedelta(days=day_i)).isoformat()
        days_ahead   = (target_start + timedelta(days=day_i) - today).days

        # Anomaly decay: exp(-days_ahead / τ), clamped so negative days_ahead
        # (trip already started) still get partial anomaly
        decay = float(np.exp(-max(0, days_ahead) / TAU))

        # Short-term trend taper: linearly fades out after TREND_HORIZON days
        taper = max(0.0, 1.0 - days_ahead / TREND_HORIZON)

        def predict(col, lo, hi):
            base    = _clim_mean(clim_target_df, day_i, col)
            corrected = (base
                         + anomalies.get(col, 0.0) * decay
                         + slopes.get(col, 0.0) * days_ahead * taper)
            if lo is not None:
                corrected = max(lo, corrected)
            if hi is not None:
                corrected = min(hi, corrected)
            return round(corrected, 1)

        temp_max = predict("temp_max",         -80,  60)
        temp_min = predict("temp_min",         -80,  60)
        cloud    = predict("cloud_cover_mean",   0, 100)
        precip   = predict("precipitation_mm",   0, None)
        wind     = predict("wind_speed_max",     0, None)
        uv       = predict("uv_index_max",       0,  16)

        # Enforce physical constraint
        if temp_max < temp_min + 0.5:
            temp_max = round(temp_min + 0.5, 1)

        # Weather code: use climatological mode (categorical, no anomaly logic)
        code = _clim_mode(clim_target_df, day_i)

        forecasts.append(DayForecast(
            date=current_date,
            uv_index_max=uv,
            cloud_cover_mean=cloud,
            temp_min=temp_min,
            temp_max=temp_max,
            precipitation_mm=precip,
            wind_speed_max=wind,
            weather_code=code,
        ))

    return forecasts
