import urllib.request
import urllib.parse
import json
from datetime import date, timedelta
from models import DayForecast

MAX_FORECAST_DAYS = 16


def _build_url(lat: float, lon: float, timezone: str, start: str, end: str, historical: bool) -> str:
    base = "https://archive-api.open-meteo.com/v1/archive" if historical else "https://api.open-meteo.com/v1/forecast"
    params = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "start_date": start,
        "end_date": end,
        "daily": ",".join([
            "uv_index_max",
            "cloud_cover_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "weather_code",
        ]),
    })
    return f"{base}?{params}"


def _fetch_single(lat: float, lon: float, timezone: str,
                  start_date: str, end_date: str, historical: bool) -> list:
    """Fetch one contiguous date range from either the archive or forecast API."""
    url = _build_url(lat, lon, timezone, start_date, end_date, historical)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise ConnectionError(f"Could not reach weather service: {e}")

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        return []

    def _val(key, i, default=0.0):
        v = daily.get(key, [])
        return v[i] if i < len(v) and v[i] is not None else default

    forecasts = []
    for i, d in enumerate(dates):
        forecasts.append(DayForecast(
            date=d,
            uv_index_max=_val("uv_index_max", i),
            cloud_cover_mean=_val("cloud_cover_mean", i),
            temp_min=_val("temperature_2m_min", i),
            temp_max=_val("temperature_2m_max", i),
            precipitation_mm=_val("precipitation_sum", i),
            wind_speed_max=_val("wind_speed_10m_max", i),
            weather_code=int(_val("weather_code", i)),
        ))
    return forecasts


def get_forecast(lat: float, lon: float, timezone: str, start_date: str, end_date: str) -> list:
    """
    Fetch daily weather forecast from Open-Meteo.
    Automatically switches to historical API for past dates.
    Splits mixed past/future ranges across both APIs.
    Returns list of DayForecast.
    """
    today = date.today()
    start = date.fromisoformat(start_date)
    end   = date.fromisoformat(end_date)

    # Clamp to 16-day forecast limit
    max_end = today + timedelta(days=MAX_FORECAST_DAYS - 1)
    if end > max_end:
        print(f"[Warning] End date clamped to forecast limit ({max_end}).")
        end = max_end

    forecasts = []

    # Past segment → archive API (up to and including yesterday)
    yesterday = today - timedelta(days=1)
    if start <= yesterday:
        past_end = min(end, yesterday)
        forecasts += _fetch_single(lat, lon, timezone,
                                   start.isoformat(), past_end.isoformat(),
                                   historical=True)

    # Future segment → forecast API (today onwards)
    if end >= today:
        future_start = max(start, today)
        forecasts += _fetch_single(lat, lon, timezone,
                                   future_start.isoformat(), end.isoformat(),
                                   historical=False)

    if not forecasts:
        raise ValueError("No forecast data returned for this location/date range.")

    return forecasts