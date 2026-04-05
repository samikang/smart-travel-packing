import urllib.request
import urllib.parse
import json


def get_location(city: str) -> dict:
    """
    Convert city name to lat/lon/timezone using Open-Meteo geocoding API.
    Returns dict with keys: latitude, longitude, timezone, country, name
    Raises ValueError if city not found, ConnectionError on network failure.
    """
    url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise ConnectionError(f"Could not reach geocoding service: {e}")

    results = data.get("results")
    if not results:
        raise ValueError(
            f"City '{city}' not found. Try a more specific name, e.g. 'Paris, France'."
        )

    loc = results[0]
    return {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": loc.get("timezone", "UTC"),
        "country": loc.get("country", ""),
        "name": loc.get("name", city),
    }
