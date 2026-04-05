from dataclasses import dataclass, field


@dataclass
class DayForecast:
    date: str
    uv_index_max: float
    cloud_cover_mean: float   # percent 0-100
    temp_min: float           # Celsius
    temp_max: float           # Celsius
    precipitation_mm: float
    wind_speed_max: float     # km/h
    weather_code: int = 0     # WMO weather code


@dataclass
class TripContext:
    city: str
    country: str
    purpose: str = ""         # "business" | "tourism" | "visiting"


@dataclass
class DayRecommendation:
    date: str
    clothing: list = field(default_factory=list)
    packing: list  = field(default_factory=list)
    alerts: list   = field(default_factory=list)
    summary: str   = ""
