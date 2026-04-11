"""
Recommendation Engine — Multi-Label Classifier
===============================================
Architecture:
  - Fixed vocabulary of 37 clothing/packing items
  - Features: temperature stats, precipitation, wind, UV, cloud cover,
    snow/thunder flags, trip purpose (10 features total)
  - Labels: binary per item (include / don't include)
  - Training data: 15,000 synthetic samples generated from weather rules,
    with random noise to learn smooth decision boundaries instead of hard cutoffs
  - Auto-trains on first run; subsequent runs load from cache (~50ms)

Model type options (pass model_type= to recommend_day / train_and_save):
  "knn"           (default) MultiOutputClassifier(KNeighborsClassifier) — instance-based;
                  stores training data and looks up k most similar weather conditions.
  "lgbm"          MultiOutputClassifier(LGBMClassifier) — gradient
                  boosting; highest accuracy on synthetic data.
  "random_forest" MultiOutputClassifier(RandomForestClassifier) — bagging
                  ensemble; no learning-rate tuning needed, robust to noise.
  "rules"         Direct rule evaluation via _rule_labels — no ML at all;
                  instant, deterministic, useful as a reference baseline.

Alerts remain rule-based — safety-critical info should never be ML-gated.
"""

import numpy as np
import joblib
from pathlib import Path
from models import DayForecast, TripContext, DayRecommendation

MODEL_PATH = Path(__file__).parent / "model" / "recommender.joblib"  # legacy / lgbm

_MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATHS = {
    "lgbm":          _MODEL_DIR / "recommender.joblib",
    "random_forest": _MODEL_DIR / "recommender_random_forest.joblib",
    "knn":           _MODEL_DIR / "recommender_knn.joblib",
}

VALID_MODEL_TYPES = ("lgbm", "random_forest", "knn", "rules")

# ── Item vocabulary ───────────────────────────────────────────────────────────
# Order matters: indices 0..N_CLOTHING-1 = clothing, rest = packing

CLOTHING_ITEMS = [
    "Heavy winter coat",
    "Thermal underlayer",
    "Warm sweater or fleece",
    "Gloves and scarf",
    "Insulated boots",
    "Light jacket or fleece",
    "Long-sleeve shirt",
    "Jeans or trousers",
    "T-shirt or short sleeves",
    "Lightweight breathable clothing",
    "Shorts or light trousers",
    "Waterproof jacket",
    "Windproof jacket",
    "Waterproof snow boots",
    "Thermal socks",
    "Business attire",
    "Formal shoes",
    "Comfortable walking shoes",
    "Casual wear",
    "Smart casual outfit",
    "Smart jacket",
]

PACKING_ITEMS = [
    "Compact umbrella",
    "Waterproof bag cover",
    "Sunscreen SPF 50+",
    "Sunscreen SPF 30+",
    "Sunglasses",
    "Wide-brim hat",
    "Hand warmers",
    "Laptop bag",
    "Power adapter",
    "Business cards",
    "Day backpack",
    "Phone charger / power bank",
    "City map or offline maps",
    "Reusable water bottle",
    "Small gift (optional)",
    "Phone charger",
]

ALL_ITEMS = CLOTHING_ITEMS + PACKING_ITEMS
N_CLOTHING = len(CLOTHING_ITEMS)
ITEM_TO_IDX = {item: i for i, item in enumerate(ALL_ITEMS)}

SNOW_CODES    = {71, 73, 75, 77, 85, 86}
THUNDER_CODES = {95, 96, 99}

# WMO code → short description (used in day summary)
WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ heavy hail",
}

PURPOSE_ENC = {"business": 0, "tourism": 1, "visiting": 2}

FEATURE_NAMES = [
    "temp_avg", "temp_min", "temp_max", "log_precip",
    "wind_speed", "uv_index", "cloud_cover",
    "is_snow", "is_thunder", "purpose_enc",
]


# ── Feature extraction ────────────────────────────────────────────────────────

def _features(forecast: DayForecast, purpose: str):
    """
    Returns a single-row pandas DataFrame with named features.
    Using DataFrame (not raw array) keeps feature names consistent
    between training and inference, avoiding sklearn warnings.
    """
    import pandas as pd
    temp_avg = (forecast.temp_min + forecast.temp_max) / 2.0
    return pd.DataFrame([{
        "temp_avg":    temp_avg,
        "temp_min":    forecast.temp_min,
        "temp_max":    forecast.temp_max,
        "log_precip":  np.log1p(forecast.precipitation_mm),
        "wind_speed":  forecast.wind_speed_max,
        "uv_index":    forecast.uv_index_max,
        "cloud_cover": forecast.cloud_cover_mean,
        "is_snow":     float(forecast.weather_code in SNOW_CODES),
        "is_thunder":  float(forecast.weather_code in THUNDER_CODES),
        "purpose_enc": float(PURPOSE_ENC.get(purpose, 0)),
    }])


# ── Label generation (rules → synthetic training labels) ─────────────────────

def _rule_labels(temp_avg, temp_min, temp_max, log_precip, wind, uv, cloud,
                 is_snow, is_thunder, purpose_enc) -> np.ndarray:
    """
    Generate ground-truth label vector from weather rules.
    Used to produce synthetic training data for LightGBM.
    Returns binary numpy array of shape (len(ALL_ITEMS),).
    """
    precip = np.expm1(log_precip)
    eff_uv = uv * (1.0 - cloud / 100.0 * 0.7)
    labels = np.zeros(len(ALL_ITEMS), dtype=np.float32)

    def on(*names):
        for n in names:
            if n in ITEM_TO_IDX:
                labels[ITEM_TO_IDX[n]] = 1.0

    # Temperature layers
    if temp_avg < 5:
        on("Heavy winter coat", "Thermal underlayer", "Warm sweater or fleece",
           "Gloves and scarf", "Insulated boots")
    elif temp_avg < 10:
        on("Heavy winter coat", "Warm sweater or fleece", "Gloves and scarf")
    elif temp_avg < 17:
        on("Light jacket or fleece", "Long-sleeve shirt", "Jeans or trousers")
    elif temp_avg < 24:
        on("Light jacket or fleece", "Long-sleeve shirt", "Jeans or trousers")
    else:
        on("T-shirt or short sleeves", "Lightweight breathable clothing", "Shorts or light trousers")

    # Rain
    if precip > 20:
        on("Waterproof jacket", "Compact umbrella", "Waterproof bag cover")
    elif precip > 1:
        on("Compact umbrella")

    # UV
    if eff_uv > 6:
        on("Sunscreen SPF 50+", "Sunglasses", "Wide-brim hat")
    elif eff_uv > 3:
        on("Sunscreen SPF 30+", "Sunglasses")

    # Wind
    if wind > 40:
        on("Windproof jacket")

    # Snow
    if is_snow:
        on("Waterproof snow boots", "Thermal socks", "Hand warmers")

    # Thunder (pack rain gear regardless of precip amount)
    if is_thunder:
        on("Waterproof jacket", "Compact umbrella")

    # Trip purpose
    purpose = int(round(purpose_enc))
    if purpose == 0:  # business
        on("Business attire", "Formal shoes", "Laptop bag", "Power adapter", "Business cards")
        if precip > 1:
            on("Waterproof bag cover")
    elif purpose == 1:  # tourism
        on("Comfortable walking shoes", "Casual wear", "Day backpack",
           "Phone charger / power bank", "City map or offline maps")
        if temp_max > 24:
            on("Reusable water bottle")
    elif purpose == 2:  # visiting
        on("Smart casual outfit", "Small gift (optional)", "Phone charger")
        if temp_max < 17:
            on("Smart jacket")

    return labels


# ── Synthetic training data generation ───────────────────────────────────────

def _generate_training_data(n_samples: int = 15_000, seed: int = 42):
    """
    Sample synthetic (features, labels) pairs across the full weather space.
    Adds Gaussian noise to continuous features so LightGBM learns smooth
    decision boundaries rather than memorising the hard threshold cutoffs.
    """
    rng = np.random.default_rng(seed)

    temp_avg   = rng.uniform(-25, 45, n_samples)
    temp_range = rng.uniform(2, 16, n_samples)
    temp_min   = temp_avg - temp_range / 2
    temp_max   = temp_avg + temp_range / 2
    precip     = rng.exponential(scale=5.0, size=n_samples).clip(0, 80)
    log_precip = np.log1p(precip)
    wind       = rng.uniform(0, 80, n_samples)
    uv         = rng.uniform(0, 12, n_samples)
    cloud      = rng.uniform(0, 100, n_samples)
    is_snow    = ((temp_avg < 3) & (rng.random(n_samples) < 0.5)).astype(float)
    is_thunder = (rng.random(n_samples) < 0.05).astype(float)
    purpose    = rng.integers(0, 3, n_samples).astype(float)

    import pandas as pd
    X = pd.DataFrame(np.column_stack([
            temp_avg, temp_min, temp_max, log_precip,
            wind, uv, cloud, is_snow, is_thunder, purpose,
        ]), columns=FEATURE_NAMES)

    # Generate clean labels from rules, then add label noise (~3%) for robustness
    X_arr = X.to_numpy()
    Y = np.array([
        _rule_labels(*X_arr[i])
        for i in range(n_samples)
    ])
    noise_mask = rng.random(Y.shape) < 0.03
    Y[noise_mask] = 1.0 - Y[noise_mask]

    return X, Y


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_save(verbose: bool = True, model_type: str = "knn") -> object:
    """
    Generate synthetic training data, fit the selected classifier, and persist
    to the corresponding model file via joblib.

    model_type: "knn" (default), "lgbm", "random_forest"
                ("rules" needs no training — pass it to recommend_day directly)
    """
    from sklearn.multioutput import MultiOutputClassifier

    print("[Recommender] Generating training data...")
    X, Y = _generate_training_data()
    print(f"[Recommender] Training '{model_type}' on {len(X):,} samples × {Y.shape[1]} items...")

    if model_type == "lgbm":
        from lightgbm import LGBMClassifier
        clf = MultiOutputClassifier(LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            verbose=-1,
        ), n_jobs=-1)

    elif model_type == "random_forest":
        # Bagging ensemble — robust to label noise, no learning-rate tuning needed.
        # RandomForestClassifier supports multi-output natively but we wrap it for
        # a consistent interface with lgbm/knn.
        from sklearn.ensemble import RandomForestClassifier
        clf = MultiOutputClassifier(RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        ), n_jobs=-1)

    elif model_type == "knn":
        # Instance-based: no explicit training, stores the dataset and finds
        # k most similar weather conditions at query time.
        from sklearn.neighbors import KNeighborsClassifier
        clf = MultiOutputClassifier(KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            metric="euclidean",
        ))

    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Choose from: {list(MODEL_PATHS)}"
        )

    clf.fit(X, Y)

    path = MODEL_PATHS[model_type]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    if verbose:
        print(f"[Recommender] Model saved → {path}")
    return clf


# ── Lazy model loading ────────────────────────────────────────────────────────

_model_cache: dict = {}


def _load_or_train(model_type: str = "knn"):
    """Load model from disk if available, otherwise train and save it."""
    if model_type == "rules":
        return None  # rules mode needs no model

    if model_type in _model_cache:
        return _model_cache[model_type]

    path = MODEL_PATHS.get(model_type)
    if path is None:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Choose from: {VALID_MODEL_TYPES}"
        )

    if path.exists():
        print(f"[Recommender] Loading '{model_type}' model from {path}")
        _model_cache[model_type] = joblib.load(path)
    else:
        print(f"[Recommender] No saved model for '{model_type}' — training now...")
        _model_cache[model_type] = train_and_save(model_type=model_type)
    return _model_cache[model_type]


# ── Alerts (rule-based — safety info should not be ML-gated) ─────────────────

def _alerts(forecast: DayForecast) -> list:
    alerts = []
    eff_uv = forecast.uv_index_max * (1 - forecast.cloud_cover_mean / 100 * 0.7)
    if forecast.precipitation_mm > 20:
        alerts.append(f"Heavy rain ({forecast.precipitation_mm:.0f}mm) — dress waterproof")
    elif forecast.precipitation_mm > 1:
        alerts.append(f"Rain expected ({forecast.precipitation_mm:.0f}mm) — bring umbrella")
    if eff_uv > 6:
        alerts.append(f"Very high UV (index {forecast.uv_index_max:.0f}) — sun protection essential")
    elif eff_uv > 3:
        alerts.append(f"Moderate UV (index {forecast.uv_index_max:.0f}) — apply sunscreen outdoors")
    if forecast.wind_speed_max > 60:
        alerts.append(f"Strong winds ({forecast.wind_speed_max:.0f} km/h) — secure loose items")
    elif forecast.wind_speed_max > 40:
        alerts.append(f"Gusty winds ({forecast.wind_speed_max:.0f} km/h) expected")
    if forecast.weather_code in SNOW_CODES:
        alerts.append("Snow forecast — wear waterproof boots and warm layers")
    if forecast.weather_code in THUNDER_CODES:
        alerts.append("Thunderstorm risk — stay indoors during storms")
    return alerts


def _day_summary(forecast: DayForecast) -> str:
    desc = WMO_CODES.get(forecast.weather_code, f"Code {forecast.weather_code}")
    temp_str = f"{forecast.temp_min:.0f}–{forecast.temp_max:.0f}°C"
    rain_str = f", {forecast.precipitation_mm:.0f}mm rain" if forecast.precipitation_mm > 1 else ""
    uv_str = f", UV {forecast.uv_index_max:.0f}" if forecast.uv_index_max > 0 else ""
    return f"{desc}, {temp_str}{rain_str}{uv_str}"


# ── Public API ────────────────────────────────────────────────────────────────
#class DayRecommendation:
#    date: str
#    clothing: list = field(default_factory=list)
#    packing: list = field(default_factory=list)
#    alerts: list = field(default_factory=list)
#    summary: str = ""
def suitability_scores(forecast: DayForecast, context: TripContext,
                       model_type: str = "knn") -> dict:
    """
    Return a suitability score (0.0–1.0) for every item.
    For ML models: uses predict_proba() — probability that the item is recommended.
    For rules: returns 1.0 (recommended) or 0.0 (not recommended).
    """
    feat = _features(forecast, context.purpose)

    if model_type == "rules":
        pred = _rule_labels(*feat.to_numpy()[0])
        return {ALL_ITEMS[i]: float(pred[i]) for i in range(len(ALL_ITEMS))}

    model = _load_or_train(model_type)
    proba_list = model.predict_proba(feat)  # list of 37 arrays, each shape (1, 2)
    return {
        ALL_ITEMS[i]: round(float(proba_list[i][0][1]), 3)
        for i in range(len(ALL_ITEMS))
    }


def recommend_day(forecast: DayForecast, context: TripContext,
                  model_type: str = "knn") -> DayRecommendation:
    feat = _features(forecast, context.purpose)  # single-row DataFrame

    if model_type == "rules":
        # Bypass ML entirely — evaluate _rule_labels directly.
        # Deterministic, instant, and the reference ground truth for all ML models.
        pred = _rule_labels(*feat.to_numpy()[0])
    else:
        model = _load_or_train(model_type)
        pred = model.predict(feat)[0]  # shape (37,)

    clothing = [ALL_ITEMS[i] for i in range(N_CLOTHING) if pred[i] == 1]
    packing  = [ALL_ITEMS[i] for i in range(N_CLOTHING, len(ALL_ITEMS)) if pred[i] == 1]

    return DayRecommendation(
        date=forecast.date,
        clothing=clothing,
        packing=packing,
        alerts=_alerts(forecast),
        summary=_day_summary(forecast),
    )

#{"clothing": list[str], "packing": list[str]}
def build_trip_packing_list(recommendations: list) -> dict:
    all_clothing, all_packing = [], []
    for r in recommendations:
        all_clothing.extend(r.clothing)
        all_packing.extend(r.packing)
    return {
        "clothing": list(dict.fromkeys(all_clothing)),
        "packing":  list(dict.fromkeys(all_packing)),
    }
