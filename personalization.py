"""
Personal Comfort Model (AI Personalization Layer)
===================================================
Trains a Random Forest Regressor to calculate a 'Personal Comfort Offset'
that adjusts the standard weather CLO based on user-specific traits.

The 5-point cold_tolerance scale from slot_detection maps to:
    very_cold_sensitive  → cold_tolerance feature = 1.0
    cold_sensitive       → cold_tolerance feature = 0.75
    neutral              → cold_tolerance feature = 0.5
    heat_sensitive       → cold_tolerance feature = 0.25
    very_heat_sensitive  → cold_tolerance feature = 0.0

Offset range: -0.15 CLO (runs very hot) to +0.25 CLO (very cold sensitive).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

MODEL_DIR  = Path(__file__).parent / "data" / "model"
MODEL_PATH = MODEL_DIR / "personalization_rf.joblib"

# ── 5-point cold_tolerance → numeric feature ──────────────────────────────────
_COLD_TOLERANCE_MAP = {
    "very_cold_sensitive":  1.00,
    "cold_sensitive":       0.75,
    "neutral":              0.50,
    "heat_sensitive":       0.25,
    "very_heat_sensitive":  0.00,
    # Legacy / fallback labels from old slot_detection versions
    "gets_cold":            0.75,
    "runs_hot":             0.25,
    "standard":             0.50,
    "i run hot":            0.25,
    "i get cold easily":    0.75,
}

_ACTIVITY_MAP = {
    "relaxed":       0.0,
    "moderate":      0.5,
    "highly_active": 1.0,
    # Pill label variants
    "relaxed / sightseeing":  0.0,
    "relaxed/sightseeing":    0.0,
    "highly active / hiking": 1.0,
    "highly active/hiking":   1.0,
}


# ── Synthetic Data Generation ─────────────────────────────────────────────────

def _generate_synthetic_traits(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic user profiles with features normalised 0.0–1.0.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age_normalized":      rng.uniform(0.0, 1.0, n_samples),
        "cold_tolerance":      rng.uniform(0.0, 1.0, n_samples),
        "humidity_sensitivity":rng.uniform(0.0, 1.0, n_samples),
        "activity_level":      rng.uniform(0.0, 1.0, n_samples),
        "baseline_clo":        rng.uniform(0.1, 1.0, n_samples),
    })


def _calculate_target_clo_offset(df: pd.DataFrame) -> np.ndarray:
    """
    Rule-based offset generation for supervised training.
    Offset ∈ [-0.15, +0.25] CLO.
    """
    offsets = np.zeros(len(df))
    for i, row in df.iterrows():
        offset  = 0.0
        offset += (row["cold_tolerance"] - 0.5) * 0.2
        offset += (row["age_normalized"] - 0.5) * 0.1
        offset -= row["activity_level"] * 0.15
        if row["baseline_clo"] > 0.6:
            offset *= 0.5
        if row["baseline_clo"] > 0.3:
            offset += row["humidity_sensitivity"] * 0.05
        offsets[i] = np.clip(offset, -0.15, 0.25)
    return offsets


# ── Training Pipeline ─────────────────────────────────────────────────────────

def train_and_save(verbose: bool = True) -> RandomForestRegressor:
    """Trains the personalization model and persists to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[Personalization] Generating synthetic user profiles...")
    X = _generate_synthetic_traits()
    y = _calculate_target_clo_offset(X)

    if verbose:
        print("[Personalization] Training Random Forest Regressor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    if verbose:
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        print(f"[Personalization] Training complete. RMSE: {rmse:.4f}")
        print(f"[Personalization] Model saved → {MODEL_PATH}")

    joblib.dump(model, MODEL_PATH)
    return model


# ── Lazy Loading ──────────────────────────────────────────────────────────────

_model_cache = None


def _load_model() -> RandomForestRegressor:
    """Loads model from disk, training from scratch if not found."""
    global _model_cache
    if _model_cache is None:
        if MODEL_PATH.exists():
            _model_cache = joblib.load(MODEL_PATH)
        else:
            print("[Personalization] No saved model found — training now...")
            _model_cache = train_and_save()
    return _model_cache


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_clo_offset(user_prefs: dict, baseline_clo: float) -> dict:
    """
    Predicts the personal CLO offset for a user.

    Args:
        user_prefs:   Dict from slot_detection context_json. Relevant keys:
                      cold_tolerance, activity_level, age_normalized (optional),
                      humidity_sensitivity (optional).
        baseline_clo: Weather CLO from kg_rules.calculate_base_weather_clo().

    Returns:
        dict with:
            offset        — CLO adjustment (positive = needs more insulation)
            adjusted_clo  — baseline_clo + offset (floored at 0.1)
            features      — feature dict passed to SHAP for XAI explanation
    """
    model = _load_model()

    cold_raw      = user_prefs.get("cold_tolerance", "neutral")
    activity_raw  = user_prefs.get("activity_level", "moderate")

    cold_numeric     = _COLD_TOLERANCE_MAP.get(cold_raw.lower().strip(), 0.5)
    activity_numeric = _ACTIVITY_MAP.get(activity_raw.lower().strip(), 0.5)

    features = {
        "age_normalized":       user_prefs.get("age_normalized", 0.5),
        "cold_tolerance":       cold_numeric,
        "humidity_sensitivity": user_prefs.get("humidity_sensitivity", 0.5),
        "activity_level":       activity_numeric,
        "baseline_clo":         baseline_clo,
    }

    X      = pd.DataFrame([features])
    offset = float(model.predict(X)[0])
    adjusted_clo = max(0.1, round(baseline_clo + offset, 3))

    return {
        "offset":       round(offset, 3),
        "adjusted_clo": adjusted_clo,
        "features":     features,  # exposed for SHAP in xai_explain.py
    }
