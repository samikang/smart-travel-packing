"""
LIME Explainer for Clothing Recommender (Plotly Bar Chart Version)
==================================================================
For a chosen day, builds a LIME explainer for each recommended item
and returns a Plotly horizontal bar chart showing feature contributions
towards the "Recommended" class.

No HTML – clean, interactive, fits Streamlit's design perfectly.

Usage:
    figures = explain_clothing_day(date_str, recommendations, forecasts, purpose)
    for fig in figures:
        st.plotly_chart(fig, use_container_width=True)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer

from recommender import _features, ALL_ITEMS


def _build_training_data() -> pd.DataFrame:
    """
    Creates a synthetic background dataset matching the recommender's
    feature space.  Needed by LIME to understand the distribution of
    features.  1000 random samples covering the typical weather range.
    """
    rng = np.random.default_rng(42)
    n = 1000
    temp_avg   = rng.uniform(-25, 45, n)
    temp_range = rng.uniform(2, 16, n)
    temp_min   = temp_avg - temp_range / 2
    temp_max   = temp_avg + temp_range / 2
    precip     = rng.exponential(scale=5.0, size=n).clip(0, 80)
    log_precip = np.log1p(precip)
    wind       = rng.uniform(0, 80, n)
    uv         = rng.uniform(0, 12, n)
    cloud      = rng.uniform(0, 100, n)
    is_snow    = ((temp_avg < 3) & (rng.random(n) < 0.5)).astype(float)
    is_thunder = (rng.random(n) < 0.05).astype(float)
    purpose    = rng.integers(0, 3, n).astype(float)

    X = pd.DataFrame(np.column_stack([
        temp_avg, temp_min, temp_max, log_precip,
        wind, uv, cloud, is_snow, is_thunder, purpose,
    ]), columns=[
        "temp_avg", "temp_min", "temp_max", "log_precip",
        "wind_speed", "uv_index", "cloud_cover",
        "is_snow", "is_thunder", "purpose_enc",
    ])
    return X


def explain_clothing_day(date_str: str,
                         recommendations: list,
                         forecasts: list,
                         purpose: str = "tourism") -> list:
    """
    Generate LIME bar charts for every clothing/packing item recommended
    for the given date.

    Args:
        date_str:        e.g. "2026-08-13"
        recommendations: list of DayRecommendation objects
        forecasts:       list of DayForecast objects
        purpose:         "business", "tourism", or "visiting"

    Returns:
        List of Plotly Figure objects, one per recommended item, or empty list
        if no items to explain.
    """
    # 1. Find matching forecast and recommendation
    fc = None
    rec = None
    for f in forecasts:
        if f.date == date_str:
            fc = f
            break
    for r in recommendations:
        if r.date == date_str:
            rec = r
            break
    if not fc or not rec:
        return []

    # 2. Load the recommender model (KNN is default)
    from recommender import _load_or_train
    model = _load_or_train("knn")

    # 3. Build feature vector (DataFrame with named columns)
    X = _features(fc, purpose)

    # 4. Prepare LIME explainer
    background = _build_training_data()
    categorical_features = [7, 8]  # indices of is_snow, is_thunder
    explainer = LimeTabularExplainer(
        training_data=background.values,
        feature_names=list(background.columns),
        categorical_features=categorical_features,
        discretize_continuous=True,
        mode="classification",
        class_names=["Not recommended", "Recommended"],
    )

    figures = []

    # 5. Loop over items recommended today
    for item_name in rec.clothing + rec.packing:
        if item_name not in ALL_ITEMS:
            continue
        idx = ALL_ITEMS.index(item_name)

        # Sub‑model for this specific item (MultiOutputClassifier)
        sub_model = model.estimators_[idx]

        # Probability prediction wrapper for LIME
        def predict_fn(data_2d):
            proba = sub_model.predict_proba(data_2d)  # shape (n, 2)
            # LIME expects columns for both classes
            return np.column_stack([1 - proba[:, 1], proba[:, 1]])

        # Explain the single instance
        exp = explainer.explain_instance(
            X.values[0],
            predict_fn,
            num_features=5,
            labels=(1,),   # explain why class 1 (Recommended) was predicted
        )

        # Extract feature names and weights for class 1
        # exp.as_list() returns list of (feature_name, weight) sorted by abs weight
        feature_weights = exp.as_list(label=1)

        # Build a horizontal bar chart with Plotly
        if feature_weights:
            names  = [fw[0] for fw in feature_weights]
            values = [fw[1] for fw in feature_weights]
            # Colour positive green, negative red
            colours = ["#22c55e" if v >= 0 else "#ef4444" for v in values]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=names,
                x=values,
                orientation="h",
                marker_color=colours,
                text=[f"{v:+.3f}" for v in values],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Why was <b>{item_name}</b> recommended?",
                xaxis_title="Contribution to recommendation",
                yaxis=dict(autorange="reversed"),  # highest feature on top
                height=200 + 20 * len(names),
                margin=dict(l=10, r=10, t=40, b=20),
                font=dict(size=12),
            )
            # Get the model's predicted probability for class "Recommended"
            prob_rec = predict_fn(X.values)[0, 1]   # probability of being recommended
            fig.update_layout(
                title=f"Why was <b>{item_name}</b> recommended?<br>"
                      f"<sup>Predicted probability: {prob_rec:.2%}</sup>",
            )
            figures.append(fig)

    return figures