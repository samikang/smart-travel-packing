"""
Visualization Engine
Renders weather forecasts, CLO gaps, and SHAP XAI plots.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from models import DayForecast

def render_weather_plot(forecasts: list, city: str):
    """Interactive Plotly chart for historical weather predictions."""
    if not forecasts:
        return
        
    dates = [f.date for f in forecasts]
    temp_max = [f.temp_max for f in forecasts]
    temp_min = [f.temp_min for f in forecasts]
    precip = [f.precipitation_mm for f in forecasts]
    
    fig = go.Figure()
    
    # Temperature trace
    fig.add_trace(go.Scatter(
        x=dates, y=temp_max, name="Max Temp (°C)",
        mode='lines+markers', line=dict(color='red', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=temp_min, name="Min Temp (°C)",
        mode='lines+markers', fill='tonexty', line=dict(color='blue', width=2)
    ))
    
    # Precipitation trace on secondary y-axis
    fig.add_trace(go.Bar(
        x=dates, y=precip, name="Rain (mm)",
        yaxis='y2', marker_color='lightblue', opacity=0.6
    ))
    
    fig.update_layout(
        title=f"Historical Weather Prediction: {city}",
        yaxis=dict(title="Temperature (°C)"),
        yaxis2=dict(title="Precipitation (mm)", overlaying='y', side='right', range=[0, max(precip)*1.5 if max(precip) > 0 else 10]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

def render_shap_plot(shap_data: dict):
    """Renders the SHAP force plot for Personal Comfort Offset."""
    if not shap_data:
        st.warning("SHAP data not available.")
        return
        
    try:
        import shap
        import matplotlib.pyplot as plt
        
        # Reconstruct the Explanation object for plotting
        expl = shap.Explanation(
            values=shap_data["shap_values"],
            base_values=shap_data["base_value"],
            data=shap_data["feature_values"],
            feature_names=shap_data["feature_names"]
        )
        
        fig, ax = plt.subplots(figsize=(10, 2))
        # Force plot requires JS in notebooks, but matplotlib works natively in Streamlit
        shap.waterfall_plot(expl, max_display=6, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        st.caption("**XAI Insight:** Features pushing the CLO requirement higher (red) vs lower (blue) based on your personal profile.")
        
    except Exception as e:
        st.error(f"Could not render SHAP plot: {e}")

def render_luggage_breakdown(optimized_items: list, limits: dict):
    """Visualizes the weight distribution of optimized items."""
    if not optimized_items:
        return
        
    df = pd.DataFrame(optimized_items)
    fig = go.Figure(go.Pie(
        labels=df["name"], 
        values=df["weight_g"], 
        textinfo='label+percent',
        hole=0.4
    ))
    
    total_w = sum(i["weight_g"] for i in optimized_items) / 1000
    limit_w = limits.get("checked_kg", 20)
    
    fig.update_layout(
        title=f"Luggage Allocation ({total_w:.1f}kg / {limit_w}kg limit)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_clo_metrics(base_clo: float, adjusted_clo: float,
                       clo_assessment: dict) -> None:
    """
    Renders Base CLO, Adjusted CLO, and wardrobe gap metrics as
    Streamlit metric cards below the weather plot (Section 3).

    Args:
        base_clo:       Weather-derived CLO from kg_rules.calculate_base_weather_clo().
        adjusted_clo:   Personalization-adjusted CLO from personalization.predict_clo_offset().
        clo_assessment: Dict from kg_rules.assess_wardrobe_suitability() with keys:
                        achievable_clo, clo_gap, warnings, recommendations.
    """
    achievable = clo_assessment.get("achievable_clo", 0.0)
    gap        = clo_assessment.get("clo_gap", 0.0)
    warnings   = clo_assessment.get("warnings", [])
    recs       = clo_assessment.get("recommendations", [])

    st.markdown("#### 🌡️ CLO Insulation Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Base Weather CLO",
        value=f"{base_clo:.2f}",
        help="Minimum insulation required based on coldest forecast day (ISO 9920).",
    )
    col2.metric(
        label="Your Adjusted CLO",
        value=f"{adjusted_clo:.2f}",
        delta=f"{adjusted_clo - base_clo:+.2f} from personal profile",
        delta_color="inverse",
        help="Adjusted for your cold/heat sensitivity and activity level.",
    )
    col3.metric(
        label="Wardrobe CLO",
        value=f"{achievable:.2f}",
        delta=f"Gap: {gap:+.2f}" if gap != 0 else "✅ Sufficient",
        delta_color="inverse" if gap > 0 else "normal",
        help="Total insulation your uploaded wardrobe provides.",
    )

    # Warnings and recommendations
    if warnings:
        for w in warnings:
            st.warning(f"⚠️ {w}")
    if recs:
        for r in recs:
            if "adequately" in r:
                st.success(f"✅ {r}")
            else:
                st.info(f"💡 {r}")

    # CLO layering bar chart
    if clo_assessment.get("item_breakdown"):
        breakdown = clo_assessment["item_breakdown"]
        items_df  = [
            {"Item": d["label"], "CLO": d["calculated_clo"], "Layer": d["category"]}
            for d in breakdown if d["calculated_clo"] > 0
        ]
        if items_df:
            import plotly.express as px
            import pandas as pd
            df  = pd.DataFrame(items_df)
            fig = px.bar(
                df, x="Item", y="CLO", color="Layer",
                title="Wardrobe CLO by Layer",
                color_discrete_map={
                    "base": "#60a5fa", "mid": "#34d399",
                    "outer": "#f87171", "feet": "#a78bfa",
                    "acc": "#fbbf24",
                },
                height=280,
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=60),
                xaxis_tickangle=-30,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)
