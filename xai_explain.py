"""
XAI Explanations (Reasoning + Optimization)
Provides human-readable reasoning using SHAP for Member C's Personal Comfort Model,
and LLM narrative generation for the overall packing logic.
"""
import numpy as np
import pandas as pd
from services import llm_client

def generate_personalization_shap(features_dict: dict, rf_model) -> dict:
    """
    Generates SHAP values for the Personal Comfort Random Forest.
    Returns data needed to render a Plotly/SHAP force plot in the UI.
    """
    try:
        import shap
        # TreeExplainer is highly optimized for Random Forests
        explainer = shap.TreeExplainer(rf_model)
        
        # Convert features dict back to DataFrame
        X = pd.DataFrame([features_dict])
        
        # Calculate SHAP values
        shap_values = explainer(X)
        
        # Extract base value and feature values for the single prediction
        base_value = float(explainer.expected_value[0])
        feature_names = list(features_dict.keys())
        feature_values = list(features_dict.values())
        # shap_vals = shap_values.values[0].tolist()
        shap_vals = shap_values.values[0] # stays as numpy array
        
        return {
            "base_value": base_value,
            "feature_names": feature_names,
            "feature_values": feature_values,
            "shap_values": shap_vals,
            "plot_data": shap_values # Pass raw object for st.pyplot if needed
        }
    except ImportError:
        print("[XAI Error] shap not installed.")
        return None
    except Exception as e:
        print(f"[XAI Error] generating SHAP: {e}")
        return None

def generate_xai_narrative(context: dict, weather_highlights: str, 
                            personalization_data: dict, 
                            clo_assessment: dict,
                            optimization_summary: dict) -> str:
    """
    Uses Groq LLM to weave raw JSON outputs into a cohesive, human-readable explanation.
    """
    if not llm_client.is_llm_available():
        return _fallback_narrative(clo_assessment, personalization_data, optimization_summary)

    from langchain_core.prompts import ChatPromptTemplate
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are PackPal's XAI explainer. Turn technical packing data into a concise, empathetic 2-paragraph summary for the user."),
        ("human", """
        Trip: {context}
        Weather: {weather}
        Personal Comfort Adjustments: {personalization}
        Wardrobe CLO Analysis: {clo_analysis}
        Optimization Result: {optimization}
        
        Write the explanation addressing:
        1. Why specific items were recommended based on weather.
        2. How their personal profile (cold tolerance/activity) changed the packing list.
        3. Why items were removed by the optimization algorithm (if any).
        """)
    ])
    
    chain = prompt | llm_client.get_llm(temperature=0.5)
    
    try:
        response = chain.invoke({
            "context": str(context),
            "weather": weather_highlights,
            "personalization": f"Adjusted CLO by {personalization_data.get('offset', 0)} based on user traits.",
            "clo_analysis": f"Thermal gap: {clo_assessment.get('clo_gap', 0)}. Warnings: {clo_assessment.get('warnings', [])}",
            "optimization": f"Removed: {optimization_summary.get('removed_items', [])}. Reason: {optimization_summary.get('basic_explanation', '')}"
        })
        return response.content
    except Exception as e:
        print(f"[XAI LLM Error] {e}")
        return _fallback_narrative(clo_assessment, personalization_data, optimization_summary)

# ==========================================
# # [FALLBACK MODE] Rule-Based Text Generation
# ==========================================

def _fallback_narrative(clo_assessment: dict, personalization_data: dict, optimization_summary: dict) -> str:
    print("[FALLBACK MODE] Groq unavailable. Using rule-based XAI narrative.")
    text = "### Packing Analysis Summary\n\n"
    
    offset = personalization_data.get('offset', 0)
    if offset > 0.05:
        text += "**Personalization:** Because you tend to get cold easily, we increased your insulation requirements.\n\n"
    elif offset < -0.05:
        text += "**Personalization:** Because you are highly active and run hot, we reduced heavy layers.\n\n"
        
    gap = clo_assessment.get('clo_gap', 0)
    if gap > 0:
        text += f"**Thermal Warning:** Your wardrobe is slightly under-insulated (CLO gap of {gap:.2f}). {clo_assessment.get('recommendations', [''])[0]}\n\n"
    else:
        text += "**Wardrobe Status:** Your clothes perfectly match the predicted weather conditions.\n\n"
        
    removed = optimization_summary.get('removed_items', [])
    if removed:
        text += f"**Optimization:** To meet baggage limits, we removed: {', '.join(removed)}."
        
    return text