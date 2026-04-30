"""
Knowledge Graph Rules + Seeding
=================================
Two responsibilities:
  1. seed_kg()            — Called once on app startup. Merges all required
                            nodes and relationships into Neo4j AuraDB using
                            the data defined in neo4j_schema_design.txt.
                            Safe to re-run (uses MERGE, not CREATE).
  2. CLO inference logic  — calculate_base_weather_clo(), assess_wardrobe_suitability()
                            Use the Neo4j graph + CV attributes to determine
                            thermal adequacy and layering recommendations.

Usage in streamlit_app.py:
    import kg_rules
    kg_rules.seed_kg()  # called once at startup
"""

from __future__ import annotations
from typing import List

# ── Garment data: all 37 items from recommender.py ALL_ITEMS ─────────────────
# (name, category, base_clo)  — used for seeding AND local fallback
_GARMENT_SEED = [
    # Base tops
    ("T-shirt or short sleeves",        "base",  0.09),
    ("Long-sleeve shirt",               "base",  0.12),
    ("Thermal underlayer",              "base",  0.20),
    ("Lightweight breathable clothing", "base",  0.05),
    ("Business attire",                 "base",  0.25),
    ("Casual wear",                     "base",  0.15),
    ("Smart casual outfit",             "base",  0.20),
    # Base bottoms
    ("Jeans or trousers",               "base",  0.20),
    ("Shorts or light trousers",        "base",  0.06),
    # Mid layer
    ("Warm sweater or fleece",          "mid",   0.30),
    ("Light jacket or fleece",          "mid",   0.25),
    ("Windproof jacket",                "mid",   0.25),
    # Outer layer
    ("Heavy winter coat",               "outer", 0.35),
    ("Waterproof jacket",               "outer", 0.20),
    ("Smart jacket",                    "outer", 0.30),
    # Footwear
    ("Insulated boots",                 "feet",  0.40),
    ("Comfortable walking shoes",       "feet",  0.04),
    ("Waterproof snow boots",           "feet",  0.35),
    ("Formal shoes",                    "feet",  0.04),
    # Accessories
    ("Gloves and scarf",                "acc",   0.10),
    ("Thermal socks",                   "feet",  0.05),
    # Gear / packing items (CLO = 0.0 — not thermal garments)
    ("Compact umbrella",                "gear",  0.00),
    ("Waterproof bag cover",            "gear",  0.00),
    ("Sunscreen SPF 50+",               "gear",  0.00),
    ("Sunscreen SPF 30+",               "gear",  0.00),
    ("Sunglasses",                      "gear",  0.00),
    ("Wide-brim hat",                   "gear",  0.00),
    ("Hand warmers",                    "gear",  0.00),
    ("Laptop bag",                      "gear",  0.00),
    ("Power adapter",                   "gear",  0.00),
    ("Business cards",                  "gear",  0.00),
    ("Day backpack",                    "gear",  0.00),
    ("Phone charger / power bank",      "gear",  0.00),
    ("City map or offline maps",        "gear",  0.00),
    ("Reusable water bottle",           "gear",  0.00),
    ("Small gift (optional)",           "gear",  0.00),
    ("Phone charger",                   "gear",  0.00),
]

# ── Airline baggage data ──────────────────────────────────────────────────────
# (iata, name, [(fare_class, checked_kg, carry_on_kg), ...])
_AIRLINE_SEED = [
    ("SQ", "Singapore Airlines", [
        ("economy",         25, 7),
        ("premium_economy", 35, 7),
        ("business",        35, 7),
        ("first",           50, 7),
    ]),
    ("TR", "Scoot", [
        ("economy",  15, 10),
        ("business", 25, 10),
    ]),
    ("3K", "Jetstar", [
        ("economy",  15, 7),
        ("business", 30, 7),
    ]),
    ("AK", "AirAsia", [
        ("economy",  15, 7),
        ("business", 30, 7),
    ]),
    ("CX", "Cathay Pacific", [
        ("economy",  23, 7),
        ("business", 30, 7),
        ("first",    40, 7),
    ]),
    ("EK", "Emirates", [
        ("economy",  25, 7),
        ("business", 40, 7),
        ("first",    50, 7),
    ]),
    ("QF", "Qantas", [
        ("economy",  23, 7),
        ("business", 40, 7),
        ("first",    50, 7),
    ]),
    ("LH", "Lufthansa", [
        ("economy",  23, 8),
        ("business", 32, 8),
        ("first",    40, 8),
    ]),
    ("DL", "Delta Air Lines", [
        ("economy",  23, 7),
        ("business", 32, 7),
        ("first",    45, 7),
    ]),
    ("QR", "Qatar Airways", [
        ("economy",  23, 7),
        ("business", 40, 7),
        ("first",    50, 7),
    ]),
    ("U2", "EasyJet", [
        ("economy",  23, 15),
    ]),
]

# ── WeatherCondition + CLO layering seed ─────────────────────────────────────
# (condition_type, temp_min, temp_max, [(garment_name, layer_role), ...])
_WEATHER_CONDITION_SEED = [
    ("freezing", -30,  0, [
        ("Thermal underlayer",    "base"),
        ("Warm sweater or fleece","mid"),
        ("Heavy winter coat",     "outer"),
        ("Insulated boots",       "feet"),
        ("Gloves and scarf",      "acc"),
    ]),
    ("cold",  0, 10, [
        ("Long-sleeve shirt",     "base"),
        ("Warm sweater or fleece","mid"),
        ("Heavy winter coat",     "outer"),
        ("Insulated boots",       "feet"),
    ]),
    ("cool", 10, 17, [
        ("Long-sleeve shirt",     "base"),
        ("Jeans or trousers",     "base"),
        ("Light jacket or fleece","mid"),
    ]),
    ("mild", 17, 24, [
        ("Long-sleeve shirt",     "base"),
        ("Jeans or trousers",     "base"),
        ("Light jacket or fleece","outer"),
    ]),
    ("warm", 24, 32, [
        ("T-shirt or short sleeves",   "base"),
        ("Shorts or light trousers",   "base"),
    ]),
    ("hot",  32, 50, [
        ("Lightweight breathable clothing", "base"),
        ("Shorts or light trousers",        "base"),
    ]),
    ("rainy", -10, 50, [
        ("Waterproof jacket",  "outer"),
        ("Compact umbrella",   "gear"),
    ]),
    ("snowy", -30, 3, [
        ("Heavy winter coat",      "outer"),
        ("Waterproof snow boots",  "feet"),
        ("Thermal socks",          "feet"),
    ]),
]

# ── Seeding ───────────────────────────────────────────────────────────────────

def seed_kg() -> None:
    """
    Merges all PackPal nodes and relationships into Neo4j.
    Safe to call on every app startup — MERGE is idempotent.
    Silently skips if Neo4j is unreachable (fallback mode).
    """
    try:
        from services.kg_client import _get_driver
        driver = _get_driver()
        if not driver:
            print("[KG Seed] Neo4j unavailable — skipping seed.")
            return

        with driver.session() as session:
            # ── Constraints ───────────────────────────────────────────────────
            for label, prop in [("GarmentType",      "name"),
                                 ("FareClass",        "type"),
                                 ("Airline",          "iata"),
                                 ("WeatherCondition", "condition_type")]:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )

            # ── GarmentType nodes ─────────────────────────────────────────────
            for name, category, base_clo in _GARMENT_SEED:
                session.run(
                    "MERGE (g:GarmentType {name: $name}) "
                    "SET g.category = $category, g.base_clo = $base_clo",
                    name=name, category=category, base_clo=base_clo,
                )

            # ── FareClass nodes ───────────────────────────────────────────────
            for fc in ("economy", "premium_economy", "business", "first"):
                session.run("MERGE (:FareClass {type: $type})", type=fc)

            # ── Airline nodes + HAS_LIMIT relationships ───────────────────────
            for iata, name, limits in _AIRLINE_SEED:
                session.run(
                    "MERGE (a:Airline {iata: $iata}) SET a.name = $name",
                    iata=iata, name=name,
                )
                for fare_class, checked_kg, carry_on_kg in limits:
                    session.run(
                        """
                        MATCH (a:Airline {iata: $iata})
                        MATCH (f:FareClass {type: $fare_class})
                        MERGE (a)-[r:HAS_LIMIT]->(f)
                        SET r.checked_kg   = $checked_kg,
                            r.carry_on_kg  = $carry_on_kg
                        """,
                        iata=iata, fare_class=fare_class,
                        checked_kg=checked_kg, carry_on_kg=carry_on_kg,
                    )

            # ── WeatherCondition nodes + REQUIRES relationships ───────────────
            for ctype, t_min, t_max, garments in _WEATHER_CONDITION_SEED:
                session.run(
                    "MERGE (w:WeatherCondition {condition_type: $ctype}) "
                    "SET w.temp_min = $t_min, w.temp_max = $t_max",
                    ctype=ctype, t_min=t_min, t_max=t_max,
                )
                for garment_name, layer_role in garments:
                    session.run(
                        """
                        MATCH (w:WeatherCondition {condition_type: $ctype})
                        MATCH (g:GarmentType {name: $garment_name})
                        MERGE (w)-[r:REQUIRES]->(g)
                        SET r.layer_role = $layer_role
                        """,
                        ctype=ctype, garment_name=garment_name,
                        layer_role=layer_role,
                    )

        print("[KG Seed] ✅ Knowledge graph seeded successfully.")
        # TEMP – verify Neo4j is alive
        from services.kg_client import _get_driver
        d = _get_driver()
        print("[DIAG] Neo4j driver:", d)
        if d:
            with d.session() as s:
                res = s.run("MATCH (g:GarmentType) RETURN count(g) AS cnt")
                print("[DIAG] GarmentType nodes:", res.single()["cnt"])
        # END TEMP

    except Exception as e:
        print(f"[KG Seed] Warning: {e} — app will continue with fallback data.")


# ── Weather to Base CLO Mapping (ISO 9920 simplified) ─────────────────────────

def calculate_base_weather_clo(forecasts: list) -> float:
    """
    Calculates the required baseline CLO for the trip based on historical forecast.
    Uses the coldest expected temperature to ensure thermal safety.
    Wind chill adds a small CLO buffer above 30 km/h.

    Returns:
        float: Base CLO value (e.g. 0.1 for hot, 1.2 for sub-zero).
    """
    if not forecasts:
        return 0.3  # default mild

    min_temp  = min(f.temp_min for f in forecasts)
    avg_wind  = sum(f.wind_speed_max for f in forecasts) / len(forecasts)

    # Simplified lookup based on physiological comfort models
    if min_temp > 25:
        base_clo = 0.1
    elif min_temp > 18:
        base_clo = 0.3
    elif min_temp > 10:
        base_clo = 0.5
    elif min_temp > 0:
        base_clo = 0.8
    else:
        base_clo = 1.2

    # Wind chill increases required insulation
    if avg_wind > 30:
        base_clo += 0.1

    return round(base_clo, 2)


# ── Dynamic Wardrobe Assessment ───────────────────────────────────────────────

def assess_wardrobe_suitability(
    wardrobe_items: list,
    required_clo: float,
    forecasts: list,
) -> dict:
    """
    Checks if CV-analysed wardrobe items meet the required CLO.
    Handles base / mid / outer layering logic.
    Queries Neo4j for ASHRAE base CLO values; falls back to local table.

    Args:
        wardrobe_items: List of dicts from db_client.get_wardrobe_items() or
                        st.session_state.wardrobe_items. Each must have
                        item_data_json.detected_label and optionally .thickness.
        required_clo:   Target CLO from calculate_base_weather_clo() adjusted
                        by personalization offset.
        forecasts:      List of DayForecast objects (used to check rain risk).

    Returns:
        dict with keys: required_clo, achievable_clo, clo_gap,
                        warnings, recommendations, item_breakdown.
    """
    from services import kg_client
    # -- Diagnostic (remove after debugging) --
    from services.kg_client import _get_driver
    print("[DIAGNOSTIC] Neo4j driver:", _get_driver())
    # -- end diagnostic --

    # Determine if waterproof outer layer is needed
    max_rain = max((f.precipitation_mm for f in forecasts), default=0.0)
    needs_waterproof = max_rain > 5.0

    base_clo_sum  = 0.0
    mid_clo_sum   = 0.0
    outer_clo_sum = 0.0
    details: list = []

    for item in wardrobe_items:
        # Handle both DB row format and direct dict format
        item_data  = item.get("item_data_json", item)
        label      = item_data.get("detected_label", "Unknown")

        # Fetch base CLO from Neo4j (or local fallback)
        base_clo   = kg_client.get_ashrae_base_clo(label)
        print(f"[DIAG] CLO for {label}: {base_clo} (KG)" if _get_driver() else "[DIAG] CLO for {label}: {base_clo} (FALLBACK)")
        # Apply thickness multiplier from CV output
        thickness  = item_data.get("thickness", "medium")
        multiplier = {"thin": 0.7, "medium": 1.0, "thick": 1.3}.get(thickness, 1.0)
        final_clo  = base_clo * multiplier

        # Categorise by CLO band (mirrors ASHRAE layer bands)
        if final_clo <= 0.15:
            base_clo_sum  += final_clo
            category = "base"
        elif final_clo <= 0.35:
            mid_clo_sum   += final_clo
            category = "mid"
        else:
            outer_clo_sum += final_clo
            category = "outer"

        is_waterproof = (
            "waterproof" in label.lower()
            or item_data.get("material", "").lower() in ("nylon", "gore-tex")
        )

        details.append({
            "label":         label,
            "category":      category,
            "calculated_clo": round(final_clo, 2),
            "is_waterproof": is_waterproof,
        })

    total_achievable = base_clo_sum + mid_clo_sum + outer_clo_sum
    gap = required_clo - total_achievable

    warnings: list        = []
    recommendations: list = []

    # Thermal gap check
    if gap > 0:
        warnings.append(
            f"Thermal gap: your wardrobe provides {total_achievable:.2f} CLO "
            f"but {required_clo:.2f} is required for the coldest day."
        )
        if mid_clo_sum < 0.3:
            recommendations.append("Add a mid-layer (e.g. fleece or sweater).")
        if outer_clo_sum < 0.2:
            recommendations.append("Add an outer layer (e.g. jacket or coat).")
    else:
        recommendations.append(
            "Your uploaded wardrobe adequately covers the thermal requirements."
        )

    # Rain / waterproof check
    if needs_waterproof and not any(
        d["is_waterproof"] and d["category"] == "outer" for d in details
    ):
        warnings.append(
            "High rain forecasted but no waterproof outer layer detected."
        )
        recommendations.append(
            "Consider adding a waterproof shell to your packing list."
        )

    return {
        "required_clo":    round(required_clo,     2),
        "achievable_clo":  round(total_achievable,  2),
        "clo_gap":         round(gap,               2),
        "warnings":        warnings,
        "recommendations": recommendations,
        "item_breakdown":  details,
    }


def recommend_layering(forecasts: list, adjusted_clo: float) -> str:
    """
    Uses Neo4j to recommend garment layering combinations based on the
    coldest forecast day.  Falls back to a simple rule-based table if
    Neo4j is unreachable.

    Args:
        forecasts:     List of DayForecast objects.
        adjusted_clo:  Personalised CLO target (weather CLO +/- personal offset).

    Returns:
        A human-readable string describing the recommended layering, or
        empty string if no data is available.
    """
    if not forecasts:
        return ""

    min_temp = min(f.temp_min for f in forecasts)

    # 1. Try Neo4j graph traversal
    try:
        from services.kg_client import _get_driver
        driver = _get_driver()
        if driver:
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (w:WeatherCondition)
                    WHERE w.temp_min <= $min_temp AND w.temp_max >= $min_temp
                    MATCH (w)-[:REQUIRES]->(g:GarmentType)
                    RETURN w.condition_type AS condition,
                           g.name AS garment,
                           g.category AS category
                    ORDER BY condition, category
                    """,
                    min_temp=min_temp
                )
                records = list(result)
                if records:
                    layers = {}
                    for rec in records:
                        cat = rec["category"]
                        layers.setdefault(cat, []).append(rec["garment"])
                    parts = []
                    for cat in ["base", "mid", "outer", "feet", "acc"]:
                        if cat in layers:
                            parts.append(f"{cat.title()}: {', '.join(layers[cat])}")
                    if parts:
                        advice = "; ".join(parts)
                        return (f"For temperatures around {min_temp:.0f}°C, "
                                f"the Knowledge Graph recommends: {advice}.")
    except Exception as e:
        print(f"[KG Layering] Neo4j error: {e}")

    # 2. Fallback: rule-based layering table
    if min_temp < 0:
        base = "Thermal underlayer, Long-sleeve shirt"
        mid  = "Warm sweater or fleece"
        outer = "Heavy winter coat"
        feet = "Insulated boots, Thermal socks"
        acc  = "Gloves and scarf"
    elif min_temp < 10:
        base = "Long-sleeve shirt, Jeans or trousers"
        mid  = "Warm sweater or fleece"
        outer = "Heavy winter coat"
        feet = "Insulated boots or Comfortable walking shoes"
        acc  = "Gloves and scarf"
    elif min_temp < 17:
        base = "Long-sleeve shirt, Jeans or trousers"
        mid  = "Light jacket or fleece"
        outer = "Light jacket or fleece"
        feet = "Comfortable walking shoes"
        acc  = ""
    elif min_temp < 24:
        base = "T-shirt or short sleeves, Jeans or trousers"
        mid  = ""
        outer = "Light jacket or fleece"
        feet = "Comfortable walking shoes"
        acc  = ""
    else:
        base = "Lightweight breathable clothing, Shorts or light trousers"
        mid  = ""
        outer = ""
        feet = "Comfortable walking shoes"
        acc  = ""

    parts = []
    for label, val in [("Base", base), ("Mid", mid),
                        ("Outer", outer), ("Feet", feet), ("Accessories", acc)]:
        if val:
            parts.append(f"{label}: {val}")

    return (
        f"For temperatures around {min_temp:.0f}°C: "
        + "; ".join(parts) + "."
        if parts else ""
    )
