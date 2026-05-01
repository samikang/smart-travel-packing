"""
Neo4j Knowledge Graph Client
Handles connection pooling and queries for Airline Limits & ASHRAE CLO data.
"""
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DEPS_AVAILABLE

_driver = None

def _get_driver():
    global _driver
    if _driver is None and DEPS_AVAILABLE["neo4j"] and NEO4J_URI and NEO4J_PASSWORD:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver

def get_baggage_limit(airline_name: str, fare_class: str) -> dict:
    """
    Queries KG for baggage limits based on airline and fare class.
    Returns dict: {"checked_kg": float, "cabin_kg": float, "checked_dims_cm": list}
    """
    driver = _get_driver()
    if not driver:
        return _fallback_baggage_limit(fare_class)

    query = """
    MATCH (a:Airline {name: $airline_name})-[r:HAS_LIMIT]->(f:FareClass {type: $fare_class})
    RETURN r.checked_kg AS checked_kg, r.cabin_kg AS cabin_kg, r.checked_dims_cm AS dims
    """
    try:
        with driver.session() as session:
            result = session.run(query, airline_name=airline_name, fare_class=fare_class)
            record = result.single()
            if record:
                return {
                    "checked_kg": record["checked_kg"],
                    "cabin_kg": record["cabin_kg"],
                    "checked_dims_cm": record["dims"]
                }
    except Exception as e:
        print(f"[Neo4j Error] {e}")
    
    return _fallback_baggage_limit(fare_class)

def get_ashrae_base_clo(garment_name: str) -> float:
    """
    Queries KG for the standard ASHRAE CLO insulation value of a garment.
    """
    driver = _get_driver()
    if not driver:
        return _fallback_ashrae_clo(garment_name)

    query = """
    MATCH (g:GarmentType {name: $garment_name})
    RETURN g.base_clo AS base_clo
    """
    try:
        with driver.session() as session:
            result = session.run(query, garment_name=garment_name)
            record = result.single()
            if record:
                return record["base_clo"]
    except Exception as e:
        print(f"[Neo4j Error] {e}")

    return _fallback_ashrae_clo(garment_name)

def close_driver():
    global _driver
    if _driver:
        _driver.close()

# ==========================================
# # [FALLBACK MODE] Local Heuristics
# ==========================================
def _fallback_baggage_limit(fare_class: str) -> dict:
    print("[FALLBACK MODE] Neo4j unreachable. Using standard IATA fallback limits.")
    limits = {
        "economy": {"checked_kg": 23, "cabin_kg": 7, "checked_dims_cm": [158, 158, 158]},
        "premium_economy": {"checked_kg": 35, "cabin_kg": 10, "checked_dims_cm": [158, 158, 158]},
        "business": {"checked_kg": 40, "cabin_kg": 14, "checked_dims_cm": [158, 158, 158]},
        "first": {"checked_kg": 50, "cabin_kg": 18, "checked_dims_cm": [158, 158, 158]},
    }
    return limits.get(fare_class, limits["economy"])

def _fallback_ashrae_clo(garment_name: str) -> float:
    print("[FALLBACK MODE] Neo4j unreachable. Using local ASHRAE dictionary.")
    # Minimal fallback dictionary for core items
    fallback_map = {
        "T-shirt or short sleeves": 0.09,
        "Heavy winter coat": 0.35,
        "Waterproof jacket": 0.20,
        "Warm sweater or fleece": 0.30,
    }
    for key, val in fallback_map.items():
        if key.lower() in garment_name.lower():
            return val
    return 0.15 # Default generic middle-layer CLO