"""
Supabase Database Client
=========================
Manages Trip Sessions and Wardrobe Items using JSONB for NoSQL flexibility.

Supabase schema (trips table columns after ALTER TABLE):
    id                   uuid PK
    session_id           text
    status               text  (gathering_context | slots_complete | optimized)
    context_json         jsonb  — slot data (destination, dates, purpose, airline, etc.)
    forecast_json        jsonb  — list of DayForecast dicts
    recommendations_json jsonb  — list of DayRecommendation dicts
    packing_list_json    jsonb  — editable packing list items
    chat_history_json    jsonb  — full message history for session restore
    optimization_json    jsonb  — GA + knapsack + XAI results
    created_at           timestamptz

wardrobe_items table:
    id             uuid PK
    trip_id        uuid FK → trips.id
    item_data_json jsonb  — CV output (label, weight_g, volume_l, r2_url, etc.)
    created_at     timestamptz
"""

import json
from typing import List, Optional
from config.settings import SUPABASE_URL, SUPABASE_KEY, DEPS_AVAILABLE

_client = None


def _get_client():
    """Lazily initialises the Supabase client on first call."""
    global _client
    if _client is None and DEPS_AVAILABLE["supabase"] and SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── Trip CRUD ─────────────────────────────────────────────────────────────────

def create_trip(session_id: str, context_json: dict) -> Optional[str]:
    """
    Inserts a new trip row and returns its UUID.
    Called as soon as all core slots are filled.
    """
    client = _get_client()
    if not client:
        return _fallback_create_trip(session_id, context_json)

    response = client.table("trips").insert({
        "session_id":   session_id,
        "status":       "gathering_context",
        "context_json": context_json,
    }).execute()

    return response.data[0]["id"] if response.data else None


def update_trip_context(trip_id: str, new_context: dict, status: str) -> None:
    """
    Safely MERGES new_context into the existing context_json.
    Prevents accidental data wipes if only partial slots arrive.
    """
    client = _get_client()

    existing = {}
    if client:
        resp = client.table("trips").select("context_json").eq("id", trip_id).execute()
        if resp.data and resp.data[0].get("context_json"):
            existing = resp.data[0]["context_json"]
    else:
        fp = _FALLBACK_DIR / f"{trip_id}.json"
        if fp.exists():
            existing = json.loads(fp.read_text()).get("context_json", {})

    existing.update(new_context)

    if client:
        client.table("trips").update({
            "context_json": existing,
            "status":       status,
        }).eq("id", trip_id).execute()
    else:
        _fallback_update_trip(trip_id, existing, status)


def save_forecast(trip_id: str, forecast_data: list) -> None:
    """Persists weather forecast list (DayForecast dicts) to forecast_json."""
    client = _get_client()
    if client:
        client.table("trips").update(
            {"forecast_json": forecast_data}
        ).eq("id", trip_id).execute()
    else:
        _fallback_update_column(trip_id, "forecast_json", forecast_data)


def save_recommendations(trip_id: str, recommendations_data: list) -> None:
    """Persists per-day clothing recommendations to recommendations_json."""
    client = _get_client()
    if client:
        client.table("trips").update(
            {"recommendations_json": recommendations_data}
        ).eq("id", trip_id).execute()
    else:
        _fallback_update_column(trip_id, "recommendations_json", recommendations_data)


def save_packing_list(trip_id: str, packing_list: list) -> None:
    """Persists the editable packing list to packing_list_json."""
    client = _get_client()
    if client:
        client.table("trips").update(
            {"packing_list_json": packing_list}
        ).eq("id", trip_id).execute()
    else:
        _fallback_update_column(trip_id, "packing_list_json", packing_list)


def save_chat_history(trip_id: str, messages: list) -> None:
    """Persists full chat message history to chat_history_json."""
    client = _get_client()
    if client:
        client.table("trips").update(
            {"chat_history_json": messages}
        ).eq("id", trip_id).execute()
    else:
        _fallback_update_column(trip_id, "chat_history_json", messages)


def save_optimization(trip_id: str, optimization_result: dict) -> None:
    """Persists GA + knapsack + XAI results to optimization_json."""
    client = _get_client()
    if client:
        client.table("trips").update({
            "optimization_json": optimization_result,
            "status":            "optimized",
        }).eq("id", trip_id).execute()
    else:
        _fallback_update_column(trip_id, "optimization_json", optimization_result)


def get_user_trips(session_id: str) -> List[dict]:
    """
    Retrieves all trips for this browser session, newest first.
    Used to populate the session history sidebar.
    """
    client = _get_client()
    if not client:
        return _fallback_get_trips(session_id)

    response = (
        client.table("trips")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .execute()
    )
    return response.data if response.data else []


def get_trip(trip_id: str) -> Optional[dict]:
    """
    Retrieves a single full trip row by UUID.
    Used for full session restore when clicking a historical trip.
    """
    client = _get_client()
    if not client:
        return _fallback_get_trip(trip_id)

    response = client.table("trips").select("*").eq("id", trip_id).execute()
    return response.data[0] if response.data else None


# ── Wardrobe Items ────────────────────────────────────────────────────────────

def add_wardrobe_item(trip_id: str, item_data_json: dict) -> Optional[str]:
    """
    Inserts a CV-analysed wardrobe item.
    item_data_json: detected_label, r2_image_url, thickness,
                    calculated_clo, calculated_weight_g, calculated_volume_l.
    """
    client = _get_client()
    if not client:
        return _fallback_add_item(trip_id, item_data_json)

    response = client.table("wardrobe_items").insert({
        "trip_id":        trip_id,
        "item_data_json": item_data_json,
    }).execute()

    return response.data[0]["id"] if response.data else None


def get_wardrobe_items(trip_id: str) -> List[dict]:
    """Retrieves all wardrobe items for a trip."""
    client = _get_client()
    if not client:
        return _fallback_get_items(trip_id)

    response = (
        client.table("wardrobe_items")
        .select("*")
        .eq("trip_id", trip_id)
        .execute()
    )
    return response.data if response.data else []


# ── Fallback: Local File System ───────────────────────────────────────────────

import os
from pathlib import Path

_FALLBACK_DIR = Path("./data/local_fallback_db")
_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)


def _fallback_create_trip(session_id: str, context_json: dict) -> str:
    print("[FALLBACK] Supabase unreachable — saving trip locally.")
    trip_id = f"local_{session_id}_{len(list(_FALLBACK_DIR.glob('*.json')))}"
    fp      = _FALLBACK_DIR / f"{trip_id}.json"
    fp.write_text(json.dumps({
        "id":                   trip_id,
        "session_id":           session_id,
        "status":               "gathering_context",
        "context_json":         context_json,
        "forecast_json":        [],
        "recommendations_json": [],
        "packing_list_json":    [],
        "chat_history_json":    [],
        "optimization_json":    {},
        "items":                [],
    }), encoding="utf-8")
    return trip_id


def _fallback_update_trip(trip_id: str, context_json: dict, status: str) -> None:
    fp = _FALLBACK_DIR / f"{trip_id}.json"
    if fp.exists():
        data = json.loads(fp.read_text())
        data["context_json"] = context_json
        data["status"]       = status
        fp.write_text(json.dumps(data), encoding="utf-8")


def _fallback_update_column(trip_id: str, column: str, value) -> None:
    fp = _FALLBACK_DIR / f"{trip_id}.json"
    if fp.exists():
        data         = json.loads(fp.read_text())
        data[column] = value
        fp.write_text(json.dumps(data), encoding="utf-8")


def _fallback_get_trips(session_id: str) -> List[dict]:
    return sorted(
        [json.loads(fp.read_text())
         for fp in _FALLBACK_DIR.glob(f"local_{session_id}_*.json")],
        key=lambda x: x.get("id", ""), reverse=True,
    )


def _fallback_get_trip(trip_id: str) -> Optional[dict]:
    fp = _FALLBACK_DIR / f"{trip_id}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


def _fallback_add_item(trip_id: str, item_data_json: dict) -> str:
    fp = _FALLBACK_DIR / f"{trip_id}.json"
    if fp.exists():
        data    = json.loads(fp.read_text())
        item_id = f"item_{len(data.get('items', []))}"
        data.setdefault("items", []).append(
            {"id": item_id, "item_data_json": item_data_json}
        )
        fp.write_text(json.dumps(data), encoding="utf-8")
        return item_id
    return "item_fallback"


def _fallback_get_items(trip_id: str) -> List[dict]:
    fp = _FALLBACK_DIR / f"{trip_id}.json"
    return json.loads(fp.read_text()).get("items", []) if fp.exists() else []
