"""
PackPal — Main Streamlit Application
======================================
Layout (SIDEBAR):
  Branding → New Trip → Current Trip (slot chips) → Jump to Section → Past Trips

MAIN AREA (5 sections with HTML anchors for scroll‑links):
  1. Trip Details (chat)           – always visible
  2. Clothing Image Upload         – always visible once slots are complete
  3. Weather Forecast              – visible after slots complete
  4. Recommended Packing List      – visible after slots complete (editable)
  5. Recommendation Explanations   – visible after optimisation (XAI)

Run:
    streamlit run streamlit_app.py
"""

import os
import uuid
import warnings
import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

st.set_page_config(
    page_title="PackPal — Smart Travel Packer",
    page_icon="🧳",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 2rem !important; }

/* Slot chips */
.slot-chip {
    display: inline-block; background: #e0e7ff; color: #3730a3;
    border-radius: 20px; padding: 2px 10px; font-size: 12px;
    font-weight: 500; margin: 2px 3px;
}
.slot-chip.missing { background: #fee2e2; color: #991b1b; }

/* Section anchor headers */
.section-anchor {
    font-size: 16px; font-weight: 700; color: #1e293b;
    padding: 4px 0;
}

/* Jump to Section — button-like pills, no underline */
.nav-btn {
    display: block;
    padding: 6px 12px; margin: 3px 0;
    border-radius: 8px;
    border: 1px solid #d1d5db;
    background: white;
    color: #1e293b !important;
    text-decoration: none !important;
    font-size: 13px; font-weight: 500;
    transition: background 0.15s, border-color 0.15s;
}
.nav-btn:hover { background: #f1f5f9; border-color: #6366f1; color: #6366f1 !important; }
.nav-btn-disabled {
    display: block;
    padding: 6px 12px; margin: 3px 0;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    color: #9ca3af !important;
    text-decoration: none !important;
    font-size: 13px; font-weight: 400;
    pointer-events: none;
}

section[data-testid="stSidebar"] { background-color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# ── Guarded imports ───────────────────────────────────────────────────────────
try:
    from services import db_client, storage
    SERVICES_OK = True
except ImportError as _e:
    print(f"[App] services warning: {_e}"); SERVICES_OK = False

try:
    from services.kg_client import get_baggage_limit
    KG_OK = True
except ImportError:
    KG_OK = False

try:
    import kg_rules
    kg_rules.seed_kg()
    KG_RULES_OK = True
except Exception as _e:
    print(f"[App] kg_rules warning: {_e}"); KG_RULES_OK = False

try:
    from slot_detection import extract_slots
    SLOT_OK = True
except ImportError as _e:
    print(f"[App] slot_detection warning: {_e}"); SLOT_OK = False

try:
    from geocoder import get_location
    from historical_forecast import get_historical_forecast
    from recommender import recommend_day, build_trip_packing_list, calculate_needed_quantity
    from models import TripContext, DayForecast, DayRecommendation
    BACKEND_OK = True
except ImportError as _e:
    print(f"[App] backend warning: {_e}"); BACKEND_OK = False

# ── Computer Vision imports ─────────────────────────────────────────
# Primary CV backend: cloth‑tool (EfficientNet‑B0 custom classifier)
# Fallback: YOLO detection + rule‑based size estimation
try:
    from image_recognition import (
        _detect_yolo,                 # fallback YOLO label detection
        estimate_dimensions_from_cv,  # fallback weight/volume estimator
        _detect_cloth_tool,           # primary: run the cloth‑tool model
        _cloth_tool_readable_label,   # format cloth‑tool prediction into text
        _CLOTH_MATERIAL_MAP,          # material_group → fabric key
        _CLOTH_THICKNESS_MAP,         # weight_class → thickness string
        CLOTH_TOOL_MODEL,             # path to the trained model checkpoint
    )
    CV_OK = True
except ImportError as _e:
    print(f"[App] CV warning: {_e}")
    CV_OK = False
try:
    from packing_optimizer import optimise_dynamic_items, _item_weight
    OPT_OK = True
except ImportError as _e:
    print(f"[App] optimizer warning: {_e}"); OPT_OK = False

try:
    from ui.visualization import (
        render_weather_plot, render_clo_metrics,
        render_shap_plot, render_luggage_breakdown,
    )
    from ui.chat_engine import render_chat, process_pill_if_clicked
    from ui.packing_list import render_packing_editor
    from xai_explain import generate_xai_narrative, generate_personalization_shap
    from personalization import predict_clo_offset, _load_model as load_personalization_model
    UI_OK = True
except ImportError as _e:
    print(f"[App] UI warning: {_e}"); UI_OK = False


# ════════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "browser_session_id":   (
            st.query_params.get("sid") or
            st.query_params.setdefault("sid", str(uuid.uuid4())) or
            st.query_params.get("sid")
        ),
        "trip_id":              None,
        "trip_status":          "new",
        "slots":                {},
        "forecasts":            [],
        "recommendations":      [],
        "trip_packing":         {},
        "packing_list_items":   [],
        "optimization_result":  None,
        "base_clo":             0.3,
        "adjusted_clo":         0.3,
        "personalization_data": {},
        "clo_assessment":       {},
        "xai_narrative":        "",
        "shap_data":            None,
        "wardrobe_items":       [],
        "pending_uploads":      [],
        "processed_filenames":  set(),    # prevents re‑adding same files on rerun
        "baggage_limits":       {},
        "messages":             [],
        "pill_clicked":         None,
        "needs_processing":     False,
        "all_trips":            [],
        "sec3_open":            False,
        "sec4_open":            False,
        "sec5_open":            False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Cold‑start warmup ─────────────────────────────────────────────────────────
if "warmup_done" not in st.session_state:
    try:
        from services import llm_client as _llm
        _llm.get_llm()
    except Exception:
        pass
    try:
        if SERVICES_OK:
            db_client._get_client()
    except Exception:
        pass
    st.session_state.warmup_done = True


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _greeting():
    return {
        "role": "assistant",
        "content": (
            "👋 Hi! I'm **PackPal**, your smart travel packing assistant.\n\n"
            "Tell me about your upcoming trip — where are you headed and when? "
            "I'll create a personalised packing list based on the weather forecast!"
        ),
        "render_pills": None,
    }


def _start_new_trip():
    """Resets all session‑state variables for a brand‑new trip."""
    reset = {
        "trip_id": None, "slots": {}, "forecasts": [], "recommendations": [],
        "trip_packing": {}, "packing_list_items": [], "optimization_result": None,
        "wardrobe_items": [], "pending_uploads": [], "processed_filenames": set(),
        "baggage_limits": {},
        "personalization_data": {}, "clo_assessment": {}, "xai_narrative": "",
        "shap_data": None, "pill_clicked": None, "needs_processing": False,
        "base_clo": 0.3, "adjusted_clo": 0.3, "trip_status": "new",
        "sec3_open": False, "sec4_open": False, "sec5_open": False,
    }
    for k, v in reset.items():
        st.session_state[k] = v
    st.session_state.messages = [_greeting()]


def _load_trip_from_db(trip: dict):
    """
    Restores a trip from Supabase JSON into session state.
    Converts stored dicts back to their proper dataclass objects
    (DayForecast, DayRecommendation) so that downstream code works.
    """
    ctx = trip.get("context_json", {})
    st.session_state.trip_id             = trip.get("id")
    st.session_state.trip_status         = trip.get("status", "gathering_context")
    st.session_state.slots               = ctx

    # Convert forecast dicts → DayForecast objects (required by render_weather_plot etc.)
    raw_forecasts = trip.get("forecast_json", [])
    if raw_forecasts:
        st.session_state.forecasts = [DayForecast(**f) for f in raw_forecasts]
    else:
        st.session_state.forecasts = []

    # Convert recommendation dicts → DayRecommendation objects (required by LIME explainer)
    raw_recs = trip.get("recommendations_json", [])
    if raw_recs:
        st.session_state.recommendations = [DayRecommendation(**r) for r in raw_recs]
    else:
        st.session_state.recommendations = []

    st.session_state.trip_packing        = trip.get("packing_list_json",    {})
    st.session_state.optimization_result = trip.get("optimization_json") or None

    restored = trip.get("chat_history_json", [])
    st.session_state.messages = restored or [
        {"role": "assistant",
         "content": "Welcome back! Your trip has been restored. How can I help?",
         "render_pills": None}
    ]

    # Auto‑open sections that have data
    st.session_state.sec3_open = bool(st.session_state.forecasts)
    st.session_state.sec4_open = bool(st.session_state.packing_list_items)
    st.session_state.sec5_open = bool(st.session_state.optimization_result)


def _load_all_trips():
    """
    Fetches ALL trips from Supabase (no session filter) so that past trips
    always appear in the sidebar regardless of browser session ID.
    """
    if SERVICES_OK:
        try:
            # We ignore session_id and just get all trips.
            # In a real multi‑user app you'd filter by user, but for this project
            # showing all trips is simplest and ensures nothing is hidden.
            from supabase import create_client
            from config.settings import SUPABASE_URL, SUPABASE_KEY
            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            resp = client.table("trips").select("*").order("created_at", desc=True).execute()
            st.session_state.all_trips = resp.data if resp.data else []
        except Exception as _e:
            print(f"[App] load trips: {_e}")


def _compress_image(file_bytes: bytes, max_dim: int = 1024, quality: int = 85) -> bytes:
    """
    Resize and recompress an uploaded image for storage and analysis.

    - Resizes so the longest side ≤ ``max_dim`` pixels.
    - Converts to RGB (strips alpha channel) so the image can be safely
      saved as JPEG regardless of whether the original was PNG, WebP, etc.
    - Saves as JPEG with the given ``quality``.

    Returns the compressed JPEG bytes.
    """
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(file_bytes))
    img.thumbnail((max_dim, max_dim))

    # JPEG does not support alpha transparency.
    # Remove any alpha channel by converting to RGB before saving.
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
def _estimate_initial_quantity(item_name: str, category: str,
                               trip_days: int, min_temp: float) -> int:
    """
    Returns a reasonable starting quantity for a packing list item
    based on trip length, temperature, and item category.
    These are soft defaults; users can override in the editable table.
    """
    name_lower = item_name.lower()

    # ── Clothing ─────────────────────────────────────────────────────────
    if category in ("Clothing", "Your Wardrobe"):
        # Tops: one for every 2 days, minimum 2
        if any(k in name_lower for k in ("t-shirt", "shirt", "top", "blouse",
                                          "sweater", "fleece", "pullover",
                                          "breathable", "thermal", "sleeve")):
            return max(2, trip_days // 2)

        # Bottoms: one for every 3-4 days, minimum 1
        if any(k in name_lower for k in ("jeans", "trousers", "pants",
                                          "shorts", "skirt")):
            return max(1, (trip_days // 3) if trip_days >= 3 else 1)

        # Outerwear: usually one piece is enough
        if any(k in name_lower for k in ("coat", "jacket", "windproof",
                                          "waterproof", "raincoat")):
            # If very cold, maybe suggest 2 heavy coats, else 1
            if min_temp < 0:
                return 2 if "winter" in name_lower or "heavy" in name_lower else 1
            return 1

        # Footwear: 1 pair, except for warm trips where sandals might be used
        if "shoe" in name_lower or "boot" in name_lower or "sneaker" in name_lower:
            return 1

        # Accessories: gloves, scarves, hats – typically 1 set
        if any(k in name_lower for k in ("glove", "scarf", "hat", "sock",
                                           "hand warmer")):
            # Socks: one pair per 2 days
            if "sock" in name_lower:
                return max(3, trip_days // 2)
            return 1

        # Business / formal / smart items – 1 outfit per 2 days
        if any(k in name_lower for k in ("business", "formal", "smart")):
            return max(1, trip_days // 2)

        # Default for any other clothing
        return 1

    # ── Gear / packing items ─────────────────────────────────────────────
    # Most gear items need only one
    if "umbrella" in name_lower:
        return 1
    if "sunscreen" in name_lower:
        return max(1, trip_days // 5)      # one bottle per 5 days
    if "water bottle" in name_lower:
        return max(1, trip_days // 10)
    if "charger" in name_lower or "adapter" in name_lower:
        return 1
    if "backpack" in name_lower or "bag" in name_lower:
        return 1
    if "map" in name_lower or "card" in name_lower:
        return 1

    # Generic fallback
    return 1

def _slot_chip(label: str, value, missing: bool = False) -> str:
    cls     = "slot-chip missing" if missing else "slot-chip"
    display = value if value else "—"
    return f'<span class="{cls}"><b>{label}:</b> {display}</span>'


def _slots_complete() -> bool:
    s = st.session_state.slots
    return all(s.get(f) for f in ("destination", "start_date", "end_date", "purpose"))


def _section_anchor(anchor_id: str, emoji: str, title: str):
    st.markdown(
        f'<div id="{anchor_id}" class="section-anchor">{emoji} {title}</div>',
        unsafe_allow_html=True,
    )


# ── First load ────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    _start_new_trip()
_load_all_trips()

# Auto‑open sections
if _slots_complete() and st.session_state.forecasts:
    st.session_state.sec3_open = True
if _slots_complete() and st.session_state.packing_list_items:
    st.session_state.sec4_open = True
if st.session_state.optimization_result:
    st.session_state.sec5_open = True


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧳 PackPal")
    st.caption("Smart Travel Packing Assistant")
    st.divider()

    # 1. New Trip button
    if st.button("＋ New Trip", type="primary", use_container_width=True):
        _start_new_trip()
        st.rerun()

    st.markdown(" ")

    # 2. Current Trip (slot chips)
    with st.expander("📍 **Current Trip**", expanded=True):
        s = st.session_state.slots
        if not s:
            st.caption("Chat to start capturing trip details.")
        else:
            chips = ""
            chips += _slot_chip("Destination", s.get("destination"), not s.get("destination"))
            chips += _slot_chip("From",        s.get("start_date"),  not s.get("start_date"))
            chips += _slot_chip("To",          s.get("end_date"),    not s.get("end_date"))
            chips += _slot_chip("Purpose",     s.get("purpose"),     not s.get("purpose"))
            if s.get("airline"):
                chips += _slot_chip("Airline",    s.get("airline"))
            if s.get("fare_class"):
                chips += _slot_chip("Class",      s.get("fare_class"))
            if s.get("baggage_weight_limit"):
                chips += _slot_chip("Baggage",    f"{s.get('baggage_weight_limit')} kg")
            if s.get("cold_tolerance"):
                chips += _slot_chip("Comfort",    s.get("cold_tolerance").replace("_", " "))
            if s.get("traveller_count"):
                chips += _slot_chip("Travellers", s.get("traveller_count"))
            st.markdown(chips, unsafe_allow_html=True)

        if st.session_state.baggage_limits:
            lim = st.session_state.baggage_limits
            st.success(
                f"✈️ Checked: **{lim.get('checked_kg','?')} kg** | "
                f"Carry‑on: **{lim.get('carry_on_kg','?')} kg**"
            )

    # 3. Jump to Section links (styled as pills)
    st.divider()
    slots_done = _slots_complete()
    opt_done   = bool(st.session_state.optimization_result)

    st.markdown("**Jump to Section**")
    st.markdown('<a href="#sec1" class="nav-btn">💬 Trip Details</a>', unsafe_allow_html=True)
    
    sec2_cls = "nav-btn" if slots_done else "nav-btn-disabled"
    sec3_cls = "nav-btn" if slots_done else "nav-btn-disabled"
    sec4_cls = "nav-btn" if slots_done else "nav-btn-disabled"
    sec5_cls = "nav-btn" if opt_done   else "nav-btn-disabled"
    st.markdown(f'<a href="#sec2" class="{sec2_cls}">👗 Clothing Image Upload</a>', unsafe_allow_html=True)
    st.markdown(f'<a href="#sec3" class="{sec3_cls}">🌤️ Weather Forecast</a>', unsafe_allow_html=True)
    st.markdown(f'<a href="#sec4" class="{sec4_cls}">📋 Recommended Packing List</a>', unsafe_allow_html=True)
    st.markdown(f'<a href="#sec5" class="{sec5_cls}">🧠 Recommendation Explanations</a>', unsafe_allow_html=True)

    # 4. Past Trips (if any)
    if st.session_state.all_trips:
        st.divider()
        st.markdown("**Past Trips**")
        for trip in st.session_state.all_trips:
            ctx   = trip.get("context_json", {})
            dest  = ctx.get("destination", "Unknown")
            start = ctx.get("start_date", "")
            end   = ctx.get("end_date",   "")
            label = f"📍 {dest}  {start} → {end}"
            is_active = trip.get("id") == st.session_state.trip_id
            if st.button(
                f"{'▶ ' if is_active else ''}{label}",
                key=f"trip_{trip['id']}",
                use_container_width=True,
            ):
                if SERVICES_OK:
                    full = db_client.get_trip(trip["id"])
                    if full:
                        _load_trip_from_db(full)
                        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("## 🧳 PackPal — Smart Travel Packer")
st.caption("Tell me about your trip and I'll handle everything else.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Chat (always visible)
# ════════════════════════════════════════════════════════════════════════════════

_section_anchor("sec1", "💬", "Trip Details")

with st.expander("💬 Trip Details", expanded=True):

    if UI_OK:
        render_chat()
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(msg.get("content", ""))

    if UI_OK and process_pill_if_clicked():
        st.session_state.needs_processing = True

    if prompt := st.chat_input("Tell me about your trip..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "render_pills": None}
        )
        st.session_state.needs_processing = True

# ── Process latest user message ───────────────────────────────────────────────
if st.session_state.needs_processing:
    user_input = ""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            user_input = msg["content"]
            break

    if user_input and SLOT_OK:
        with st.spinner("Thinking..."):
            history_without_last = st.session_state.messages[:-1]
            result = extract_slots(history_without_last, user_input)

            for field in ("destination", "country", "start_date", "end_date",
                          "purpose", "airline", "fare_class", "baggage_weight_limit",
                          "cold_tolerance", "activity_level", "traveller_count"):
                val = getattr(result, field, None)
                if val is not None:
                    st.session_state.slots[field] = val

            result.missing_slots = [
                f for f in result.missing_slots
                if not st.session_state.slots.get(f)
            ]

            st.session_state.messages.append({
                "role":         "assistant",
                "content":      result.next_prompt,
                "render_pills": result.render_pills,
            })

            if (KG_OK
                    and st.session_state.slots.get("airline")
                    and st.session_state.slots.get("fare_class")
                    and not st.session_state.baggage_limits):
                try:
                    lim = get_baggage_limit(
                        st.session_state.slots["airline"],
                        st.session_state.slots["fare_class"],
                    )
                    st.session_state.baggage_limits = lim
                    if not st.session_state.slots.get("baggage_weight_limit"):
                        st.session_state.slots["baggage_weight_limit"] = lim.get("checked_kg")
                except Exception as _e:
                    print(f"[App] Baggage lookup: {_e}")

            if _slots_complete() and not st.session_state.trip_id and SERVICES_OK:
                trip_id = db_client.create_trip(
                    st.session_state.browser_session_id,
                    st.session_state.slots,
                )
                st.session_state.trip_id     = trip_id
                st.session_state.trip_status = "gathering_context"
                st.session_state.sec3_open   = True
                st.session_state.sec4_open   = True

            if st.session_state.trip_id and SERVICES_OK:
                db_client.save_chat_history(
                    st.session_state.trip_id, st.session_state.messages
                )
                db_client.update_trip_context(
                    st.session_state.trip_id,
                    st.session_state.slots,
                    "gathering_context",
                )

    st.session_state.needs_processing = False
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Wardrobe Upload
# ════════════════════════════════════════════════════════════════════════════════

if _slots_complete():
    _section_anchor("sec2", "👗", "Clothing Image Upload")

    with st.expander("👗 Clothing Image Upload", expanded=True):

        uploaded_files = st.file_uploader(
            "Upload clothing photos (PNG / JPG / WEBP, max 2 MB each)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="main_uploader",
            help="Upload anytime — analysis runs when you click the button below.",
        )

        MAX_UPLOAD_BYTES = 2 * 1024 * 1024

        if uploaded_files:
            tmp_dir = Path(tempfile.gettempdir()) / "packpal_uploads"
            tmp_dir.mkdir(exist_ok=True)
            for f in uploaded_files:
                if f.name not in st.session_state.processed_filenames:
                    raw_bytes = f.read()
                    if len(raw_bytes) > MAX_UPLOAD_BYTES:
                        st.warning(f"⚠️ {f.name} exceeds 2 MB and was skipped.")
                        continue
                    compressed = _compress_image(raw_bytes)

                    upload_result = (
                        storage.upload_image(
                            compressed, f.name,
                            trip_context=st.session_state.slots,
                        ) if SERVICES_OK else {"url": "", "size_bytes": len(compressed)}
                    )

                    tmp_path = tmp_dir / f"{uuid.uuid4()}_{f.name}"
                    tmp_path.write_bytes(compressed)

                    st.session_state.pending_uploads.append({
                        "path":       str(tmp_path),
                        "name":       f.name,
                        "r2_url":     upload_result["url"],
                        "size_bytes": upload_result["size_bytes"],
                    })
                    st.session_state.processed_filenames.add(f.name)

        # ── Unified wardrobe grid ─────────────────────────────────────────────
        cv_lookup = {
            item["item_data_json"].get("r2_image_url"): item["item_data_json"]
            for item in st.session_state.wardrobe_items
        }

        # Gather images from R2 (source of truth)
        if SERVICES_OK and st.session_state.slots:
            r2_images = storage.list_images_in_folder(st.session_state.slots)
        else:
            r2_images = []

        display_items = []
        if r2_images:
            for img in r2_images:
                cv = cv_lookup.get(img["url"])
                display_items.append({
                    "url":        img["url"],
                    "size_bytes": img["size_bytes"],
                    "cv":         cv,
                })
        else:
            # offline fallback
            for item in st.session_state.pending_uploads:
                url = item.get("r2_url") or item.get("path", "")
                display_items.append({
                    "url":        url,
                    "size_bytes": item.get("size_bytes", 0),
                    "cv":         None,
                })
            for item in st.session_state.wardrobe_items:
                d   = item["item_data_json"]
                url = d.get("r2_image_url") or d.get("local_path", "")
                if not any(di["url"] == url for di in display_items):
                    display_items.append({
                        "url":        url,
                        "size_bytes": d.get("r2_size_bytes", 0),
                        "cv":         d,
                    })

        if display_items:
            pending_count  = sum(1 for d in display_items if d["cv"] is None)
            analysed_count = sum(1 for d in display_items if d["cv"] is not None)
            st.markdown(
                f"**{len(display_items)} item(s)** — "
                f"{analysed_count} analysed, {pending_count} pending"
            )
            wcols = st.columns(5)
            for i, d in enumerate(display_items):
                col = wcols[i % 5]
                try:
                    col.image(d["url"], width=90)
                except Exception:
                    col.markdown("🖼️")
                size_kb = d["size_bytes"] / 1024
                if d["cv"]:
                    col.caption(
                        f"**{d['cv']['detected_label']}**  \n"
                        f"CLO: {d['cv']['calculated_clo']} | "
                        f"{d['cv']['calculated_weight_g']}g  \n"
                        f"{size_kb:.0f} KB"
                    )
                else:
                    col.caption(f"Pending  \n{size_kb:.0f} KB")

        # ── Analyse button ────────────────────────────────────────────────────
        # pending items are those without CV data
        pending = [d for d in display_items if d["cv"] is None]

        c1, c2 = st.columns([2, 5])
        analyse_btn = c1.button(
            "🧠 Analyse Clothing",
            type="primary",
            disabled=(not pending or not CV_OK),
        )
        if not CV_OK:
            c2.warning("image_recognition module unavailable.")
        elif not pending:
            c2.caption("Upload photos above to enable analysis.")

        if analyse_btn and CV_OK and pending:
            # Build list of pending uploads to process
            pending_uploads_list = [
                pu for pu in st.session_state.pending_uploads
                if pu.get("r2_url") in {d["url"] for d in display_items if d["cv"] is None}
            ]
            prog = st.progress(0, text="Starting CV analysis...")
            for idx, item in enumerate(pending_uploads_list):
                img_path = Path(item["path"])
                try:
                    prog.progress(
                        int((idx / len(pending_uploads_list)) * 100),
                        text=f"Analysing {item['name']}...",
                    )
                    compressed = img_path.read_bytes()
                    r2_url     = item.get("r2_url", "")
                    size_bytes = item.get("size_bytes", len(compressed))

                    # ── Garment label detection ──────────────────────────
                    # Primary: use the custom cloth‑tool classifier when its
                    # trained model file exists. It provides 9 attribute
                    # predictions plus weight & volume directly.
                    if CLOTH_TOOL_MODEL.exists():
                        raw = _detect_cloth_tool(img_path)
                        if raw:
                            # Convert raw prediction to a readable label
                            clean_label = _cloth_tool_readable_label(raw)
                            # Map model outputs to internal thickness string
                            thickness = _CLOTH_THICKNESS_MAP.get(
                                raw.get("weight_class", "medium"), "medium"
                            )
                            weight_g = int(raw.get("approx_weight_g", 400))
                            volume_l = round(raw.get("approx_volume_L", 2.0), 2)
                        else:
                            clean_label = "Unknown Garment"
                            thickness   = "medium"
                            weight_g    = 400
                            volume_l    = 2.0
                    else:
                        # Fallback: YOLO detection + rule‑based estimation
                        label_raw   = _detect_yolo(img_path)
                        clean_label = label_raw.split("(")[0].strip() or "Unknown Garment"
                        dims        = estimate_dimensions_from_cv(img_path, clean_label)
                        thickness   = dims.get("thickness", "medium")
                        weight_g    = dims.get("weight_g", 400)
                        volume_l    = dims.get("volume_l", 2.0)

                    # ── CLO insulation lookup ────────────────────────────
                    # Uses the Neo4j Knowledge Graph (GarmentType node) if
                    # available; otherwise a built‑in ASHRAE table.
                    base_clo_val = 0.15
                    if KG_OK:
                        try:
                            from services.kg_client import get_ashrae_base_clo
                            base_clo_val = get_ashrae_base_clo(clean_label)
                        except Exception:
                            pass

                    # ── Save the analysis result ─────────────────────────
                    record = {"item_data_json": {
                        "r2_image_url":        r2_url,
                        "r2_size_bytes":       size_bytes,
                        "detected_label":      clean_label,
                        "thickness":           thickness,
                        "calculated_clo":      base_clo_val,
                        "calculated_weight_g": weight_g,
                        "calculated_volume_l": volume_l,
                    }}
                    st.session_state.wardrobe_items.append(record)
                    if SERVICES_OK and st.session_state.trip_id:
                        db_client.add_wardrobe_item(
                            st.session_state.trip_id, record["item_data_json"]
                        )
                    # Remove from pending_uploads
                    st.session_state.pending_uploads = [
                        p for p in st.session_state.pending_uploads
                        if p.get("r2_url") != r2_url
                    ]
                except Exception as _e:
                    st.error(f"Failed on {item['name']}: {_e}")

            prog.progress(100, text="Analysis complete!")
            # Force the packing list to rebuild on next run
            # so that newly analysed wardrobe items are included.
            st.session_state.packing_list_items = []
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Weather + CLO
# ════════════════════════════════════════════════════════════════════════════════

if _slots_complete():
    _section_anchor("sec3", "🌤️", "Weather Forecast")

    with st.expander("🌤️ Weather Forecast",
                     expanded=st.session_state.sec3_open):

        if not st.session_state.forecasts and BACKEND_OK:
            with st.spinner("Fetching historical weather data..."):
                try:
                    s   = st.session_state.slots
                    loc = get_location(s["destination"])
                    if not s.get("country"):
                        st.session_state.slots["country"] = loc.get("country", "")

                    forecasts = get_historical_forecast(
                        lat=loc["latitude"], lon=loc["longitude"],
                        timezone=loc["timezone"],
                        start_date=s["start_date"], end_date=s["end_date"],
                        n_years=10,
                    )
                    st.session_state.forecasts = forecasts

                    base_clo = (
                        kg_rules.calculate_base_weather_clo(forecasts)
                        if KG_RULES_OK else 0.3
                    )
                    st.session_state.base_clo = base_clo

                    if UI_OK:
                        pers = predict_clo_offset(st.session_state.slots, base_clo)
                        st.session_state.personalization_data = pers
                        st.session_state.adjusted_clo         = pers["adjusted_clo"]
                        try:
                            rf    = load_personalization_model()
                            st.session_state.shap_data = generate_personalization_shap(
                                pers["features"], rf
                            )
                        except Exception as _e:
                            print(f"[App] SHAP: {_e}")

                    if KG_RULES_OK and st.session_state.wardrobe_items:
                        st.session_state.clo_assessment = kg_rules.assess_wardrobe_suitability(
                            st.session_state.wardrobe_items,
                            st.session_state.adjusted_clo,
                            forecasts,
                        )

                    if SERVICES_OK and st.session_state.trip_id:
                        db_client.save_forecast(
                            st.session_state.trip_id,
                            [asdict(f) for f in forecasts],
                        )
                        db_client.update_trip_context(
                            st.session_state.trip_id,
                            st.session_state.slots,
                            "slots_complete",
                        )
                    st.session_state.trip_status = "slots_complete"

                except Exception as _e:
                    st.error(f"Weather fetch failed: {_e}")

        if st.session_state.forecasts and UI_OK:
            render_weather_plot(
                st.session_state.forecasts,
                st.session_state.slots.get("destination", ""),
            )
            render_clo_metrics(
                st.session_state.base_clo,
                st.session_state.adjusted_clo,
                st.session_state.clo_assessment,
            )
        elif not BACKEND_OK:
            st.warning("Backend modules unavailable — weather forecast cannot be generated.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Packing List
# ════════════════════════════════════════════════════════════════════════════════

if _slots_complete():
    _section_anchor("sec4", "📋", "Recommended Packing List")

    with st.expander("📋 Recommended Packing List",
                     expanded=st.session_state.sec4_open):

        if not st.session_state.packing_list_items and BACKEND_OK and st.session_state.forecasts:
            s   = st.session_state.slots
            ctx = TripContext(
                purpose=s.get("purpose", "tourism"),
                city=s.get("destination", ""),
                country=s.get("country", ""),
            )
            recs    = [recommend_day(f, ctx) for f in st.session_state.forecasts]
            ml_pack = build_trip_packing_list(recs)
            st.session_state.recommendations = recs

            draft = []
            # Compute trip length and coldest temp to estimate quantities
            if st.session_state.forecasts:
                trip_days = len(st.session_state.forecasts)
                min_temp  = min(f.temp_min for f in st.session_state.forecasts)
            else:
                trip_days = 7   # sensible defaults when no forecast yet
                min_temp  = 10

            for name in ml_pack.get("clothing", []):
                # Use the shared quantity function that considers trip duration and purpose
                qty = calculate_needed_quantity(name, trip_days, ctx.purpose)
                draft.append({
                    "name": name, "category": "Clothing", "quantity": qty,
                    "weight_g": int(_item_weight(name) * 1000) if OPT_OK else 300,
                    "volume_l": round(_item_weight(name) * 4, 1) if OPT_OK else 1.0,
                })
            for name in ml_pack.get("packing", []):
                qty = calculate_needed_quantity(name, trip_days, ctx.purpose)

                draft.append({
                    "name": name, "category": "Gear", "quantity": qty,
                    "weight_g": int(_item_weight(name) * 1000) if OPT_OK else 150,
                    "volume_l": round(_item_weight(name) * 4, 1) if OPT_OK else 0.5,
                })
            existing_names = {d["name"].lower() for d in draft}
            for wi in st.session_state.wardrobe_items:
                d     = wi["item_data_json"]
                label = d.get("detected_label", "")
                if label.lower() not in existing_names:
                    qty = calculate_needed_quantity(label, trip_days, ctx.purpose)
                    draft.append({
                        "name": label, "category": "Your Wardrobe", "quantity": qty,
                        "weight_g": d.get("calculated_weight_g", 300),
                        "volume_l": d.get("calculated_volume_l", 1.0),
                    })
            st.session_state.packing_list_items = draft
            if SERVICES_OK and st.session_state.trip_id:
                db_client.save_packing_list(st.session_state.trip_id, draft)
                db_client.save_recommendations(
                    st.session_state.trip_id,
                    [asdict(r) for r in recs],
                )

        # KG layering suggestion
        if st.session_state.forecasts and KG_RULES_OK:
            try:
                layering_advice = kg_rules.recommend_layering(
                    st.session_state.forecasts,
                    st.session_state.adjusted_clo,
                )
                if layering_advice:
                    st.success(f"🧥 **Layering advice from Knowledge Graph:** {layering_advice}")
            except Exception as _e:
                print(f"[App] KG layering failed: {_e}")

        for w in st.session_state.clo_assessment.get("warnings", []):
            st.warning(f"⚠️ {w}")

        if UI_OK and st.session_state.packing_list_items:
            final_items = render_packing_editor(st.session_state.packing_list_items)
            st.session_state.packing_list_items = final_items
            if SERVICES_OK and st.session_state.trip_id:
                db_client.save_packing_list(st.session_state.trip_id, final_items)

            st.markdown("---")
            weight_limit = float(
                st.session_state.slots.get("baggage_weight_limit")
                or st.session_state.baggage_limits.get("checked_kg", 20)
            )
            st.caption(f"Baggage weight limit: **{weight_limit} kg**")

            if st.button("✂️ Optimise for Baggage Limits", type="primary",
                         disabled=not OPT_OK):
                with st.spinner("Running Genetic Algorithm + 3D Volume Knapsack..."):
                    try:
                        s   = st.session_state.slots
                        ctx = TripContext(
                            purpose=s.get("purpose", "tourism"),
                            city=s.get("destination", ""),
                            country=s.get("country", ""),
                        )
                        result = optimise_dynamic_items(
                            wardrobe_items       = st.session_state.wardrobe_items,
                            ml_recommended_items = [
                                i["name"] for i in st.session_state.packing_list_items
                                if i.get("category") != "Your Wardrobe"
                            ],
                            weight_limit_kg = weight_limit,
                            volume_limit_l  = 40.0,
                            forecasts       = st.session_state.forecasts,
                            context         = ctx,
                        )
                        result_dict = result.to_dict()
                        st.session_state.optimization_result = result_dict
                        st.session_state.trip_status         = "optimized"
                        st.session_state.sec5_open           = True

                        if UI_OK:
                            highlights = "; ".join(
                                f"{f.date}: {f.temp_max}°C, {f.precipitation_mm}mm rain"
                                for f in st.session_state.forecasts[:3]
                            )
                            st.session_state.xai_narrative = generate_xai_narrative(
                                context              = st.session_state.slots,
                                weather_highlights   = highlights,
                                personalization_data = st.session_state.personalization_data,
                                clo_assessment       = st.session_state.clo_assessment,
                                optimization_summary = result_dict.get("stage3_summary", {}),
                            )

                        if SERVICES_OK and st.session_state.trip_id:
                            db_client.save_optimization(
                                st.session_state.trip_id, result_dict
                            )
                        st.rerun()

                    except Exception as _e:
                        st.error(f"Optimisation failed: {_e}")
        elif not BACKEND_OK:
            st.info("Complete slot detection to generate the draft packing list.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — XAI (SHAP, Luggage, LIME)
# ════════════════════════════════════════════════════════════════════════════════

if st.session_state.optimization_result:
    _section_anchor("sec5", "🧠", "Recommendation Explanations")

    with st.expander("🧠 Recommendation Explanations",
                     expanded=st.session_state.sec5_open):

        st.success("✅ Packing optimisation complete!")

        if UI_OK:
            tab1, tab2, tab3, tab4 = st.tabs([
                "🧠 AI Personalisation (SHAP)",
                "🧳 Luggage Allocation",
                "📝 XAI Explanation",
                "🔍 Clothing Recommender (LIME)",
            ])

            with tab1:
                if st.session_state.shap_data:
                    p = st.session_state.personalization_data
                    c1, c2 = st.columns(2)
                    c1.metric("Base Weather CLO",  f"{st.session_state.base_clo:.2f}")
                    c2.metric("Your Adjusted CLO", f"{st.session_state.adjusted_clo:.2f}",
                              delta=f"{p.get('offset', 0):+.3f}")
                    render_shap_plot(st.session_state.shap_data)
                else:
                    st.info("SHAP data not available — personalisation model may still be training.")

            with tab2:
                res        = st.session_state.optimization_result
                stage2     = res.get("stage2_knapsack", {})
                final_list = stage2.get("final_list", [])
                if final_list and OPT_OK:
                    chart_items = [
                        {"name": n, "weight_g": _item_weight(n) * 1000}
                        for n in final_list
                    ]
                    render_luggage_breakdown(chart_items, stage2)
                    stage3  = res.get("stage3_summary", {})
                    removed = stage3.get("removed_items", [])
                    if removed:
                        st.warning(f"**Removed:** {', '.join(removed)}")
                    st.info(stage3.get("basic_explanation", ""))
                else:
                    st.info("No optimisation result yet.")

            with tab3:
                if st.session_state.xai_narrative:
                    st.markdown(st.session_state.xai_narrative)
                else:
                    st.info("XAI narrative will appear here after optimisation.")

            with tab4:
                # LIME tab: pick a day, then show horizontal bar charts for each recommended item
                if st.session_state.recommendations and st.session_state.forecasts:
                    try:
                        from lime_explainer import explain_clothing_day
                        rec_dates = [r.date for r in st.session_state.recommendations]
                        selected_date = st.selectbox(
                            "Pick a day to explain:",
                            rec_dates,
                            key="lime_date",
                        )
                        if selected_date:
                            with st.spinner("Generating LIME explanation..."):
                                # Returns a list of Plotly figures
                                figures = explain_clothing_day(
                                    selected_date,
                                    st.session_state.recommendations,
                                    st.session_state.forecasts,
                                    st.session_state.slots.get("purpose", "tourism"),
                                )
                            if figures:
                                for fig in figures:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No items to explain.")
                    except Exception as _e:
                        st.warning(f"LIME explanation unavailable: {_e}")
                else:
                    st.info("LIME explanation will be available after recommendations are generated.")
        else:
            st.json(st.session_state.optimization_result)