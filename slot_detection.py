"""
Slot Detection & Extraction Engine
=====================================
Uses LangChain + Groq (llama-3.3-70b) for conversational slot filling,
with SBERT semantic validation and Pydantic schema enforcement.

Full slot list (per session_clarifications.txt):
    destination      — city (compulsory); country auto-extracted via geocoder
    start_date       — YYYY-MM-DD (default year is current or next year)
    end_date         — YYYY-MM-DD
    purpose          — normalised to: business | tourism | visiting
    airline          — free text, validated against KG
    fare_class       — economy | premium_economy | business | first
    baggage_weight_limit — auto-filled from KG; user can override
    cold_tolerance   — very_cold_sensitive | cold_sensitive | neutral |
                       heat_sensitive | very_heat_sensitive (5-point scale)
    activity_level   — relaxed | moderate | highly_active
    traveller_count  — integer (captured; used for future enhancement)

Changes from previous version:
- SYSTEM_PROMPT now injects today's actual date so LLM uses the correct year
- Date validation blocks past dates (not just past years)
- Singapore / city-state / province detection added — prompts for city if needed
- Fallback also validates dates are in the future
"""

import json
import re
from datetime import date as dt_date
from typing import List, Optional
from pydantic import BaseModel, Field

from config.settings import DEPS_AVAILABLE
from services import llm_client


# ── SBERT Semantic Normalisation Engine ──────────────────────────────────────
# One canonical anchor phrase per valid slot value.
# SBERT embeds user input + anchors, cosine similarity picks the winner.
# No hardcoded string-match dictionaries — new phrasings work automatically.

_SLOT_ANCHORS = {
    "purpose": {
        "business":  "a business, work, conference or corporate trip",
        "tourism":   "a holiday, vacation, leisure or sightseeing trip",
        "visiting":  "visiting family, friends or relatives",
    },
    "cold_tolerance": {
        "very_cold_sensitive":  "I get very cold easily, I always need extra layers",
        "cold_sensitive":       "I tend to feel cold more than most people",
        "neutral":              "I am comfortable in most temperatures",
        "heat_sensitive":       "I tend to feel warm and overheat sometimes",
        "very_heat_sensitive":  "I overheat very easily, I always feel too hot",
    },
    "activity_level": {
        "relaxed":       "relaxed sightseeing, leisure and light walking",
        "moderate":      "moderate walking, some activities, typical tourist pace",
        "highly_active": "hiking, trekking, sports, cycling or intense physical activity",
    },
    "fare_class": {
        "economy":          "economy class, standard cabin, cheapest seat",
        "premium_economy":  "premium economy, extra legroom, mid-tier cabin",
        "business":         "business class, lie-flat seat, premium cabin",
        "first":            "first class, luxury suite, top cabin",
    },
    "airline": {
        "Singapore Airlines": "Singapore Airlines SIA SQ full service premium carrier",
        "Scoot":              "Scoot TR budget low cost airline Singapore",
        "Jetstar":            "Jetstar 3K budget low cost airline",
        "AirAsia":            "AirAsia AK budget low cost airline Asia",
        "Cathay Pacific":     "Cathay Pacific CX Hong Kong full service airline",
        "Emirates":           "Emirates EK Dubai luxury full service airline",
        "Qantas":             "Qantas QF Australian national full service airline",
        "Lufthansa":          "Lufthansa LH German full service airline",
        "Delta Air Lines":    "Delta DL American US full service airline",
        "Qatar Airways":      "Qatar Airways QR Doha premium full service airline",
        "EasyJet":            "EasyJet U2 European budget low cost airline",
    },
}

_SBERT_MODEL  = None   # lazy singleton
_SBERT_CACHE  = {}     # cache of pre-computed anchor embeddings per slot
_SBERT_THRESHOLD = 0.25  # minimum cosine similarity to accept a match


def _get_sbert():
    """Loads SBERT model once and caches it."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("[SBERT] Loading all-MiniLM-L6-v2 (first time only)...")
            _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            print("[SBERT] Model ready.")
        except Exception as e:
            print(f"[SBERT] Could not load model: {e}")
            _SBERT_MODEL = None
    return _SBERT_MODEL


def _anchor_embeddings(slot: str):
    """Returns cached embeddings for a slot's anchor phrases."""
    if slot not in _SBERT_CACHE:
        model = _get_sbert()
        if model is None:
            return None, None
        anchors     = _SLOT_ANCHORS[slot]
        keys        = list(anchors.keys())
        phrases     = list(anchors.values())
        embeddings  = model.encode(phrases, convert_to_tensor=True)
        _SBERT_CACHE[slot] = (keys, embeddings)
    return _SBERT_CACHE[slot]


def semantic_normalise(slot: str, raw_value: str) -> str:
    """
    Maps any raw user/LLM string to the canonical slot value using SBERT.

    Args:
        slot:      One of: purpose | cold_tolerance | activity_level |
                   fare_class | airline
        raw_value: Raw string from LLM or user input.

    Returns:
        Canonical value string (e.g. "business", "very_cold_sensitive"),
        or raw_value unchanged if model unavailable or similarity too low.
    """
    if not raw_value or slot not in _SLOT_ANCHORS:
        return raw_value

    model = _get_sbert()
    if model is None:
        return raw_value  # graceful fallback — no crash

    try:
        import torch
        keys, anchor_embs = _anchor_embeddings(slot)
        if keys is None:
            return raw_value

        query_emb = model.encode(raw_value, convert_to_tensor=True)

        # Cosine similarity
        cos_scores = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), anchor_embs
        )
        best_idx   = int(cos_scores.argmax())
        best_score = float(cos_scores[best_idx])

        if best_score >= _SBERT_THRESHOLD:
            canonical = keys[best_idx]
            print(f"[SBERT] '{raw_value}' → '{canonical}' (score={best_score:.2f})")
            return canonical
        else:
            print(f"[SBERT] Low confidence for '{raw_value}' (score={best_score:.2f}) — keeping raw")
            return raw_value

    except Exception as e:
        print(f"[SBERT] Normalisation error: {e}")
        return raw_value




class TripSlots(BaseModel):
    """Strict schema the LLM must populate. All fields Optional until confirmed."""
    destination:          Optional[str]   = Field(None, description="City name, e.g. 'Tokyo'")
    country:              Optional[str]   = Field(None, description="Country, auto-extracted from city")
    start_date:           Optional[str]   = Field(None, description="YYYY-MM-DD format")
    end_date:             Optional[str]   = Field(None, description="YYYY-MM-DD format")
    purpose:              Optional[str]   = Field(None, description="business | tourism | visiting")
    airline:              Optional[str]   = Field(None, description="Airline name if mentioned")
    fare_class:           Optional[str]   = Field(None, description="economy | premium_economy | business | first")
    baggage_weight_limit: Optional[float] = Field(None, description="Checked baggage limit in kg")
    cold_tolerance:       Optional[str]   = Field(None,
        description="very_cold_sensitive | cold_sensitive | neutral | heat_sensitive | very_heat_sensitive")
    activity_level:       Optional[str]   = Field(None, description="relaxed | moderate | highly_active")
    traveller_count:      Optional[int]   = Field(None, description="Number of travellers")
    missing_slots:        List[str]       = Field(default_factory=list)
    next_prompt:          str             = Field("", description="Natural conversational question for the user")
    render_pills:         Optional[dict]  = Field(None,
        description="e.g. {'fare_class': ['economy', 'business', 'first']}")


# ── System Prompt builder ─────────────────────────────────────────────────────
# Built as a function so today's date is always current at call time,
# not frozen at module import time.

def _build_system_prompt() -> str:
    today     = dt_date.today()
    today_str = today.strftime("%d %B %Y")   # e.g. "25 April 2026"
    # Default year: if we're in Oct-Dec, default to next year; otherwise current year
    default_year = today.year + 1 if today.month >= 10 else today.year

    return f"""
You are PackPal, a friendly smart travel packing assistant.
Gather trip details through natural conversation. ONE question at a time.

TODAY'S DATE: {today_str}
DEFAULT YEAR FOR DATES: {default_year}

SLOT PRIORITY ORDER (ask in this sequence if missing):
1. destination (specific city — NOT just a country or region)
2. start_date and end_date together
3. purpose
4. airline + fare_class together (one question)
5. cold_tolerance (personal comfort preference)
6. activity_level
7. traveller_count (optional — ask last)

CRITICAL RULES:
1. CITY REQUIRED: destination must be a specific city, not a country, state, province,
   or prefecture. Examples of INVALID destinations: "Japan", "California", "Kanto",
   "Ontario". VALID: "Tokyo", "Los Angeles", "Singapore" (Singapore is both a city
   and country — accept it as-is). If the user gives a region, ask for the city.
2. DATES: must be output in YYYY-MM-DD format. Be forgiving with user input:
   - If the user gives a full date like "20 Dec" or "Dec 20" or "20/12" without a year,
     automatically assume {default_year} and output "{default_year}-12-20".
   - If the user gives "20 Dec 2026", output "2026-12-20".
   - If the user gives just a month like "December", ask for the specific day.
   - Dates MUST be in the future (after {today_str}). If the user gives a past date,
     tell them politely and ask for a future date.
   - NEVER leave start_date or end_date null just because the year was not stated —
     always assume {default_year} for the missing year.
3. PURPOSE: normalise to exactly one of: business, tourism, visiting.
4. AIRLINE: accept free text. Do not force a choice.
5. FARE CLASS: offer pills — {{"fare_class": ["economy", "premium_economy", "business", "first"]}}.
6. COLD TOLERANCE: offer pills — {{"cold_tolerance": ["Very cold sensitive", "Cold sensitive", "Neutral", "Heat sensitive", "Very heat sensitive"]}}.
   Map display labels to internal values: very_cold_sensitive | cold_sensitive | neutral | heat_sensitive | very_heat_sensitive.
7. ACTIVITY LEVEL: offer pills — {{"activity_level": ["Relaxed / Sightseeing", "Moderate", "Highly Active / Hiking"]}}.
   Map to: relaxed | moderate | highly_active.
8. next_prompt MUST always be a warm, natural human-readable sentence — never a raw field name.
9. MANDATORY: You MUST ALWAYS append a valid JSON object at the very end of EVERY response,
   no exceptions. Even if you only have one thing to say, still end with the JSON block.
   Never omit the JSON. Never say you cannot provide JSON.
10. Keep missing_slots updated. Core required slots: destination, start_date, end_date, purpose.
"""


# ── Main extraction function ──────────────────────────────────────────────────

def extract_slots(chat_history: List[dict], user_input: str) -> TripSlots:
    """
    Processes user input and returns extracted slots + UI instructions.

    Args:
        chat_history: List of {"role": "user"|"assistant", "content": str}
        user_input:   Latest user message text.

    Returns:
        TripSlots with filled fields, missing_slots, next_prompt, render_pills.
    """
    if not llm_client.is_llm_available():
        return _fallback_extraction(chat_history, user_input)

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    llm   = llm_client.get_llm(temperature=0.1)

    try:
        # Build message list directly — avoids LangChain template parsing
        # which would mis-interpret the JSON examples in the system prompt
        # as template variables (e.g. {"fare_class": ...} → variable error).
        system_msg = SystemMessage(content=_build_system_prompt())

        lc_messages = [system_msg]
        for msg in chat_history:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))
        lc_messages.append(HumanMessage(content=user_input))

        raw_response  = llm.invoke(lc_messages)
        response_text = raw_response.content

        # If LLM returned a conversational reply with no JSON block,
        # send a follow-up message explicitly requesting the JSON output.
        if "{" not in response_text:
            retry_messages = lc_messages + [
                AIMessage(content=response_text),
                HumanMessage(content=(
                    "Please now output the JSON slot object as required, "
                    "reflecting everything gathered so far in this conversation."
                )),
            ]
            raw_response  = llm.invoke(retry_messages)
            response_text = raw_response.content

        # Bulletproof JSON extraction
        start_idx = response_text.find("{")
        if start_idx == -1:
            raise ValueError("No JSON block found in LLM response.")

        decoder   = json.JSONDecoder()
        data, _   = decoder.raw_decode(response_text, start_idx)
        conv_text = response_text[:start_idx].strip()

        # ── Date validation + auto-correction ────────────────────────────────
        today        = dt_date.today()
        default_year = today.year + 1 if today.month >= 10 else today.year

        for date_field in ("start_date", "end_date"):
            val = data.get(date_field)
            if val and "-" in str(val):
                try:
                    parsed = dt_date.fromisoformat(str(val))
                    if parsed <= today:
                        # Try auto-correcting by bumping to default year
                        corrected = parsed.replace(year=default_year)
                        if corrected > today:
                            print(f"[Slot Detection] Auto-corrected {val} → {corrected}")
                            data[date_field] = corrected.isoformat()
                        else:
                            # Even with corrected year it's still in the past — ask user
                            print(f"[Slot Detection] Blocked past date: {val}")
                            data[date_field] = None
                            data.setdefault("missing_slots", [])
                            if date_field not in data["missing_slots"]:
                                data["missing_slots"].append(date_field)
                            data["next_prompt"] = (
                                f"That date looks like it's already passed. "
                                f"Today is {today.strftime('%d %B %Y')} — "
                                f"could you give me your travel dates?"
                            )
                except ValueError:
                    data[date_field] = None

        # ── SBERT semantic normalisation for all categorical slots ───────────
        # Replaces hardcoded string-match dictionaries.
        # Works for any phrasing — "work trip", "trekking", "SIA", etc.
        for slot in ("purpose", "cold_tolerance", "activity_level",
                     "fare_class", "airline"):
            raw = data.get(slot)
            if raw:
                data[slot] = semantic_normalise(slot, str(raw))

        # Smart fallback: use conversational text as next_prompt if LLM left blank
        if conv_text and not data.get("next_prompt"):
            data["next_prompt"] = conv_text

        # Safety: always have a next_prompt if slots are still missing
        if not data.get("next_prompt") and data.get("missing_slots"):
            data["next_prompt"] = (
                f"Just a couple more things — I still need: "
                f"{', '.join(data['missing_slots'])}."
            )

        return TripSlots(**data)

    except Exception as e:
        print(f"[Slot Detection Error] {e}")
        return _fallback_extraction(chat_history, user_input)


# ── Fallback: Regex / Keyword extraction ──────────────────────────────────────

def _fallback_extraction(chat_history: List[dict], user_input: str) -> TripSlots:
    """
    Regex + keyword fallback when Groq is unavailable.
    """
    print("[Slot Detection] FALLBACK MODE — Groq unavailable. Using regex.")
    slots = TripSlots()
    text  = user_input.lower()
    today        = dt_date.today()
    default_year = today.year + 1 if today.month >= 10 else today.year

    # Dates — extract YYYY-MM-DD and validate/auto-correct
    dates_found = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", user_input)
    for d_str in dates_found:
        try:
            parsed = dt_date.fromisoformat(d_str)
            if parsed <= today:
                parsed = parsed.replace(year=default_year)
            if parsed > today:
                if slots.start_date is None:
                    slots.start_date = parsed.isoformat()
                elif slots.end_date is None:
                    slots.end_date = parsed.isoformat()
        except ValueError:
            pass

    # Destination – pick capitalised words but ignore common English pronouns
    IGNORE_WORDS = {
        "I", "I'm", "Ive", "I'll", "I'd", "We", "We're", "Weve", "We'll",
        "You", "You're", "Youve", "You'll", "He", "She", "It", "They",
        "Me", "Myself", "Him", "Her", "Us", "Them", "My", "Our", "Your",
    }
    cap_words = [
        w for w in user_input.split()
        if w and w[0].isupper() and w.strip(",.?!") not in IGNORE_WORDS
    ]
    if cap_words:
        slots.destination = " ".join(cap_words)

    # Purpose — use SBERT for semantic matching
    # Looks for purpose-like phrases anywhere in the input
    purpose_hints = re.findall(
        r"(business|work|conference|holiday|vacation|tourism|visit|leisure|"
        r"sightseeing|meeting|corporate|family|friends|trekking|hiking)",
        text, re.IGNORECASE
    )
    if purpose_hints:
        slots.purpose = semantic_normalise("purpose", " ".join(purpose_hints))

    # Traveller count
    count_match = re.search(r"\b(\d+)\s*(person|people|traveller|passenger)", text)
    if count_match:
        slots.traveller_count = int(count_match.group(1))

    # Determine missing required slots
    required = ["destination", "start_date", "end_date", "purpose"]
    slots.missing_slots = [s for s in required if getattr(slots, s) is None]

    # Guide to next missing slot
    if not slots.missing_slots and not slots.fare_class:
        slots.missing_slots.append("fare_class")
        slots.render_pills = {"fare_class": ["economy", "premium_economy", "business", "first"]}
        slots.next_prompt  = "What cabin class will you be flying in?"
    elif not slots.missing_slots and not slots.cold_tolerance:
        slots.missing_slots.append("cold_tolerance")
        slots.render_pills = {"cold_tolerance": [
            "Very cold sensitive", "Cold sensitive", "Neutral",
            "Heat sensitive", "Very heat sensitive",
        ]}
        slots.next_prompt = "How do you usually handle cold weather?"
    elif slots.missing_slots:
        slots.next_prompt = (
            f"Could you tell me your {slots.missing_slots[0].replace('_', ' ')}?"
        )
    else:
        slots.next_prompt = "Great, I think I have everything I need!"

    return slots