"""
Unit tests for slot detection (regex-based fallback extraction).

Validates that the fallback correctly extracts required slots
and fills a TripSlots object. The LLM‑based extraction is tested
implicitly during UAT.

Usage:
    pytest tests/test_slot_detection.py -v
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from slot_detection import _fallback_extraction, TripSlots


def test_fallback_extracts_destination():
    """Should extract the first capitalised word as the destination."""
    result = _fallback_extraction([], "I am going to Tokyo next month")
    assert result.destination == "Tokyo"


def test_fallback_extracts_dates():
    """Should extract two YYYY-MM-DD dates as start and end."""
    result = _fallback_extraction([], "2026-08-11 to 2026-08-20")
    assert result.start_date == "2026-08-11"
    assert result.end_date == "2026-08-20"


def test_fallback_extracts_purpose():
    """Should recognise tourism‑related keywords."""
    result = _fallback_extraction([], "I'm planning a holiday trip")
    assert result.purpose == "tourism"


def test_fallback_missing_slots():
    """Should list slots that are still missing after extraction."""
    result = _fallback_extraction([], "I want to see Tokyo")
    assert "start_date" in result.missing_slots
    assert "end_date" in result.missing_slots
    assert "purpose" in result.missing_slots


def test_fallback_output_is_tripslots():
    """The result must always be a TripSlots instance."""
    result = _fallback_extraction([], "Hello")
    assert isinstance(result, TripSlots)