"""
Evaluate LLM + SBERT slot detection – Precision@1.

Uses the full extract_slots function (Groq LLM + SBERT) to process
a small set of test utterances and compares the extracted slots
against expected values.

Requires a valid GROQ_API_KEY in your .env file.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from slot_detection import extract_slots

# Test utterances with expected slot values
test_cases = [
    (
        "I am going to Tokyo from 2026-08-11 to 2026-08-20 for a holiday",
        {"destination": "Tokyo", "start_date": "2026-08-11",
         "end_date": "2026-08-20", "purpose": "tourism"},
    ),
    (
        "Business trip to Singapore on 2026-08-01 returning 2026-08-05",
        {"destination": "Singapore", "start_date": "2026-08-01",
         "end_date": "2026-08-05", "purpose": "business"},
    ),
    (
        "I'll be visiting my family in London from next week",
        {"destination": "London", "purpose": "visiting"},
    ),
]

correct = 0
total = 0

for utterance, expected in test_cases:
    # extract_slots expects the full chat history and the latest user message
    result = extract_slots([], utterance)

    for field, exp_val in expected.items():
        total += 1
        if getattr(result, field) == exp_val:
            correct += 1

precision = correct / total if total else 0.0
print(f"LLM + SBERT Slot Detection Precision@1: {precision:.2f} ({correct}/{total})")