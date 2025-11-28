"""Export utilities for Phase 1 outputs."""
from __future__ import annotations

import json
from typing import Iterable, List, Optional

from .bio import EntitySpan


def build_json_record(
    *,
    transcript_id: str,
    tokens: List[str],
    speaker: str,
    intent: str,
    spans: Iterable[EntitySpan],
) -> dict:
    """Build a JSON-serializable record from tokens and spans."""

    full_text = " ".join(tokens)
    entities = [
        {"text": span.text, "label": span.label}
        for span in spans
    ]
    return {
        "transcript_id": transcript_id,
        "full_text": full_text,
        "speaker": speaker,
        "intent": intent,
        "entities": entities,
    }


def write_json(records: List[dict], output_path: str, indent: Optional[int] = 2) -> None:
    """Write a list of dictionaries to a JSON file."""

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=indent)
