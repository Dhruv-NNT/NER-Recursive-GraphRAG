"""Phase 1 pipeline: load CSV, merge BIO tags, validate, and export JSON."""
from __future__ import annotations

from typing import List, Optional

from ..data.loader import CsvLoader
from ..preprocess.bio import merge_bio_spans
from ..preprocess.validator import validate_token_coverage
from ..preprocess.json_export import build_json_record, write_json


def process_csv_to_json(
    csv_path: str, output_path: str, limit: Optional[int] = None
) -> List[dict]:
    """Run the full Phase 1 preprocessing pipeline.

    Args:
        csv_path: Path to the input CSV file.
        output_path: Destination path for the JSON output.
        limit: Optional limit on number of rows to process.

    Returns:
        List of JSON-compatible dictionaries.
    """

    loader = CsvLoader(csv_path)
    json_records: List[dict] = []

    for record in loader.load(limit=limit):
        spans = merge_bio_spans(record.tokens, record.tags)
        validate_token_coverage(record.tokens, spans)
        json_record = build_json_record(
            transcript_id=record.transcript_id or "",  # fallback empty string if None
            tokens=record.tokens,
            speaker=record.speaker,
            intent=record.intent,
            spans=spans,
        )
        json_records.append(json_record)

    write_json(json_records, output_path)
    return json_records


def demo(csv_path: str, limit: int = 2) -> None:
    """Run a demo on a small subset and print the resulting JSON records."""

    records = process_csv_to_json(csv_path, output_path="/tmp/phase1_demo.json", limit=limit)
    for rec in records:
        print(rec)
