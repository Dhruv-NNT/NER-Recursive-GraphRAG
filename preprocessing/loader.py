"""
CSV loader for ATC transcripts.

The file in this project stores one transcript per row where the tokens and
BIO tags are space separated strings.  This module converts a row into a
structured dataclass so the remaining pipeline can focus on higher level
logic instead of worrying about CSV edge cases (missing fields, whitespace,
etc.).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ATCTranscript:
    """
    Structured representation of one labeled transcript.

    Using a dataclass makes the objects easy to inspect while keeping the code
    terse.  Each list is already tokenized so downstream steps can operate on
    them directly.
    """

    transcript_id: str
    tokens: List[str]
    tags: List[str]
    speaker: str
    intent: str


class CSVLoader:
    """
    Load ATC transcripts from a CSV file with robustness checks.

    The loader collects non-fatal issues instead of raising immediately so that
    downstream code can decide how to react.  For example, a row with a length
    mismatch is skipped but the rest of the file is still processed.
    """

    def __init__(self, csv_path: str | Path, strict: bool = False) -> None:
        self.csv_path = Path(csv_path)
        self.strict = strict
        self.errors: List[str] = []

    def load(self, limit: Optional[int] = None) -> List[ATCTranscript]:
        """
        Load rows from the CSV file.

        Args:
            limit: Optional cap on number of rows to load.

        Returns:
            List of ATCTranscript instances.
        """
        records: List[ATCTranscript] = []
        # enumerate gives us a 1-based index that we reuse as the fallback id
        for idx, row in enumerate(self._iter_rows(), start=1):
            try:
                record = self._parse_row(row, idx)
            except ValueError as exc:  # malformed row
                message = f"Row {idx}: {exc}"
                if self.strict:
                    raise ValueError(message) from exc
                self.errors.append(message)
                continue

            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        return records

    def _iter_rows(self) -> Iterable[dict]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        with self.csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield row

    def _parse_row(self, row: dict, idx: int) -> ATCTranscript:
        tokens_raw = (row.get("transcripts") or "").strip()
        tags_raw = (row.get("tags") or "").strip()
        speaker = (row.get("speaker") or "").strip()
        intent = (row.get("intent") or "").strip()

        if not tokens_raw or not tags_raw:
            raise ValueError("Missing tokens or tags")
        if not speaker or not intent:
            raise ValueError("Missing speaker or intent")

        # The CSV uses space separated tokens/tags, so a simple split is enough.
        tokens = tokens_raw.split()
        tags = tags_raw.split()

        if len(tokens) != len(tags):
            raise ValueError(
                f"Token/tag length mismatch ({len(tokens)} vs {len(tags)})"
            )

        # Some files might not carry an explicit id column, so fall back to the row number.
        transcript_id = row.get("id") or str(idx)

        return ATCTranscript(
            transcript_id=transcript_id,
            tokens=tokens,
            tags=tags,
            speaker=speaker,
            intent=intent,
        )
