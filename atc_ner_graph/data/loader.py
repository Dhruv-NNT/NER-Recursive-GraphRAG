"""Data loading utilities for ATC transcript datasets.

This module focuses on Phase 1 ingestion: reading CSV files and
returning structured rows with tokens, BIO tags, speaker, and intent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional
import pandas as pd


@dataclass
class TranscriptRecord:
    """Structured representation of a single transcript row."""

    tokens: List[str]
    tags: List[str]
    speaker: str
    intent: str
    transcript_id: Optional[str] = None


class CsvLoader:
    """CSV reader that validates token/tag alignment and missing fields."""

    def __init__(self, path: str):
        self.path = path

    def load(self, limit: Optional[int] = None) -> Iterable[TranscriptRecord]:
        """Yield ``TranscriptRecord`` objects from the CSV file.

        Args:
            limit: Optional maximum number of rows to return.

        Yields:
            ``TranscriptRecord`` for each valid row.

        Raises:
            ValueError: if tokens and tags are misaligned in length.
        """

        df = pd.read_csv(self.path)
        if limit is not None:
            df = df.head(limit)

        for idx, row in df.iterrows():
            try:
                tokens = self._safe_split(row.get("transcripts"))
                tags = self._safe_split(row.get("tags"))
                speaker = self._safe_strip(row.get("speaker"))
                intent = self._safe_strip(row.get("intent"))
            except ValueError as exc:
                raise ValueError(f"Row {idx}: {exc}") from exc

            if len(tokens) != len(tags):
                raise ValueError(
                    f"Row {idx}: token/tag length mismatch ({len(tokens)} vs {len(tags)})"
                )

            yield TranscriptRecord(
                tokens=tokens,
                tags=tags,
                speaker=speaker,
                intent=intent,
                transcript_id=str(idx),
            )

    @staticmethod
    def _safe_split(value: Optional[str]) -> List[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("missing required text field")
        value = str(value).strip()
        if not value:
            raise ValueError("empty text field")
        return value.split()

    @staticmethod
    def _safe_strip(value: Optional[str]) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            raise ValueError("missing required field")
        return str(value).strip()
