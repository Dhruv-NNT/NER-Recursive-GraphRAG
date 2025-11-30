"""
Small helper for writing the processed transcript examples to disk.

Each example is already JSON serializable thanks to the dataclasses in
`bio.py`, so this module simply handles file creation and formatting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .bio import TranscriptExample


class JSONExporter:
    """Writes processed transcript examples to disk."""

    @staticmethod
    def write_examples(
        examples: Iterable[TranscriptExample], output_path: str | Path
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = [example.to_json() for example in examples]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
