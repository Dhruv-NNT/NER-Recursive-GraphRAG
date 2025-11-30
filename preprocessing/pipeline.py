"""
High-level entry point for Phase 1.

The `Phase1Pipeline` class assembles the components in this package to perform
three steps:
    1. Load transcripts from the CSV.
    2. Merge BIO tags into span-level entities.
    3. Optionally write the structured data to JSON for later phases.
"""

from __future__ import annotations

import json
from typing import List, Optional

from .loader import CSVLoader
from .json_export import JSONExporter
from .bio import BIOConverter, TranscriptExample


class Phase1Pipeline:
    """Runs the Phase 1 ingestion and preprocessing steps."""

    def __init__(self, csv_path: str, strict: bool = False) -> None:
        self.csv_path = csv_path
        self.strict = strict

    def run(
        self, limit: Optional[int] = None, output_path: Optional[str] = None
    ) -> List[TranscriptExample]:
        loader = CSVLoader(self.csv_path, strict=self.strict)
        transcripts = loader.load(limit=limit)
        if loader.errors:
            # surface data issues but keep going
            for error in loader.errors:
                print(f"[CSVLoader warning] {error}")

        examples = [BIOConverter.build_example(entry) for entry in transcripts]

        if output_path:
            JSONExporter.write_examples(examples, output_path)

        return examples

    def demo(self, limit: int = 3) -> None:
        """Prints a few processed examples for quick inspection."""
        examples = self.run(limit=limit)
        for example in examples:
            print(json.dumps(example.to_json(), indent=2))
