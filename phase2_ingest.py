"""
Command-line helper for Phase 2 (graph ingestion).

The script can either:
    * Re-run Phase 1 on the CSV and ingest the in-memory output.
    * Or read a previously exported JSON file.
It then ensures the Neo4j schema exists and upserts every transcript/entity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from graph_ingestion import (
    GraphIngestor,
    Neo4jConfig,
    Neo4jConnector,
    SchemaManager,
)
from preprocessing import Phase1Pipeline
from preprocessing.bio import TranscriptExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 ingestion: CSV -> JSON -> Neo4j"
    )
    parser.add_argument("--csv", type=str, default="generation_train 2.csv")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Strict CSV parsing")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="Helloworld@123")
    parser.add_argument("--neo4j-database", type=str, default=None)
    return parser.parse_args()


def load_examples(args: argparse.Namespace) -> List[TranscriptExample]:
    if args.json:
        # Reading from JSON avoids reprocessing the CSV when iterating quickly.
        data = json.loads(Path(args.json).read_text(encoding="utf-8"))
        return [TranscriptExample.from_json(item) for item in data]

    pipeline = Phase1Pipeline(args.csv, strict=args.strict)
    return pipeline.run(limit=args.limit)


def main() -> None:
    args = parse_args()
    examples = load_examples(args)
    config = Neo4jConfig(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )

    with Neo4jConnector(config) as connector:
        SchemaManager(connector).ensure_constraints()
        GraphIngestor(connector).ingest_examples(examples)
    print(f"Ingested {len(examples)} transcripts into Neo4j.")


if __name__ == "__main__":
    main()
