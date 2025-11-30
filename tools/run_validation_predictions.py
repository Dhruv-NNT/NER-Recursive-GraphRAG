"""
Batch driver to run GraphRAG over validation transcripts and save predictions.

Reads a CSV with a `transcripts` column (default: generation_val 1.csv),
queries the GraphRAG pipeline row-by-row, and writes a CSV containing the
original text plus the JSON prediction and any warnings.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

try:  # allow running without installing the package
    from graph_rag import GraphRAGConfig, build_graphrag_with_llamaindex
except ModuleNotFoundError:  # pragma: no cover - runtime convenience
    repo_root = Path(__file__).resolve().parents[1]
    import sys

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from graph_rag import GraphRAGConfig, build_graphrag_with_llamaindex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GraphRAG predictions over a validation CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="generation_val 1.csv",
        help="Path to the validation CSV containing a 'transcripts' column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the CSV with model predictions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process (defaults to all rows).",
    )
    parser.add_argument(
        "--cypher-top-k",
        type=int,
        default=None,
        help="Optional override for Cypher retrieval depth.",
    )
    parser.add_argument(
        "--vector-top-k",
        type=int,
        default=None,
        help="Optional override for vector retrieval depth.",
    )
    return parser.parse_args()


def read_transcripts(csv_path: Path, limit: int | None) -> List[str]:
    transcripts: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "transcripts" not in (reader.fieldnames or []):
            raise ValueError("Input CSV must include a 'transcripts' column.")
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            transcripts.append(row.get("transcripts", "") or "")
    return transcripts


def write_predictions(out_path: Path, rows: List[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["row_id", "transcript", "prediction_json", "warnings"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    transcripts = read_transcripts(input_path, args.limit)

    config = GraphRAGConfig.from_env()
    try:
        rag = build_graphrag_with_llamaindex(config)
    except Exception as exc:
        raise SystemExit(f"Failed to build GraphRAG: {exc}")

    results: List[dict] = []
    for idx, text in enumerate(transcripts):
        try:
            res = rag.query(
                text,
                cypher_top_k=args.cypher_top_k,
                vector_top_k=args.vector_top_k,
            )
            results.append(
                {
                    "row_id": idx,
                    "transcript": text,
                    "prediction_json": res.answer,
                    "warnings": " | ".join(res.warnings) if res.warnings else "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "row_id": idx,
                    "transcript": text,
                    "prediction_json": "",
                    "warnings": f"error: {exc}",
                }
            )

    write_predictions(Path(args.output), results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
