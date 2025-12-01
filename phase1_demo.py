"""
Quick demo script for Phase 1 preprocessing.

Run this file to load a few rows from the CSV, convert them into the JSON
structure, and print the result.  It is handy on the HPC cluster when you want
to spot-check the preprocessing output without running the heavier phases.
"""

from __future__ import annotations

import argparse

from preprocessing import Phase1Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 preprocessing on ATC transcripts."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="generation_train 2.csv",
        help="Path to the source CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of rows to process for the demo.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = Phase1Pipeline(args.csv)
    examples = pipeline.run(limit=args.limit, output_path=args.output)
    for example in examples:
        print(example.to_json())


if __name__ == "__main__":
    main()
