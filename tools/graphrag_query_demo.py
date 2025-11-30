"""
GraphRAG demo: ask a question over Neo4j using LlamaIndex + Ollama.

Usage (after ingesting data via phase2_ingest.py and starting Neo4j/Ollama):
    python tools/graphrag_query_demo.py \\
      --question "Who cleared Redcap 1727 to taxi via whiskey?"

Pass --validate-only to skip retrieval and only check that imports/config work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

try:
    from graph_rag import GraphRAGConfig, build_graphrag_with_llamaindex
except ModuleNotFoundError:  # pragma: no cover - allow running without install
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from graph_rag import GraphRAGConfig, build_graphrag_with_llamaindex


def build_config(args: argparse.Namespace) -> GraphRAGConfig:
    base = GraphRAGConfig.from_env()
    return GraphRAGConfig(
        neo4j_uri=args.neo4j_uri or base.neo4j_uri,
        neo4j_user=args.neo4j_user or base.neo4j_user,
        neo4j_password=args.neo4j_password or base.neo4j_password,
        neo4j_database=args.neo4j_database or base.neo4j_database,
        ollama_model=args.ollama_model or base.ollama_model,
        ollama_base_url=args.ollama_base_url or base.ollama_base_url,
        embedding_model=args.embedding_model or base.embedding_model,
        cypher_top_k=base.cypher_top_k if args.cypher_top_k is None else args.cypher_top_k,
        vector_top_k=base.vector_top_k if args.vector_top_k is None else args.vector_top_k,
        transcript_limit=(
            base.transcript_limit if args.transcript_limit is None else args.transcript_limit
        ),
        system_prompt=base.system_prompt if args.system_prompt is None else args.system_prompt,
    )


def format_context(label: str, chunks) -> str:
    lines: List[str] = [label]
    if not chunks:
        lines.append("  (none)")
        return "\n".join(lines)
    for idx, chunk in enumerate(chunks, start=1):
        uid = chunk.metadata.get("uid") or chunk.node_id or idx
        snippet = (chunk.text or "").replace("\n", " ").strip()
        lines.append(f"  {idx}. [{chunk.source}] uid={uid} text={snippet}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GraphRAG query demo using Neo4j + LlamaIndex + Ollama"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Who cleared Redcap 1727 to taxi via whiskey?",
        help="Natural language question to ask over the graph.",
    )
    parser.add_argument("--neo4j-uri", type=str, default=None)
    parser.add_argument("--neo4j-user", type=str, default=None)
    parser.add_argument("--neo4j-password", type=str, default=None)
    parser.add_argument("--neo4j-database", type=str, default=None)
    parser.add_argument("--ollama-model", type=str, default=None)
    parser.add_argument("--ollama-base-url", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--cypher-top-k", type=int, default=None)
    parser.add_argument("--vector-top-k", type=int, default=None)
    parser.add_argument(
        "--transcript-limit",
        type=int,
        default=None,
        help="Limit transcripts pulled for the vector index (speeds up demos).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional override for the answering prompt.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip retrieval/LLM calls; useful for quick import checks.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the full prompt (merged context) sent to the LLM.",
    )
    args = parser.parse_args()

    config = build_config(args)
    if args.validate_only:
        print("Config parsed successfully. Skipping retrieval (--validate-only).")
        return 0

    try:
        rag = build_graphrag_with_llamaindex(
            config, transcript_limit=config.transcript_limit
        )
    except Exception as exc:  # pragma: no cover - runtime dependency issues
        print(f"Failed to build GraphRAG: {exc}", file=sys.stderr)
        return 1

    try:
        result = rag.query(
            args.question,
            cypher_top_k=config.cypher_top_k,
            vector_top_k=config.vector_top_k,
        )
    except Exception as exc:  # pragma: no cover - runtime dependency issues
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1

    if result.warnings:
        print("=== Warnings ===")
        for w in result.warnings:
            print(f"- {w}")

    print(format_context("=== Cypher context ===", result.cypher_context))
    print(format_context("=== Vector context ===", result.vector_context))
    if args.show_prompt:
        print("\n=== LLM prompt (merged context) ===")
        print(result.prompt)
    print("\n=== Answer ===")
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
