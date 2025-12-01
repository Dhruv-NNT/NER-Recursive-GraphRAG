"""
Graph RAG utilities built on top of Neo4j, LlamaIndex, and Ollama.

Call `build_graphrag_with_llamaindex` to spin up the full pipeline backed by
real services, or instantiate `GraphRAG` directly with stub retrievers for
testing.
"""

from .config import GraphRAGConfig
from .cypher_retriever import CypherHintRetriever, CYPHER_HINT_QUERY
from .llamaindex_adapters import build_graphrag_with_llamaindex
from .pipeline import ContextChunk, GraphRAG, GraphRAGResult

__all__ = [
    "ContextChunk",
    "CypherHintRetriever",
    "CYPHER_HINT_QUERY",
    "GraphRAG",
    "GraphRAGConfig",
    "GraphRAGResult",
    "build_graphrag_with_llamaindex",
]
