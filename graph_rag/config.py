"""
Configuration container for the Graph RAG pipeline.

Values mirror the knobs referenced in Graph_RAG_Using_LLamaIndex.pdf so that
environment variables or CLI flags can be wired in without repeating boilerplate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GraphRAGConfig:
    """Connection + model settings for Neo4j, Ollama, and LlamaIndex."""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "Helloworld@123"
    neo4j_database: str | None = None
    ollama_model: str = "gemma3:12b"
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cypher_top_k: int = 5
    vector_top_k: int = 10
    transcript_limit: int | None = 500
    system_prompt: str = (
        "You are an Air Traffic Communication data structuring assistant. "
        "For each transmission, you must output JSON only with keys full_text, speaker, intent, "
        "entities (list of {text,label}). Labels: speaker in {CONTROLLER, PILOT}; "
        "intent in {GREETING, READBACK, TAXI, TRAFFIC, FREQUENCY, STANDBY, OTHER}; "
        "entity labels in [CALLSIGN, FREQUENCY, TAXIWAY, ACTION, O, GATE, CONTROLLER, VEHICLE, QUALIFIER, GREETING]. "
        "Use Cypher graph examples (highest scores first), then vector hints, then language priors. "
        "If unsure, leave text/label empty rather than guessing."
    )
    llm_timeout: float = 120.0
    neo4j_timeout: float | None = None

    @classmethod
    def from_env(cls, prefix: str = "NEO4J") -> "GraphRAGConfig":
        """Create a config object from environment variables."""

        transcript_limit_env = os.getenv("TRANSCRIPT_LIMIT")
        transcript_limit = (
            int(transcript_limit_env)
            if transcript_limit_env is not None
            else cls.transcript_limit
        )

        return cls(
            neo4j_uri=os.getenv(f"{prefix}_URI", cls.neo4j_uri),
            neo4j_user=os.getenv(f"{prefix}_USER", cls.neo4j_user),
            neo4j_password=os.getenv(f"{prefix}_PASSWORD", cls.neo4j_password),
            neo4j_database=os.getenv(f"{prefix}_DATABASE") or None,
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            cypher_top_k=int(os.getenv("CYPHER_TOP_K", cls.cypher_top_k)),
            vector_top_k=int(os.getenv("VECTOR_TOP_K", cls.vector_top_k)),
            transcript_limit=transcript_limit,
            system_prompt=os.getenv("GRAPH_RAG_SYSTEM_PROMPT", cls.system_prompt),
            llm_timeout=float(os.getenv("GRAPH_RAG_LLM_TIMEOUT", str(cls.llm_timeout))),
            neo4j_timeout=(
                float(os.getenv("GRAPH_RAG_NEO4J_TIMEOUT"))
                if os.getenv("GRAPH_RAG_NEO4J_TIMEOUT") is not None
                else cls.neo4j_timeout
            ),
        )
