"""
Deterministic Cypher retrieval tuned for ATC transcript similarity.

This retriever always runs the same parameterized Cypher query, using simple
token overlap against Transcript.tokens and Entity.text to surface examples
that look like the incoming transmission. The goal is to give the LLM concrete
graph-grounded hints (speaker/intent/entities) without relying on a
text-to-Cypher model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from graph_ingestion.config import Neo4jConfig
from graph_ingestion.connector import Neo4jConnector

# We intentionally avoid APOC so this works on stock Neo4j installations.
CYPHER_HINT_QUERY = r"""
WITH toLower($query) AS q
WITH q,
     [w IN split(replace(replace(replace(q, ',', ' '), '.', ' '), '-', ' '), ' ')
      WHERE size(w) > 0] AS terms
MATCH (t:Transcript)
WITH t, terms, coalesce(t.tokens, []) AS tokens
OPTIONAL MATCH (t)-[:CONTAINS]->(e:Entity)-[:IS_A]->(c:Category)
WITH t, tokens, terms,
     collect(DISTINCT e) AS ents,
     collect(DISTINCT c.name) AS categories
WITH t, ents, categories, terms,
     size([term IN terms WHERE term IN tokens]) AS token_overlap,
     size([term IN terms WHERE any(e IN ents WHERE term IN split(coalesce(e.text, ''), ' '))]) AS entity_overlap
WITH t, ents, categories,
     (token_overlap * 1.0) + (entity_overlap * 1.5) AS score
WHERE score > 0
RETURN t.uid AS uid,
       t.text AS text,
       coalesce(t.intent, '') AS intent,
       coalesce(t.speaker, '') AS speaker,
       categories AS categories,
       [e IN ents | {text: e.text, label: e.category}] AS entities,
       coalesce(t.tokens, []) AS tokens,
       score
ORDER BY score DESC, uid
LIMIT coalesce($limit, 5)
"""


@dataclass
class CypherHit:
    """Simple container so GraphRAG can normalize Cypher results."""

    text: str
    metadata: dict
    score: float | None = None
    node_id: str | None = None


class CypherHintRetriever:
    """
    Fixed-query Cypher retriever returning transcript + entity context.

    The retriever mirrors the vector retriever API (retrieve(query, top_k=?))
    so the GraphRAG pipeline can treat both sources identically.
    """

    name = "CypherHintRetriever"

    def __init__(self, neo4j_config: Neo4jConfig, default_top_k: int = 5) -> None:
        self.neo4j_config = neo4j_config
        self.default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        similarity_top_k: Optional[int] = None,
    ) -> List[CypherHit]:
        limit = top_k or similarity_top_k or self.default_top_k

        params = {"query": query, "limit": limit}
        with Neo4jConnector(self.neo4j_config) as connector:
            with connector.session() as session:
                records = session.run(CYPHER_HINT_QUERY, params)
                return [self._record_to_hit(record) for record in records]

    @staticmethod
    def _record_to_hit(record) -> CypherHit:
        uid = record.get("uid")
        metadata = {
            "uid": uid,
            "intent": record.get("intent", ""),
            "speaker": record.get("speaker", ""),
            "categories": record.get("categories") or [],
            "entities": record.get("entities") or [],
            "tokens": record.get("tokens") or [],
            "full_text": record.get("text", ""),
            "score": record.get("score"),
        }
        return CypherHit(
            text=record.get("text", ""),
            metadata=metadata,
            score=record.get("score"),
            node_id=uid,
        )
