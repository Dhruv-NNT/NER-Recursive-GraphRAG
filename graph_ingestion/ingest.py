"""
Functions that push the Phase 1 JSON output into Neo4j.

The ingestor merges Transcript, IntentNode, Entity, and Category nodes.  It
also maintains the canonical relationships so later retrieval agents can query
the graph without worrying about duplicates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from preprocessing.bio import TranscriptExample

from .connector import Neo4jConnector


class GraphIngestor:
    """Upserts transcripts, entities, and categories into Neo4j."""

    def __init__(self, connector: Neo4jConnector) -> None:
        self.connector = connector

    def ingest_examples(self, examples: Iterable[TranscriptExample]) -> None:
        with self.connector.session() as session:
            for example in examples:
                session.execute_write(
                    self._upsert_transcript, example.to_json()
                )

    def ingest_from_json(self, json_path: str | Path) -> None:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        examples = [TranscriptExample.from_json(item) for item in data]
        self.ingest_examples(examples)

    @staticmethod
    def _upsert_transcript(tx, payload: dict) -> None:
        uid = payload["transcript_id"]
        # UNWIND lets Neo4j handle each entity span in a single query call.
        tx.run(
            """
            MERGE (t:Transcript {uid: $uid})
            ON CREATE SET t.text = $text, t.speaker = $speaker, t.intent = $intent
            ON MATCH SET t.text = $text, t.speaker = $speaker, t.intent = $intent
            """,
            uid=uid,
            text=payload.get("full_text", ""),
            speaker=payload.get("speaker", ""),
            intent=payload.get("intent", ""),
        )

        tx.run(
            """
            MATCH (t:Transcript {uid: $uid})
            MERGE (intent:IntentNode {name: $intent})
            MERGE (t)-[:HAS_INTENT]->(intent)
            """,
            uid=uid,
            intent=payload.get("intent", ""),
        )

        entities = payload.get("entities", [])
        # Transcripts occasionally contain only 'O' tags; in that case we simply store the node.
        if not entities:
            return

        tx.run(
            """
            MATCH (t:Transcript {uid: $uid})
            UNWIND $entities AS entity
            MERGE (c:Category {name: entity.label})
            MERGE (e:Entity {text: entity.text, category: entity.label})
            ON CREATE SET e.normalizedText = toLower(entity.text)
            MERGE (t)-[:CONTAINS]->(e)
            MERGE (e)-[:IS_A]->(c)
            """,
            uid=uid,
            entities=entities,
        )
