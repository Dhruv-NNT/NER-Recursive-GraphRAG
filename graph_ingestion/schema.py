"""
Creates the constraints needed by the ATC knowledge graph.

Creating uniqueness constraints up front prevents duplicated transcripts or
entities and also speeds up later MATCH/MERGE statements.
"""

from __future__ import annotations

from typing import Iterable

from .connector import Neo4jConnector


class SchemaManager:
    """Creates constraints and indexes required by the ATC graph schema."""

    CONSTRAINT_STATEMENTS: Iterable[str] = (
        """
        CREATE CONSTRAINT transcript_uid_unique IF NOT EXISTS
        FOR (t:Transcript)
        REQUIRE t.uid IS UNIQUE
        """,
        """
        CREATE CONSTRAINT entity_text_category_unique IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE (e.text, e.category) IS UNIQUE
        """,
        """
        CREATE CONSTRAINT category_name_unique IF NOT EXISTS
        FOR (c:Category)
        REQUIRE c.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT intent_name_unique IF NOT EXISTS
        FOR (i:IntentNode)
        REQUIRE i.name IS UNIQUE
        """,
    )

    def __init__(self, connector: Neo4jConnector) -> None:
        self.connector = connector

    def ensure_constraints(self) -> None:
        with self.connector.session() as session:
            for statement in self.CONSTRAINT_STATEMENTS:
                # run each constraint individually so errors do not block the others
                session.execute_write(
                    lambda tx, stmt=statement: tx.run(stmt.strip())
                )
