"""
Lightweight configuration object for connecting to Neo4j.

The class centralizes the URI, username, password, and optional database name
so that scripts and unit tests can instantiate the connector with one object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    """Simple container for Neo4j connection parameters."""

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "letmein"
    database: str | None = None

    @classmethod
    def from_env(cls, prefix: str = "NEO4J") -> "Neo4jConfig":
        """
        Create a config object by reading environment variables.

        Supported variables (with defaults):
            - {prefix}_URI
            - {prefix}_USER
            - {prefix}_PASSWORD
            - {prefix}_DATABASE
        """
        return cls(
            uri=os.getenv(f"{prefix}_URI", cls.uri),
            user=os.getenv(f"{prefix}_USER", cls.user),
            password=os.getenv(f"{prefix}_PASSWORD", cls.password),
            database=os.getenv(f"{prefix}_DATABASE") or None,
        )

    def auth(self) -> tuple[str, str]:
        return (self.user, self.password)
