"""
Glue between our code and the official Neo4j driver.

Neo4j is optional until Phase 2 is used, so this file includes an import guard
that raises a friendly error if the `neo4j` package has not been installed.
"""

from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - import guard for optional dependency
    from neo4j import Driver, GraphDatabase, Session
except ImportError as exc:  # pragma: no cover
    Driver = Session = None  # type: ignore
    GraphDatabase = None  # type: ignore
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None

from .config import Neo4jConfig


class Neo4jConnector:
    """Lazy Neo4j driver wrapper with context-manager support."""

    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self._driver: Optional[Driver] = None

    def connect(self) -> Driver:
        if GraphDatabase is None:
            raise ImportError(
                "The 'neo4j' package is required. Install via 'pip install neo4j'."
            ) from _IMPORT_ERROR
        if self._driver is None:
            # Create the driver lazily so scripts that only need preprocessing
            # do not require Neo4j dependencies.
            self._driver = GraphDatabase.driver(
                self.config.uri, auth=self.config.auth()
            )
        return self._driver

    def session(self) -> Session:
        driver = self.connect()
        if self.config.database:
            return driver.session(database=self.config.database)
        return driver.session()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jConnector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
