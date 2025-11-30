"""
Static description of the ATC property graph schema.

We keep a lightweight, code-level schema so GraphRAG can operate even when
Neo4j instances do not have the APOC plugin installed.  The structure mirrors
the output shape of `Neo4jPropertyGraphStore.refresh_schema`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class NodeProperty:
    property: str
    type: str


@dataclass(frozen=True)
class RelationshipDescription:
    start: str
    type: str
    end: str


NODE_PROPERTIES: Dict[str, List[NodeProperty]] = {
    "Transcript": [
        NodeProperty("uid", "STRING"),
        NodeProperty("text", "STRING"),
        NodeProperty("speaker", "STRING"),
        NodeProperty("intent", "STRING"),
        NodeProperty("tokens", "LIST"),
        NodeProperty("bio_tags", "LIST"),
    ],
    "Entity": [
        NodeProperty("text", "STRING"),
        NodeProperty("category", "STRING"),
        NodeProperty("normalizedText", "STRING"),
        NodeProperty("surfaceText", "STRING"),
    ],
    "Category": [NodeProperty("name", "STRING")],
    "IntentNode": [NodeProperty("name", "STRING")],
}

RELATIONSHIPS: List[RelationshipDescription] = [
    RelationshipDescription("Transcript", "CONTAINS", "Entity"),
    RelationshipDescription("Entity", "IS_A", "Category"),
    RelationshipDescription("Transcript", "HAS_INTENT", "IntentNode"),
]


def build_structured_schema() -> dict:
    """Return a dict compatible with Neo4jPropertyGraphStore.structured_schema."""

    node_props = {
        label: [asdict(prop) for prop in props] for label, props in NODE_PROPERTIES.items()
    }
    rel_props = {rel.type: [] for rel in RELATIONSHIPS}

    return {
        "node_props": node_props,
        "rel_props": rel_props,
        "relationships": [asdict(rel) for rel in RELATIONSHIPS],
        "metadata": {"constraint": [], "index": []},
    }

