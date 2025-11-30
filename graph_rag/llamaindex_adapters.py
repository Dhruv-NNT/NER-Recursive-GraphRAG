"""
Lazy LlamaIndex imports and factory helpers for GraphRAG.

The functions here mirror the steps in Graph_RAG_Using_LLamaIndex.pdf while
handling version differences between LlamaIndex releases.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable, Tuple

import re

from graph_ingestion.config import Neo4jConfig
from graph_ingestion.connector import Neo4jConnector

from .config import GraphRAGConfig
from .cypher_retriever import CypherHintRetriever
from .pipeline import GraphRAG

LLAMA_INDEX_IMPORT_HINT = """\
Install the LlamaIndex extras listed in requirements.txt, for example:

    pip install llama-index llama-index-graph-stores-neo4j \\
        llama-index-llms-ollama llama-index-embeddings-huggingface
"""


def _import_attr(path: str) -> Any:
    module_name, attr = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr)


def _first_available(paths: Iterable[str]) -> Any:
    errors: list[Exception] = []
    for path in paths:
        try:
            return _import_attr(path)
        except Exception as exc:  # pragma: no cover - import resolution
            errors.append(exc)
            continue
    raise ImportError(LLAMA_INDEX_IMPORT_HINT) from errors[-1]


def load_llamaindex_classes(include_property_graph: bool = True) -> Dict[str, Any]:
    """
    Return a dict of LlamaIndex classes, tolerant to minor path changes.

    The property-graph-related imports can be skipped (include_property_graph=False)
    when only vector retrieval is needed, which reduces optional dependency
    requirements in lightweight environments.
    """

    try:
        classes = {
            "Settings": _first_available(["llama_index.core.Settings"]),
            "VectorStoreIndex": _first_available(
                ["llama_index.core.VectorStoreIndex"]
            ),
            "Document": _first_available(["llama_index.core.Document"]),
            "Ollama": _first_available(["llama_index.llms.ollama.Ollama"]),
            "HuggingFaceEmbedding": _first_available(
                ["llama_index.embeddings.huggingface.HuggingFaceEmbedding"]
            ),
        }
        if include_property_graph:
            classes.update(
                {
                    "StorageContext": _first_available(
                        ["llama_index.core.storage.storage_context.StorageContext"]
                    ),
                    "PropertyGraphIndex": _first_available(
                        ["llama_index.core.indices.property_graph.PropertyGraphIndex"]
                    ),
                    "Neo4jGraphStore": _first_available(
                        [
                            "llama_index.graph_stores.neo4j.Neo4jPropertyGraphStore",
                            "llama_index.graph_stores.neo4j.Neo4jGraphStore",
                        ]
                    ),
                    "TextToCypherRetriever": _first_available(
                        [
                            "llama_index.core.indices.property_graph.sub_retrievers.text_to_cypher.TextToCypherRetriever",
                            "llama_index.graph_stores.neo4j.TextToCypherRetriever",
                        ]
                    ),
                }
            )
        return classes
    except Exception as exc:  # pragma: no cover - dependency missing at runtime
        raise ImportError(LLAMA_INDEX_IMPORT_HINT) from exc


def _neo4j_config_from_rag(config: GraphRAGConfig) -> Neo4jConfig:
    return Neo4jConfig(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
        database=config.neo4j_database,
    )


def _normalize_text(text: str) -> str:
    """
    Lowercase, strip punctuation, and collapse whitespace for stable embeddings.
    """

    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_transcripts(
    config: GraphRAGConfig, limit: int | None
) -> Iterable[Tuple[str, str, str, str, list, list]]:
    query = """
    MATCH (t:Transcript)
    OPTIONAL MATCH (t)-[:CONTAINS]->(e:Entity)
    WHERE t.text IS NOT NULL
    WITH t,
         collect(DISTINCT e.category) AS categories,
         collect({text: e.text, label: e.category}) AS entities
    RETURN t.uid AS uid,
           t.text AS text,
           coalesce(t.intent, "") AS intent,
           coalesce(t.speaker, "") AS speaker,
           categories AS categories,
           entities AS entities
    ORDER BY t.uid
    """
    if limit is not None:
        query += "\nLIMIT $limit"

    with Neo4jConnector(_neo4j_config_from_rag(config)) as connector:
        with connector.session() as session:
            result = (
                session.run(query, limit=limit)
                if limit is not None
                else session.run(query)
            )
            for record in result:
                yield (
                    record["uid"],
                    record["text"],
                    record["intent"],
                    record["speaker"],
                    record["categories"] or [],
                    [e for e in record["entities"] if e.get("text")],
                )


def _init_text_to_cypher(cls: Any, property_index: Any, llm: Any, top_k: int) -> Any:
    """
    Instantiate TextToCypherRetriever across LlamaIndex versions.

    Some releases expect `property_graph_index`, others accept `graph_store` or
    `property_graph_store`. We try all variants with common top-k parameter
    names before giving up.
    """

    candidates = [
        {"property_graph_index": property_index},
        {"graph_store": getattr(property_index, "property_graph_store", None)},
        {"property_graph_store": getattr(property_index, "property_graph_store", None)},
    ]
    topk_params = ("max_paths", "max_path_length", "top_k", "similarity_top_k")

    last_exc: Exception | None = None
    for base_kwargs in candidates:
        if not all(base_kwargs.values()):
            continue
        for param in topk_params:
            try:
                return cls(llm=llm, **base_kwargs, **{param: top_k})
            except TypeError as exc:
                last_exc = exc
                continue
        try:
            return cls(llm=llm, **base_kwargs)
        except TypeError as exc:
            last_exc = exc
            continue

    if last_exc is not None:  # pragma: no cover - unexpected API change
        raise last_exc
    raise RuntimeError("Failed to initialize TextToCypherRetriever with available kwargs.")


def _init_vector_retriever(vector_index: Any, top_k: int) -> Any:
    for param in ("similarity_top_k", "top_k"):
        try:
            return vector_index.as_retriever(**{param: top_k})
        except TypeError:
            continue
    return vector_index.as_retriever()


def _build_property_index(
    cls: Any,
    graph_store: Any,
    llm: Any,
    embed_model: Any,
    storage_context: Any,
) -> Any:
    """
    Build a PropertyGraphIndex in a version-tolerant way.

    Newer LlamaIndex builds provide `from_existing` helpers; older ones accept
    `property_graph_store` in the constructor. We try the common forms before
    raising an error.
    """

    # Prefer the documented path if available.
    for method_name in ("from_existing", "from_existing_graph"):
        factory = getattr(cls, method_name, None)
        if callable(factory):
            try:
                return factory(
                    property_graph_store=graph_store,
                    llm=llm,
                    embed_model=embed_model,
                )
            except TypeError:
                try:
                    return factory(
                        graph_store=graph_store,
                        llm=llm,
                        embed_model=embed_model,
                    )
                except Exception:
                    pass

    # Fall back to ctor with explicit graph store.
    ctor_kwargs_options = [
        {"property_graph_store": graph_store},
        {"graph_store": graph_store},
        {"storage_context": storage_context, "llm": llm, "embed_model": embed_model},
    ]

    last_exc: Exception | None = None
    for kwargs in ctor_kwargs_options:
        try:
            return cls(llm=llm, embed_model=embed_model, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue

    if last_exc is not None:  # pragma: no cover - unexpected API drift
        raise last_exc
    raise RuntimeError("Unable to construct PropertyGraphIndex with available parameters.")


def build_graphrag_with_llamaindex(
    config: GraphRAGConfig,
    *,
    transcript_limit: int | None = None,
) -> GraphRAG:
    """
    Build a GraphRAG instance backed by real LlamaIndex components.

    Args:
        config: GraphRAGConfig with Neo4j/Ollama/embedding parameters.
        transcript_limit: Optional override for how many transcripts to load
            into the vector index. Defaults to config.transcript_limit.
    """

    classes = load_llamaindex_classes(include_property_graph=False)

    llm = classes["Ollama"](
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        request_timeout=config.llm_timeout,
        stream=False,
    )
    embed_model = classes["HuggingFaceEmbedding"](model_name=config.embedding_model)

    settings = classes["Settings"]
    settings.llm = llm
    settings.embed_model = embed_model

    cypher_retriever = CypherHintRetriever(
        _neo4j_config_from_rag(config), default_top_k=config.cypher_top_k
    )

    docs = []
    for uid, text, intent, speaker, categories, entities in _fetch_transcripts(
        config, limit=transcript_limit if transcript_limit is not None else config.transcript_limit
    ):
        if not text or not str(text).strip():
            continue

        entity_strings = [
            _normalize_text(f"entity {ent['text']} label {ent['label']}")
            for ent in entities
            if ent.get("text")
        ]
        text_parts = [
            _normalize_text(text),
            _normalize_text(f"intent {intent}") if intent else "",
            _normalize_text(f"speaker {speaker}") if speaker else "",
            _normalize_text("categories " + " ".join(categories))
            if categories
            else "",
            " ".join(entity_strings) if entity_strings else "",
        ]
        doc_text = " ".join(part for part in text_parts if part)

        docs.append(
            classes["Document"](
                text=doc_text,
                metadata={
                    "uid": uid,
                    "intent": intent,
                    "speaker": speaker,
                    "categories": categories,
                    "entities": entities,
                    "full_text": text,
                },
            )
        )
    if not docs:
        raise RuntimeError(
            "No Transcript nodes found in Neo4j; ingest data via phase2_ingest.py first."
        )

    vector_index = classes["VectorStoreIndex"].from_documents(
        docs, embed_model=embed_model
    )
    vector_retriever = _init_vector_retriever(vector_index, config.vector_top_k)

    return GraphRAG(
        config=config,
        llm=llm,
        cypher_retriever=cypher_retriever,
        vector_retriever=vector_retriever,
    )
