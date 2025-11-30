"""
Core Graph RAG orchestration.

The pipeline mirrors the steps outlined in Graph_RAG_Using_LLamaIndex.pdf:
    1. Run a fixed Cypher query to fetch structurally similar transcripts
       (token/entity overlap) from Neo4j.
    2. Pull transcript snippets from a vector index for extra grounding.
    3. Feed both sources into an Ollama-backed LLM to synthesize an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from .config import GraphRAGConfig


def _extract_text(node: Any) -> str:
    """Best-effort extraction of readable text from a retriever hit."""

    for attr in ("text", "get_text", "get_content", "content", "source_text"):
        value = getattr(node, attr, None)
        if callable(value):
            value = value()
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_metadata(obj: Any) -> dict:
    metadata = getattr(obj, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _format_chunk_lines(chunk: "ContextChunk") -> list[str]:
    """
    Build a structured, JSON-like view of a chunk for the LLM prompt.
    """

    meta = chunk.metadata or {}
    uid = meta.get("uid") or chunk.node_id or "unknown"
    full_text = meta.get("full_text") or chunk.text or ""
    speaker = meta.get("speaker")
    intent = meta.get("intent")
    entities = meta.get("entities") or []
    categories = meta.get("categories") or []

    lines = [f"- uid: {uid}", f"  full_text: {full_text}"]
    if speaker:
        lines.append(f"  speaker: {speaker}")
    if intent:
        lines.append(f"  intent: {intent}")
    if categories:
        lines.append(f"  categories: {', '.join(categories)}")
    if entities:
        lines.append("  entities:")
        for ent in entities:
            text = ent.get("text") if isinstance(ent, dict) else None
            label = ent.get("label") if isinstance(ent, dict) else None
            if text or label:
                lines.append(f"    - text: {text or ''}, label: {label or ''}")
    if chunk.score is not None:
        lines.append(f"  score: {chunk.score}")
    return lines


def _normalize_hit(hit: Any, source: str) -> "ContextChunk":
    node = getattr(hit, "node", hit)
    metadata = _extract_metadata(node) or _extract_metadata(hit)
    node_id = (
        metadata.get("uid")
        or getattr(node, "node_id", None)
        or getattr(node, "id_", None)
    )
    score = getattr(hit, "score", None)
    return ContextChunk(
        text=_extract_text(node),
        source=source,
        metadata=metadata,
        score=score,
        node_id=node_id,
    )


@dataclass
class ContextChunk:
    """Normalized view of a single retrieval hit."""

    text: str
    source: str
    metadata: dict
    score: float | None = None
    node_id: str | None = None


@dataclass
class GraphRAGResult:
    """Structured response from the GraphRAG pipeline."""

    question: str
    answer: str
    cypher_context: List[ContextChunk]
    vector_context: List[ContextChunk]
    prompt: str
    warnings: List[str]

    @property
    def merged_context(self) -> List[ContextChunk]:
        """Deduplicate contexts while preserving source provenance."""

        combined: List[ContextChunk] = []
        seen: set[str] = set()
        for chunk in self.cypher_context + self.vector_context:
            key = chunk.node_id or chunk.text
            if key in seen:
                continue
            seen.add(key)
            combined.append(chunk)
        return combined


class GraphRAG:
    """Orchestrates Neo4j + LlamaIndex retrieval with Ollama synthesis."""

    def __init__(
        self,
        config: GraphRAGConfig,
        *,
        llm: Any,
        cypher_retriever: Any | None,
        vector_retriever: Any | None,
        text_to_cypher: Any | None = None,
    ) -> None:
        self.config = config
        self._llm = llm
        # Keep a back-compat alias so older callers can still pass text_to_cypher.
        self._cypher_retriever = cypher_retriever or text_to_cypher
        self._vector_retriever = vector_retriever

    def _call_llm(self, prompt: str) -> str:
        if hasattr(self._llm, "complete"):
            response = self._llm.complete(prompt)
            if hasattr(response, "text"):
                return str(response.text)
            return str(response)
        if callable(self._llm):
            return str(self._llm(prompt))
        raise TypeError("LLM client must expose complete(prompt) or be callable.")

    def _run_retriever(
        self,
        retriever: Any | None,
        question: str,
        top_k: Optional[int],
    ) -> List[ContextChunk]:
        if retriever is None:
            return []

        hits: Sequence[Any] | Any
        if top_k is not None:
            for param in ("top_k", "similarity_top_k"):
                try:
                    hits = retriever.retrieve(question, **{param: top_k})
                    break
                except TypeError:
                    continue
            else:
                try:
                    hits = retriever.retrieve(question)
                except TypeError:
                    try:
                        hits = retriever.retrieve(query_str=question)
                    except Exception as exc:  # pragma: no cover - unexpected API
                        raise RuntimeError(
                            f"Retriever {retriever} does not accept supported params"
                        ) from exc
        else:
            hits = retriever.retrieve(question)

        if not isinstance(hits, Sequence):
            hits = [hits]

        source = (
            getattr(retriever, "name", None)
            or getattr(retriever, "__class__", type("Anon", (), {})).__name__
        )
        return [_normalize_hit(hit, source=source) for hit in hits]

    def _build_prompt(
        self,
        question: str,
        cypher_context: List[ContextChunk],
        vector_context: List[ContextChunk],
    ) -> str:
        lines: list[str] = [self.config.system_prompt.strip(), ""]
        lines.append("Target transmission to label:")
        lines.append(f"- full_text: {question.strip()}")
        lines.append("")
        lines.append("Reasoning order for hints:")
        lines.append(
            "1) Cypher graph examples (score shown per chunk). Higher score = closer structure."
        )
        lines.append("2) Vector snippets (use when Cypher lacks coverage).")
        lines.append(
            "3) ATC language priors (controllers often lead with a position/callsign; pilots often end with theirs)."
        )
        lines.append(
            "If evidence conflicts, trust the highest score; if nothing is clear, leave fields empty instead of guessing."
        )
        lines.append("")
        lines.append("Speaker cues:")
        lines.append(
            "- CONTROLLER transmissions often begin with the aircraft callsign or a controller role (e.g., 'singapore ground')."
        )
        lines.append(
            "- PILOT transmissions frequently end with their callsign and read back prior instructions."
        )
        lines.append("")
        lines.append("Intent cues:")
        lines.append("- GREETING: salutations like good day/evening/morning/thanks.")
        lines.append("- READBACK: repeats instructions and ends with callsign acknowledgment.")
        lines.append("- TAXI: issues taxi/hold short routing; callsign usually near start.")
        lines.append("- TRAFFIC: references other aircraft or explicitly says 'traffic'.")
        lines.append("- FREQUENCY: mentions/contact frequency handoff (word 'decimal' may be missing).")
        lines.append("- STANDBY: asks to wait/standby.")
        lines.append("- OTHER: any transmission not fitting the above.")
        lines.append("")
        lines.append("Entity label reminders:")
        lines.append("- CALLSIGN: airline name plus 1-4 digits (spoken or numeric).")
        lines.append("- FREQUENCY: controller frequencies such as 'one two four decimal three'.")
        lines.append("- TAXIWAY: letters like whiskey/sierra/tango/papa/quebec with optional digits.")
        lines.append("- ACTION: verbs like push, pushback, start, taxi, hold short, monitor, turn (all tenses).")
        lines.append("- O: filler/non-aviation words.")
        lines.append("- GATE: bay/stand/parking stand followed by numbers.")
        lines.append("- CONTROLLER: positions such as singapore ground/tower/ground.")
        lines.append("- VEHICLE: other aircraft or vehicles referenced.")
        lines.append("- QUALIFIER: left/right/north/south/east/west/ahead/behind/lima, etc.")
        lines.append("- GREETING: greeting phrases/thanks.")
        lines.append("- If an entity is uncertain, output {'text': '', 'label': ''} rather than hallucinating.")
        lines.append("")
        lines.append("Desired JSON structure example:")
        lines.append(
            '{"full_text":"<transcript>", "speaker":"CONTROLLER", "intent":"TAXI", '
            '"entities":[{"text":"qantas five two","label":"CALLSIGN"}, {"text":"hold short","label":"ACTION"}]}'
        )
        lines.append("Always copy the target transcript into full_text verbatim.")
        lines.append("")

        if cypher_context:
            lines.append("Graph examples from Cypher matches (structural hints):")
            for chunk in cypher_context:
                lines.extend(_format_chunk_lines(chunk))
            lines.append("")

        if vector_context:
            lines.append("Transcript snippets (vector search hints):")
            for chunk in vector_context:
                lines.extend(_format_chunk_lines(chunk))
            lines.append("")

        lines.append(
            "Use the above hints plus ATC structure to fill speaker, intent, and entities for the target "
            "transmission. Return exactly one JSON object with keys: full_text, speaker, intent, entities "
            "(list of {text, label}). Use empty strings or empty list when information is unknown. Output JSON only. Answer:"
        )
        return "\n".join(lines)

    def query(
        self,
        question: str,
        *,
        cypher_top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
    ) -> GraphRAGResult:
        vector_k = self.config.vector_top_k if vector_top_k is None else vector_top_k
        cypher_k = self.config.cypher_top_k if cypher_top_k is None else cypher_top_k
        warnings: list[str] = []

        cypher_context: List[ContextChunk] = []
        try:
            cypher_context = self._run_retriever(
                self._cypher_retriever,
                question,
                cypher_k,
            )
        except Exception as exc:
            cypher_context = []
            warnings.append(f"Cypher retrieval failed: {exc}")

        try:
            vector_context = self._run_retriever(
                self._vector_retriever,
                question,
                vector_k,
            )
        except Exception as exc:
            vector_context = []
            warnings.append(f"Vector retrieval failed: {exc}")

        prompt = self._build_prompt(question, cypher_context, vector_context)
        try:
            answer = self._call_llm(prompt)
        except Exception as exc:
            warnings.append(f"LLM call failed: {exc}")
            answer = ""

        return GraphRAGResult(
            question=question,
            answer=answer.strip(),
            cypher_context=cypher_context,
            vector_context=vector_context,
            prompt=prompt,
            warnings=warnings,
        )
