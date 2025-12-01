"""
Helpers for working with BIO labels.

The ATC dataset stores entities as BIO tags.  For downstream processing it is
easier to operate on phrase level spans (e.g., "hold short" with label ACTION).
This module converts between those representations and enforces that no tokens
are lost in the process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .loader import ATCTranscript


@dataclass
class EntitySpan:
    """Represents a merged entity span."""

    text: str
    label: str
    start: int | None = None
    end: int | None = None  # exclusive token index

    def to_json(self) -> dict:
        return {"text": self.text, "label": self.label}

    @staticmethod
    def from_json(payload: dict) -> "EntitySpan":
        return EntitySpan(
            text=payload.get("text", ""),
            label=payload.get("label", ""),
            start=payload.get("start"),
            end=payload.get("end"),
        )


@dataclass
class TranscriptExample:
    """Final JSON-ready representation of a transcript."""

    transcript_id: str
    tokens: List[str]
    tags: List[str]
    full_text: str
    speaker: str
    intent: str
    entities: List[EntitySpan]

    def to_json(self) -> dict:
        return {
            "transcript_id": self.transcript_id,
            "tokens": self.tokens,
            "tags": self.tags,
            "full_text": self.full_text,
            "speaker": self.speaker,
            "intent": self.intent,
            "entities": [entity.to_json() for entity in self.entities],
        }

    @staticmethod
    def from_json(payload: dict) -> "TranscriptExample":
        entities = [
            EntitySpan.from_json(entity) for entity in payload.get("entities", [])
        ]
        return TranscriptExample(
            transcript_id=str(payload.get("transcript_id", "")),
            tokens=list(payload.get("tokens", [])),
            tags=list(payload.get("tags", [])),
            full_text=payload.get("full_text", ""),
            speaker=payload.get("speaker", ""),
            intent=payload.get("intent", ""),
            entities=entities,
        )


class BIOConverter:
    """Converts BIO tags into entity spans with validation."""

    @staticmethod
    def merge(tokens: Iterable[str], tags: Iterable[str]) -> List[EntitySpan]:
        tokens_list = list(tokens)
        tags_list = list(tags)
        entity_spans = BIOConverter._merge_entity_spans(tokens_list, tags_list)
        o_spans = BIOConverter._merge_o_spans(tokens_list, tags_list)

        all_spans = sorted(
            entity_spans + o_spans,
            key=lambda span: span.start if span.start is not None else -1,
        )

        BIOConverter.validate_coverage(tokens_list, tags_list, all_spans)
        return all_spans

    @staticmethod
    def _merge_entity_spans(tokens: List[str], tags: List[str]) -> List[EntitySpan]:
        spans: List[EntitySpan] = []
        current_tokens: List[str] = []
        current_label: str | None = None
        start_idx: int | None = None

        for idx, (token, tag) in enumerate(zip(tokens, tags)):
            if not tag or tag == "O":
                if current_tokens:
                    spans.append(
                        EntitySpan(
                            text=" ".join(current_tokens),
                            label=current_label or "",
                            start=start_idx or 0,
                            end=idx,
                        )
                    )
                    current_tokens = []
                    current_label = None
                    start_idx = None
                continue

            prefix, _, label = tag.partition("-")
            label = label or ""

            # A "B" tag starts a new entity even if the label is the same.
            if prefix == "B" or current_label is None or label != current_label:
                if current_tokens:
                    spans.append(
                        EntitySpan(
                            text=" ".join(current_tokens),
                            label=current_label or "",
                            start=start_idx or 0,
                            end=idx,
                        )
                    )
                current_tokens = [token]
                current_label = label
                start_idx = idx
                continue

            # prefix == "I" and labels match
            if start_idx is None:
                start_idx = idx
            current_tokens.append(token)

        if current_tokens:
            spans.append(
                EntitySpan(
                    text=" ".join(current_tokens),
                    label=current_label or "",
                    start=start_idx or 0,
                    end=len(tokens),
                )
            )
        return spans

    @staticmethod
    def _merge_o_spans(tokens: List[str], tags: List[str]) -> List[EntitySpan]:
        spans: List[EntitySpan] = []
        for idx, tag in enumerate(tags):
            if tag == "O":
                spans.append(
                    EntitySpan(
                        text=tokens[idx],
                        label="O",
                        start=idx,
                        end=idx + 1,
                    )
                )
        return spans

    @staticmethod
    def validate_coverage(
        tokens: List[str], tags: List[str], spans: List[EntitySpan]
    ) -> None:
        """
        Ensures every entity-tagged token is covered by exactly one span.

        Raises:
            ValueError if coverage fails.
        """
        # Track whether each token index is already claimed by an entity span.
        coverage: List[str | None] = [None] * len(tokens)
        for span in spans:
            if span.start is None or span.end is None:
                raise ValueError("Span boundaries missing for coverage validation")
            if span.start < 0 or span.end > len(tokens):
                raise ValueError(
                    f"Span {span} is out of bounds for {len(tokens)} tokens"
                )
            for idx in range(span.start, span.end):
                if coverage[idx] is not None:
                    raise ValueError(
                        f"Token index {idx} covered by multiple spans"
                    )
                coverage[idx] = span.label or ""

        for idx, tag in enumerate(tags):
            span_label = coverage[idx]
            if tag == "O":
                if span_label != "O":
                    raise ValueError(
                        f"Non-entity token at position {idx} missing 'O' coverage"
                    )
                continue

            if not tag.startswith(("B-", "I-")):
                raise ValueError(f"Unexpected tag format: {tag}")

            _, _, label = tag.partition("-")
            label = label or ""
            if span_label != label:
                raise ValueError(
                    f"Entity token at position {idx} covered by '{span_label}' "
                    f"instead of '{label}'"
                )

    @staticmethod
    def build_example(transcript: ATCTranscript) -> TranscriptExample:
        spans = BIOConverter.merge(transcript.tokens, transcript.tags)
        full_text = " ".join(transcript.tokens)
        return TranscriptExample(
            transcript_id=transcript.transcript_id,
            tokens=list(transcript.tokens),
            tags=list(transcript.tags),
            full_text=full_text,
            speaker=transcript.speaker,
            intent=transcript.intent,
            entities=spans,
        )
