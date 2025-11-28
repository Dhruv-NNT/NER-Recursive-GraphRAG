"""BIO tag processing and span merging utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass
class EntitySpan:
    """Phrase-level entity derived from BIO tags."""

    text: str
    label: str
    start: int
    end: int  # exclusive


def merge_bio_spans(tokens: List[str], tags: List[str]) -> List[EntitySpan]:
    """Merge BIO tags into entity spans.

    Args:
        tokens: Tokenized transcript.
        tags: BIO tags aligned with tokens.

    Returns:
        List of ``EntitySpan`` objects representing merged entities.

    Raises:
        ValueError: on unsupported tag sequences.
    """

    if len(tokens) != len(tags):
        raise ValueError("Token and tag lengths must match for merging")

    spans: List[EntitySpan] = []
    current_tokens: List[str] = []
    current_label: str | None = None
    start_idx: int | None = None

    def close_span(end_idx: int) -> None:
        nonlocal current_tokens, current_label, start_idx
        if current_label is None or start_idx is None:
            return
        spans.append(
            EntitySpan(
                text=" ".join(current_tokens),
                label=current_label,
                start=start_idx,
                end=end_idx,
            )
        )
        current_tokens = []
        current_label = None
        start_idx = None

    for idx, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag == "O" or not tag:
            close_span(idx)
            continue

        if tag.startswith("B-"):
            close_span(idx)
            current_label = tag[2:]
            current_tokens = [tok]
            start_idx = idx
        elif tag.startswith("I-"):
            if current_label != tag[2:]:
                raise ValueError(
                    f"Invalid I- tag at position {idx}: expected I-{current_label} but got {tag}"
                )
            current_tokens.append(tok)
        else:
            raise ValueError(f"Unsupported tag '{tag}' at position {idx}")

    close_span(len(tokens))
    return spans


def reconstruct_tokens_from_spans(
    tokens: List[str], spans: Iterable[EntitySpan]
) -> List[Tuple[int, str]]:
    """Reconstruct token sequence using span boundaries for coverage validation.

    Returns list of (index, token) pairs in the order recovered.
    """

    sorted_spans = sorted(spans, key=lambda s: s.start)
    reconstructed: List[Tuple[int, str]] = []
    cursor = 0

    for span in sorted_spans:
        if span.start < cursor:
            raise ValueError("Overlapping spans detected during reconstruction")
        if span.start < 0 or span.end > len(tokens):
            raise ValueError("Span indices out of token range")

        # add non-entity tokens between cursor and span start
        for i in range(cursor, span.start):
            reconstructed.append((i, tokens[i]))

        # add tokens inside the span
        for i in range(span.start, span.end):
            reconstructed.append((i, tokens[i]))

        cursor = span.end

    # add remaining tokens after last span
    for i in range(cursor, len(tokens)):
        reconstructed.append((i, tokens[i]))

    return reconstructed
