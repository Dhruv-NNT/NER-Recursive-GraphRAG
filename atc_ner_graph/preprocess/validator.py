"""Validation utilities to ensure token coverage is preserved."""
from __future__ import annotations

from typing import Iterable, List

from .bio import EntitySpan, reconstruct_tokens_from_spans


def validate_token_coverage(tokens: List[str], spans: Iterable[EntitySpan]) -> None:
    """Validate that no tokens are lost when applying entity spans.

    The function ensures spans do not overlap, stay within bounds, and
    that a reconstruction using spans plus non-entity tokens exactly
    matches the original token ordering.

    Raises:
        ValueError: if coverage is incomplete or spans are invalid.
    """

    reconstructed = reconstruct_tokens_from_spans(tokens, list(spans))
    if len(reconstructed) != len(tokens):
        raise ValueError("Reconstructed token count does not match original")

    for (idx, tok), original in zip(reconstructed, tokens):
        if tok != original:
            raise ValueError(
                f"Token mismatch at position {idx}: reconstructed '{tok}' vs original '{original}'"
            )

    # ensure all indices appear exactly once in order
    indices = [idx for idx, _ in reconstructed]
    if indices != list(range(len(tokens))):
        raise ValueError("Token indices are missing or out of order after reconstruction")
