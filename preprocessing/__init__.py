"""
Preprocessing package that turns the raw CSV rows into
clean, JSON-friendly training examples.

This module exposes the most common classes so import sites can do:

    from preprocessing import CSVLoader, BIOConverter
"""

from .loader import ATCTranscript, CSVLoader
from .bio import BIOConverter, EntitySpan, TranscriptExample
from .json_export import JSONExporter
from .pipeline import Phase1Pipeline

__all__ = [
    "ATCTranscript",
    "CSVLoader",
    "BIOConverter",
    "EntitySpan",
    "TranscriptExample",
    "JSONExporter",
    "Phase1Pipeline",
]
