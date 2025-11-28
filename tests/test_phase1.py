"""
Basic checks for the Phase 1 preprocessing pipeline.

These tests use tiny hand-crafted examples so we can reason about the expected
entities without loading the full CSV file.  They guard against regressions in
the BIO merging logic and the transcript dataclass conversion.
"""

from preprocessing.loader import ATCTranscript
from preprocessing.bio import BIOConverter


def test_merge_and_coverage():
    tokens = ["hold", "short", "runway", "two"]
    tags = ["B-ACTION", "I-ACTION", "B-TAXIWAY", "I-TAXIWAY"]
    spans = BIOConverter.merge(tokens, tags)

    assert len(spans) == 2
    assert spans[0].text == "hold short"
    assert spans[0].label == "ACTION"
    assert spans[0].start == 0 and spans[0].end == 2


def test_build_example_roundtrip():
    transcript = ATCTranscript(
        transcript_id="demo-1",
        tokens=["redcap", "one", "two", "standby"],
        tags=["B-CALLSIGN", "I-CALLSIGN", "I-CALLSIGN", "B-ACTION"],
        speaker="controller",
        intent="standby",
    )
    example = BIOConverter.build_example(transcript)

    assert example.full_text == "redcap one two standby"
    assert len(example.entities) == 2
    assert example.entities[1].text == "standby"
    assert example.entities[1].label == "ACTION"
