"""
Basic checks for the Phase 1 preprocessing pipeline.

These tests use tiny hand-crafted examples so we can reason about the expected
entities without loading the full CSV file.  They guard against regressions in
the BIO merging logic and the transcript dataclass conversion.
"""

from preprocessing.loader import ATCTranscript
from preprocessing.bio import BIOConverter


def test_merge_and_coverage():
    tokens = ["hold", "short", "please", "idle", "runway", "two"]
    tags = ["B-ACTION", "I-ACTION", "O", "O", "B-TAXIWAY", "I-TAXIWAY"]
    spans = BIOConverter.merge(tokens, tags)

    assert len(spans) == 4
    assert spans[0].text == "hold short"
    assert spans[0].label == "ACTION"
    assert spans[0].start == 0 and spans[0].end == 2
    assert spans[1].label == "O"
    assert spans[1].text == "please"
    assert spans[1].start == 2 and spans[1].end == 3
    assert spans[2].label == "O"
    assert spans[2].text == "idle"
    assert spans[2].start == 3 and spans[2].end == 4


def test_build_example_roundtrip():
    transcript = ATCTranscript(
        transcript_id="demo-1",
        tokens=["redcap", "one", "idle", "standby"],
        tags=["B-CALLSIGN", "I-CALLSIGN", "O", "B-ACTION"],
        speaker="controller",
        intent="standby",
    )
    example = BIOConverter.build_example(transcript)

    assert example.full_text == "redcap one idle standby"
    assert example.tokens == ["redcap", "one", "idle", "standby"]
    assert example.tags == ["B-CALLSIGN", "I-CALLSIGN", "O", "B-ACTION"]
    assert len(example.entities) == 3
    assert example.entities[1].label == "O"
    assert example.entities[2].text == "standby"
    assert example.entities[2].label == "ACTION"
