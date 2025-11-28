from atc_ner_graph.preprocess.bio import merge_bio_spans
from atc_ner_graph.preprocess.validator import validate_token_coverage


def test_merge_and_coverage():
    tokens = ["hold", "short", "runway", "two", "two"]
    tags = ["B-ACTION", "I-ACTION", "B-RUNWAY", "I-RUNWAY", "I-RUNWAY"]

    spans = merge_bio_spans(tokens, tags)
    assert len(spans) == 2
    assert spans[0].text == "hold short"
    assert spans[0].label == "ACTION"
    assert spans[1].text == "runway two two"
    assert spans[1].label == "RUNWAY"

    # should not raise
    validate_token_coverage(tokens, spans)
