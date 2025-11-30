"""
Unit tests for Neo4j ingestion helpers.

Neo4j is not available during unit tests, so we replace the driver/session with
simple dummy classes that record the Cypher queries issued by the ingestor.
"""

from graph_ingestion.ingest import GraphIngestor
from preprocessing.bio import EntitySpan, TranscriptExample


class DummyTx:
    def __init__(self) -> None:
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query.strip(), params))


class DummySession:
    def __init__(self) -> None:
        self.write_calls = []

    def execute_write(self, func, payload):
        tx = DummyTx()
        func(tx, payload)
        self.write_calls.append((func.__name__, payload, tx))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyConnector:
    def __init__(self) -> None:
        self.session_obj = DummySession()

    def session(self):
        return self.session_obj


def build_example() -> TranscriptExample:
    # Keep the example tiny so assertions are easy to read.
    return TranscriptExample(
        transcript_id="demo-graph",
        tokens=["hold", "short", "wait", "runway"],
        tags=["B-ACTION", "I-ACTION", "O", "B-TAXIWAY"],
        full_text="hold short wait runway",
        speaker="controller",
        intent="taxi",
        entities=[
            EntitySpan(text="HOLD SHORT", label="ACTION"),
            EntitySpan(text="Wait", label="O"),
            EntitySpan(text="Runway", label="TAXIWAY"),
        ],
    )


def test_upsert_transcript_emits_expected_queries():
    payload = build_example().to_json()
    tx = DummyTx()
    GraphIngestor._upsert_transcript(tx, payload)

    assert len(tx.queries) == 3
    upsert_query, params = tx.queries[0]
    assert "MERGE (t:Transcript" in upsert_query
    assert params["uid"] == "demo-graph"
    assert params["tokens"] == ["hold", "short", "wait", "runway"]
    assert params["bio_tags"] == ["B-ACTION", "I-ACTION", "O", "B-TAXIWAY"]

    entities_query, entities_params = tx.queries[2]
    assert "UNWIND $entities AS entity" in entities_query
    lowered = [entity["text"] for entity in entities_params["entities"]]
    assert lowered == ["hold short", "wait", "runway"]
    assert entities_params["entities"][0]["surface_text"] == "HOLD SHORT"


def test_ingest_examples_invokes_write_per_example():
    connector = DummyConnector()
    ingestor = GraphIngestor(connector)
    examples = [build_example(), build_example()]

    ingestor.ingest_examples(examples)

    assert len(connector.session_obj.write_calls) == 2
