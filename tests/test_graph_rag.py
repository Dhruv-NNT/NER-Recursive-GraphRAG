from dataclasses import dataclass

from graph_rag.config import GraphRAGConfig
from graph_rag.pipeline import GraphRAG, GraphRAGResult


@dataclass
class DummyNode:
    text: str
    metadata: dict


@dataclass
class DummyHit:
    node: DummyNode
    score: float | None = None


class DummyRetriever:
    def __init__(self, name: str, payloads):
        self.name = name
        self.payloads = payloads
        self.top_k_seen: int | None = None

    def retrieve(self, query: str, **kwargs):
        # Track whatever top-k parameter the pipeline passes through.
        self.top_k_seen = kwargs.get("top_k") or kwargs.get("similarity_top_k")
        assert query  # sanity check
        return self.payloads


class DummyLLM:
    def __init__(self, text: str = "dummy answer"):
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str):
        self.prompts.append(prompt)

        class Response:
            def __init__(self, text: str) -> None:
                self.text = text

        return Response(self.text)


def test_graphrag_builds_prompt_and_merges_context():
    cypher_hits = [
        DummyHit(node=DummyNode("taxi via alpha", {"uid": "T1"}), score=0.8),
    ]
    vector_hits = [
        DummyHit(node=DummyNode("taxi via alpha", {"uid": "T1"}), score=0.7),
        DummyHit(node=DummyNode("hold short runway", {"uid": "T2"}), score=0.9),
    ]

    cypher_retriever = DummyRetriever("CypherRetriever", cypher_hits)
    vector_retriever = DummyRetriever("VectorRetriever", vector_hits)

    config = GraphRAGConfig(cypher_top_k=1, vector_top_k=2)
    llm = DummyLLM("final answer")
    rag = GraphRAG(
        config=config,
        llm=llm,
        cypher_retriever=cypher_retriever,
        vector_retriever=vector_retriever,
    )

    result = rag.query("find taxi clearances")
    assert isinstance(result, GraphRAGResult)
    assert result.answer == "final answer"
    assert len(result.vector_context) == 2
    # Dedup keeps unique (uid, source) tuples.
    assert len(result.merged_context) == 2
    assert {"T1", "T2"} == {c.metadata["uid"] for c in result.merged_context}
    assert "Target transmission to label" in result.prompt
    assert "Graph examples from Cypher matches" in result.prompt
    assert "Transcript snippets" in result.prompt
    assert "Reasoning order for hints" in result.prompt
    assert '{"full_text":"' in result.prompt
    assert "JSON object" in result.prompt
    assert "find taxi clearances" in result.prompt
    assert cypher_retriever.top_k_seen == 1
    assert vector_retriever.top_k_seen == 2
    # Ensure prompt was sent to the LLM client.
    assert llm.prompts and result.prompt in llm.prompts
    assert result.warnings == []


def test_graphrag_handles_missing_channels():
    config = GraphRAGConfig()
    llm = DummyLLM("ok")
    rag = GraphRAG(
        config=config,
        llm=llm,
        cypher_retriever=None,
        vector_retriever=None,
    )
    result = rag.query("any traffic")
    assert result.vector_context == []
    assert result.answer == "ok"
    assert result.warnings == []


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://test:9999")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("CYPHER_TOP_K", "7")
    monkeypatch.setenv("GRAPH_RAG_SYSTEM_PROMPT", "custom prompt")
    monkeypatch.setenv("GRAPH_RAG_LLM_TIMEOUT", "15")

    cfg = GraphRAGConfig.from_env()
    assert cfg.neo4j_uri == "bolt://test:9999"
    assert cfg.ollama_model == "llama3"
    assert cfg.cypher_top_k == 7
    assert cfg.system_prompt == "custom prompt"
    assert cfg.llm_timeout == 15.0
