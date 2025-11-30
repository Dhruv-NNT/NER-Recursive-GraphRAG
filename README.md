# ATC Graph-Augmented NER / NLU

Simple notes on how to run the current phases of the project and what the data looks like.

---

## 1. What this repo does right now

- **Phase 1** (preprocessing): reads `generation_train 2.csv`, keeps every token with its BIO tag, merges B/I spans into phrase-level entities, writes JSON, and validates that no tokens are lost.
- **Phase 2** (graph ingestion): pushes the JSON output into Neo4j. Each `Transcript` node stores the raw text, tokens, BIO tags, speaker, intent, and all spans (including per-token `O` spans) through `CONTAINS` edges.
- Later phases (graph retrieval + prediction + evaluation) have placeholders only.

---

## 2. Requirements & quick setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

External services:
1. **Neo4j** (Phase 2)
   ```bash
   docker run --rm -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/Helloworld@123 neo4j:5.20
   ```
2. **Ollama** (used in later work, no code yet): plan assumes `gemma3:12b` served from `http://localhost:11434`.

---

## 3. Phase 1 – Preprocessing

Command-line demo:
```bash
python phase1_demo.py \
  --csv "generation_train 2.csv" \
  --limit 5 \
  --output data/train_examples.json
```

### JSON structure

Each row becomes:
```json
{
  "transcript_id": "42",
  "tokens": ["redcap", "one", "idle", "standby"],
  "tags": ["B-CALLSIGN", "I-CALLSIGN", "O", "B-ACTION"],
  "full_text": "redcap one idle standby",
  "speaker": "controller",
  "intent": "standby",
  "entities": [
    {"text": "redcap one", "label": "CALLSIGN"},
    {"text": "idle", "label": "O"},
    {"text": "standby", "label": "ACTION"}
  ]
}
```
- Every token is kept in `tokens` + `tags`.
- `entities` now includes single-token `O` spans so the JSON (and graph) explicitly records background words. B/I labels stay merged for downstream convenience.

### Tests

```bash
pytest tests/test_phase1.py
```

---

## 4. Phase 2 – Neo4j ingestion

Ingest from CSV (reruns Phase 1 in-memory):
```bash
python phase2_ingest.py \
  --csv "generation_train 2.csv" \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password Helloworld@123
```

Reusing a saved JSON:
```bash
python phase2_ingest.py \
  --json data/train_examples.json \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password Helloworld@123
```

What gets stored:
- `Transcript` nodes: `uid`, `text`, `speaker`, `intent`, `tokens` (list), `bio_tags` (list).
- `IntentNode` nodes plus `HAS_INTENT` edges.
- `Entity` nodes (deduplicated by `text + category`) and `Category` nodes, connected via `CONTAINS` and `IS_A`.
- Because `entities` includes `O` spans, even background words show up as `Entity` nodes labeled `O`, keeping the graph consistent with the JSON.

### Tests

```bash
pytest tests/test_graph_ingest.py
```

---

## 5. Graph RAG (Neo4j + LlamaIndex + Ollama)

The new `graph_rag/` package follows the workflow described in
“Graph_RAG_Using_LLamaIndex.pdf”:

1. Text-to-Cypher retrieval against Neo4j for structured facts.
2. Vector search over transcript text for extra grounding.
3. Ollama-powered answer synthesis with both context streams.

Sample usage (requires Phase 2 ingestion plus running Neo4j/Ollama):

```python
from graph_rag import GraphRAGConfig, build_graphrag_with_llamaindex

config = GraphRAGConfig.from_env()
rag = build_graphrag_with_llamaindex(config)
result = rag.query("Who cleared Redcap 1727 to taxi via whiskey?")
print(result.answer)
```

> **Note**  
> `graph_rag/schema.py` ships with a static Neo4j schema so the pipeline does not
> need APOC procedures (the default `Neo4jPropertyGraphStore` normally calls
> `apoc.meta.data`). If you add new node labels or relationships, update that
> schema helper or enable the APOC plugin in Neo4j.

CLI demo:

```bash
python tools/graphrag_query_demo.py \
  --question "Who cleared Redcap 1727 to taxi via whiskey?"
```

Unit tests (use lightweight stubs, no external services required):

```bash
pytest tests/test_graph_rag.py
```

---

## 6. Next steps (not yet implemented)

1. Build the three-agent retrieval/prediction workflow that uses Neo4j + Ollama.
2. Hook predictions into `2-validate-NLU.ipynb`, making sure we can convert span outputs back to BIO for metric calculations.
3. Document Ollama installation and provide detailed runbooks for the agent orchestration once built.

Until then, rerun Phase 1 whenever the CSV changes, regenerate the JSON, and ingest it into Neo4j to keep the graph synced.
