# Development Status

This document explains, in plain words, what already exists in the repository and how it maps to the planned multi-phase build for the Graph-Augmented NER / NLU system for Air Traffic Control (ATC) transcripts.

---

## 1. Roadmap Snapshot

| Phase | Goal | Current Status |
| --- | --- | --- |
| Phase 0 – Environment & Dependencies | Describe Python + service setup, list required packages, make it easy to install Neo4j and Ollama. | Requirements listed in `requirements.txt`, minimal install docs placed in `ROOTLESS_DOCKER.md`; more detailed run books still needed. |
| Phase 1 – Data Ingestion & Preprocessing | Load the CSV, merge BIO tags into entity spans, export JSON, and validate token coverage. | ✅ Completed in `preprocessing/` with demo script `phase1_demo.py` and unit tests in `tests/test_phase1.py`. |
| Phase 2 – Knowledge Graph Construction | Connect to Neo4j, create schema constraints, and ingest Phase 1 JSON. | ✅ Core ingestion module lives in `graph_ingestion/` with CLI helper `phase2_ingest.py` and tests in `tests/test_graph_ingest.py`. |
| Phase 3 – Retrieval & Prediction Pipeline | Build the three-agent workflow (Segmenter → Grounder → Predictor) that calls Neo4j + Ollama. | 🚧 GraphRAG retrieval implemented in `graph_rag/`; multi-agent prediction remains in `predictions/`. |
| Phase 4 – Evaluation & Metrics | Wire the system into `2-validate-NLU.ipynb`, support BIO ↔ span conversion, compute metrics. | ⏳ Not implemented yet. `evaluation/__init__.py` is a placeholder. |

---

## 2. Environment & Dependency Notes (Phase 0)

- **Python packages**: Exact versions live in `requirements.txt`. Key ones already used today are `pandas`, `numpy`, `neo4j`, `langchain`, `langgraph`, `sentence-transformers`, `transformers`, `torch`, `scikit-learn`, and `pytest`.
- **Installation command**: `pip install -r requirements.txt`. (Already run on your local Mac, so no extra step needed.)
- **Neo4j**:
  - Default URI `bolt://localhost:7687`, username `neo4j`, password `Helloworld@123`.
  - Run locally with Docker:
    ```bash
    docker run --rm -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/Helloworld@123 neo4j:5.20
    ```
  - Config handled by `graph_ingestion.config.Neo4jConfig`.
- **Ollama**:
  - Not used yet, but plan assumes local endpoint at `http://localhost:11434` with model `gemma3:12b`.
  - Install instructions still pending for a later phase.
- **Dataset**: `generation_train 2.csv` (train) and `generation_val 1.csv` (validation) already live at repo root.

---

### 2.1 Docker Installation on macOS (Apple Silicon or Intel)

1. **Download Docker Desktop**
   - Visit https://docs.docker.com/desktop/install/mac-install/.
   - Pick the DMG that matches your chip (Apple Silicon or Intel).
2. **Install**
   - Open the DMG, drag `Docker.app` into `Applications`, then launch it.
   - macOS will ask for privileged access the first time; grant it so Docker can manage its internal VM.
3. **Initial configuration**
   - Wait until the whale icon in the menu bar shows “Docker Engine is running”.
   - Optional: In *Settings → Resources* allocate ~4 CPU / 8 GB RAM if you plan to run Neo4j plus other containers concurrently.
4. **Verify from terminal**
   ```bash
   docker --version
   docker run hello-world
   ```
   - The `hello-world` container prints a success message if everything works.
5. **Run Neo4j container locally**
   ```bash
   docker run --rm -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/AlphaGamma@890 neo4j:5.20
   ```
   - Access https://localhost:7474 in a browser, log in with `neo4j / Helloworld@123`, and the database is ready for Phase 2 ingestion.

That is all you need for Docker on the Mac. Future services (Neo4j, Ollama, etc.) can reuse the same installation.

---

## 3. Phase 1 – Data Ingestion & Preprocessing (Implemented)

Goal: Convert the CSV rows into JSON-ready transcript objects with entity spans while guaranteeing no tokens disappear.

### Core modules

1. `preprocessing/loader.py`
   - `CSVLoader` reads the CSV safely (handles missing columns, trims whitespace, ensures equal token/tag counts).
   - `ATCTranscript` dataclass stores one transcript with tokens, tags, speaker, and intent.
2. `preprocessing/bio.py`
   - `BIOConverter.merge` turns BIO tags into `EntitySpan` objects.
   - Each span stores text, label, and token boundaries; `validate_coverage` ensures every entity token is covered once.
   - `TranscriptExample` encapsulates the final JSON payload per transcript.
3. `preprocessing/json_export.py`
   - `JSONExporter.write_examples` dumps processed examples to disk with pretty JSON formatting.
4. `preprocessing/pipeline.py`
   - `Phase1Pipeline` orchestrates: load CSV → build examples → optionally save JSON → print demo objects.
   - Includes `demo()` helper to print a few samples for sanity checks.

### Support scripts

- `phase1_demo.py`
  - Command line wrapper around `Phase1Pipeline`.
  - Usage:
    ```bash
    python phase1_demo.py --csv "generation_train 2.csv" --limit 5 --output data/train_examples.json
    ```
  - Prints JSON to stdout and optionally writes to file.

### Testing

- `tests/test_phase1.py`
  - `test_merge_and_coverage` creates a small hand-crafted example and asserts the merged spans + coverage.
  - `test_build_example_roundtrip` verifies `BIOConverter.build_example` returns the expected `TranscriptExample`.
  - Tests run with `pytest`.

### What still needs to happen later

- Span → BIO conversion utility (needed once the evaluation notebook is wired).
- Integration with evaluation script to confirm metrics.

---

## 4. Phase 2 – Knowledge Graph Construction (Implemented)

Goal: Push Phase 1 JSON output into Neo4j with strict deduplication and schema constraints.

### Core modules

1. `graph_ingestion/config.py`
   - `Neo4jConfig` dataclass stores URI, user, password, database, plus `from_env()` helper.
2. `graph_ingestion/connector.py`
   - `Neo4jConnector` lazily creates the official Neo4j driver and exposes `.session()` context manager.
3. `graph_ingestion/schema.py`
   - `SchemaManager.ensure_constraints()` issues Cypher statements for unique Transcript, Entity, Category, and Intent nodes.
4. `graph_ingestion/ingest.py`
   - `GraphIngestor.ingest_examples()` iterates over processed transcripts and merges Transcript, IntentNode, Entity, Category nodes.
   - `_upsert_transcript` handles relationships: `CONTAINS`, `IS_A`, and `HAS_INTENT`.
   - Accepts either in-memory examples or the JSON file from Phase 1 via `ingest_from_json`.

### Support script

- `phase2_ingest.py`
  - CLI entry point: optionally reruns Phase 1 or reads a JSON dump, then ingests everything into Neo4j.
  - Example:
    ```bash
    python phase2_ingest.py --csv "generation_train 2.csv" --limit 100 \
      --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password Helloworld@123
    ```

### Testing

- `tests/test_graph_ingest.py`
  - Uses dummy connector/session classes to capture Cypher statements.
  - Confirms `_upsert_transcript` emits the expected sequence (Transcript → Intent → Entities).
  - Ensures `ingest_examples` calls `execute_write` once per transcript.

### Outstanding items

- No automated integration test with a real Neo4j instance yet.
- Need documentation on running ingestion inside Docker for air-gapped hosts.

---

## 5. Phase 3 – Retrieval & Prediction Pipeline (In Progress)

- **GraphRAG retrieval**: Implemented in `graph_rag/` with Neo4j + LlamaIndex + Ollama. Demo script lives at `tools/graphrag_query_demo.py`; stubbed unit tests at `tests/test_graph_rag.py` run without external services.
- **APOC-free schema**: `graph_rag/schema.py` provides a static schema so property-graph retrieval works even when the APOC plugin is not installed. Update this file (or enable APOC) if the graph structure changes.
- **Agents still pending**: Segmenter → Grounder → Predictor flow has not been built yet. Needs span ↔ BIO converters, heuristics for unseen call signs, and prediction orchestration in `predictions/`.
- **Next implementation tasks**:
  - Design the Segmenter and Grounder agents to call into the new GraphRAG context.
  - Wire the Predictor to output JSON with coverage checks using Ollama.
  - Add repair heuristics and coverage validation before scoring.

---

## 6. Phase 4 – Evaluation & Metrics (Not Started)

- Notebook `2-validate-NLU.ipynb` exists but is not connected yet.
- Still need utilities to convert span predictions back to BIO tags so the notebook can score precision/recall/F1.
- Coverage validator from Phase 1 will be reused, but no automated evaluation hooks exist yet.

---

## 7. Data & Outputs on Disk

- **Input CSVs**: `generation_train 2.csv`, `generation_val 1.csv`.
- **Generated JSON (example)**: Running `phase1_demo.py --output data/train_examples.json` will produce JSON ready for Phase 2.
- **Graph artifacts**: Stored inside Neo4j; no serialized dump included yet.
- **Tests**: All located under `tests/`, runnable via `pytest`.

---

## 8. Immediate Next Steps

1. Document Ollama setup (pull `gemma3:12b`, verify endpoint, show sample curl call).
2. Add embedding store + retrieval design notes for the Segmenter agent before coding Phase 3.
3. Build span ↔ BIO conversion utilities and hook them into both prediction and evaluation flows.
4. Expand evaluation notebook instructions so anyone can reproduce metrics end-to-end.

This concludes the current status overview. The project now has a stable base for Phase 3 and Phase 4 work.
