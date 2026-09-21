# Zepto Support Assistant (RAG service)

A small GenAI service for Zepto: 8 policy documents embedded locally with
`all-MiniLM-L6-v2` and stored in ChromaDB, a LangGraph flow that routes each query and
retrieves grounded context, a Pydantic-enforced JSON output, and a FastAPI `POST /ask`
endpoint that also ships as a Docker image.

**Default behaviour is fully offline.** Every LLM call is gated behind `MOCK_LLM`.
Unset (or `MOCK_LLM=1`) = deterministic rule-based mock, no API key, no network to any LLM
provider. Only `MOCK_LLM=0` calls a real LLM (optional extension).

## Project layout

| File | Purpose |
|---|---|
| `docs/doc_01.txt … doc_08.txt` | The corpus, verbatim from the assignment |
| `config.py` | Paths, model name, `TOP_K`, `is_mock()` toggle |
| `ingest.py` | **Ingestion + embedding**: load, chunk, embed, upsert into ChromaDB |
| `retriever.py` | **Retrieval**: embed query, cosine top-3 from ChromaDB |
| `intent.py` | Keyword heuristic used by `classify_intent` in mock mode |
| `prompts.py` | Structured prompt template (role–context–task–format–length + negative constraint + few-shot) |
| `llm.py` | Optional real-LLM client (Groq), schema validation and retry logic |
| `graph.py` | LangGraph `StateGraph`: 3 nodes + conditional edge |
| `schemas.py` | Pydantic `AskRequest` / `AskResponse` |
| `main.py` | FastAPI app (`POST /ask`, `GET /health`) |
| `record_examples.py` | Runs the two demo calls and writes their JSON into this README |
| `tests/` | Unit tests (intent, retry logic, graph routing, API, real retrieval) |
| `Dockerfile` | Container image (serves on port 7860) |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # first run downloads all-MiniLM-L6-v2 (~90 MB)
python ingest.py                     # optional: builds ./chroma_db (also auto-built on API start)
uvicorn main:app --port 8000         # MOCK_LLM left unset -> mock mode
```

Then open `http://localhost:8000/docs` or use `curl` as below.

## Example calls (MOCK_LLM at its default)

Recorded by `python record_examples.py` (which calls the real app and real retrieval and
pastes the raw JSON here).

<!-- EXAMPLES:START -->
_Not recorded yet: run `python record_examples.py` once after installing the requirements._
<!-- EXAMPLES:END -->

## Architecture: the RAG pipeline

```
docs/doc_01..08.txt
      │  ingest.load_documents()
      ▼
 [1 INGESTION]   ingest.chunk_text()  → one chunk per document  (ids: doc_01_c0 … doc_08_c0)
      │
      ▼
 [2 EMBEDDING]   ingest.embed_texts() → all-MiniLM-L6-v2 (sentence-transformers, local, 384-d)
      │
      ▼
 ChromaDB PersistentClient ── collection "zepto_policies" (cosine space) ── ./chroma_db
      ▲
      │ query embedding
 POST /ask ─► LangGraph:  classify_intent ──policy_question──► retrieve_and_answer ──► END
                              │                                  [3 RETRIEVAL + 4 GENERATION]
                              └──general_question──► direct_answer ──► END
                                                                  │
                                       Pydantic AskResponse {answer, sources, confidence}
```

**1. Ingestion (`ingest.py`).** `load_documents()` reads the 8 files in `docs/`.
`chunk_text()` keeps each document as a single chunk (they are ~400–500 characters);
passing `chunk_size` switches to fixed-size overlapping windows. Chunk ids look like
`doc_05_c0` and carry `doc_id`/`title` metadata.

**2. Embedding (`ingest.py`).** `embed_texts()` encodes every chunk locally with
`all-MiniLM-L6-v2` (normalised vectors). `build_index()` upserts ids, texts, metadata and
embeddings into the ChromaDB collection **`zepto_policies`**, created with cosine distance.
`main.py` calls `ensure_index()` on startup, and the Dockerfile pre-builds the index.

**3. Retrieval (`retriever.py`, called from the `retrieve_and_answer` node in `graph.py`).**
`classify_intent` first labels the query. For `policy_question`, the conditional edge routes
to `retrieve_and_answer`, which embeds the query with the same model and queries ChromaDB for
the **top-3** chunks by cosine similarity. For `general_question` the edge routes to
`direct_answer` and no retrieval happens. Retrieval runs for real in both modes.

**4. Generation (`retrieve_and_answer` / `direct_answer` nodes; prompt in `prompts.py`).**
The chosen node produces `answer`, `sources` and `confidence`, which `graph.run_query()`
validates through the Pydantic `AskResponse` model before FastAPI returns it.

### What branches on `MOCK_LLM`

Only the *generation* step inside each node (routing and retrieval are identical in both modes):

| Node | Default / `MOCK_LLM=1` (graded baseline) | Optional `MOCK_LLM=0` |
|---|---|---|
| `classify_intent` | Keyword heuristic (`intent.py`), no LLM call | LLM classifies; falls back to the heuristic if the reply is unusable |
| `retrieve_and_answer` | `"Based on the retrieved context: " + first 200 chars of top chunk`; `sources` = retrieved chunk ids; `confidence = 1.0` | LLM answers grounded only in the retrieved chunks using the structured prompt from `prompts.py` |
| `direct_answer` | Fixed string `"I can only answer questions about Zepto policies right now."`, `sources=[]`, `confidence=1.0` | LLM answers directly, no retrieval |

**Schema enforcement.** In mock mode the response is populated deterministically from code,
so there is nothing to fail validation. In the `MOCK_LLM=0` path, `llm.generate_structured()`
parses and validates the raw output against `AskResponse`. On failure it retries up to
**2 more times** with a corrective instruction appended, then returns a clearly marked error
response (`answer` starts with `ERROR:`, `confidence = 0.0`, `sources = []`). Citations returned
by the LLM are additionally filtered to chunk ids that were actually retrieved.

### Structured prompt (`prompts.py`)

`build_rag_prompt()` assembles, in order: **ROLE**, **CONTEXT** (retrieved chunks tagged with
ids), **TASK** (including the negative constraint *"Do NOT answer using information that is not
present in the provided context"*), **FORMAT** (single JSON object), **LENGTH** (≤3 sentences),
and a **few-shot EXAMPLE** (a worked cancellation question), followed by the customer's question.
This prompt is used only by the optional `MOCK_LLM=0` path.

## Routing behaviour to be aware of

The mock classifier is a plain substring match on the eight required keywords, so it is
deliberately simple. A policy-style question without a keyword (e.g. *"Is there phone
support?"*, *"How do I get my money back?"*) is classified `general_question`, and a
query containing a keyword in a non-policy sense would go to retrieval. The real-LLM
classifier (`MOCK_LLM=0`) handles these better.

## Docker

```bash
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support

curl -s -X POST http://localhost:7860/ask \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the delivery fee for orders below INR 149?"}'
```

The image bakes in the embedding model and the ChromaDB index, and leaves `MOCK_LLM` unset,
so the container works offline in the default mock mode.

## Tests

```bash
pytest -q
```

`test_retrieval_real.py` uses the real embedding model (skipped automatically if
`sentence-transformers` is not installed); the rest run without any model download.

## Optional extensions (not required for grading)

**Real LLM (Groq free tier).** Create a free key at console.groq.com, then:

```bash
export GROQ_API_KEY=...        # never commit this
export MOCK_LLM=0
# optional: export GROQ_MODEL=<any model available on your Groq account>
uvicorn main:app --port 8000
```

**Hugging Face Spaces.** Push this repo to a Docker Space on the free community CPU tier and
add `GROQ_API_KEY` as a Space secret (only needed with `MOCK_LLM=0`). Space README front-matter
needs `sdk: docker` and `app_port: 7860`. Live URL / tier: _not deployed_.
