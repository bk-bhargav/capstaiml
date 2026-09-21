"""INGESTION + EMBEDDING stages.

load_documents -> chunk_text -> embed (all-MiniLM-L6-v2, local) -> ChromaDB collection.

Run directly (`python ingest.py`) to (re)build the index. The FastAPI app also calls
ensure_index() on startup, so the index is built automatically if missing.
"""
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME, DOC_TITLES, DOCS_DIR, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedder():
    # Imported lazily: torch import is slow and not needed by unit tests.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Local embeddings, L2-normalised (cosine distance in Chroma also normalises)."""
    vecs = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False)
    )


def get_collection():
    """The `zepto_policies` collection, configured for cosine similarity."""
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def load_documents(docs_dir: Path = DOCS_DIR) -> Dict[str, str]:
    """Read docs/doc_XX.txt -> {doc_id: text}."""
    return {
        p.stem: p.read_text(encoding="utf-8").strip()
        for p in sorted(Path(docs_dir).glob("doc_*.txt"))
    }


def chunk_text(text: str, chunk_size: Optional[int] = None, overlap: int = 40) -> List[str]:
    """Default: one chunk per document (each doc is only ~400-500 chars).

    Pass chunk_size to get fixed-size character windows with overlap instead.
    """
    if not chunk_size or len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def build_index(chunk_size: Optional[int] = None) -> int:
    """Chunk + embed + store all documents. Idempotent (upsert). Returns chunk count."""
    ids, texts, metas = [], [], []
    for doc_id, text in load_documents().items():
        for i, chunk in enumerate(chunk_text(text, chunk_size)):
            ids.append(f"{doc_id}_c{i}")
            texts.append(chunk)
            metas.append(
                {"doc_id": doc_id, "title": DOC_TITLES.get(doc_id, doc_id), "chunk_index": i}
            )
    col = get_collection()
    col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embed_texts(texts))
    return len(ids)


def ensure_index() -> int:
    """Build the index only if the collection is empty. Returns collection size."""
    col = get_collection()
    if col.count() == 0:
        build_index()
    return col.count()


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks into collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
