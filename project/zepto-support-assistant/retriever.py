"""RETRIEVAL stage: embed the query, cosine top-k from ChromaDB.

Always runs for real (in mock and real-LLM modes): needs no API key or network.
"""
from typing import Dict, List

from config import TOP_K
from ingest import embed_texts, get_collection


def retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    """Return the top-k chunks, most similar first.

    Each item: {id, doc_id, title, text, score} where score = 1 - cosine distance.
    """
    res = get_collection().query(
        query_embeddings=embed_texts([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": cid,
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "text": text,
            "score": round(1.0 - dist, 4),
        }
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
