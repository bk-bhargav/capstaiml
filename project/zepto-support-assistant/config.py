"""Central configuration. All LLM behaviour is gated by the MOCK_LLM env var."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# Human-readable titles for the 8 corpus documents (stored as chunk metadata).
DOC_TITLES = {
    "doc_01": "Delivery Policy",
    "doc_02": "Returns & Refunds",
    "doc_03": "Membership Tiers",
    "doc_04": "Order Tracking",
    "doc_05": "Order Cancellation Policy",
    "doc_06": "Damaged or Missing Items",
    "doc_07": "Gift Cards",
    "doc_08": "Customer Support Hours",
}

# Optional real-LLM extension (only used when MOCK_LLM=0).
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_SCHEMA_RETRIES = 2  # extra attempts after the first one


def is_mock() -> bool:
    """True (the graded default) unless MOCK_LLM=0 is set explicitly.

    Read at call time, not import time, so tests / operators can flip it.
    """
    return os.getenv("MOCK_LLM", "1").strip() != "0"
