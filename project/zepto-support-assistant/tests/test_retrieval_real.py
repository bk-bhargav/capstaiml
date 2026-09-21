"""Integration: real embeddings + real ChromaDB (needs sentence-transformers + model download)."""
import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("chromadb")

import ingest  # noqa: E402
import retriever  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def index(tmp_path_factory):
    import config

    tmp = tmp_path_factory.mktemp("chroma")
    ingest.get_client.cache_clear()
    config.CHROMA_DIR = str(tmp)
    ingest.CHROMA_DIR = str(tmp)
    assert ingest.build_index() == 8
    yield
    ingest.get_client.cache_clear()


CASES = [
    ("What is the delivery fee for a small order?", "doc_01"),
    ("Can I return opened personal care items?", "doc_02"),
    ("How much does Zepto Pass+ cost per month?", "doc_03"),
    ("The rider hasn't moved for 25 minutes, what should I do?", "doc_04"),
    ("Can I cancel my order after it is packed?", "doc_05"),
    ("An item was missing from my order", "doc_06"),
    ("Which gift card denominations exist?", "doc_07"),
    ("Is there phone support for customers?", "doc_08"),
]


@pytest.mark.parametrize("query,expected", CASES)
def test_correct_doc_in_top3(query, expected):
    hits = retriever.retrieve(query, k=3)
    assert len(hits) == 3
    assert expected in [h["doc_id"] for h in hits]


def test_all_eight_docs_indexed():
    assert ingest.get_collection().count() == 8
