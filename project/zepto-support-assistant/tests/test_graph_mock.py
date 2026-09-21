"""Graph routing + mock-mode outputs. Retrieval is stubbed so no model download is needed;
the real-retrieval check lives in test_retrieval_real.py."""
import pytest

import graph
import llm

CHUNKS = [
    {"id": "doc_01_c0", "doc_id": "doc_01", "title": "Delivery Policy",
     "text": "Zepto delivers grocery and household essentials " * 8, "score": 0.8},
    {"id": "doc_03_c0", "doc_id": "doc_03", "title": "Membership Tiers", "text": "b", "score": 0.5},
    {"id": "doc_08_c0", "doc_id": "doc_08", "title": "Support", "text": "c", "score": 0.4},
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)          # default = mock
    monkeypatch.setattr(graph.retriever, "retrieve", lambda q, k=3: CHUNKS)

    def boom(*a, **k):
        raise AssertionError("LLM must not be called in mock mode")

    monkeypatch.setattr(llm, "call_llm", boom)
    monkeypatch.setattr(graph, "_GRAPH", None)


def test_graph_has_three_named_nodes():
    nodes = set(graph.build_graph().get_graph().nodes)
    assert {"classify_intent", "retrieve_and_answer", "direct_answer"} <= nodes


def test_policy_query_routes_to_retrieval():
    r = graph.run_query("What is the delivery fee?")
    assert r.answer == "Based on the retrieved context: " + CHUNKS[0]["text"][:200]
    assert r.sources == ["doc_01_c0", "doc_03_c0", "doc_08_c0"]
    assert r.confidence == 1.0


def test_general_query_routes_to_direct_answer():
    r = graph.run_query("What is the capital of France?")
    assert r.answer == "I can only answer questions about Zepto policies right now."
    assert r.sources == [] and r.confidence == 1.0
