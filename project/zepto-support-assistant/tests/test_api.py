import pytest
from fastapi.testclient import TestClient

import graph
import main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setattr(main, "ensure_index", lambda: 8)
    monkeypatch.setattr(graph, "_GRAPH", None)
    monkeypatch.setattr(
        graph.retriever, "retrieve",
        lambda q, k=3: [{"id": "doc_01_c0", "doc_id": "doc_01", "title": "t", "text": "hello", "score": 0.9}],
    )
    with TestClient(main.app) as c:
        yield c


def test_ask_policy(client):
    body = client.post("/ask", json={"query": "delivery fee?"}).json()
    assert set(body) == {"answer", "sources", "confidence"}
    assert body["sources"] == ["doc_01_c0"]


def test_ask_general(client):
    body = client.post("/ask", json={"query": "tell me a joke"}).json()
    assert body["sources"] == []


def test_ask_rejects_bad_request(client):
    assert client.post("/ask", json={"query": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
