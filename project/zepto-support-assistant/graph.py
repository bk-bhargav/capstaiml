"""LangGraph orchestration: classify_intent -> (conditional) -> retrieve_and_answer | direct_answer.

Only the *generation* inside each node branches on MOCK_LLM. The routing edge and the
retrieval step are identical in both modes.
"""
from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph

import retriever
from config import is_mock
from intent import keyword_intent
from prompts import build_direct_prompt, build_rag_prompt
from schemas import AskResponse

MOCK_DIRECT_ANSWER = "I can only answer questions about Zepto policies right now."
SNIPPET_CHARS = 200


class GraphState(TypedDict, total=False):
    query: str
    intent: str                 # "policy_question" | "general_question"
    retrieved: List[Dict]       # chunks from ChromaDB (policy path only)
    answer: str
    sources: List[str]
    confidence: float


# ------------------------------------------------------------------ nodes
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]
    if is_mock():
        return {"intent": keyword_intent(query)}          # no LLM call
    import llm  # optional extension, lazy import

    try:
        return {"intent": llm.classify_with_llm(query)}
    except Exception:
        return {"intent": keyword_intent(query)}          # safe fallback


def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    chunks = retriever.retrieve(query)                    # always real, both modes
    sources = [c["id"] for c in chunks]

    if is_mock():
        snippet = chunks[0]["text"][:SNIPPET_CHARS] if chunks else ""
        return {
            "retrieved": chunks,
            "answer": f"Based on the retrieved context: {snippet}",
            "sources": sources,
            "confidence": 1.0,
        }

    import llm  # optional extension

    result = llm.generate_structured(build_rag_prompt(query, chunks))
    # Keep only citations that really came from retrieval.
    cited = [s for s in result.sources if s in sources]
    return {
        "retrieved": chunks,
        "answer": result.answer,
        "sources": cited,
        "confidence": result.confidence,
    }


def direct_answer(state: GraphState) -> GraphState:
    if is_mock():
        return {"answer": MOCK_DIRECT_ANSWER, "sources": [], "confidence": 1.0}

    import llm  # optional extension

    result = llm.generate_structured(build_direct_prompt(state["query"]))
    return {"answer": result.answer, "sources": [], "confidence": result.confidence}


# ---------------------------------------------------------------- routing
def route_by_intent(state: GraphState) -> str:
    """Conditional-edge function; independent of MOCK_LLM."""
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_and_answer", retrieve_and_answer)
    g.add_node("direct_answer", direct_answer)
    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    g.add_edge("retrieve_and_answer", END)
    g.add_edge("direct_answer", END)
    return g.compile()


_GRAPH = None


def run_query(query: str) -> AskResponse:
    """Run the graph and return the Pydantic-validated final response."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    final = _GRAPH.invoke({"query": query})
    return AskResponse(
        answer=final["answer"],
        sources=final.get("sources", []),
        confidence=final["confidence"],
    )
