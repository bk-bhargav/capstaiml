"""Keyword-heuristic intent classifier (the mock-mode / graded classify_intent logic).

Kept dependency-free so it can be unit-tested without torch/chromadb/langgraph.
"""

POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)


def keyword_intent(query: str) -> str:
    """Return 'policy_question' if any keyword is a substring of the lowercased query."""
    q = query.lower()
    return "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"
