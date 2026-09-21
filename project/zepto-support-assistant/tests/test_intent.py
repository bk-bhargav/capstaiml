import pytest

from intent import keyword_intent


@pytest.mark.parametrize(
    "q",
    [
        "What is the delivery fee?",
        "Can I RETURN this item?",
        "How long does a refund take?",
        "Tell me about membership tiers",
        "Where is my order tracking?",
        "I want to cancel my order",
        "Can I combine a gift card with UPI?",
        "What are your support hours?",
    ],
)
def test_policy_keywords(q):
    assert keyword_intent(q) == "policy_question"


@pytest.mark.parametrize("q", ["What is the capital of France?", "Tell me a joke", "hello"])
def test_general(q):
    assert keyword_intent(q) == "general_question"
