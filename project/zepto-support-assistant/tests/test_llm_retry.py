"""Schema-enforcement + retry logic of the optional MOCK_LLM=0 path (LLM is faked)."""
import json

import llm

GOOD = json.dumps({"answer": "ok", "sources": ["doc_01_c0"], "confidence": 0.9})


def test_valid_first_try():
    calls = []
    out = llm.generate_structured("p", llm=lambda p: calls.append(p) or GOOD)
    assert out.answer == "ok" and len(calls) == 1


def test_recovers_after_two_bad_outputs():
    replies = iter(["not json", '{"answer": "x"}', GOOD])
    prompts = []

    def fake(p):
        prompts.append(p)
        return next(replies)

    out = llm.generate_structured("p", llm=fake)
    assert out.answer == "ok"
    assert len(prompts) == 3                      # 1 + 2 retries
    assert "previous reply was invalid" in prompts[1]   # corrective instruction added


def test_gives_up_with_marked_error():
    calls = []
    out = llm.generate_structured("p", llm=lambda p: calls.append(p) or "garbage")
    assert len(calls) == 3
    assert out.answer.startswith("ERROR:") and out.confidence == 0.0 and out.sources == []


def test_fenced_json_is_accepted():
    out = llm.generate_structured("p", llm=lambda p: "```json\n" + GOOD + "\n```")
    assert out.answer == "ok"


def test_confidence_out_of_range_triggers_retry():
    bad = json.dumps({"answer": "x", "sources": [], "confidence": 7})
    replies = iter([bad, GOOD])
    out = llm.generate_structured("p", llm=lambda p: next(replies))
    assert out.confidence == 0.9
