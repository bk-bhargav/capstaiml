"""OPTIONAL real-LLM extension (only reached when MOCK_LLM=0).

Nothing in this module is imported or called in the default mock mode, and the `groq`
package is imported lazily so mock mode does not even need it installed.

Schema enforcement: the raw LLM output must parse into AskResponse. On failure we retry
up to MAX_SCHEMA_RETRIES (=2) more times with a corrective instruction; if it still fails
we return a clearly marked error response.
"""
import json
import os
import re
from typing import Callable, Optional

from pydantic import ValidationError

from config import GROQ_MODEL, MAX_SCHEMA_RETRIES
from prompts import CORRECTIVE_INSTRUCTION
from schemas import AskResponse


def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Single chat-completion call to Groq's free tier (GROQ_API_KEY env var)."""
    from groq import Groq  # lazy import

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("MOCK_LLM=0 requires GROQ_API_KEY to be set")
    resp = Groq(api_key=key).chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_response(raw: str) -> AskResponse:
    """Strip stray markdown fences, then validate against the Pydantic schema."""
    cleaned = _FENCE.sub("", raw.strip()).strip()
    return AskResponse.model_validate(json.loads(cleaned))


def error_response(reason: str) -> AskResponse:
    return AskResponse(
        answer=f"ERROR: could not produce a valid structured answer ({reason}).",
        sources=[],
        confidence=0.0,
    )


def generate_structured(
    prompt: str, llm: Optional[Callable[[str], str]] = None
) -> AskResponse:
    """Call the LLM and enforce the schema: 1 attempt + up to 2 corrective retries."""
    llm = llm or call_llm
    current_prompt = prompt
    last_error = "unknown"
    for _attempt in range(1 + MAX_SCHEMA_RETRIES):
        try:
            raw = llm(current_prompt)
            return parse_response(raw)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = str(exc).splitlines()[0][:200]
            current_prompt = (
                prompt + "\n\n" + CORRECTIVE_INSTRUCTION.format(error=last_error)
            )
        except Exception as exc:  # network / auth / rate limit: not a schema problem
            return error_response(f"LLM call failed: {type(exc).__name__}")
    return error_response(last_error)


def classify_with_llm(query: str, llm: Optional[Callable[[str], str]] = None) -> str:
    """LLM-based intent classification; raises ValueError on an unusable reply."""
    llm = llm or call_llm
    prompt = (
        "Classify the customer query as exactly one label: policy_question (about "
        "Zepto delivery, returns, refunds, membership, tracking, cancellation, gift "
        "cards or support hours) or general_question (anything else). "
        "Reply with only the label.\n\nQuery: " + query + "\nLabel:"
    )
    label = llm(prompt).strip().lower()
    if "policy_question" in label:
        return "policy_question"
    if "general_question" in label:
        return "general_question"
    raise ValueError(f"unusable classification: {label!r}")
