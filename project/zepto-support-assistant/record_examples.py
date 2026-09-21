"""Runs the two demo calls against the FastAPI app (in-process, MOCK_LLM at its default)
and writes the raw JSON responses into README.md between the EXAMPLES markers.

    python record_examples.py
"""
import json
import os
import re
from pathlib import Path

os.environ.pop("MOCK_LLM", None)  # make sure we record the graded default (mock) mode

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

EXAMPLES = [
    ("Policy question (routes to retrieve_and_answer)", "What is the delivery fee for orders below INR 149?"),
    ("General question (routes to direct_answer)", "What is the capital of France?"),
]


def main() -> None:
    blocks = []
    with TestClient(app) as client:  # triggers startup -> ensure_index()
        for title, q in EXAMPLES:
            r = client.post("/ask", json={"query": q})
            assert r.status_code == 200, r.text
            body = json.dumps(r.json(), indent=2, ensure_ascii=False)
            req = json.dumps({"query": q})
            blocks.append(
                f"**{title}**\n\n```bash\ncurl -s -X POST http://localhost:8000/ask "
                f"-H 'Content-Type: application/json' -d '{req}'\n```\n\n```json\n{body}\n```\n"
            )
    text = "\n".join(blocks)
    print(text)

    readme = Path(__file__).with_name("README.md")
    md = readme.read_text(encoding="utf-8")
    new = re.sub(
        r"(<!-- EXAMPLES:START -->).*?(<!-- EXAMPLES:END -->)",
        lambda m: f"{m.group(1)}\n{text}\n{m.group(2)}",
        md,
        flags=re.DOTALL,
    )
    readme.write_text(new, encoding="utf-8")
    print("README.md updated.")


if __name__ == "__main__":
    main()
