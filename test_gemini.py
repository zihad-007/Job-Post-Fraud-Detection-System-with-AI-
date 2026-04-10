"""
Lightweight Gemini smoke test that avoids hard‑coding credentials.

The test only runs when `GEMINI_API_KEY` is present in the environment.
Otherwise it is skipped so local/dev environments without keys do not fail.
"""

import os
import pytest

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


@pytest.mark.skipif(genai is None, reason="google-genai not installed")
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping live Gemini call",
)
def test_gemini_hello_world():
    api_key = os.environ["GEMINI_API_KEY"]
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents="Say 'hello' in one short sentence.")

    text = getattr(response, "text", "") or ""
    assert text and "hello" in text.lower()
