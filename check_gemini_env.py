"""
Quick environment probe to debug Gemini integration.

Run:
    python check_gemini_env.py

It reports:
    - Python executable in use
    - google-genai import status and version
    - whether GEMINI_API_KEY is set
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    print(f"Python: {sys.executable}")
    print(f"sys.path entries: {len(sys.path)}")

    try:
        from google import genai  # type: ignore

        print("google.genai: OK")
        version = getattr(genai, "__version__", "unknown")
        print(f"google-genai version: {version}")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"google.genai import FAILED: {exc.__class__.__name__}: {exc}")
        return

    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    print(f"GEMINI_API_KEY present: {api_key_set}")


if __name__ == "__main__":
    main()
