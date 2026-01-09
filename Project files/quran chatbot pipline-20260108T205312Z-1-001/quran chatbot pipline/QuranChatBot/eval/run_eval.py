"""Simple evaluation harness for QuranChatBot.

This script reads a list of queries from ``sample_queries.jsonl`` and exercises the
pipeline via the FastAPI ``answer_query`` function. It validates that each
response conforms to the expected schema and prints a summary of successes and
failures.

Usage:

    python -m eval.run_eval

Before running this script ensure that you have:

* Started Qdrant (via ``docker compose up -d``).
* Indexed the ayahs and words collections using ``indexing/index_ayahs.py`` and
  ``indexing/index_words.py``.
* Exported your ``OPENAI_API_KEY`` in the environment.

The evaluation does not assert the correctness of the answers; it only
verifies that the response object is well formed and matches the user’s
requested language. Use this as a smoke test when upgrading the pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from typing import Any

from ..app.fastapi_app import answer_query, QueryRequest, QuranAnswer, PronunciationAnswer


def main() -> None:
    # Locate the sample queries file relative to this script
    samples_path = Path(__file__).resolve().parents[0] / "sample_queries.jsonl"
    queries = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            queries.append(json.loads(line))

    total = len(queries)
    passed = 0
    for idx, q in enumerate(queries, 1):
        req = QueryRequest(
            query=q["query"],
            lang=q.get("lang", "en"),
            top_k=q.get("top_k"),
            show_tafsir=q.get("show_tafsir", False),
        )
        try:
            resp = answer_query(req)
        except Exception as exc:
            print(f"[{idx}/{total}] ERROR processing '{q['query']}': {exc}")
            continue

        data: Any = resp
        # Determine whether we got a pronunciation or ayah answer
        if isinstance(data, PronunciationAnswer):
            # Response should have a non-empty pronunciation list or caution
            assert hasattr(data, "pronunciation"), "PronunciationAnswer missing pronunciation field"
            # answer_language should be 'en' regardless of input
            assert data.answer_language == "en", f"Pronunciation answer language mismatch: {data.answer_language}"
        elif isinstance(data, QuranAnswer):
            # Validate mandatory fields
            assert data.answer_language == req.lang, (
                f"Answer language mismatch: expected {req.lang}, got {data.answer_language}"
            )
            assert data.ayah_ref and ":" in data.ayah_ref, "Invalid ayah_ref format"
            assert data.ayah_arabic, "Missing Arabic text"
            assert data.translation, "Missing translation"
            assert data.response, "Missing response text"
            assert data.retrieval, "Missing retrieval info"
        else:
            print(f"[{idx}/{total}] Unknown response type for '{q['query']}'")
            continue
        passed += 1
        print(f"[{idx}/{total}] OK – '{q['query']}'")
    print(f"\nCompleted {total} queries: {passed} passed, {total - passed} failed")


if __name__ == "__main__":
    main()