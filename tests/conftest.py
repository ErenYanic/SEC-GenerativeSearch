"""Shared pytest fixtures for SEC-GenerativeSearch.

Currently focused on foundation tests. As more subsystems land,
fixtures for HTTP clients, mocked
providers, and temporary ChromaDB/SQLite stores will be added here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Remove all env vars that influence Settings defaults.

    Used by settings tests so that one test's env var does not leak into
    another.  Pydantic Settings reads os.environ at instantiation time,
    so isolating that environment keeps tests deterministic.
    """
    prefixes = (
        "EDGAR_",
        "EMBEDDING_",
        "CHUNKING_",
        "DB_",
        "LLM_",
        "PROVIDER_",
        "RAG_",
        "SEARCH_",
        "LOG_FILE_",
        "LOG_LEVEL",
        "LOG_REDACT_QUERIES",
        "HUGGING_FACE_",
        "API_",
    )
    for key in list(os.environ.keys()):
        if key.startswith(prefixes):
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch
