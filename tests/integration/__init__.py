"""Cross-subsystem integration tests.

These tests wire the real pipeline, storage, retrieval, and RAG
components together over a temporary on-disk ChromaDB + SQLite stack.
Only the true external boundaries are faked: the EDGAR HTML source,
the embedding model, and the LLM API.

The suite is marked ``@pytest.mark.integration``.  It needs no network
and no external service, so it stays in the default CI test run.
"""
