"""Security tests for header redaction (10A.7)."""

from __future__ import annotations

import pytest

from sec_generative_search.api.access_log import (
    redact_header_value,
    redact_headers,
)


@pytest.mark.security
class TestRedactHeaderValue:
    def test_authorization_is_masked(self) -> None:
        out = redact_header_value("Authorization", "Bearer sk-1234567890abcdef")
        assert "sk-1234567890abcdef" not in out
        assert out.endswith("cdef")

    def test_cookie_is_masked(self) -> None:
        out = redact_header_value("Cookie", "sec_rag_session=ABCDE_long_value_xyz123")
        assert "ABCDE_long_value_xyz123" not in out

    def test_provider_key_header_is_masked(self) -> None:
        out = redact_header_value("X-Provider-Key-Openai", "sk-actual-secret-1234")
        assert "sk-actual-secret-1234" not in out

    def test_provider_key_prefix_match_case_insensitive(self) -> None:
        out = redact_header_value("x-provider-key-anthropic", "sk-anthropic-abcd")
        assert "sk-anthropic-abcd" not in out

    def test_edgar_identity_is_fully_suppressed(self) -> None:
        # Email and name MUST be entirely redacted — not even tail-shown.
        assert redact_header_value("X-Edgar-Email", "user@example.com") == "***"
        assert redact_header_value("X-Edgar-Name", "Jane Doe") == "***"

    def test_unknown_header_passes_through(self) -> None:
        assert redact_header_value("X-Custom-Header", "free text") == "free text"

    def test_short_value_is_fully_redacted(self) -> None:
        # ``mask_secret`` redacts entirely below 8 chars.
        out = redact_header_value("Authorization", "abc")
        assert "abc" not in out


@pytest.mark.security
class TestRedactHeaders:
    def test_dict_input(self) -> None:
        out = redact_headers(
            {
                "Authorization": "Bearer secret_long_token",
                "X-Provider-Key-Openai": "sk-real-key-12345",
                "X-Custom": "hello",
                "X-Edgar-Name": "Jane Doe",
            }
        )
        assert "secret_long_token" not in out["Authorization"]
        assert "sk-real-key-12345" not in out["X-Provider-Key-Openai"]
        assert out["X-Custom"] == "hello"
        assert out["X-Edgar-Name"] == "***"

    def test_list_of_tuples_input(self) -> None:
        out = redact_headers(
            [
                ("Cookie", "sec_rag_session=abcdef0123456789long"),
                ("X-API-Key", "api-secret-12345"),
            ]
        )
        assert "abcdef0123456789long" not in out["Cookie"]
        assert "api-secret-12345" not in out["X-API-Key"]
