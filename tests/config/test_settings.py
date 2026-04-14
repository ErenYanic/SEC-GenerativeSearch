"""Tests for hierarchical Pydantic Settings.

Covers Phase 1.3: the three new nested settings classes (LLMSettings,
ProviderSettings, RAGSettings) and the security-relevant behaviours of
DatabaseSettings (path traversal guard, encryption key resolution).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sec_generative_search.config.settings import (
    DatabaseSettings,
    LLMSettings,
    ProviderSettings,
    RAGSettings,
    Settings,
    reload_settings,
    resolve_encryption_key_from_values,
)

# ---------------------------------------------------------------------------
# LLMSettings
# ---------------------------------------------------------------------------


class TestLLMSettings:
    def test_defaults(self, clean_env: pytest.MonkeyPatch) -> None:
        s = LLMSettings()
        assert s.default_provider == "openai"
        assert s.default_model is None
        assert s.temperature == pytest.approx(0.1)
        assert s.max_output_tokens == 2048
        assert s.streaming is True

    def test_env_override(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("LLM_DEFAULT_PROVIDER", "anthropic")
        clean_env.setenv("LLM_DEFAULT_MODEL", "claude-sonnet-4")
        clean_env.setenv("LLM_TEMPERATURE", "0.7")
        clean_env.setenv("LLM_MAX_OUTPUT_TOKENS", "4096")
        clean_env.setenv("LLM_STREAMING", "false")

        s = LLMSettings()
        assert s.default_provider == "anthropic"
        assert s.default_model == "claude-sonnet-4"
        assert s.temperature == pytest.approx(0.7)
        assert s.max_output_tokens == 4096
        assert s.streaming is False


# ---------------------------------------------------------------------------
# ProviderSettings
# ---------------------------------------------------------------------------


class TestProviderSettings:
    def test_defaults(self, clean_env: pytest.MonkeyPatch) -> None:
        s = ProviderSettings()
        assert s.timeout == 60
        assert s.max_retries == 3
        assert s.retry_backoff_base == pytest.approx(2.0)
        assert s.circuit_breaker_threshold == 5
        assert s.circuit_breaker_reset == 60
        assert s.cost_tracking_enabled is True

    def test_env_override(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("PROVIDER_TIMEOUT", "30")
        clean_env.setenv("PROVIDER_MAX_RETRIES", "5")
        clean_env.setenv("PROVIDER_COST_TRACKING_ENABLED", "false")

        s = ProviderSettings()
        assert s.timeout == 30
        assert s.max_retries == 5
        assert s.cost_tracking_enabled is False


# ---------------------------------------------------------------------------
# RAGSettings
# ---------------------------------------------------------------------------


class TestRAGSettings:
    def test_defaults(self, clean_env: pytest.MonkeyPatch) -> None:
        s = RAGSettings()
        assert s.context_token_budget == 6000
        assert s.citation_mode == "inline"
        assert s.default_answer_mode == "concise"
        assert s.refusal_enabled is True
        assert s.chunk_overlap_tokens == 50
        # Chat history MUST default to off (security/privacy baseline, Phase 3.3).
        assert s.chat_history_enabled is False
        assert s.chat_history_max_turns == 10

    def test_env_override(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("RAG_CONTEXT_TOKEN_BUDGET", "8000")
        clean_env.setenv("RAG_CITATION_MODE", "footnote")
        clean_env.setenv("RAG_DEFAULT_ANSWER_MODE", "analytical")
        clean_env.setenv("RAG_REFUSAL_ENABLED", "false")

        s = RAGSettings()
        assert s.context_token_budget == 8000
        assert s.citation_mode == "footnote"
        assert s.default_answer_mode == "analytical"
        assert s.refusal_enabled is False


# ---------------------------------------------------------------------------
# Settings composition
# ---------------------------------------------------------------------------


class TestSettingsComposition:
    def test_all_sections_present(self, clean_env: pytest.MonkeyPatch) -> None:
        s = reload_settings()
        # Existing sections (carried over)
        assert s.edgar is not None
        assert s.embedding is not None
        assert s.chunking is not None
        assert s.database is not None
        assert s.search is not None
        assert s.log_file is not None
        assert s.hugging_face is not None
        assert s.api is not None
        # New sections (Phase 1.3)
        assert isinstance(s.llm, LLMSettings)
        assert isinstance(s.provider, ProviderSettings)
        assert isinstance(s.rag, RAGSettings)

    def test_singleton_returns_same_instance_until_reload(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        from sec_generative_search.config.settings import get_settings

        first = get_settings()
        second = get_settings()
        assert first is second

        reloaded = reload_settings()
        assert reloaded is not first


# ---------------------------------------------------------------------------
# DatabaseSettings security — path traversal + encryption key resolution
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestDatabasePathSecurity:
    """Security: DB_*_PATH env vars must not escape the working directory."""

    def test_relative_path_inside_cwd_allowed(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("DB_CHROMA_PATH", "./data/chroma_db")
        clean_env.setenv("DB_METADATA_DB_PATH", "./data/metadata.sqlite")
        # Should not raise.
        DatabaseSettings()

    def test_absolute_path_outside_cwd_rejected(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("DB_CHROMA_PATH", "/tmp/escape_chroma")  # noqa: S108
        with pytest.raises(ValueError, match="outside the project directory"):
            DatabaseSettings()

    def test_relative_traversal_rejected(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("DB_CHROMA_PATH", "../../../etc/sneaky_chroma")
        with pytest.raises(ValueError, match="outside the project directory"):
            DatabaseSettings()

    def test_symlink_in_path_rejected(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        """A symlink in the database path must be refused.

        We make the symlink target stay INSIDE cwd so the prior
        "outside project" check passes — isolating the symlink check
        as the one that fires.
        """
        cwd = Path.cwd()
        host_dir = cwd / "_test_symlink_host"
        real_target = cwd / "_test_symlink_target"
        host_dir.mkdir(exist_ok=True)
        real_target.mkdir(exist_ok=True)
        link = host_dir / "link"
        try:
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(real_target)

            clean_env.setenv("DB_CHROMA_PATH", str(link / "chroma_db"))
            with pytest.raises(ValueError, match="symlink"):
                DatabaseSettings()
        finally:
            if link.is_symlink():
                link.unlink()
            if host_dir.exists():
                host_dir.rmdir()
            if real_target.exists():
                real_target.rmdir()


@pytest.mark.security
class TestEncryptionKeyResolution:
    """Security: DB_ENCRYPTION_KEY / DB_ENCRYPTION_KEY_FILE mutual exclusion
    and file validation — prevents silent misconfiguration of the at-rest
    encryption key (Phase 3.5).
    """

    def test_direct_key_used(self) -> None:
        assert resolve_encryption_key_from_values("secret", None) == "secret"

    def test_no_source_returns_none(self) -> None:
        assert resolve_encryption_key_from_values(None, None) is None

    def test_both_sources_is_error(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_encryption_key_from_values("secret", "/some/file")

    def test_key_file_read(self, tmp_path: Path) -> None:
        key_file = tmp_path / "db.key"
        key_file.write_text("my-secret-key\n")
        assert resolve_encryption_key_from_values(None, str(key_file)) == "my-secret-key"

    def test_key_file_missing_is_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.key"
        with pytest.raises(ValueError, match="does not exist"):
            resolve_encryption_key_from_values(None, str(missing))

    def test_key_file_empty_is_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.key"
        empty.write_text("")
        with pytest.raises(ValueError, match="empty"):
            resolve_encryption_key_from_values(None, str(empty))

    def test_key_file_is_directory_is_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a file"):
            resolve_encryption_key_from_values(None, str(tmp_path))


# ---------------------------------------------------------------------------
# ApiSettings — empty-string normalisation (security-adjacent)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestApiSettings:
    def test_empty_string_api_key_becomes_none(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        """An empty API_KEY must NOT accidentally enable a zero-length credential."""
        clean_env.setenv("API_KEY", "")
        s = Settings()
        assert s.api.key is None
        assert s.api.admin_key is None
