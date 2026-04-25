"""Tests for hierarchical Pydantic Settings.

Covers the nested settings classes (LLMSettings, ProviderSettings,
RAGSettings) and the security-relevant behaviours of DatabaseSettings
(path traversal guard, encryption key resolution).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
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
        # Chat history MUST default to off by default.
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
# EmbeddingSettings — provider validator + local-only knob guard
# ---------------------------------------------------------------------------


class TestEmbeddingSettingsDefaults:
    def test_defaults(self, clean_env: pytest.MonkeyPatch) -> None:
        s = EmbeddingSettings()
        # ``local`` is the default so an out-of-the-box install does not
        # require a hosted-provider credential to construct the settings.
        assert s.provider == "local"
        assert s.model_name == "google/embeddinggemma-300m"
        assert s.device == "auto"
        assert s.batch_size == 32
        assert s.idle_timeout_minutes == 0

    def test_env_override_hosted_provider(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("EMBEDDING_PROVIDER", "openai")
        clean_env.setenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        s = EmbeddingSettings()
        assert s.provider == "openai"
        assert s.model_name == "text-embedding-3-small"


class TestEmbeddingSettingsProviderValidator:
    """The ``provider`` field must refuse unknown registry names."""

    def test_unknown_provider_rejected(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("EMBEDDING_PROVIDER", "acme-corp")
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingSettings()
        # The error names the offending value and includes the
        # registry's list so a typo is fixable without hunting through
        # docs.
        message = str(exc_info.value)
        assert "acme-corp" in message
        assert "Known embedding providers" in message

    def test_empty_string_rejected(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("EMBEDDING_PROVIDER", "")
        with pytest.raises(ValidationError):
            EmbeddingSettings()

    def test_local_accepted_even_when_extra_missing(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        """``include_unavailable=True`` is what makes this work.

        A user may configure ``EMBEDDING_PROVIDER=local`` in ``.env``
        before installing ``[local-embeddings]``.  Settings load must
        still succeed — the factory surfaces a clear install hint when
        an embedder is actually built.
        """
        import importlib.util

        from sec_generative_search.providers.registry import ProviderRegistry

        real_find_spec = importlib.util.find_spec
        clean_env.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **kw: (
                None if name == "sentence_transformers" else real_find_spec(name, *a, **kw)
            ),
        )
        ProviderRegistry._reset_availability_cache()

        clean_env.setenv("EMBEDDING_PROVIDER", "local")
        # No error — the validator accepts gated entries.
        s = EmbeddingSettings()
        assert s.provider == "local"

    def test_registry_is_the_source_of_truth(self, clean_env: pytest.MonkeyPatch) -> None:
        """Every registry-known embedding provider loads cleanly."""
        from sec_generative_search.providers.registry import (
            ProviderRegistry,
            ProviderSurface,
        )

        registered = {
            entry.name
            for entry in ProviderRegistry.all_entries(
                ProviderSurface.EMBEDDING, include_unavailable=True
            )
        }
        # A future new vendor must not require a manual test edit here —
        # the loop covers whatever the registry advertises.
        for name in registered:
            clean_env.setenv("EMBEDDING_PROVIDER", name)
            # Hosted providers need a compatible model slug; using the
            # provider's default removes the coupling with this test.
            clean_env.delenv("EMBEDDING_MODEL_NAME", raising=False)
            # For hosted providers we also need to use a valid model
            # from the provider's catalogue — use the local default for
            # 'local' and the registry-known default otherwise.
            cls = ProviderRegistry.all_entries(ProviderSurface.EMBEDDING, include_unavailable=True)
            model = next(
                (
                    e.provider_cls.default_model
                    for e in cls
                    if e.name == name and e.provider_cls.default_model
                ),
                None,
            )
            if model is not None:
                clean_env.setenv("EMBEDDING_MODEL_NAME", model)
            s = EmbeddingSettings()
            assert s.provider == name


class TestEmbeddingSettingsLocalOnlyKnobGuard:
    """Non-default local-only knobs must be rejected for hosted providers."""

    def test_local_provider_accepts_local_knobs(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("EMBEDDING_PROVIDER", "local")
        clean_env.setenv("EMBEDDING_DEVICE", "cuda")
        clean_env.setenv("EMBEDDING_BATCH_SIZE", "64")
        clean_env.setenv("EMBEDDING_IDLE_TIMEOUT_MINUTES", "5")
        s = EmbeddingSettings()
        assert s.device == "cuda"
        assert s.batch_size == 64
        assert s.idle_timeout_minutes == 5

    def test_hosted_provider_rejects_device(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("EMBEDDING_PROVIDER", "openai")
        clean_env.setenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        clean_env.setenv("EMBEDDING_DEVICE", "cuda")
        with pytest.raises(ValidationError, match="device='cuda'"):
            EmbeddingSettings()

    def test_hosted_provider_rejects_batch_size(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("EMBEDDING_PROVIDER", "openai")
        clean_env.setenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        clean_env.setenv("EMBEDDING_BATCH_SIZE", "128")
        with pytest.raises(ValidationError, match="batch_size=128"):
            EmbeddingSettings()

    def test_hosted_provider_rejects_idle_timeout(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("EMBEDDING_PROVIDER", "openai")
        clean_env.setenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        clean_env.setenv("EMBEDDING_IDLE_TIMEOUT_MINUTES", "5")
        with pytest.raises(ValidationError, match="idle_timeout_minutes=5"):
            EmbeddingSettings()

    def test_hosted_provider_error_names_every_offender(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        """Multiple non-default knobs must all appear in the error."""
        clean_env.setenv("EMBEDDING_PROVIDER", "gemini")
        clean_env.setenv("EMBEDDING_MODEL_NAME", "text-embedding-004")
        clean_env.setenv("EMBEDDING_DEVICE", "cuda")
        clean_env.setenv("EMBEDDING_BATCH_SIZE", "128")
        clean_env.setenv("EMBEDDING_IDLE_TIMEOUT_MINUTES", "5")
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingSettings()
        message = str(exc_info.value)
        assert "device='cuda'" in message
        assert "batch_size=128" in message
        assert "idle_timeout_minutes=5" in message

    def test_hosted_provider_with_all_defaults_passes(self, clean_env: pytest.MonkeyPatch) -> None:
        """The guard only fires for *non-default* knobs."""
        clean_env.setenv("EMBEDDING_PROVIDER", "openai")
        clean_env.setenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        # device/batch_size/idle_timeout_minutes all left at defaults.
        s = EmbeddingSettings()
        assert s.provider == "openai"


@pytest.mark.security
class TestEmbeddingSettingsCredentialHygiene:
    """``EmbeddingSettings`` must never expose credential-shaped fields."""

    def test_no_credential_fields(self, clean_env: pytest.MonkeyPatch) -> None:
        # The same hint list the core-types security test uses — keep in
        # sync so a future rename can only widen the guard.
        bad = ("api_key", "secret", "password", "credential", "bearer", "auth_token")
        for name in EmbeddingSettings.model_fields:
            for hint in bad:
                assert hint not in name.lower(), (
                    f"EmbeddingSettings.{name} looks credential-bearing; "
                    f"credentials must not live on settings."
                )


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
        # New sections
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


class TestDatabaseDeploymentProfile:
    """Profile-driven defaults for ``max_filings`` and ``retention_max_age_days``.

    The contract: the deployment profile fills in unset fields from the
    profile-defaults table; explicit env vars always win.  Local
    profile preserves the historical static defaults so existing
    operators see zero behavioural change.
    """

    def test_local_default_preserves_historical_behaviour(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        s = DatabaseSettings()
        assert s.deployment_profile == "local"
        assert s.max_filings == 2500
        assert s.retention_max_age_days == 0

    def test_team_profile_supplies_defaults(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("DB_DEPLOYMENT_PROFILE", "team")
        s = DatabaseSettings()
        assert s.deployment_profile == "team"
        assert s.max_filings == 10000
        assert s.retention_max_age_days == 90

    def test_cloud_profile_supplies_defaults(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("DB_DEPLOYMENT_PROFILE", "cloud")
        s = DatabaseSettings()
        assert s.deployment_profile == "cloud"
        assert s.max_filings == 10000
        assert s.retention_max_age_days == 30

    def test_explicit_max_filings_wins_over_profile(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("DB_DEPLOYMENT_PROFILE", "team")
        clean_env.setenv("DB_MAX_FILINGS", "20000")
        s = DatabaseSettings()
        assert s.max_filings == 20000
        # Retention still pulled from profile defaults — only the
        # explicitly-overridden field bypasses the profile.
        assert s.retention_max_age_days == 90

    def test_explicit_retention_zero_disables_eviction_in_team(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        """Operator opts a team-sized deployment out of eviction."""
        clean_env.setenv("DB_DEPLOYMENT_PROFILE", "team")
        clean_env.setenv("DB_RETENTION_MAX_AGE_DAYS", "0")
        s = DatabaseSettings()
        assert s.retention_max_age_days == 0
        assert s.max_filings == 10000

    def test_explicit_retention_overrides_cloud_profile(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("DB_DEPLOYMENT_PROFILE", "cloud")
        clean_env.setenv("DB_RETENTION_MAX_AGE_DAYS", "7")
        s = DatabaseSettings()
        assert s.retention_max_age_days == 7

    def test_unknown_profile_rejected(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        clean_env.setenv("DB_DEPLOYMENT_PROFILE", "edge")
        with pytest.raises(ValidationError, match="DB_DEPLOYMENT_PROFILE"):
            DatabaseSettings()

    @pytest.mark.security
    def test_negative_retention_rejected(
        self,
        clean_env: pytest.MonkeyPatch,
    ) -> None:
        """Defence-in-depth: a negative cutoff would invert the WHERE clause
        and delete recent filings.  Rejected at settings load."""
        clean_env.setenv("DB_RETENTION_MAX_AGE_DAYS", "-1")
        with pytest.raises(ValidationError, match=">= 0"):
            DatabaseSettings()


@pytest.mark.security
class TestEncryptionKeyResolution:
    """Security: DB_ENCRYPTION_KEY / DB_ENCRYPTION_KEY_FILE mutual exclusion
    and file validation — prevents silent misconfiguration of the at-rest
    encryption key.
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
