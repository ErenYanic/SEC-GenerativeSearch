"""Tests for :mod:`sec_generative_search.providers.factory`.

Covers:

- Default resolver shape: provider-specific env-var lookup, empty-string
    coercion to ``None``, unknown-name pass-through.
- ``build_embedder`` construction flow through
  :class:`ProviderRegistry`, including the forbidden-outside-the-factory
  contract (exercised by real provider classes, not bypassed by stubs).
- Hosted-provider missing-credential path surfaces a
  :class:`ConfigurationError` with the expected env var name.
- ``local`` tolerates a missing credential via the sentinel path.
- Custom resolver overrides the default.
- Security: factory never retains the resolved credential after the
  call, and the ``ConfigurationError`` message never echoes any
  credential material (the default resolver is the only seam that
  produces the value, and it never passes the value to the error
  branch — fail-fast guards that invariant).
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from typing import Any

import pytest

from sec_generative_search.config.settings import EmbeddingSettings
from sec_generative_search.core.exceptions import ConfigurationError
from sec_generative_search.providers.factory import (
    build_embedder,
    default_api_key_resolver,
)
from sec_generative_search.providers.openai import OpenAIEmbeddingProvider
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)

_LOCAL_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
_requires_local = pytest.mark.skipif(
    not _LOCAL_AVAILABLE,
    reason="LocalEmbeddingProvider requires the [local-embeddings] extra",
)


@pytest.fixture(autouse=True)
def _reset_registry_cache() -> None:
    ProviderRegistry._reset_availability_cache()


@pytest.fixture(autouse=True)
def _clean_embedding_and_credential_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[pytest.MonkeyPatch]:
    """Strip every env var that would leak the developer's ``.env``.

    Pydantic settings read ``.env`` eagerly; without this fixture the
    ``EmbeddingSettings(...)`` calls below inherit the local machine's
    ``EMBEDDING_BATCH_SIZE=8`` (or similar) and trip the hosted-provider
    guard.  The credential env vars are stripped for the same reason —
    a populated ``OPENAI_API_KEY`` on the dev machine would paper over
    the "missing credential" tests.
    """
    for key in list(os.environ.keys()):
        if key.startswith("EMBEDDING_") or key in {
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "MISTRAL_API_KEY",
            "DASHSCOPE_API_KEY",
            "HF_TOKEN",
        }:
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch


# ---------------------------------------------------------------------------
# default_api_key_resolver
# ---------------------------------------------------------------------------


class TestDefaultApiKeyResolver:
    @pytest.mark.parametrize(
        "provider,env_var",
        [
            ("openai", "OPENAI_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
            ("mistral", "MISTRAL_API_KEY"),
            ("qwen", "DASHSCOPE_API_KEY"),
            ("local", "HF_TOKEN"),
        ],
    )
    def test_reads_expected_env_var(
        self,
        provider: str,
        env_var: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(env_var, "sk-value-from-env")
        assert default_api_key_resolver(provider) == "sk-value-from-env"

    def test_unknown_provider_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Unregistered names never raise from the resolver; the factory
        # decides what to do with a missing credential.
        assert default_api_key_resolver("not-a-real-provider") is None

    def test_empty_string_env_value_coerces_to_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Blank ``OPENAI_API_KEY=`` must not enable a zero-length key."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        assert default_api_key_resolver("openai") is None

    def test_missing_env_var_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert default_api_key_resolver("openai") is None


# ---------------------------------------------------------------------------
# build_embedder — happy path
# ---------------------------------------------------------------------------


class TestBuildEmbedderHosted:
    def test_builds_openai_embedding_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-ABCDEFGHIJKLMNOP")

        # Construct settings directly (no env I/O) so the test does not
        # depend on ``.env`` contents.
        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        embedder = build_embedder(settings)

        assert isinstance(embedder, OpenAIEmbeddingProvider)
        # The factory must wire the configured model through.
        assert embedder.get_dimension() == 1536

    def test_raises_without_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        with pytest.raises(ConfigurationError) as exc_info:
            build_embedder(settings)
        # The message names the env var so the user can fix it without
        # reading the factory source.
        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_custom_resolver_is_used(self) -> None:
        calls: list[str] = []

        def resolver(name: str) -> str | None:
            calls.append(name)
            return "sk-from-custom-resolver-1234"

        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        embedder = build_embedder(settings, api_key_resolver=resolver)
        assert isinstance(embedder, OpenAIEmbeddingProvider)
        # The resolver was called exactly once with the configured
        # provider name.
        assert calls == ["openai"]

    def test_resolver_none_for_hosted_provider_raises(self) -> None:
        settings = EmbeddingSettings(provider="gemini", model_name="text-embedding-004")
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            build_embedder(settings, api_key_resolver=lambda _name: None)


# ---------------------------------------------------------------------------
# build_embedder — local path
# ---------------------------------------------------------------------------


@_requires_local
class TestBuildEmbedderLocal:
    def test_local_tolerates_missing_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        settings = EmbeddingSettings(
            provider="local", model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        embedder = build_embedder(settings)
        # The provider accepts ``None`` via its ``_NO_TOKEN_SENTINEL``
        # path; the factory must not intercept it with a
        # ConfigurationError.
        from sec_generative_search.providers.local import LocalEmbeddingProvider

        assert isinstance(embedder, LocalEmbeddingProvider)


class TestBuildEmbedderExtrasGating:
    def test_missing_extras_surfaces_registry_key_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``[local-embeddings]`` installed, the factory must
        surface the registry's install-hint ``KeyError`` — this is the
        contract that lets settings load succeed while construction
        fails with actionable guidance.
        """
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **kw: (
                None if name == "sentence_transformers" else real_find_spec(name, *a, **kw)
            ),
        )
        ProviderRegistry._reset_availability_cache()

        settings = EmbeddingSettings(provider="local", model_name="google/embeddinggemma-300m")
        with pytest.raises(KeyError, match="optional extras"):
            build_embedder(settings)


# ---------------------------------------------------------------------------
# build_embedder — construction seam & security
# ---------------------------------------------------------------------------


class TestFactoryIsTheSoleSeam:
    """The factory must route through ProviderRegistry so callers can
    swap the resolver without touching any call site."""

    def test_factory_delegates_to_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patching ``ProviderRegistry.get_class`` proves the factory
        uses the registry rather than importing adapters directly."""
        calls: list[tuple[str, ProviderSurface]] = []

        def stub_class(name: str, surface: ProviderSurface) -> type:
            calls.append((name, surface))
            return OpenAIEmbeddingProvider

        monkeypatch.setattr(ProviderRegistry, "get_class", stub_class)

        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        build_embedder(settings, api_key_resolver=lambda _n: "sk-test-1234ABCD")
        assert calls == [("openai", ProviderSurface.EMBEDDING)]


@pytest.mark.security
class TestFactoryCredentialHygiene:
    """The factory must not retain the credential beyond construction."""

    def test_configuration_error_does_not_echo_credential_material(self) -> None:
        """When the resolver returns ``None`` we know no credential is
        in flight, but assert that the error text also carries none of
        the bad-actor needles downstream logs grep for."""

        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        with pytest.raises(ConfigurationError) as exc_info:
            build_embedder(settings, api_key_resolver=lambda _n: None)
        rendered = str(exc_info.value).lower()
        # The env-var name mention is fine; credentials themselves must
        # not appear.
        for needle in ("sk-", "bearer", "secret=", "password"):
            assert needle not in rendered

    def test_module_carries_no_credentials_at_rest(self) -> None:
        """Module-level state must not accumulate keys across calls.

        A naive caching layer inside the factory could inadvertently
        hold credentials in memory after the embedder has been dropped.
        The factory is stateless by contract — assert it.
        """
        import sec_generative_search.providers.factory as factory_module

        # Whitelist of acceptable module-level attributes.
        module_attrs: dict[str, Any] = {
            name: getattr(factory_module, name)
            for name in dir(factory_module)
            if not name.startswith("_")
        }
        for name, value in module_attrs.items():
            # Strings at module scope must not look like keys.
            if isinstance(value, str):
                assert "sk-" not in value.lower()
                assert "bearer " not in value.lower()
            # Dict values at module scope (the env-var mapping) must be
            # env-var *names*, not credentials.
            if isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str):
                        # Env var names end in _KEY or _TOKEN by
                        # convention; a literal credential would not.
                        assert v.isupper(), (
                            f"factory module dict {name!r} contains a "
                            f"lower-case value {v!r} — the mapping must "
                            "carry env-var names only."
                        )
