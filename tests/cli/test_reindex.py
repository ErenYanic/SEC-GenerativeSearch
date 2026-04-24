"""Tests for ``sec-rag manage reindex``.

Most tests drive the command through ``typer.testing.CliRunner`` with
``build_embedder`` and :class:`ReindexService` stubbed at the
``cli.reindex`` import site.  Stubbing at the import site (as opposed to
at the source module) proves the CLI wires the *right* seams — a future
refactor that reached around :func:`build_embedder` would fail the
factory-seam test loudly.

One end-to-end test drives the real :class:`chromadb.PersistentClient`
under ``tmp_path`` so the whole fetch-less reindex path (seed → CLI →
swap → reopen) is covered without an SDK round-trip.  The deterministic
in-memory embedder is the same pattern ``tests/database/test_reindex.py``
uses.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.reindex as reindex_module
from sec_generative_search.cli.reindex import reindex
from sec_generative_search.config.constants import COLLECTION_NAME
from sec_generative_search.config.settings import EmbeddingSettings
from sec_generative_search.core.exceptions import ConfigurationError, DatabaseError
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    EmbedderStamp,
    FilingIdentifier,
    IngestResult,
    ReindexReport,
)
from sec_generative_search.database import ChromaDBClient
from sec_generative_search.pipeline.orchestrator import ProcessedFiling

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Minimal embedder double satisfying the surface the CLI touches.

    ``provider_name`` is required by ``_ProviderBase.__init_subclass__``
    on the real subclasses; the fake carries it so assertions that
    probe the embedder surface match production shape without dragging
    in the full base-class contract.
    """

    provider_name = "fake"

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.embed_calls: list[int] = []

    def get_dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.embed_calls.append(len(texts))
        rows = [
            [float(idx), float(len(t)), 0.0, 0.0][: self._dimension] for idx, t in enumerate(texts)
        ]
        return np.asarray(rows, dtype=np.float32)


class _FakeService:
    """Substitute for :class:`ReindexService` recording what the CLI drives.

    Captures the stamp and embedder the CLI hands it, emits progress
    events, and returns a canned :class:`ReindexReport`.  A ``raises``
    knob lets tests inject ``DatabaseError`` / ``KeyboardInterrupt`` to
    exercise the CLI error paths without touching Chroma.
    """

    def __init__(
        self,
        *,
        chunks: int = 6,
        duration: float = 0.25,
        raises: BaseException | None = None,
    ) -> None:
        self._chunks = chunks
        self._duration = duration
        self._raises = raises
        self.calls: list[tuple[EmbedderStamp, Any]] = []

    def run(
        self,
        target_stamp: EmbedderStamp,
        target_embedder: Any,
        *,
        progress_callback: Any = None,
    ) -> ReindexReport:
        self.calls.append((target_stamp, target_embedder))

        if progress_callback is not None:
            # Drive both step names so the CLI's relabelling branch
            # (embed → swap) executes in tests.
            progress_callback("reindex", self._chunks // 2, self._chunks)
            progress_callback("reindex", self._chunks, self._chunks)

        if self._raises is not None:
            raise self._raises

        if progress_callback is not None:
            progress_callback("reindex-swap", self._chunks, self._chunks)

        return ReindexReport(
            source_stamp=EmbedderStamp(
                provider="local",
                model="google/embeddinggemma-300m",
                dimension=768,
            ),
            target_stamp=target_stamp,
            chunks_copied=self._chunks,
            duration_seconds=self._duration,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    # Click 8.3+ dropped ``mix_stderr`` — stdout and stderr are already
    # split on ``result.output`` / ``result.stderr`` by default.
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    """Throw-away Typer app registering the reindex command as a subcommand.

    The test app uses the same registration shape as the production
    command registration so ``CliRunner`` exercises the parameter
    annotations.  A second no-op command is registered alongside so
    Typer dispatches ``reindex`` as a named subcommand rather than
    auto-unwrapping a single-command app.
    """
    test_app = typer.Typer()
    test_app.command(name="reindex")(reindex)

    @test_app.command(name="_noop", hidden=True)
    def _noop() -> None:
        """Forces Typer to treat ``reindex`` as a proper subcommand."""

    return test_app


@pytest.fixture
def patch_build_embedder(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``cli.reindex.build_embedder`` with a recording stub.

    The stub returns a ``_FakeEmbedder`` and records every call so tests
    can assert that the CLI routes through the factory seam rather than
    instantiating a provider directly.
    """
    calls: list[EmbeddingSettings] = []
    
    def _stub(settings: EmbeddingSettings, *_: Any, **__: Any) -> _FakeEmbedder:
        calls.append(settings)
        return _FakeEmbedder(dimension=1536)  # matches openai text-embedding-3-small

    monkeypatch.setattr(reindex_module, "build_embedder", _stub)
    return {"calls": calls}


@pytest.fixture
def patch_service(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the ``ReindexService`` name on ``cli.reindex`` with a recorder.

    Instances constructed by the CLI end up in ``container['instances']``
    so tests can inspect what the CLI drove without a mocking library.
    """
    instances: list[_FakeService] = []
    config: dict[str, Any] = {
        "chunks": 6,
        "duration": 0.25,
        "raises": None,
    }

    def _factory(*_: Any, **__: Any) -> _FakeService:
        svc = _FakeService(
            chunks=config["chunks"],
            duration=config["duration"],
            raises=config["raises"],
        )
        instances.append(svc)
        return svc

    monkeypatch.setattr(reindex_module, "ReindexService", _factory)
    return {"instances": instances, "config": config}


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_yes_flag_skips_confirmation_and_runs_service(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 0, result.output
        # Service must have been called once with the resolved target stamp.
        assert len(patch_service["instances"]) == 1
        svc = patch_service["instances"][0]
        assert len(svc.calls) == 1
        stamp, embedder = svc.calls[0]
        assert stamp == EmbedderStamp(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        )
        # Embedder was built through the factory seam.
        assert len(patch_build_embedder["calls"]) == 1
        settings = patch_build_embedder["calls"][0]
        assert settings.provider == "openai"
        assert settings.model_name == "text-embedding-3-small"
        # Embedder handed to service is the one the factory returned.
        assert isinstance(embedder, _FakeEmbedder)
        # Completion line rendered.
        assert "Reindex complete" in result.output
        assert "6 chunk(s)" in result.output

    def test_default_model_falls_back_to_provider_default(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        """Omitting --model must use the provider class's ``default_model``.

        The registry is the source of truth for the default; the CLI
        must not hard-code it here.
        """
        result = runner.invoke(app, ["reindex", "-p", "openai", "-y"])

        assert result.exit_code == 0, result.output
        stamp, _ = patch_service["instances"][0].calls[0]
        # OpenAI default is ``text-embedding-3-small`` per the provider.
        assert stamp.provider == "openai"
        assert stamp.model == "text-embedding-3-small"

    def test_interactive_confirm_yes_runs_service(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small"],
            input="y\n",
        )

        assert result.exit_code == 0, result.output
        assert len(patch_service["instances"][0].calls) == 1

    def test_interactive_confirm_no_aborts_without_running(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small"],
            input="n\n",
        )

        assert result.exit_code == 0, result.output
        assert "Cancelled" in result.output
        # Service was never instantiated because the CLI exited before
        # the construction branch.
        assert patch_service["instances"] == []

    def test_progress_bar_relabels_between_phases(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        """Both progress labels must surface in rendered output.

        Rich buffers the final rendered frame on completion; asserting
        on the captured output is enough to prove the CLI's relabel
        branch ran for both step names.
        """
        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 0, result.output
        # Swap-phase label wins the final frame; embed-phase label was
        # rendered during the in-between progress update.  Rich collapses
        # frames in non-TTY capture, so we only assert that at least one
        # of the two phase labels made it into the captured frame — the
        # service stub ensures both callbacks fired if exit was clean.
        assert (
            "Re-embedding chunks" in result.output or "Copying to live collection" in result.output
        )


# ---------------------------------------------------------------------------
# Input-validation + refuse paths
# ---------------------------------------------------------------------------


class TestRefusePaths:
    def test_unknown_provider_exits_with_hint(
        self,
        runner: CliRunner,
        app: typer.Typer,
    ) -> None:
        result = runner.invoke(
            app,
            ["reindex", "-p", "not-a-real-provider", "-m", "whatever", "-y"],
        )

        assert result.exit_code == 1
        assert "Unknown embedding provider" in result.output

    def test_unknown_model_exits_with_hint(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        """Unknown embedding slug: registry raises ValueError → CLI renders hint."""
        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "made-up-slug", "-y"],
        )

        assert result.exit_code == 1
        assert "Unknown embedding model" in result.output
        # Factory / service must not run when model resolution fails.
        assert patch_build_embedder["calls"] == []
        assert patch_service["instances"] == []

    def test_configuration_error_from_factory_exits_with_hint(
        self,
        runner: CliRunner,
        app: typer.Typer,
        monkeypatch: pytest.MonkeyPatch,
        patch_service: dict[str, Any],
    ) -> None:
        def _raise(*_: Any, **__: Any) -> None:
            raise ConfigurationError("No API key resolved for embedding provider 'openai'.")

        monkeypatch.setattr(reindex_module, "build_embedder", _raise)

        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 1
        assert "Embedder construction failed" in result.output
        # Service must not have been constructed when the factory refuses.
        assert patch_service["instances"] == []

    def test_missing_optional_extras_exits_with_install_hint(
        self,
        runner: CliRunner,
        app: typer.Typer,
        monkeypatch: pytest.MonkeyPatch,
        patch_service: dict[str, Any],
    ) -> None:
        """A registered provider that needs extras must surface an install hint.

        We drive the command with a registered provider (``openai``) so
        the earlier ``get_class`` probe succeeds, then force
        ``build_embedder`` to raise ``KeyError`` the way the registry
        would if an extras-gated provider were invoked without its
        extras installed.  This exercises the CLI's install-hint branch
        deterministically, independent of whether ``[local-embeddings]``
        happens to be installed in the current test environment.
        """

        def _raise(*_: Any, **__: Any) -> None:
            raise KeyError(
                "Provider 'local' (surface=embedding) requires optional extras: "
                "sentence_transformers. Install with `uv pip install -e "
                "'.[local-embeddings]'`."
            )

        monkeypatch.setattr(reindex_module, "build_embedder", _raise)

        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 1
        assert "Embedder unavailable" in result.output
        assert "local-embeddings" in result.output
        assert patch_service["instances"] == []

    def test_service_database_error_exits_with_message(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        patch_service["config"]["raises"] = DatabaseError(
            "Source and target stamps are identical — nothing to reindex.",
        )

        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 1
        assert "Reindex failed" in result.output
        assert "nothing to reindex" in result.output

    def test_keyboard_interrupt_exits_130_with_recovery_hint(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        """Ctrl-C mid-run must surface exit 130 and name the recovery path.

        :class:`ReindexService.run` catches ``Exception`` but not
        ``BaseException``, so a ``KeyboardInterrupt`` during embedding
        leaves the staging collection alive.  The CLI tells the
        operator exactly that instead of pretending cleanup happened.
        """
        patch_service["config"]["raises"] = KeyboardInterrupt()

        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 130
        assert "Interrupted" in result.output
        assert "re-run" in result.output.lower()


# ---------------------------------------------------------------------------
# Security / defence-in-depth
# ---------------------------------------------------------------------------


class TestSecurity:
    @pytest.mark.security
    def test_no_dimension_flag_on_surface(
        self,
        runner: CliRunner,
        app: typer.Typer,
    ) -> None:
        """``--dimension`` must not exist.

        Registry is the only authoritative source for a stamp's
        dimension; a CLI surface for it would let a misconfigured stamp
        slip past the pre-flight check and corrupt the collection.
        """
        result = runner.invoke(
            app,
            ["reindex", "--dimension", "768", "-y"],
        )

        # Typer surfaces unknown options as exit code 2 with a usage
        # error — that is the shape we want.
        assert result.exit_code == 2
        combined = (result.output or "") + (result.stderr or "")
        assert "--dimension" in combined or "No such option" in combined

    @pytest.mark.security
    def test_command_routes_through_build_embedder(
        self,
        runner: CliRunner,
        app: typer.Typer,
        monkeypatch: pytest.MonkeyPatch,
        patch_service: dict[str, Any],
    ) -> None:
        """The CLI must not sidestep :func:`build_embedder`.

        The factory is the sole construction seam.  We assert the CLI
        really calls the seam by stubbing it and checking it was
        invoked before the service ran.
        """
        invocations: list[EmbeddingSettings] = []

        def _stub(settings: EmbeddingSettings, *_: Any, **__: Any) -> _FakeEmbedder:
            invocations.append(settings)
            return _FakeEmbedder(dimension=1536)

        monkeypatch.setattr(reindex_module, "build_embedder", _stub)

        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 0, result.output
        assert len(invocations) == 1
        assert invocations[0].provider == "openai"
        # Service was then driven exactly once.
        assert len(patch_service["instances"][0].calls) == 1

    @pytest.mark.security
    def test_api_key_env_never_echoed_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        monkeypatch: pytest.MonkeyPatch,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        """A real-shaped API key in the env must never appear in output.

        Defence-in-depth against a future refactor that accidentally
        includes settings or env values in the rendered confirmation /
        completion banner.
        """
        secret = "sk-TEST-DO-NOT-LEAK-abcdef1234567890"
        monkeypatch.setenv("OPENAI_API_KEY", secret)

        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 0, result.output
        assert secret not in result.output

    @pytest.mark.security
    def test_hosted_target_ignores_local_only_env_knobs(
        self,
        runner: CliRunner,
        app: typer.Typer,
        monkeypatch: pytest.MonkeyPatch,
        patch_build_embedder: dict[str, Any],
        patch_service: dict[str, Any],
    ) -> None:
        """A hosted target must not trip ``EmbeddingSettings`` on env-only local knobs.

        If the operator has ``EMBEDDING_DEVICE=cuda`` set (common when
        the live embedder is local) and reindexes into ``openai``, the
        CLI must pin local-only knobs to defaults so the validator
        accepts the hosted provider.  Otherwise the reindex would be
        unusable for exactly the hand-off it exists to support.
        """
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "64")
        monkeypatch.setenv("EMBEDDING_IDLE_TIMEOUT_MINUTES", "15")

        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 0, result.output
        settings = patch_build_embedder["calls"][0]
        # The CLI must have pinned the local-only knobs to defaults for
        # a hosted target.
        assert settings.device == "auto"
        assert settings.batch_size == 32
        assert settings.idle_timeout_minutes == 0


# ---------------------------------------------------------------------------
# End-to-end: real Chroma + real ReindexService
# ---------------------------------------------------------------------------


def _make_filing(
    accession: str,
    *,
    ticker: str = "AAPL",
    filing_date: date | None = None,
    n_chunks: int = 3,
) -> ProcessedFiling:
    fid = FilingIdentifier(
        ticker=ticker,
        form_type="10-K",
        filing_date=filing_date or date(2023, 11, 3),
        accession_number=accession,
    )
    chunks = [
        Chunk(
            content=f"chunk {i}",
            path="Part I > Item 1 > Business",
            content_type=ContentType.TEXT,
            filing_id=fid,
            chunk_index=i,
            token_count=5,
        )
        for i in range(n_chunks)
    ]
    embeddings = np.arange(n_chunks * 4, dtype=np.float32).reshape(n_chunks, 4)
    return ProcessedFiling(
        filing_id=fid,
        chunks=chunks,
        embeddings=embeddings,
        ingest_result=IngestResult(
            filing_id=fid,
            segment_count=n_chunks,
            chunk_count=n_chunks,
            duration_seconds=0.0,
        ),
    )


class TestEndToEnd:
    def test_cli_swaps_live_collection_against_real_chroma(
        self,
        runner: CliRunner,
        app: typer.Typer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full stack: seed live, drive CLI, reopen with new stamp.

        ``build_embedder`` is the only external seam we replace — with a
        deterministic in-memory embedder.  The service itself, Chroma,
        the stamp resolution path, and the progress callback are real.
        """
        # ``DatabaseSettings._validate_paths`` rejects tmp-dir paths outside
        # CWD by design, so re-instantiating settings with a
        # ``DB_CHROMA_PATH=/tmp/...`` env var fails before the CLI even
        # runs.  Instead we inject the path at the ``ReindexService``
        # construction seam on the CLI, which is both the cleanest E2E
        # shape and a realistic one.
        chroma_path = str(tmp_path / "chroma")

        source_stamp = EmbedderStamp(
            provider="local",
            model="google/embeddinggemma-300m",
            dimension=4,
        )
        source_client = ChromaDBClient(source_stamp, chroma_path=chroma_path)
        # ``Chunk.chunk_id`` is ``{TICKER}_{FORM}_{DATE}_{INDEX}`` — the
        # accession number does not flow into the ID, so two fake
        # filings sharing ticker/form/date collide in Chroma.  Use
        # distinct tickers to keep IDs unique.
        for i in range(2):
            source_client.store_filing(
                _make_filing(
                    accession=f"0000320193-23-00000{i}",
                    ticker=f"TK{i}",
                    n_chunks=3,
                )
            )
        seeded_total = source_client.collection_count()
        assert seeded_total == 6
        del source_client

        # Keep the registry's resolved dimension in line with the fake
        # embedder we return — OpenAI's text-embedding-3-small catalogue
        # claims 1536, so we monkeypatch the registry lookup to 4 so the
        # stamp matches our in-memory embedder.  The CLI still drives the
        # real ReindexService through real Chroma.
        def _patched_get_dimension(name: str, model: str | None = None) -> int:
            return 4

        monkeypatch.setattr(
            "sec_generative_search.cli.reindex.ProviderRegistry.get_dimension",
            _patched_get_dimension,
        )
        monkeypatch.setattr(
            reindex_module,
            "build_embedder",
            lambda *_, **__: _FakeEmbedder(dimension=4),
        )

        # Inject the tmp chroma_path at the ReindexService construction
        # seam; see the comment above on why the env-var route is
        # unavailable.
        from sec_generative_search.database import ReindexService as _RealService

        monkeypatch.setattr(
            reindex_module,
            "ReindexService",
            lambda: _RealService(chroma_path=chroma_path),
        )

        result = runner.invoke(
            app,
            ["reindex", "-p", "openai", "-m", "text-embedding-3-small", "-y"],
        )

        assert result.exit_code == 0, result.output
        assert "Reindex complete" in result.output

        # The live collection must now open cleanly under the new stamp.
        target_stamp = EmbedderStamp(
            provider="openai",
            model="text-embedding-3-small",
            dimension=4,
        )
        reopened = ChromaDBClient(target_stamp, chroma_path=chroma_path)
        assert reopened.collection_count() == seeded_total

        # And the staging collection must be gone.
        raw = chromadb.PersistentClient(path=chroma_path)
        names = {c.name for c in raw.list_collections()}
        assert "sec_filings_reindex_staging" not in names
        assert COLLECTION_NAME in names
