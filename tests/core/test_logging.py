"""Tests for logging configuration."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

from sec_generative_search.core import logging as sgs_logging
from sec_generative_search.core.logging import (
    LOGGER_NAME,
    audit_log,
    configure_logging,
    get_logger,
    redact_for_log,
    suppress_third_party_loggers,
)


@pytest.fixture(autouse=True)
def _reset_logging_config() -> None:
    """Reset the module-level ``_logging_configured`` flag between tests."""
    # Pre-test: clear handlers and flag
    sgs_logging._logging_configured = False
    logging.getLogger(LOGGER_NAME).handlers.clear()
    yield
    # Post-test: same
    sgs_logging._logging_configured = False
    logging.getLogger(LOGGER_NAME).handlers.clear()


class TestLoggerName:
    def test_package_logger_uses_new_namespace(self) -> None:
        assert LOGGER_NAME == "sec_generative_search"

    def test_get_logger_prefixes_module_name(self) -> None:
        logger = get_logger("pipeline.fetch")
        assert logger.name == "sec_generative_search.pipeline.fetch"

    def test_get_logger_respects_full_prefix(self) -> None:
        logger = get_logger("sec_generative_search.core.logging")
        assert logger.name == "sec_generative_search.core.logging"


class TestConfiguration:
    def test_configures_once_and_is_idempotent(self) -> None:
        configure_logging(level=logging.DEBUG, use_rich=False)
        handlers_after_first = list(logging.getLogger(LOGGER_NAME).handlers)
        configure_logging(level=logging.INFO, use_rich=False)
        handlers_after_second = list(logging.getLogger(LOGGER_NAME).handlers)
        # Idempotent: the second call must not add new handlers.
        assert len(handlers_after_first) == len(handlers_after_second)

    def test_file_handler_added_when_env_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        log_file = tmp_path / "nested" / "run.log"
        monkeypatch.setenv("LOG_FILE_PATH", str(log_file))
        configure_logging(level=logging.DEBUG, use_rich=False)

        handlers = logging.getLogger(LOGGER_NAME).handlers
        # Must have at least a stream handler + a rotating file handler.
        assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers)
        # Parent directory was created automatically.
        assert log_file.parent.is_dir()


class TestRedactForLog:
    """Security: redact_for_log must not leak the original value when enabled."""

    def test_no_redaction_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOG_REDACT_QUERIES", raising=False)
        assert redact_for_log("hello") == "hello"

    @pytest.mark.parametrize("flag", ["1", "true", "yes", "TRUE", "Yes"])
    def test_redaction_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag: str,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", flag)
        out = redact_for_log("revenue forecast for AAPL")
        assert out.startswith("<redacted:")
        assert out.endswith(">")
        # Hash prefix must be deterministic.
        expected = hashlib.sha256(b"revenue forecast for AAPL").hexdigest()[:8]
        assert expected in out
        # Original must not appear anywhere.
        assert "revenue forecast" not in out
        assert "AAPL" not in out

    def test_redaction_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        # Same input → same redaction (preserves log correlation).
        assert redact_for_log("query X") == redact_for_log("query X")

    @pytest.mark.parametrize("flag", ["0", "false", "no", "", "maybe"])
    def test_redaction_disabled_for_falsy_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag: str,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", flag)
        assert redact_for_log("hello") == "hello"


class TestAuditLog:
    def test_audit_entry_uses_warning_level_and_prefix(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # ``configure_logging`` sets ``propagate = False`` on the package
        # logger, so caplog (which attaches to root) can't see records.
        # Enable propagation for the duration of this test so the
        # WARNING reaches caplog's handler.
        configure_logging(level=logging.DEBUG, use_rich=False)
        pkg_logger = logging.getLogger(LOGGER_NAME)
        pkg_logger.propagate = True
        audit_logger_name = f"{LOGGER_NAME}.security.audit"

        try:
            with caplog.at_level(logging.WARNING, logger=audit_logger_name):
                audit_log(
                    action="delete_filing",
                    client_ip="127.0.0.1",
                    endpoint="/api/filings/AAPL",
                    detail="accession=0000320193-23-000077",
                )
        finally:
            pkg_logger.propagate = False

        records = [r for r in caplog.records if r.name == audit_logger_name]
        assert records, "audit_log should emit on security.audit logger"
        msg = records[0].getMessage()
        assert "SECURITY_AUDIT:" in msg
        assert "action=delete_filing" in msg
        assert "client=127.0.0.1" in msg


class TestSuppressThirdPartyLoggers:
    def test_sets_noisy_loggers_to_warning(self) -> None:
        # Flip them to DEBUG first to verify the call actually changes them.
        for name in ("chromadb", "httpx", "sentence_transformers"):
            logging.getLogger(name).setLevel(logging.DEBUG)

        suppress_third_party_loggers()

        for name in ("chromadb", "httpx", "sentence_transformers"):
            assert logging.getLogger(name).level == logging.WARNING
