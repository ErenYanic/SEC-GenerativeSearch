"""Tests for logging configuration."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from sec_generative_search.core import logging as sgs_logging
from sec_generative_search.core.correlation import bind_correlation_id
from sec_generative_search.core.logging import (
    LOGGER_NAME,
    CorrelationIdFilter,
    JsonFormatter,
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


def _make_record(msg: str = "hello %s", *args: object) -> logging.LogRecord:
    return logging.LogRecord(
        name="sec_generative_search.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


class TestCorrelationIdFilter:
    def test_injects_dash_without_scope(self) -> None:
        record = _make_record()
        assert CorrelationIdFilter().filter(record) is True
        assert record.correlation_id == "-"

    def test_injects_bound_id(self) -> None:
        record = _make_record()
        with bind_correlation_id("cid-12345678"):
            CorrelationIdFilter().filter(record)
        assert record.correlation_id == "cid-12345678"


class TestJsonFormatter:
    def test_emits_fixed_field_set(self) -> None:
        record = _make_record("processed %s filings", 3)
        CorrelationIdFilter().filter(record)
        line = JsonFormatter().format(record)
        payload = json.loads(line)
        assert payload["level"] == "INFO"
        assert payload["logger"] == "sec_generative_search.test"
        assert payload["message"] == "processed 3 filings"
        assert payload["correlation_id"] == "-"
        assert set(payload) == {"ts", "level", "logger", "correlation_id", "message"}

    def test_carries_bound_correlation_id(self) -> None:
        record = _make_record("x")
        with bind_correlation_id("req-abcdef12"):
            CorrelationIdFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
        assert payload["correlation_id"] == "req-abcdef12"

    @pytest.mark.security
    def test_does_not_serialise_arbitrary_extra(self) -> None:
        # A stray ``extra`` must never reach the JSON stream — it could
        # smuggle a ticker / query / secret into the aggregator.
        record = _make_record("x")
        record.ticker = "AAPL"  # type: ignore[attr-defined]
        record.api_key = "sk-secret"  # type: ignore[attr-defined]
        line = JsonFormatter().format(record)
        assert "AAPL" not in line
        assert "sk-secret" not in line

    def test_includes_exception_text_when_present(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="sec_generative_search.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError" in payload["exc"]


class TestLogFormatSelection:
    def test_json_format_uses_json_formatter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_FORMAT", "json")
        configure_logging(level=logging.INFO, use_rich=False)
        handlers = logging.getLogger(LOGGER_NAME).handlers
        assert any(isinstance(h.formatter, JsonFormatter) for h in handlers)

    def test_console_format_does_not_use_json_formatter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_FORMAT", "console")
        configure_logging(level=logging.INFO, use_rich=False)
        handlers = logging.getLogger(LOGGER_NAME).handlers
        assert not any(isinstance(h.formatter, JsonFormatter) for h in handlers)

    def test_unknown_format_falls_back_to_console(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_FORMAT", "yaml-nonsense")
        configure_logging(level=logging.INFO, use_rich=False)
        handlers = logging.getLogger(LOGGER_NAME).handlers
        assert not any(isinstance(h.formatter, JsonFormatter) for h in handlers)

    def test_every_handler_carries_correlation_filter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LOG_FILE_PATH", str(tmp_path / "run.log"))
        configure_logging(level=logging.INFO, use_rich=False)
        handlers = logging.getLogger(LOGGER_NAME).handlers
        assert handlers
        for handler in handlers:
            assert any(isinstance(f, CorrelationIdFilter) for f in handler.filters)


class TestSuppressThirdPartyLoggers:
    def test_sets_noisy_loggers_to_warning(self) -> None:
        # Flip them to DEBUG first to verify the call actually changes them.
        for name in ("chromadb", "httpx", "sentence_transformers"):
            logging.getLogger(name).setLevel(logging.DEBUG)

        suppress_third_party_loggers()

        for name in ("chromadb", "httpx", "sentence_transformers"):
            assert logging.getLogger(name).level == logging.WARNING
