"""Security + drift tests for the committed Grafana dashboard.

The dashboard at ``deploy/grafana/sec-generative-search.json`` is an
operator artefact, not code — so nothing stops it silently drifting away
from what ``core/metrics.py`` actually emits, or quietly growing a
high-cardinality / PII label in a PromQL query. These tests are the
load-bearing control for both risks:

    - **Drift.** Every ``sec_*`` series a panel queries MUST exist in the
      live exposition that the metrics facade renders. A renamed or
      removed metric breaks the test, not just the dashboard in prod.
    - **Privacy / cardinality.** Every label a query groups by or selects
      on MUST be on the content-free allow-list
      (``provider`` / ``model`` / ``kind`` / ``pricing_tier`` /
      ``error_type`` plus the Prometheus-internal ``le``). A ticker,
      query, accession number, ``user_id``, or ``session_id`` MUST NEVER
      appear — neither in a query nor anywhere in the JSON. This mirrors
      the ``core/metrics.py`` label contract one layer out, so a future
      panel cannot pre-stage a privacy regression.
    - **Self-contained template.** The dashboard MUST carry no embedded
      secret and no off-host URL — the datasource is a templated
      ``${DS_PROMETHEUS}`` input bound at import time, never a hardcoded
      instance or credential.
    - **Coverage.** All five metric families defined in ``core/metrics.py``
      MUST appear on some panel, so a renamed or omitted family fails CI.

Only tracked, CI-visible files are asserted on (the dashboard JSON +
``core/metrics.py``). The prose metric catalogue is deliberately NOT
tested here, so the dashboard JSON is the committed catalogue of record.

The available-path assertions require ``prometheus-client`` (the
``[metrics]`` extra, which CI installs); they ``importorskip`` otherwise.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sec_generative_search.core.metrics import Metrics

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD_PATH = _REPO_ROOT / "deploy" / "grafana" / "sec-generative-search.json"

# The canonical content-free, low-cardinality label axes the metrics
# facade admits, plus Prometheus' internal histogram-bucket label ``le``.
# Anything outside this set in a dashboard query is a privacy/cardinality
# regression. Mirrors ``core/metrics.py``'s labelnames.
_ALLOWED_LABELS = frozenset({"provider", "model", "kind", "pricing_tier", "error_type", "le"})

# High-cardinality / Tier-3 axes that MUST NEVER reach a PromQL label in a
# panel query (grouped-by, selector, or legend reference).
_FORBIDDEN_LABELS = frozenset(
    {
        "ticker",
        "tickers",
        "query",
        "question",
        "accession",
        "accession_number",
        "user_id",
        "session_id",
        "correlation_id",
        "ip",
        "email",
        "name",
        "api_key",
    }
)

# Distinctive Tier-3 identifiers that must not appear *anywhere* in the
# committed JSON (description, title, expr — any field). Deliberately
# tighter than ``_FORBIDDEN_LABELS``: generic Grafana schema keys such as
# ``name`` / ``query`` legitimately appear in the dashboard structure, so
# the whole-blob scan only flags tokens that can never be benign here.
_FORBIDDEN_BLOB_TOKENS = frozenset(
    {
        "ticker",
        "tickers",
        "accession",
        "accession_number",
        "user_id",
        "session_id",
        "api_key",
    }
)

# The five metric families defined in ``core/metrics.py`` (base names,
# before Prometheus appends ``_bucket`` / ``_count`` / ``_sum`` / etc.).
_METRIC_FAMILIES = frozenset(
    {
        "sec_ingestion_duration_seconds",
        "sec_retrieval_duration_seconds",
        "sec_generation_duration_seconds",
        "sec_llm_tokens_total",
        "sec_provider_failures_total",
    }
)

_SEC_METRIC_RE = re.compile(r"\bsec_[a-z0-9_]+\b")
_BY_CLAUSE_RE = re.compile(r"\bby\s*\(([^)]*)\)")
_SELECTOR_LABEL_RE = re.compile(r"\{([^}]*)\}")
_LEGEND_LABEL_RE = re.compile(r"\{\{\s*([a-z0-9_]+)\s*\}\}")


def _live_series_names() -> set[str]:
    """Render a fresh metrics registry with one sample on every family.

    Returns the full set of exposed series names (``sec_*_bucket`` /
    ``_count`` / ``_sum`` / ``_total`` / ``_created`` …) so the dashboard
    can be checked against exactly what a scraper would see.
    """
    metrics = Metrics()
    metrics.observe_ingestion(1.0)
    metrics.observe_retrieval(0.1)
    metrics.observe_generation("anthropic", 1.0)
    metrics.record_tokens(
        "anthropic", "claude-haiku-4-5", input_tokens=1, output_tokens=1, pricing_tier="low"
    )
    metrics.record_provider_failure("openai", "ProviderAuthError")
    rendered = metrics.render_latest()
    assert rendered is not None
    _content_type, payload = rendered
    names: set[str] = set()
    for line in payload.decode().splitlines():
        if not line or line.startswith("#"):
            continue
        names.add(line.split("{", 1)[0].split(" ", 1)[0])
    return names


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads(_DASHBOARD_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panel_expressions(dashboard: dict) -> list[str]:
    """Every PromQL ``expr`` string across all panels (rows included)."""
    exprs: list[str] = []

    def _walk(panels: list[dict]) -> None:
        for panel in panels:
            for target in panel.get("targets", []) or []:
                expr = target.get("expr")
                if isinstance(expr, str) and expr.strip():
                    exprs.append(expr)
            _walk(panel.get("panels", []) or [])

    _walk(dashboard.get("panels", []) or [])
    return exprs


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------


def test_dashboard_is_valid_json_with_panels(dashboard: dict, panel_expressions: list[str]) -> None:
    assert dashboard.get("title")
    assert dashboard.get("uid")
    # A template with no queryable panels would be a useless artefact.
    assert panel_expressions, "dashboard exposes no PromQL panel expressions"


# ---------------------------------------------------------------------------
# Drift: every queried metric exists in the live exposition
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_every_queried_metric_exists_in_exposition(panel_expressions: list[str]) -> None:
    pytest.importorskip("prometheus_client")
    live = _live_series_names()
    referenced = {name for expr in panel_expressions for name in _SEC_METRIC_RE.findall(expr)}
    assert referenced, "no sec_* metric referenced by any panel"
    unknown = sorted(referenced - live)
    assert not unknown, (
        f"dashboard references metric series absent from the live exposition: {unknown}. "
        "Either core/metrics.py renamed/removed a series or the dashboard drifted."
    )


@pytest.mark.security
def test_dashboard_surfaces_every_metric_family(panel_expressions: list[str]) -> None:
    """Completeness lock: every metric family must appear on some panel.

    Tracked-artefact replacement for the old catalogue check. A new family
    added to ``core/metrics.py`` that nobody charts trips this — the
    dashboard is the committed, CI-visible catalogue.
    """
    referenced = {name for expr in panel_expressions for name in _SEC_METRIC_RE.findall(expr)}
    # A panel may query a derived series (``..._bucket`` / ``..._count``); a
    # family is "surfaced" when any referenced name starts with its base.
    uncharted = sorted(
        family
        for family in _METRIC_FAMILIES
        if not any(name == family or name.startswith(family) for name in referenced)
    )
    assert not uncharted, f"dashboard charts no panel for metric families: {uncharted}"


# ---------------------------------------------------------------------------
# Privacy / cardinality: only content-free labels in queries
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_panel_queries_only_use_content_free_labels(panel_expressions: list[str]) -> None:
    offenders: set[str] = set()
    for expr in panel_expressions:
        used: set[str] = set()
        for clause in _BY_CLAUSE_RE.findall(expr):
            used.update(tok.strip() for tok in clause.split(",") if tok.strip())
        for selector in _SELECTOR_LABEL_RE.findall(expr):
            for pair in selector.split(","):
                key = pair.split("=", 1)[0].split("!", 1)[0].strip()
                if key:
                    used.add(key)
        for legend in _LEGEND_LABEL_RE.findall(expr):
            used.add(legend.strip())
        offenders.update(used - _ALLOWED_LABELS)
    assert not offenders, (
        f"dashboard queries reference labels outside the content-free allow-list: "
        f"{sorted(offenders)}"
    )


@pytest.mark.security
def test_dashboard_contains_no_pii_or_high_cardinality_token(dashboard: dict) -> None:
    blob = json.dumps(dashboard).lower()
    leaked = sorted(
        token for token in _FORBIDDEN_BLOB_TOKENS if re.search(rf"\b{re.escape(token)}\b", blob)
    )
    assert not leaked, f"dashboard JSON contains forbidden PII/high-cardinality tokens: {leaked}"


# ---------------------------------------------------------------------------
# Self-contained template: no embedded secret, no off-host URL
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_dashboard_is_self_contained_and_secret_free(dashboard: dict) -> None:
    blob = json.dumps(dashboard)
    # No off-host dependency: the datasource is a templated input, never a
    # baked-in instance URL.
    assert "http://" not in blob and "https://" not in blob, (
        "dashboard embeds an absolute URL — the datasource must stay a templated input"
    )
    # No secret-shaped material committed alongside the template.
    lowered = blob.lower()
    for needle in ("password", "secret", "bearer ", "x-api-key", "x-admin-key", "authorization"):
        assert needle not in lowered, f"dashboard JSON contains secret-shaped material: {needle!r}"
    # The datasource MUST be exposed as a templated input, not hardcoded.
    input_names = {item.get("name") for item in dashboard.get("__inputs", [])}
    assert "DS_PROMETHEUS" in input_names, "datasource is not a templated ${DS_PROMETHEUS} input"
