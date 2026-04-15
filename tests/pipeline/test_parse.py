"""Tests for :mod:`sec_generative_search.pipeline.parse` (Phase 4.3).

These tests avoid coupling to the exact doc2dict output format by
monkeypatching ``html2dict`` with controlled dict fixtures and by
exercising the parser's pure helpers (``_format_table``,
``_extract_segments``) directly.  This keeps the suite insensitive to
upstream doc2dict refactors while still covering the parser's own
invariants.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from sec_generative_search.core.exceptions import ParseError
from sec_generative_search.core.types import ContentType, FilingIdentifier
from sec_generative_search.pipeline import parse as parse_module
from sec_generative_search.pipeline.parse import FilingParser


@pytest.fixture
def filing_id() -> FilingIdentifier:
    return FilingIdentifier(
        ticker="AAPL",
        form_type="10-K",
        filing_date=date(2023, 11, 3),
        accession_number="0000320193-23-000077",
    )


def _patch_html2dict(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    """Replace doc2dict's ``html2dict`` with a fixture returning ``result``."""
    monkeypatch.setattr(parse_module, "html2dict", lambda _html: result)


class TestParseErrors:
    def test_empty_html_rejected(self, filing_id: FilingIdentifier) -> None:
        parser = FilingParser()
        with pytest.raises(ParseError, match="Empty HTML"):
            parser.parse("", filing_id)

    def test_whitespace_only_html_rejected(self, filing_id: FilingIdentifier) -> None:
        parser = FilingParser()
        with pytest.raises(ParseError, match="Empty HTML"):
            parser.parse("   \n\t  ", filing_id)

    def test_doc2dict_failure_wrapped(
        self, monkeypatch: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        def _explode(_html: str) -> Any:
            raise RuntimeError("bad html")

        monkeypatch.setattr(parse_module, "html2dict", _explode)
        parser = FilingParser()

        with pytest.raises(ParseError, match="Failed to parse HTML"):
            parser.parse("<html></html>", filing_id)

    def test_empty_parse_result_rejected(
        self, monkeypatch: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        _patch_html2dict(monkeypatch, {})
        parser = FilingParser()

        with pytest.raises(ParseError, match="empty result"):
            parser.parse("<html></html>", filing_id)

    def test_no_segments_extracted_rejected(
        self, monkeypatch: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # Parse tree has structure but no text/table content.
        _patch_html2dict(monkeypatch, {"document": {"header": {"title": "Filing"}}})
        parser = FilingParser()

        with pytest.raises(ParseError, match="No segments extracted"):
            parser.parse("<html></html>", filing_id)


class TestParseHappyPath:
    def test_hierarchical_paths_and_content_types(
        self, monkeypatch: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        tree: dict[str, Any] = {
            "document": {
                "section_0": {
                    "title": "Part I",
                    "contents": {
                        "item_1a": {
                            "title": "Item 1A. Risk Factors",
                            "text": "Our business is subject to many risks.",
                            "contents": {
                                "nested": {
                                    "title": "Regulatory",
                                    "textsmall": "See footnote 3.",
                                }
                            },
                        }
                    },
                },
                "section_1": {
                    "title": "Part II",
                    "contents": {
                        "item_7": {
                            "title": "Item 7. MD&A",
                            "table": {
                                "title": "Revenue by Segment",
                                "data": [
                                    ["Segment", "2022", "2023"],
                                    ["iPhone", "205", "200"],
                                ],
                            },
                        }
                    },
                },
            }
        }
        _patch_html2dict(monkeypatch, tree)

        segments = FilingParser().parse("<html></html>", filing_id)

        paths = [s.path for s in segments]
        assert "Part I > Item 1A. Risk Factors" in paths
        assert "Part I > Item 1A. Risk Factors > Regulatory" in paths
        assert "Part II > Item 7. MD&A" in paths

        content_types = {s.content_type for s in segments}
        assert content_types == {
            ContentType.TEXT,
            ContentType.TEXTSMALL,
            ContentType.TABLE,
        }

        # Every segment must carry the source filing identifier.
        assert all(s.filing_id is filing_id for s in segments)

    def test_empty_text_fields_skipped(
        self, monkeypatch: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        tree = {
            "document": {
                "s": {
                    "title": "Item 1",
                    "text": "   ",  # whitespace-only — skip
                    "textsmall": "Real footnote.",
                }
            }
        }
        _patch_html2dict(monkeypatch, tree)

        segments = FilingParser().parse("<html></html>", filing_id)

        assert len(segments) == 1
        assert segments[0].content_type is ContentType.TEXTSMALL
        assert segments[0].content == "Real footnote."


class TestFormatTable:
    def test_structured_dict_with_all_fields(self) -> None:
        parser = FilingParser()
        table = {
            "title": "Revenue",
            "preamble": "In millions USD",
            "data": [
                ["Q1", "100", "120"],
                ["Q2", "110", "130"],
            ],
            "footnotes": ["Excludes discontinued operations."],
            "postamble": "Source: 10-K",
        }
        result = parser._format_table(table)

        assert "Revenue" in result
        assert "In millions USD" in result
        assert "Q1 | 100 | 120" in result
        assert "Q2 | 110 | 130" in result
        assert "Excludes discontinued operations." in result
        assert "Source: 10-K" in result

    def test_simple_list_of_rows(self) -> None:
        parser = FilingParser()
        table = [
            ["Segment", "2022", "2023"],
            ["iPhone", "205", "200"],
        ]
        result = parser._format_table(table)

        assert result == "Segment | 2022 | 2023\niPhone | 205 | 200"

    def test_invalid_table_returns_empty_string(self) -> None:
        parser = FilingParser()
        assert parser._format_table(None) == ""
        assert parser._format_table("not a table") == ""
