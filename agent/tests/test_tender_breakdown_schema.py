import pytest

from pydantic import ValidationError

from pdf_agent.utils.llm.schemas import parse_tender_breakdown_report


def test_tender_breakdown_schema_valid():
    payload = {
        "report_type": "tender_breakdown",
        "title": "Tender breakdown",
        "sections": [
            {"title": "Scope", "bullets": ["Item 1"], "sources": ["file#page"]},
        ],
        "assistant_followup": "Next step",
    }
    report = parse_tender_breakdown_report(payload)
    assert report.report_type == "tender_breakdown"
    assert report.sections


def test_tender_breakdown_schema_requires_report_type():
    payload = {"title": "Missing type", "sections": []}
    with pytest.raises(ValidationError):
        parse_tender_breakdown_report(payload)


def test_tender_breakdown_schema_rejects_bad_sections():
    payload = {"report_type": "tender_breakdown", "sections": "oops"}
    with pytest.raises(ValidationError):
        parse_tender_breakdown_report(payload)
