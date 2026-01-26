from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class TenderBreakdownSection(BaseModel):
    title: Optional[str] = None
    bullets: List[str] = []
    sources: List[str] = []


class TenderBreakdownReport(BaseModel):
    report_type: Literal["tender_breakdown"]
    title: Optional[str] = None
    sections: Union[List[TenderBreakdownSection], Dict[str, Union[str, List[str]]]]
    assistant_followup: Optional[str] = None


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def parse_tender_breakdown_report(raw: object) -> TenderBreakdownReport:
    if hasattr(TenderBreakdownReport, "model_validate"):
        return TenderBreakdownReport.model_validate(raw)
    return TenderBreakdownReport.parse_obj(raw)
