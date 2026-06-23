"""
Domain schemas. ViolationRecord is the single source of truth for what a
"valid" parking violation row looks like — the ingestion layer, the
training pipeline, and the API all validate against this same contract,
so there is no drift between training-time and serving-time assumptions.
"""
from __future__ import annotations

import ast
import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from config.settings import get_settings


def _parse_violation_list(raw: Any) -> list[str]:
    """The source system stores violation_type as a stringified Python list,
    e.g. '["WRONG PARKING","NO PARKING"]'. Production feeds are messy:
    sometimes it's NaN, sometimes malformed, sometimes already a list.
    Never raise — always degrade to a safe sentinel."""
    if raw is None:
        return ["UNKNOWN"]
    if isinstance(raw, list):
        return [str(v).strip().upper() for v in raw] or ["UNKNOWN"]
    if isinstance(raw, float) and math.isnan(raw):
        return ["UNKNOWN"]
    text = str(raw).strip()
    if not text or text.upper() == "NULL":
        return ["UNKNOWN"]
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and parsed:
            return [str(v).strip().upper() for v in parsed]
        return [str(parsed).strip().upper()]
    except (ValueError, SyntaxError):
        return [text.upper()]


class ViolationRecord(BaseModel):
    """Validated representation of one parking-violation incident."""

    id: str
    latitude: float
    longitude: float
    vehicle_type: str = "UNKNOWN"
    violation_type: list[str] = Field(default_factory=lambda: ["UNKNOWN"])
    junction_name: str = "No Junction"
    police_station: str = "Unknown"
    created_datetime: datetime

    @field_validator("vehicle_type", mode="before")
    @classmethod
    def _normalize_vehicle_type(cls, v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "UNKNOWN"
        s = str(v).strip().upper()
        return s if s and s != "NULL" else "UNKNOWN"

    @field_validator("violation_type", mode="before")
    @classmethod
    def _normalize_violation_type(cls, v: Any) -> list[str]:
        return _parse_violation_list(v)

    @field_validator("junction_name", mode="before")
    @classmethod
    def _normalize_junction(cls, v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "No Junction"
        s = str(v).strip()
        return s if s and s.upper() != "NULL" else "No Junction"

    @field_validator("police_station", mode="before")
    @classmethod
    def _normalize_station(cls, v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "Unknown"
        s = str(v).strip()
        return s if s else "Unknown"

    @model_validator(mode="after")
    def _validate_geofence(self) -> "ViolationRecord":
        settings = get_settings()
        if not math.isfinite(self.latitude) or not math.isfinite(self.longitude):
            raise ValueError("non-finite coordinates")
        if self.latitude == 0.0 or self.longitude == 0.0:
            raise ValueError("null-island coordinates (0,0)")
        if not (settings.lat_min <= self.latitude <= settings.lat_max):
            raise ValueError(f"latitude {self.latitude} outside service geofence")
        if not (settings.lon_min <= self.longitude <= settings.lon_max):
            raise ValueError(f"longitude {self.longitude} outside service geofence")
        return self


class RejectedRecord(BaseModel):
    """Captures *why* a row was dropped, instead of silently discarding it.
    Feeds the data-quality report so bad upstream feeds are caught early
    rather than silently degrading model quality."""

    row_index: int
    reason: str
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class IngestionReport(BaseModel):
    """Summary returned by every ingestion run — used both as a CLI log
    and as a machine-readable artifact for CI data-quality gates."""

    total_rows_seen: int
    valid_rows: int
    rejected_rows: int
    rejection_rate: float
    rejection_reasons: dict[str, int] = Field(default_factory=dict)

    @property
    def passed_quality_gate(self) -> bool:
        settings = get_settings()
        return self.rejection_rate <= settings.max_row_rejection_rate
