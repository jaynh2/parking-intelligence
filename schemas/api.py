"""API contract — request and response models for the inference service.
Kept separate from schemas/domain.py because the API shape (what a client
needs) is allowed to diverge from the internal training representation."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DispatchStatus(str, Enum):
    URGENT_DISPATCH = "URGENT_DISPATCH"
    HIGH_PRIORITY = "HIGH_PRIORITY"
    ROUTINE_PATROL = "ROUTINE_PATROL"


class HotspotSummary(BaseModel):
    cluster_id: str
    rank: int
    junction_name: str
    police_station: str
    center_latitude: float
    center_longitude: float
    total_violations: int
    priority_score: float
    status: DispatchStatus
    recommendation: str


class LeaderboardResponse(BaseModel):
    generated_at: datetime
    model_version: str
    total_hotspots: int
    hotspots: list[HotspotSummary]


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ForecastRequest(BaseModel):
    cluster_id: str
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6, description="0=Monday ... 6=Sunday")

    @field_validator("cluster_id")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("cluster_id must not be blank")
        return v


class ForecastResponse(BaseModel):
    cluster_id: str
    hour: int
    day_of_week: int
    predicted_violations: float
    model_mae: float
    model_version: str


class KPISummary(BaseModel):
    total_active_hotspots: int
    top_priority_score: float
    top_priority_junction: str
    model_mae: float
    primary_offense: str
    generated_at: datetime
    model_version: str


class HealthStatus(BaseModel):
    status: str
    model_version: str | None
    artifacts_loaded: bool
    artifact_generated_at: datetime | None
