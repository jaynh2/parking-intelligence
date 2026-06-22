"""
Service layer — translates raw artifact-store output (pandas DataFrames,
a bare XGBoost model) into the validated Pydantic contracts in schemas/api.py.

Nothing here touches pandas/XGBoost specifics from the API layer's
perspective; inference/api.py only ever calls these functions and gets
back schema objects (or a typed exception) — keeps FastAPI handlers thin.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from config.settings import Settings, get_settings
from inference.artifact_store import ArtifactLoadError, ArtifactStore, get_artifact_store
from pipeline.forecasting import FEATURE_NAMES
from pipeline.logging_config import get_logger
from schemas.api import (
    FeatureImportance,
    ForecastRequest,
    ForecastResponse,
    HealthStatus,
    HotspotSummary,
    KPISummary,
    LeaderboardResponse,
)

logger = get_logger(__name__)


class UnknownClusterError(ValueError):
    """Raised when a forecast is requested for a cluster_id the model never saw at training time."""


def _row_to_hotspot(row: pd.Series) -> HotspotSummary:
    return HotspotSummary(
        cluster_id=row["cluster_id"],
        rank=int(row["rank"]),
        junction_name=row["junction_name"],
        police_station=row["police_station"],
        center_latitude=float(row["center_latitude"]),
        center_longitude=float(row["center_longitude"]),
        total_violations=int(row["total_violations"]),
        priority_score=float(row["priority_score"]),
        status=row["status"],
        recommendation=row["recommendation"],
    )


def get_leaderboard(limit: int | None = None, store: ArtifactStore | None = None) -> LeaderboardResponse:
    store = store or get_artifact_store()
    artifacts = store.get()
    board = artifacts.leaderboard.sort_values("rank")
    if limit is not None:
        board = board.head(limit)
    return LeaderboardResponse(
        generated_at=datetime.fromisoformat(artifacts.metrics["generated_at"]),
        model_version=artifacts.metrics["model_version"],
        total_hotspots=artifacts.metrics["total_hotspots"],
        hotspots=[_row_to_hotspot(row) for _, row in board.iterrows()],
    )


def get_hotspot(cluster_id: str, store: ArtifactStore | None = None) -> HotspotSummary:
    store = store or get_artifact_store()
    artifacts = store.get()
    match = artifacts.leaderboard.loc[artifacts.leaderboard["cluster_id"] == cluster_id]
    if match.empty:
        raise KeyError(f"Unknown cluster_id: {cluster_id!r}")
    return _row_to_hotspot(match.iloc[0])


def get_feature_importance(store: ArtifactStore | None = None) -> list[FeatureImportance]:
    store = store or get_artifact_store()
    artifacts = store.get()
    return [FeatureImportance(feature=r["feature"], importance=float(r["importance"]))
            for _, r in artifacts.feature_importance.iterrows()]


def predict(request: ForecastRequest, store: ArtifactStore | None = None) -> ForecastResponse:
    store = store or get_artifact_store()
    artifacts = store.get()
    encoder = artifacts.encoder

    if request.cluster_id not in set(encoder.classes_):
        raise UnknownClusterError(
            f"cluster_id {request.cluster_id!r} was not present in the training data for run "
            f"{artifacts.run_id}. Known clusters: {len(encoder.classes_)}. "
            "This is expected for brand-new hotspots until the next training run incorporates them."
        )

    cluster_id_encoded = int(encoder.transform([request.cluster_id])[0])
    feature_row = pd.DataFrame(
        [[cluster_id_encoded, request.day_of_week, request.hour]], columns=FEATURE_NAMES
    )
    prediction = float(artifacts.model.predict(feature_row)[0])
    prediction = max(0.0, prediction)  # violation counts can't be negative

    return ForecastResponse(
        cluster_id=request.cluster_id,
        hour=request.hour,
        day_of_week=request.day_of_week,
        predicted_violations=round(prediction, 2),
        model_mae=round(artifacts.metrics["forecast_mae"], 2),
        model_version=artifacts.metrics["model_version"],
    )


def get_kpis(store: ArtifactStore | None = None) -> KPISummary:
    store = store or get_artifact_store()
    artifacts = store.get()
    board = artifacts.leaderboard
    if board.empty:
        raise ArtifactLoadError("Leaderboard is empty; cannot compute KPIs.")

    top = board.sort_values("rank").iloc[0]
    settings: Settings = get_settings()
    primary_offense = max(settings.severity_mapping, key=settings.severity_mapping.get)

    return KPISummary(
        total_active_hotspots=int(artifacts.metrics["total_hotspots"]),
        top_priority_score=float(top["priority_score"]),
        top_priority_junction=top["junction_name"],
        model_mae=round(artifacts.metrics["forecast_mae"], 2),
        primary_offense=primary_offense,
        generated_at=datetime.fromisoformat(artifacts.metrics["generated_at"]),
        model_version=artifacts.metrics["model_version"],
    )


def health_check(store: ArtifactStore | None = None) -> HealthStatus:
    store = store or get_artifact_store()
    try:
        artifacts = store.get()
        return HealthStatus(
            status="ok",
            model_version=artifacts.metrics["model_version"],
            artifacts_loaded=True,
            artifact_generated_at=datetime.fromisoformat(artifacts.metrics["generated_at"]),
        )
    except ArtifactLoadError as exc:
        logger.warning("Health check: artifacts not ready: %s", exc)
        return HealthStatus(status="degraded", model_version=None, artifacts_loaded=False, artifact_generated_at=None)
