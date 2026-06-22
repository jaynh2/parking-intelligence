"""
Builds a tiny, *real* artifact bundle (genuinely trained via
pipeline.forecasting.train_forecast_model — not hand-faked predictions)
in a temp directory, points an ArtifactStore at it, and monkeypatches the
API/service layer's singleton lookups to use it. This exercises the same
serialize -> deserialize -> serve path the real training run uses, just
at a scale that runs in milliseconds.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from inference.artifact_store import ArtifactStore
from pipeline.forecasting import train_forecast_model


def _build_synthetic_run(tmp_path: Path, settings: Settings) -> str:
    run_id = "run_test"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)

    # Enough rows/clusters for train_test_split + LabelEncoder to behave like
    # the real pipeline, just tiny.
    rows = []
    for cluster in ["A::1", "B::2"]:
        for day in range(7):
            for hour in (8, 9, 18, 19):
                rows.append({"date": f"2024-01-{day+1:02d}", "cluster_id": cluster,
                             "day_of_week_num": day, "hour": hour,
                             "hourly_violation_count": 5 if hour in (8, 18) else 2})
    historical_trends = pd.DataFrame(rows)
    forecast_artifact = train_forecast_model(historical_trends, settings)

    leaderboard = pd.DataFrame([
        {"rank": 1, "cluster_id": "A::1", "junction_name": "MG Road", "police_station": "Upparpet",
         "center_latitude": 12.97, "center_longitude": 77.59, "total_violations": 120,
         "total_impact_score": 5000.0, "average_impact_per_vehicle": 41.6, "high_severity_incidents": 12,
         "latest_incident": pd.Timestamp("2024-04-08 10:00:00", tz="UTC"),
         "recency_factor": 0.9, "growth_rate": 1.4, "priority_score": 6300.0,
         "status": "URGENT_DISPATCH", "recommendation": "Target MG Road. Violations accelerating (+40% growth trend)."},
        {"rank": 2, "cluster_id": "B::2", "junction_name": "Silk Board", "police_station": "Whitefield",
         "center_latitude": 12.93, "center_longitude": 77.62, "total_violations": 80,
         "total_impact_score": 3000.0, "average_impact_per_vehicle": 37.5, "high_severity_incidents": 4,
         "latest_incident": pd.Timestamp("2024-04-07 22:00:00", tz="UTC"),
         "recency_factor": 0.6, "growth_rate": 0.9, "priority_score": 1620.0,
         "status": "ROUTINE_PATROL", "recommendation": "Schedule a routine check at Silk Board during predicted peak windows."},
    ])

    joblib.dump(forecast_artifact.model, run_dir / "forecast_model.joblib")
    joblib.dump(forecast_artifact.encoder, run_dir / "cluster_encoder.joblib")
    leaderboard.to_parquet(run_dir / "leaderboard.parquet", index=False)
    forecast_artifact.feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)

    metrics = {
        "run_id": run_id, "model_version": "test-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": 0.1,
        "ingestion_report": {"total_rows_seen": 100, "valid_rows": 100, "rejected_rows": 0,
                              "rejection_rate": 0.0, "rejection_reasons": {}},
        "total_hotspots": 2, "forecast_mae": forecast_artifact.mae,
        "forecast_n_train": forecast_artifact.n_train, "forecast_n_test": forecast_artifact.n_test,
        "feature_names": forecast_artifact.feature_names,
        "dataset_time_span": {"min": "2024-01-01T00:00:00+00:00", "max": "2024-04-08T00:00:00+00:00"},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))

    pointer = {"run_id": run_id, "run_dir": str(run_dir)}
    (tmp_path / "latest.json").write_text(json.dumps(pointer))
    return run_id


@pytest.fixture
def populated_store(tmp_path, settings) -> ArtifactStore:
    settings.artifact_root = tmp_path
    _build_synthetic_run(tmp_path, settings)
    return ArtifactStore(settings)


@pytest.fixture
def empty_store(tmp_path, settings) -> ArtifactStore:
    settings.artifact_root = tmp_path / "nothing_here"
    return ArtifactStore(settings)


@pytest.fixture
def client(populated_store, monkeypatch):
    import inference.api as api_module
    import inference.service as service_module

    monkeypatch.setattr(service_module, "get_artifact_store", lambda: populated_store)
    monkeypatch.setattr(api_module, "get_artifact_store", lambda: populated_store)
    return TestClient(api_module.app)


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["artifacts_loaded"] is True
    assert body["model_version"] == "test-v1"


def test_health_degraded_when_no_artifacts(empty_store, monkeypatch):
    import inference.api as api_module
    import inference.service as service_module

    monkeypatch.setattr(service_module, "get_artifact_store", lambda: empty_store)
    monkeypatch.setattr(api_module, "get_artifact_store", lambda: empty_store)
    client = TestClient(api_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["artifacts_loaded"] is False


def test_leaderboard_returns_ranked_hotspots(client):
    resp = client.get("/api/v1/leaderboard")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_hotspots"] == 2
    assert body["hotspots"][0]["cluster_id"] == "A::1"
    assert body["hotspots"][0]["rank"] == 1


def test_leaderboard_respects_limit(client):
    resp = client.get("/api/v1/leaderboard", params={"limit": 1})
    assert resp.status_code == 200
    assert len(resp.json()["hotspots"]) == 1


def test_leaderboard_limit_validation_rejects_zero(client):
    resp = client.get("/api/v1/leaderboard", params={"limit": 0})
    assert resp.status_code == 422


def test_hotspot_detail_found(client):
    resp = client.get("/api/v1/hotspots/A::1")
    assert resp.status_code == 200
    assert resp.json()["junction_name"] == "MG Road"


def test_hotspot_detail_not_found(client):
    resp = client.get("/api/v1/hotspots/NOT_REAL")
    assert resp.status_code == 404


def test_predict_known_cluster(client):
    resp = client.post("/api/v1/predict", json={"cluster_id": "A::1", "hour": 18, "day_of_week": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cluster_id"] == "A::1"
    assert body["predicted_violations"] >= 0


def test_predict_unknown_cluster_returns_422(client):
    resp = client.post("/api/v1/predict", json={"cluster_id": "GHOST::99", "hour": 18, "day_of_week": 2})
    assert resp.status_code == 422
    assert "GHOST::99" in resp.json()["detail"]


def test_predict_rejects_out_of_range_hour(client):
    resp = client.post("/api/v1/predict", json={"cluster_id": "A::1", "hour": 99, "day_of_week": 2})
    assert resp.status_code == 422


def test_feature_importance_lists_all_declared_features(client):
    resp = client.get("/api/v1/feature-importance")
    assert resp.status_code == 200
    features = {f["feature"] for f in resp.json()}
    assert features == {"cluster_id_encoded", "day_of_week_num", "hour"}


def test_kpis(client):
    resp = client.get("/api/v1/kpis")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_active_hotspots"] == 2
    assert body["top_priority_junction"] == "MG Road"


def test_map_404_when_heatmap_file_missing(client):
    # populated_store has no heatmap.html on disk (test fixture doesn't write one)
    resp = client.get("/api/v1/map")
    assert resp.status_code == 404


def test_endpoints_return_503_when_no_artifacts(empty_store, monkeypatch):
    import inference.api as api_module
    import inference.service as service_module

    monkeypatch.setattr(service_module, "get_artifact_store", lambda: empty_store)
    monkeypatch.setattr(api_module, "get_artifact_store", lambda: empty_store)
    client = TestClient(api_module.app)
    assert client.get("/api/v1/leaderboard").status_code == 503
    assert client.get("/api/v1/kpis").status_code == 503
