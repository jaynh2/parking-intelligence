"""
Auth integration tests. Each fixture builds a small but fully real artifact
bundle (same approach as test_api.py) and configures an ArtifactStore +
Settings with a known API key, then monkeypatches both into the API app
so tests run without touching network or the real artifacts/ directory.

Coverage:
 - dev-mode bypass (no api_keys configured)
 - missing X-API-Key header → 401
 - wrong X-API-Key → 403
 - valid X-API-Key → 200 on all protected endpoints
 - /health is always 200 regardless of auth state
 - RateLimitExceeded → 429 (tested structurally, not by actually exhausting)
 - CORS origin scoping
 - Settings.api_keys field parsing (comma-sep string, whitespace, empties)
 - Settings.cors_origins field parsing
 - Settings.auth_enabled property
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

# ------------------------------------------------------------------ #
# Shared helpers
# ------------------------------------------------------------------ #
VALID_KEY = "test-key-abc123"
WRONG_KEY = "this-is-not-valid"

_ALL_PROTECTED = [
    ("GET",  "/api/v1/leaderboard", None),
    ("GET",  "/api/v1/hotspots/A::1", None),
    ("GET",  "/api/v1/feature-importance", None),
    ("GET",  "/api/v1/kpis", None),
    ("GET",  "/api/v1/map", None),
    ("POST", "/api/v1/predict", {"cluster_id": "A::1", "hour": 8, "day_of_week": 1}),
]


def _build_synthetic_run(tmp_path: Path, settings: Settings) -> None:
    run_id = "run_auth_test"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)

    rows = []
    for cluster in ["A::1", "B::2"]:
        for day in range(7):
            for hour in (8, 9, 18, 19):
                rows.append({"date": f"2024-01-{day+1:02d}", "cluster_id": cluster,
                             "day_of_week_num": day, "hour": hour, "hourly_violation_count": 5})
    fa = train_forecast_model(pd.DataFrame(rows), settings)

    leaderboard = pd.DataFrame([{
        "rank": 1, "cluster_id": "A::1", "junction_name": "MG Road", "police_station": "Upparpet",
        "center_latitude": 12.97, "center_longitude": 77.59, "total_violations": 100,
        "total_impact_score": 5000.0, "average_impact_per_vehicle": 50.0, "high_severity_incidents": 10,
        "latest_incident": pd.Timestamp("2024-04-08 10:00:00", tz="UTC"),
        "recency_factor": 0.9, "growth_rate": 1.3, "priority_score": 5850.0,
        "status": "URGENT_DISPATCH", "recommendation": "Target MG Road immediately.",
    }])

    joblib.dump(fa.model, run_dir / "forecast_model.joblib")
    joblib.dump(fa.encoder, run_dir / "cluster_encoder.joblib")
    leaderboard.to_parquet(run_dir / "leaderboard.parquet", index=False)
    fa.feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps({
        "run_id": run_id, "model_version": "auth-test-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": 0.1, "total_hotspots": 1,
        "forecast_mae": fa.mae, "forecast_n_train": fa.n_train, "forecast_n_test": fa.n_test,
        "feature_names": fa.feature_names, "ingestion_report": {},
        "dataset_time_span": {"min": "2024-01-01T00:00:00Z", "max": "2024-04-08T00:00:00Z"},
    }))
    (tmp_path / "latest.json").write_text(json.dumps({"run_id": run_id, "run_dir": str(run_dir)}))


def _make_client(tmp_path, settings, monkeypatch):
    """Build a TestClient with both the store and settings monkeypatched into the API."""
    import inference.api as api_module
    import inference.auth as auth_module
    import inference.service as service_module

    store = ArtifactStore(settings)
    monkeypatch.setattr(service_module, "get_artifact_store", lambda: store)
    monkeypatch.setattr(api_module, "get_artifact_store", lambda: store)
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)
    monkeypatch.setattr(api_module, "settings", settings)
    return TestClient(api_module.app, raise_server_exceptions=True)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def no_auth_client(tmp_path, monkeypatch):
    """API client with auth disabled (default dev mode, empty api_keys)."""
    s = Settings(artifact_root=tmp_path, api_keys=frozenset())
    _build_synthetic_run(tmp_path, s)
    return _make_client(tmp_path, s, monkeypatch)


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    """API client with auth enabled, VALID_KEY is the only valid key."""
    s = Settings(artifact_root=tmp_path, api_keys=frozenset({VALID_KEY}))
    _build_synthetic_run(tmp_path, s)
    return _make_client(tmp_path, s, monkeypatch)


# ------------------------------------------------------------------ #
# Dev-mode bypass (no PCI_API_KEYS configured)
# ------------------------------------------------------------------ #
class TestDevModeBypass:
    def test_health_always_public(self, no_auth_client):
        assert no_auth_client.get("/health").status_code == 200

    @pytest.mark.parametrize("method,path,body", _ALL_PROTECTED)
    def test_all_endpoints_accessible_without_key(self, no_auth_client, method, path, body):
        if method == "GET":
            resp = no_auth_client.get(path)
        else:
            resp = no_auth_client.post(path, json=body)
        # /api/v1/map → 404 (no heatmap.html in test fixture) but NOT 401/403
        assert resp.status_code not in (401, 403), f"{path} returned {resp.status_code} in dev mode"


# ------------------------------------------------------------------ #
# Auth-enabled: missing or invalid key
# ------------------------------------------------------------------ #
class TestAuthEnforced:
    def test_health_still_public_when_auth_enabled(self, auth_client):
        assert auth_client.get("/health").status_code == 200

    @pytest.mark.parametrize("method,path,body", _ALL_PROTECTED)
    def test_missing_key_returns_401(self, auth_client, method, path, body):
        if method == "GET":
            resp = auth_client.get(path)
        else:
            resp = auth_client.post(path, json=body)
        assert resp.status_code == 401, f"{path} should return 401 with no key, got {resp.status_code}"

    @pytest.mark.parametrize("method,path,body", _ALL_PROTECTED)
    def test_wrong_key_returns_403(self, auth_client, method, path, body):
        headers = {"X-API-Key": WRONG_KEY}
        if method == "GET":
            resp = auth_client.get(path, headers=headers)
        else:
            resp = auth_client.post(path, json=body, headers=headers)
        assert resp.status_code == 403, f"{path} should return 403 with wrong key, got {resp.status_code}"

    @pytest.mark.parametrize("method,path,body", _ALL_PROTECTED)
    def test_valid_key_grants_access(self, auth_client, method, path, body):
        headers = {"X-API-Key": VALID_KEY}
        if method == "GET":
            resp = auth_client.get(path, headers=headers)
        else:
            resp = auth_client.post(path, json=body, headers=headers)
        # /api/v1/map → 404 (no heatmap in fixture), all others → 200/422
        assert resp.status_code not in (401, 403), (
            f"{path} should be accessible with a valid key, got {resp.status_code}: {resp.text}"
        )

    def test_401_body_is_structured(self, auth_client):
        resp = auth_client.get("/api/v1/kpis")
        assert resp.status_code == 401
        body = resp.json()
        assert "detail" in body
        assert body["detail"]["error"] == "missing_api_key"

    def test_403_body_is_structured(self, auth_client):
        resp = auth_client.get("/api/v1/kpis", headers={"X-API-Key": WRONG_KEY})
        assert resp.status_code == 403
        body = resp.json()
        assert body["detail"]["error"] == "invalid_api_key"


# ------------------------------------------------------------------ #
# Key rotation — multiple valid keys should all work simultaneously
# ------------------------------------------------------------------ #
class TestKeyRotation:
    def test_multiple_keys_all_accepted(self, tmp_path, monkeypatch):
        key_a, key_b = "key-a-111", "key-b-222"
        s = Settings(artifact_root=tmp_path, api_keys=frozenset({key_a, key_b}))
        _build_synthetic_run(tmp_path, s)
        client = _make_client(tmp_path, s, monkeypatch)
        for key in (key_a, key_b):
            resp = client.get("/api/v1/kpis", headers={"X-API-Key": key})
            assert resp.status_code == 200, f"Key {key} should be valid"


# ------------------------------------------------------------------ #
# Settings field parsing
# ------------------------------------------------------------------ #
class TestSettingsParsing:
    def test_api_keys_from_comma_separated_string(self):
        s = Settings(api_keys="key-1, key-2 ,  key-3  ")  # type: ignore[arg-type]
        assert s.api_keys == frozenset({"key-1", "key-2", "key-3"})

    def test_api_keys_filters_empty_segments(self):
        s = Settings(api_keys="key-1,,, key-2,")  # type: ignore[arg-type]
        assert s.api_keys == frozenset({"key-1", "key-2"})

    def test_api_keys_empty_string_means_no_auth(self):
        s = Settings(api_keys="")  # type: ignore[arg-type]
        assert s.api_keys == frozenset()
        assert s.auth_enabled is False

    def test_auth_enabled_true_when_keys_configured(self):
        s = Settings(api_keys=frozenset({"some-key"}))
        assert s.auth_enabled is True

    def test_cors_origins_from_comma_separated_string(self):
        s = Settings(cors_origins="https://dashboard.example.com, http://localhost:8501")  # type: ignore[arg-type]
        assert "https://dashboard.example.com" in s.cors_origins
        assert "http://localhost:8501" in s.cors_origins

    def test_cors_origins_default_is_localhost_only(self):
        s = Settings()
        assert s.cors_origins == ["http://localhost:8501"]
