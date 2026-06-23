from __future__ import annotations

import pandas as pd
import pytest

from pipeline.enforcement_engine import (
    _status_and_recommendation,
    build_leaderboard,
    compute_cluster_recency_and_growth,
)
from schemas.api import DispatchStatus


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_recency_factor_is_one_when_incident_is_at_max_city_time(settings):
    max_time = _ts("2024-04-08 12:00:00")
    midpoint = _ts("2024-02-01 00:00:00")
    df = pd.DataFrame({
        "cluster_id": ["A::1"] * 3,
        "created_datetime": [_ts("2024-04-08 12:00:00"), _ts("2024-01-01 00:00:00"), _ts("2024-01-01 00:00:00")],
    })
    out = compute_cluster_recency_and_growth(df, max_time, midpoint, settings)
    row = out.loc[out["cluster_id"] == "A::1"].iloc[0]
    # latest incident == max_city_time -> hours_since_last = 0 -> recency_factor = 1/(1+0) = 1.0
    assert row["recency_factor"] == pytest.approx(1.0)


def test_recency_factor_decays_with_staleness(settings):
    max_time = _ts("2024-04-08 12:00:00")
    midpoint = _ts("2024-02-01 00:00:00")
    df = pd.DataFrame({
        "cluster_id": ["STALE::1"],
        "created_datetime": [_ts("2024-04-06 12:00:00")],  # 48 hours before max_city_time
    })
    out = compute_cluster_recency_and_growth(df, max_time, midpoint, settings)
    row = out.iloc[0]
    # hours_since_last=48 -> recency_factor = 1/(1+48/24) = 1/3
    assert row["recency_factor"] == pytest.approx(1.0 / 3.0)


def test_growth_rate_defaults_to_one_with_no_older_volume(settings):
    max_time = _ts("2024-04-08 12:00:00")
    midpoint = _ts("2024-02-01 00:00:00")
    df = pd.DataFrame({
        "cluster_id": ["NEW::1", "NEW::1"],
        "created_datetime": [_ts("2024-03-01 00:00:00"), _ts("2024-03-02 00:00:00")],  # both after midpoint
    })
    out = compute_cluster_recency_and_growth(df, max_time, midpoint, settings)
    assert out.iloc[0]["growth_rate"] == pytest.approx(1.0)


def test_growth_rate_clipped_to_configured_bounds(settings):
    max_time = _ts("2024-04-08 12:00:00")
    midpoint = _ts("2024-02-01 00:00:00")
    # 1 old incident, 20 recent incidents -> raw growth_rate = 20, must clip to growth_rate_clip_max
    df = pd.DataFrame({
        "cluster_id": ["BOOM::1"] * 21,
        "created_datetime": [_ts("2024-01-01 00:00:00")] + [_ts("2024-03-01 00:00:00")] * 20,
    })
    out = compute_cluster_recency_and_growth(df, max_time, midpoint, settings)
    assert out.iloc[0]["growth_rate"] == pytest.approx(settings.growth_rate_clip_max)


def test_empty_input_returns_empty_with_correct_columns(settings):
    df = pd.DataFrame({"cluster_id": ["NOISE", "NOISE"], "created_datetime": [_ts("2024-01-01"), _ts("2024-01-02")]})
    out = compute_cluster_recency_and_growth(df, _ts("2024-04-08"), _ts("2024-02-01"), settings)
    assert out.empty
    assert list(out.columns) == ["cluster_id", "latest_incident", "recency_factor", "growth_rate"]


@pytest.mark.parametrize(
    "growth_rate,recency_factor,expected_status",
    [
        (1.5, 0.5, DispatchStatus.URGENT_DISPATCH.value),   # growth above threshold wins regardless of recency
        (1.0, 0.95, DispatchStatus.HIGH_PRIORITY.value),    # recency above threshold, growth not urgent
        (1.0, 0.5, DispatchStatus.ROUTINE_PATROL.value),    # neither threshold met
    ],
)
def test_status_thresholds(settings, growth_rate, recency_factor, expected_status):
    row = pd.Series({"growth_rate": growth_rate, "recency_factor": recency_factor, "junction_name": "MG Road"})
    status, recommendation = _status_and_recommendation(row, settings)
    assert status == expected_status
    assert "MG Road" in recommendation


def test_build_leaderboard_ranks_by_priority_score_descending(settings):
    impact = pd.DataFrame({
        "cluster_id": ["LOW::1", "HIGH::1"],
        "junction_name": ["Low St", "High St"],
        "police_station": ["A", "B"],
        "center_latitude": [12.9, 12.8],
        "center_longitude": [77.6, 77.5],
        "total_violations": [10, 50],
        "total_impact_score": [100.0, 1000.0],
        "average_impact_per_vehicle": [10.0, 20.0],
        "high_severity_incidents": [1, 5],
    })
    recency_growth = pd.DataFrame({
        "cluster_id": ["LOW::1", "HIGH::1"],
        "latest_incident": [pd.Timestamp("2024-04-01", tz="UTC")] * 2,
        "recency_factor": [0.5, 0.9],
        "growth_rate": [1.0, 1.3],
    })
    board = build_leaderboard(impact, recency_growth, settings)
    assert board.iloc[0]["cluster_id"] == "HIGH::1"
    assert board.iloc[0]["rank"] == 1
    assert board.iloc[1]["rank"] == 2
    assert board.iloc[0]["priority_score"] > board.iloc[1]["priority_score"]
    assert board.iloc[0]["status"] == DispatchStatus.URGENT_DISPATCH.value


def test_build_leaderboard_priority_score_formula(settings):
    impact = pd.DataFrame({
        "cluster_id": ["X::1"], "junction_name": ["X"], "police_station": ["A"],
        "center_latitude": [12.9], "center_longitude": [77.6],
        "total_violations": [10], "total_impact_score": [200.0],
        "average_impact_per_vehicle": [20.0], "high_severity_incidents": [2],
    })
    recency_growth = pd.DataFrame({
        "cluster_id": ["X::1"], "latest_incident": [pd.Timestamp("2024-04-01", tz="UTC")],
        "recency_factor": [0.8], "growth_rate": [1.1],
    })
    board = build_leaderboard(impact, recency_growth, settings)
    assert board.iloc[0]["priority_score"] == pytest.approx(200.0 * 0.8 * 1.1)


def test_build_leaderboard_empty_input_does_not_raise(settings):
    empty_impact = pd.DataFrame(columns=["cluster_id", "total_impact_score"])
    empty_recency = pd.DataFrame(columns=["cluster_id", "recency_factor", "growth_rate"])
    board = build_leaderboard(empty_impact, empty_recency, settings)
    assert board.empty
