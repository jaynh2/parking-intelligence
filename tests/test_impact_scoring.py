from __future__ import annotations

import pandas as pd
import pytest

from pipeline.impact_scoring import add_impact_score, add_severity, summarize_hotspot_impact


def test_add_severity_picks_highest_matching_keyword(settings):
    df = pd.DataFrame({"violation_type_clean": ["NO PARKING, WRONG PARKING", "WRONG PARKING", "UNRECOGNIZED TEXT"]})
    out = add_severity(df, settings)
    # row 0 matches both "NO PARKING" (1.5) and "WRONG PARKING" (1.0) -> max = 1.5
    assert out["severity"].iloc[0] == pytest.approx(1.5)
    assert out["severity"].iloc[1] == pytest.approx(1.0)
    # no keyword match -> falls back to default_severity
    assert out["severity"].iloc[2] == pytest.approx(settings.default_severity)


def test_add_severity_road_crossing_is_highest_weighted(settings):
    df = pd.DataFrame({"violation_type_clean": ["PARKING NEAR ROAD CROSSING"]})
    out = add_severity(df, settings)
    assert out["severity"].iloc[0] == pytest.approx(3.0)


def test_add_impact_score_noise_rows_get_density_one(hotspot_df):
    out = add_impact_score(hotspot_df)
    noise_rows = out[out["cluster_id"] == "NOISE"]
    assert (noise_rows["cluster_density"] == 1).all()


def test_add_impact_score_hotspot_density_matches_cluster_size(hotspot_df):
    out = add_impact_score(hotspot_df)
    cluster_a = out[out["cluster_id"] == "A::1"]
    # cluster A::1 appears 4 times in the fixture
    assert (cluster_a["cluster_density"] == 4).all()


def test_add_impact_score_formula(hotspot_df):
    out = add_impact_score(hotspot_df)
    row = out.iloc[0]  # cluster_id A::1, severity 1.5, vehicle_weight 2, rush_hour_factor 1.5, density 4
    expected = 4 * 1.5 * 2 * 1.5
    assert row["impact_score"] == pytest.approx(expected)


def test_summarize_hotspot_impact_excludes_noise(hotspot_df):
    scored = add_impact_score(hotspot_df)
    summary = summarize_hotspot_impact(scored)
    assert set(summary["cluster_id"]) == {"A::1", "B::2"}


def test_summarize_hotspot_impact_aggregates_correctly(hotspot_df):
    scored = add_impact_score(hotspot_df)
    summary = summarize_hotspot_impact(scored)
    cluster_a = summary.loc[summary["cluster_id"] == "A::1"].iloc[0]
    assert cluster_a["total_violations"] == 4
    assert cluster_a["junction_name"] == "MG Road"
    assert cluster_a["police_station"] == "Upparpet"
    assert cluster_a["total_impact_score"] == pytest.approx(scored.loc[scored["cluster_id"] == "A::1", "impact_score"].sum())
    # severity >= 2.0 in cluster A::1: only the 3.0 row -> 1 high-severity incident
    assert cluster_a["high_severity_incidents"] == 1


def test_summarize_hotspot_impact_empty_input_returns_correct_columns():
    empty = pd.DataFrame({
        "cluster_id": ["NOISE", "NOISE"], "junction_name": ["x", "y"], "police_station": ["a", "b"],
        "latitude": [12.9, 12.8], "longitude": [77.6, 77.5],
        "violation_type_clean": ["NO PARKING", "NO PARKING"], "severity": [1.5, 1.5],
        "vehicle_weight": [1, 1], "rush_hour_factor": [1.0, 1.0],
    })
    scored = add_impact_score(empty)
    summary = summarize_hotspot_impact(scored)
    assert summary.empty
    assert "total_impact_score" in summary.columns
