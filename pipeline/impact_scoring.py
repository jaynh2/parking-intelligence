"""
Phase 4 — Congestion Impact Scoring.

Impact Score = Cluster_Density x Severity x Vehicle_Weight x Rush_Hour_Factor

All four factors are computed in earlier phases / vectorized here;
nothing in this module re-reads or re-validates raw data, keeping each
phase a pure transform over the columns it's contractually given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import Settings, get_settings


def add_severity(df: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or get_settings()
    df = df.copy()
    text = df["violation_type_clean"].astype("string").fillna("")

    severity = pd.Series(settings.default_severity, index=df.index, dtype="float64")
    for keyword, weight in settings.severity_mapping.items():
        match = text.str.contains(keyword, regex=False)
        severity = np.where(match, np.maximum(severity, weight), severity)
    df["severity"] = severity
    return df


def add_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """Requires: cluster_id, severity, vehicle_weight, rush_hour_factor."""
    df = df.copy()
    is_hotspot = df["cluster_id"] != "NOISE"

    cluster_density = (
        df.loc[is_hotspot].groupby("cluster_id").size().rename("cluster_density").reset_index()
    )
    df = df.merge(cluster_density, on="cluster_id", how="left")
    df["cluster_density"] = df["cluster_density"].fillna(1)

    df["impact_score"] = (
        df["cluster_density"] * df["severity"] * df["vehicle_weight"] * df["rush_hour_factor"]
    )
    return df


def summarize_hotspot_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Small, per-partition rollup (one row per cluster) — safe to
    concatenate across all partitions in memory, unlike the row-level
    data it's derived from."""
    hotspot_df = df[df["cluster_id"] != "NOISE"]
    if hotspot_df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id", "junction_name", "police_station",
                "center_latitude", "center_longitude",
                "total_violations", "total_impact_score",
                "average_impact_per_vehicle", "high_severity_incidents",
            ]
        )

    def _mode_or(series: pd.Series, default: str) -> str:
        m = series.mode()
        return m.iloc[0] if not m.empty else default

    summary = hotspot_df.groupby("cluster_id").apply(
        lambda g: pd.Series(
            {
                "junction_name": _mode_or(g["junction_name"], "Unknown"),
                "police_station": _mode_or(g["police_station"], "Unknown"),
                "center_latitude": g["latitude"].mean(),
                "center_longitude": g["longitude"].mean(),
                "total_violations": len(g),
                "total_impact_score": g["impact_score"].sum(),
                "average_impact_per_vehicle": g["impact_score"].mean(),
                "high_severity_incidents": int((g["severity"] >= 2.0).sum()),
            }
        ),
        include_groups=False,
    ).reset_index()
    return summary
