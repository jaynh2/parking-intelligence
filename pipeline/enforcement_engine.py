"""
Phase 6 — Priority Enforcement Recommendation Engine.

Priority Score = Total Impact Score x Recency Factor x Growth Rate

Recency and growth both depend on a GLOBAL time reference (the
dataset-wide max/min timestamp from temporal.compute_global_time_bounds),
so this phase takes that reference as an explicit parameter rather than
recomputing it per partition — a single cheap upfront columnar scan
replaces what would otherwise require materializing the full dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import Settings, get_settings
from schemas.api import DispatchStatus

NOISE_LABEL = "NOISE"


def compute_cluster_recency_and_growth(
    df: pd.DataFrame,
    max_city_time: pd.Timestamp,
    midpoint_time: pd.Timestamp,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Per-partition rollup of (recency_factor, growth_rate, latest_incident)
    per cluster. One row per cluster — small, safe to concatenate."""
    settings = settings or get_settings()
    hotspot_df = df[df["cluster_id"] != NOISE_LABEL]
    if hotspot_df.empty:
        return pd.DataFrame(columns=["cluster_id", "latest_incident", "recency_factor", "growth_rate"])

    dt = pd.to_datetime(hotspot_df["created_datetime"], utc=True)
    hotspot_df = hotspot_df.assign(_dt=dt)

    rows = []
    for cluster_id, group in hotspot_df.groupby("cluster_id"):
        latest_incident = group["_dt"].max()
        hours_since_last = (max_city_time - latest_incident).total_seconds() / 3600.0
        recency_factor = 1.0 / (1.0 + (hours_since_last / 24.0))

        older_volume = int((group["_dt"] <= midpoint_time).sum())
        recent_volume = int((group["_dt"] > midpoint_time).sum())
        growth_rate = (recent_volume / older_volume) if older_volume > 0 else 1.0
        growth_rate = float(np.clip(growth_rate, settings.growth_rate_clip_min, settings.growth_rate_clip_max))

        rows.append(
            {
                "cluster_id": cluster_id,
                "latest_incident": latest_incident,
                "recency_factor": recency_factor,
                "growth_rate": growth_rate,
            }
        )
    return pd.DataFrame(rows)


def _status_and_recommendation(row: pd.Series, settings: Settings) -> tuple[str, str]:
    if row["growth_rate"] > settings.urgent_growth_threshold:
        pct = (row["growth_rate"] - 1) * 100
        return (
            DispatchStatus.URGENT_DISPATCH.value,
            f"Target {row['junction_name']}. Violations accelerating (+{pct:.0f}% growth trend).",
        )
    if row["recency_factor"] > settings.high_priority_recency_threshold:
        return (
            DispatchStatus.HIGH_PRIORITY.value,
            f"Monitor {row['junction_name']} closely — highly active within the past few hours.",
        )
    return (
        DispatchStatus.ROUTINE_PATROL.value,
        f"Schedule a routine check at {row['junction_name']} during predicted peak windows.",
    )


def build_leaderboard(
    hotspot_impact_summary: pd.DataFrame,
    recency_growth: pd.DataFrame,
    settings: Settings | None = None,
) -> pd.DataFrame:
    settings = settings or get_settings()
    engine_df = hotspot_impact_summary.merge(recency_growth, on="cluster_id", how="inner")
    if engine_df.empty:
        return engine_df.assign(priority_score=[], rank=[], status=[], recommendation=[])

    engine_df["priority_score"] = (
        engine_df["total_impact_score"] * engine_df["recency_factor"] * engine_df["growth_rate"]
    )
    engine_df = engine_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    engine_df.insert(0, "rank", engine_df.index + 1)

    results = engine_df.apply(lambda row: _status_and_recommendation(row, settings), axis=1)
    engine_df["status"] = [r[0] for r in results]
    engine_df["recommendation"] = [r[1] for r in results]
    return engine_df
