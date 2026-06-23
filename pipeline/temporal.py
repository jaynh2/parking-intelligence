"""
Phase 3 — Temporal Congestion Analysis.

Operates per-partition (one police jurisdiction at a time). This is safe
because a hotspot's cluster_id is namespaced by jurisdiction
("<police_station>::<local_id>") — no cluster ever spans two partitions,
so the rush-hour baseline (median violations/hour within a cluster) is
always computed from a complete, correct view of that cluster's data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.storage import list_partitions, read_partition_columns


def compute_global_time_bounds(clean_data_root: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Cheap columnar scan (reads only `created_datetime`) across every
    partition to find the dataset-wide min/max timestamp, needed for the
    recency/growth calculations in Phase 6. Avoids materializing full
    row-level data just to find two numbers."""
    global_min: pd.Timestamp | None = None
    global_max: pd.Timestamp | None = None
    for partition_dir in list_partitions(clean_data_root):
        col = read_partition_columns(partition_dir, ["created_datetime"])
        if col.empty:
            continue
        ts = pd.to_datetime(col["created_datetime"], utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        local_min, local_max = ts.min(), ts.max()
        global_min = local_min if global_min is None else min(global_min, local_min)
        global_max = local_max if global_max is None else max(global_max, local_max)
    if global_min is None or global_max is None:
        raise ValueError("No valid timestamps found across any partition")
    return global_min, global_max


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.day_name()
    df["day_of_week_num"] = dt.dt.dayofweek
    df["is_weekend"] = dt.dt.dayofweek >= 5
    df["month"] = dt.dt.month
    df["date"] = dt.dt.date
    return df


def add_rush_hour_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Rule: a violation inside a real hotspot (not noise) during an hour
    whose volume exceeds that cluster's own median hourly volume is
    flagged "rush hour" (factor 1.5); everything else is baseline (1.0)."""
    df = df.copy()
    is_hotspot = df["cluster_id"] != "NOISE"

    hourly_counts = (
        df.loc[is_hotspot]
        .groupby(["cluster_id", "hour"])
        .size()
        .rename("violations_in_hour")
        .reset_index()
    )
    cluster_medians = (
        hourly_counts.groupby("cluster_id")["violations_in_hour"]
        .median()
        .rename("cluster_median_hourly")
        .reset_index()
    )

    df = df.merge(hourly_counts, on=["cluster_id", "hour"], how="left")
    df = df.merge(cluster_medians, on="cluster_id", how="left")

    df["rush_hour_factor"] = np.where(
        is_hotspot & (df["violations_in_hour"] > df["cluster_median_hourly"]),
        1.5,
        1.0,
    )
    return df.drop(columns=["violations_in_hour", "cluster_median_hourly"])
