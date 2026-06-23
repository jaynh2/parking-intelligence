"""
Phase 1 — Geospatial Hotspot Detection.

The prototype ran HDBSCAN(metric='haversine') on the entire city-wide
dataset in one process. That is an effectively-unusable operation as the
feed grows — even BallTree-accelerated haversine HDBSCAN took ~10s for a
single 34K-row police jurisdiction in benchmarking; a naive single-pass
run over a 10x larger municipal feed does not scale.

Production fix: spatially partition by policing jurisdiction (already a
natural, non-overlapping geographic partition in the source data) and run
HDBSCAN independently per partition. Partitions are embarrassingly
parallel — dask.delayed fans them out across however many workers are
available (1 thread in this sandbox, N workers/processes in a real
deployment, with zero code changes).

Cluster assignments are written straight to a per-partition Parquet file
rather than concatenated into one global in-memory table, keeping memory
bounded by a single partition's size at every step of the pipeline —
including this one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import dask
import hdbscan
import numpy as np
import pandas as pd

from config.settings import Settings, get_settings
from pipeline.logging_config import get_logger
from pipeline.storage import list_partitions, partition_key, read_partition

logger = get_logger(__name__)

NOISE_LABEL = "NOISE"


@dataclass
class PartitionClusterStats:
    police_station: str
    n_points: int
    n_clusters: int
    n_noise: int


def _cluster_one_partition(partition_dir: Path, assignments_root: Path, settings: Settings) -> PartitionClusterStats:
    station = partition_key(partition_dir)
    df = read_partition(partition_dir)
    out_dir = Path(assignments_root) / f"{settings.cluster_partition_column}={station}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "assignments.parquet"

    if df.empty:
        pd.DataFrame(columns=["id", "cluster_id"]).to_parquet(out_path, index=False)
        return PartitionClusterStats(station, 0, 0, 0)

    if len(df) < settings.hdbscan_min_cluster_size:
        labels = pd.DataFrame({"id": df["id"], "cluster_id": NOISE_LABEL})
        labels.to_parquet(out_path, index=False)
        logger.info("Partition %s: %d rows < min_cluster_size, all noise", station, len(df))
        return PartitionClusterStats(station, len(df), 0, len(df))

    # Validation upstream stores lat/lon as pandas nullable "Float64" (capital
    # F) so we can distinguish missing from zero. That extension dtype boxes
    # values as Python floats under .to_numpy(), which breaks numpy ufuncs
    # like radians() — cast down to a plain numpy float64 array first.
    coords = np.radians(df[["latitude", "longitude"]].astype("float64").to_numpy())
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=settings.hdbscan_min_samples,
        metric="haversine",
    )
    local_labels = clusterer.fit_predict(coords)
    global_labels = np.where(
        local_labels == -1,
        NOISE_LABEL,
        [f"{station}::{lbl}" for lbl in local_labels],
    )

    pd.DataFrame({"id": df["id"].values, "cluster_id": global_labels}).to_parquet(out_path, index=False)

    n_clusters = len(set(global_labels) - {NOISE_LABEL})
    n_noise = int((global_labels == NOISE_LABEL).sum())
    logger.info("Partition %s: %d rows -> %d clusters (%d noise)", station, len(df), n_clusters, n_noise)
    return PartitionClusterStats(station, len(df), n_clusters, n_noise)


def run_partitioned_clustering(
    clean_data_root: Path, assignments_root: Path, settings: Settings | None = None
) -> list[PartitionClusterStats]:
    """Runs Phase 1 across every jurisdiction partition in parallel.
    Writes (id -> cluster_id) assignments directly to disk per partition;
    returns only small per-partition stats for logging/reporting."""
    settings = settings or get_settings()
    partitions = list_partitions(clean_data_root)
    if not partitions:
        raise FileNotFoundError(f"No partitions found under {clean_data_root}; did ingestion run?")

    logger.info("Phase 1: clustering %d jurisdiction partitions", len(partitions))
    tasks = [dask.delayed(_cluster_one_partition)(p, assignments_root, settings) for p in partitions]
    stats: list[PartitionClusterStats] = list(dask.compute(*tasks, scheduler="threads"))

    total_hotspots = sum(s.n_clusters for s in stats)
    total_noise = sum(s.n_noise for s in stats)
    total_points = sum(s.n_points for s in stats)
    logger.info(
        "Phase 1 complete: %d hotspots discovered across %d partitions, %d/%d points isolated as noise",
        total_hotspots, len(partitions), total_noise, total_points,
    )
    return stats
