"""
Partitioned Parquet storage helpers.

This is what makes the pipeline scalable: every phase after ingestion
operates on ONE partition (one police jurisdiction's worth of data) at a
time, never the whole municipal feed. Partitions are processed in
parallel via dask.delayed in geospatial.py / temporal.py. Adding more
data, more partitions, or more workers scales this horizontally without
touching the phase logic.

Layout on disk:
    <root>/<partition_col>=<safe_key>/part-0000.parquet
    <root>/<partition_col>=<safe_key>/part-0001.parquet
    ...
(Hive-style partitioning, identical convention to what Spark/Dask write,
so a real PySpark job could read this dataset directly with no changes.)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.logging_config import get_logger

logger = get_logger(__name__)


def _safe_key(value: object) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value)) or "unknown"


class ParquetPartitionWriter:
    """Appends DataFrame chunks to a Hive-partitioned Parquet dataset.
    Holds only the current chunk in memory — never the full dataset."""

    def __init__(self, root: Path, partition_col: str):
        self.root = Path(root)
        self.partition_col = partition_col
        self.root.mkdir(parents=True, exist_ok=True)
        self._part_counters: dict[str, int] = {}

    def write_chunk(self, chunk: pd.DataFrame) -> None:
        if chunk.empty:
            return
        for key, group in chunk.groupby(self.partition_col, dropna=False, observed=True):
            safe_key = _safe_key(key)
            part_dir = self.root / f"{self.partition_col}={safe_key}"
            part_dir.mkdir(parents=True, exist_ok=True)
            n = self._part_counters.get(safe_key, 0)
            out_path = part_dir / f"part-{n:04d}.parquet"
            group.to_parquet(out_path, index=False)
            self._part_counters[safe_key] = n + 1

    def __enter__(self) -> "ParquetPartitionWriter":
        return self

    def __exit__(self, *exc) -> None:
        return None


def list_partitions(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def partition_key(partition_dir: Path) -> str:
    """`police_station=Upparpet` -> `Upparpet`"""
    name = Path(partition_dir).name
    return name.split("=", 1)[1] if "=" in name else name


def read_partition(partition_dir: Path) -> pd.DataFrame:
    files = sorted(Path(partition_dir).glob("part-*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def read_partition_columns(partition_dir: Path, columns: list[str]) -> pd.DataFrame:
    """Reads only the requested columns from a partition — Parquet's
    columnar layout makes this cheap even when the partition has many
    other wide columns, useful for lightweight scans like global time
    bounds that don't need the full row."""
    files = sorted(Path(partition_dir).glob("part-*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns)
    return pd.concat((pd.read_parquet(f, columns=columns) for f in files), ignore_index=True)


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
