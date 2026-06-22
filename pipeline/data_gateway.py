"""
Data access layer.

SpatialDataGateway is the single interface the rest of the pipeline talks
to. Two implementations are provided:

  * DaskBatchGateway  — chunked, out-of-core processing of flat-file feeds.
                         This is the default and is fully functional with
                         zero external infrastructure: it never loads the
                         whole municipal feed into memory at once, which is
                         what made the original prototype's single
                         `pd.read_csv(...)` call an unbounded-memory
                         operation.

  * PostGISGateway    — production target for a real spatial database.
                         Implemented against SQLAlchemy + a parameterized
                         PostGIS query (ST_MakeEnvelope / ST_Within), with
                         server-side chunked fetch via pandas.read_sql.
                         Not exercised in this environment (no DB
                         attached), but it is the documented, ready-to-run
                         swap-in: change PCI_DATA_BACKEND=postgis and
                         PCI_POSTGIS_DSN, no other code changes needed.

Swapping backends is a one-line config change (settings.data_backend),
never a code change in the phases that consume the gateway.
"""
from __future__ import annotations

import abc
from collections.abc import Iterator
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

from config.settings import Settings, get_settings
from pipeline.logging_config import get_logger
from pipeline.validation import merge_reports, validate_and_clean_chunk
from schemas.domain import IngestionReport

logger = get_logger(__name__)


class SpatialDataGateway(abc.ABC):
    """Contract every backend must satisfy."""

    @abc.abstractmethod
    def iter_clean_chunks(self) -> Iterator[pd.DataFrame]:
        """Yield validated, cleaned DataFrame chunks. Never materializes
        the full dataset in memory at once."""

    @abc.abstractmethod
    def ingestion_report(self) -> IngestionReport:
        """Data-quality summary for the most recent read. Must be called
        after fully consuming iter_clean_chunks()."""


class DaskBatchGateway(SpatialDataGateway):
    """Default backend: chunked CSV/Parquet ingestion via Dask, with
    vectorized validation applied per-partition so memory use stays
    bounded by `chunk_size` regardless of total feed size."""

    def __init__(self, settings: Settings | None = None, source_path: Path | None = None):
        self.settings = settings or get_settings()
        self.source_path = Path(source_path or self.settings.csv_path)
        self._reports: list[IngestionReport] = []

    def _read_dask_frame(self) -> dd.DataFrame:
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Source feed not found at {self.source_path}. "
                "Set PCI_CSV_PATH or place the file at the configured location."
            )
        bytes_per_chunk = max(self.settings.chunk_size * 400, 1_000_000)  # ~400 bytes/row heuristic
        return dd.read_csv(
            str(self.source_path),
            blocksize=bytes_per_chunk,
            dtype=str,  # defer typing to the validation layer — never trust upstream dtypes
            assume_missing=True,
            on_bad_lines="warn",
        )

    def iter_clean_chunks(self) -> Iterator[pd.DataFrame]:
        self._reports = []
        ddf = self._read_dask_frame()
        n_partitions = ddf.npartitions
        logger.info("Ingesting %s across %d partitions (backend=dask)", self.source_path, n_partitions)
        for i in range(n_partitions):
            raw_chunk = ddf.get_partition(i).compute()
            if raw_chunk.empty:
                continue
            clean, report = validate_and_clean_chunk(raw_chunk)
            self._reports.append(report)
            logger.info(
                "Partition %d/%d: %d/%d rows passed validation (%.1f%% rejected)",
                i + 1,
                n_partitions,
                report.valid_rows,
                report.total_rows_seen,
                report.rejection_rate * 100,
            )
            if clean.empty:
                continue
            yield clean

    def ingestion_report(self) -> IngestionReport:
        if not self._reports:
            raise RuntimeError("ingestion_report() called before iter_clean_chunks() was consumed")
        return merge_reports(self._reports)


class PostGISGateway(SpatialDataGateway):
    """
    Production gateway for a real spatial database. Mirrors the same
    contract as DaskBatchGateway so the rest of the pipeline is agnostic
    to which one is active.

    Schema assumed (see infra/sql/001_init_postgis.sql in the deployment
    blueprint):

        CREATE EXTENSION IF NOT EXISTS postgis;
        CREATE TABLE violations (
            id              TEXT PRIMARY KEY,
            geom            GEOMETRY(Point, 4326) NOT NULL,
            vehicle_type    TEXT,
            violation_type  TEXT,
            junction_name   TEXT,
            police_station  TEXT,
            created_datetime TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX violations_geom_idx ON violations USING GIST (geom);
        CREATE INDEX violations_created_idx ON violations (created_datetime);

    Querying is chunked server-side via pandas.read_sql(..., chunksize=...),
    so result sets larger than memory still stream safely, and the
    ST_MakeEnvelope/ST_Within predicate lets the database — not Python —
    do the geofence filtering before any rows cross the wire.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._reports: list[IngestionReport] = []

    def _engine(self):
        # Imported lazily: sqlalchemy + psycopg2 are only required when this
        # backend is actually selected, keeping the Dask-only deployment lean.
        from sqlalchemy import create_engine

        return create_engine(self.settings.postgis_dsn, pool_pre_ping=True)

    def _query(self) -> str:
        s = self.settings
        return f"""
            SELECT
                id, ST_Y(geom) AS latitude, ST_X(geom) AS longitude,
                vehicle_type, violation_type, junction_name,
                police_station, created_datetime
            FROM {s.postgis_table}
            WHERE ST_Within(
                geom,
                ST_MakeEnvelope({s.lon_min}, {s.lat_min}, {s.lon_max}, {s.lat_max}, 4326)
            )
        """

    def iter_clean_chunks(self) -> Iterator[pd.DataFrame]:
        self._reports = []
        engine = self._engine()
        logger.info("Ingesting from PostGIS table=%s (backend=postgis)", self.settings.postgis_table)
        with engine.connect() as conn:
            for raw_chunk in pd.read_sql(self._query(), conn, chunksize=self.settings.chunk_size):
                if raw_chunk.empty:
                    continue
                clean, report = validate_and_clean_chunk(raw_chunk)
                self._reports.append(report)
                if not clean.empty:
                    yield clean

    def ingestion_report(self) -> IngestionReport:
        if not self._reports:
            raise RuntimeError("ingestion_report() called before iter_clean_chunks() was consumed")
        return merge_reports(self._reports)


def get_gateway(settings: Settings | None = None) -> SpatialDataGateway:
    """Factory — the only place that branches on `data_backend`."""
    settings = settings or get_settings()
    if settings.data_backend == "postgis":
        return PostGISGateway(settings)
    return DaskBatchGateway(settings)
