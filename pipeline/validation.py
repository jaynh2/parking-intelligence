"""
Batch validation & cleaning.

Row-by-row Pydantic validation of 300K+ records is correct but slow
(~50-100k rows/sec at best). For bulk ingestion we apply the *same rules*
defined in schemas.domain.ViolationRecord, but vectorized over a DataFrame
chunk so a 100M-row municipal feed stays tractable. schemas.domain stays
the single source of truth for the bounds (lat/lon/severity/etc.) — this
module never redefines a threshold, it only reads settings.

Every dropped row is accounted for in an IngestionReport so silent data
loss can never happen unnoticed.
"""
from __future__ import annotations

import ast
import math

import numpy as np
import pandas as pd

from config.settings import get_settings
from pipeline.logging_config import get_logger
from schemas.domain import IngestionReport

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "id",
    "latitude",
    "longitude",
    "vehicle_type",
    "violation_type",
    "junction_name",
    "police_station",
    "created_datetime",
]


def _safe_parse_violation(raw) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "UNKNOWN"
    text = str(raw).strip()
    if not text or text.upper() == "NULL":
        return "UNKNOWN"
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and parsed:
            return ", ".join(str(v).strip().upper() for v in parsed)
        return str(parsed).strip().upper()
    except (ValueError, SyntaxError):
        return text.upper()


def validate_and_clean_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, IngestionReport]:
    """Validate one DataFrame chunk. Returns (clean_df, report).
    Never raises on bad *data* — only on missing required *columns*,
    which indicates an upstream schema break that must fail loudly."""
    settings = get_settings()
    total = len(chunk)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in chunk.columns]
    if missing_cols:
        raise ValueError(f"Upstream feed is missing required columns: {missing_cols}")

    df = chunk.copy()
    reasons: dict[str, int] = {}

    def _track(mask: pd.Series, reason: str) -> None:
        n = int(mask.sum())
        if n:
            reasons[reason] = reasons.get(reason, 0) + n

    # --- coordinate sanity -------------------------------------------------
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    null_coords = df["latitude"].isna() | df["longitude"].isna()
    _track(null_coords, "missing_or_non_numeric_coordinates")

    finite = np.isfinite(df["latitude"].fillna(np.inf)) & np.isfinite(df["longitude"].fillna(np.inf))
    _track(~finite & ~null_coords, "non_finite_coordinates")

    zero_island = (df["latitude"] == 0.0) | (df["longitude"] == 0.0)
    _track(zero_island & ~null_coords, "null_island_coordinates")

    in_bounds = (
        df["latitude"].between(settings.lat_min, settings.lat_max)
        & df["longitude"].between(settings.lon_min, settings.lon_max)
    )
    _track(~in_bounds & ~null_coords & ~zero_island, "outside_service_geofence")

    keep_geo = null_coords.eq(False) & finite & ~zero_island & in_bounds

    # --- timestamp sanity ----------------------------------------------------
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    bad_ts = df["created_datetime"].isna()
    _track(bad_ts & keep_geo, "unparseable_timestamp")

    # --- duplicate id guard ----------------------------------------------
    dup = df["id"].duplicated(keep="first")
    _track(dup & keep_geo & ~bad_ts, "duplicate_id")

    keep = keep_geo & ~bad_ts & ~dup
    clean = df.loc[keep].copy()

    # --- normalize categorical / text fields (never drop on these) -------
    clean["vehicle_type"] = (
        clean["vehicle_type"].astype("string").str.strip().str.upper().replace({"NULL": pd.NA}).fillna("UNKNOWN")
    )
    clean["junction_name"] = (
        clean["junction_name"].astype("string").str.strip().replace({"NULL": pd.NA}).fillna("No Junction")
    )
    clean["junction_name"] = clean["junction_name"].replace("", "No Junction")
    clean["police_station"] = (
        clean["police_station"].astype("string").str.strip().replace({"NULL": pd.NA}).fillna("Unknown")
    )
    clean["violation_type_clean"] = clean["violation_type"].apply(_safe_parse_violation)

    vehicle_weights = settings.vehicle_weight_mapping
    clean["vehicle_weight"] = (
        clean["vehicle_type"].map(vehicle_weights).fillna(settings.default_vehicle_weight).astype(int)
    )

    report = IngestionReport(
        total_rows_seen=total,
        valid_rows=len(clean),
        rejected_rows=total - len(clean),
        rejection_rate=(total - len(clean)) / total if total else 0.0,
        rejection_reasons=reasons,
    )
    return clean, report


def merge_reports(reports: list[IngestionReport]) -> IngestionReport:
    total = sum(r.total_rows_seen for r in reports)
    valid = sum(r.valid_rows for r in reports)
    rejected = sum(r.rejected_rows for r in reports)
    merged_reasons: dict[str, int] = {}
    for r in reports:
        for k, v in r.rejection_reasons.items():
            merged_reasons[k] = merged_reasons.get(k, 0) + v
    return IngestionReport(
        total_rows_seen=total,
        valid_rows=valid,
        rejected_rows=rejected,
        rejection_rate=(rejected / total) if total else 0.0,
        rejection_reasons=merged_reasons,
    )
