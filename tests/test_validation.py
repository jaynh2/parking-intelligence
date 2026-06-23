from __future__ import annotations

import pandas as pd
import pytest

from pipeline.validation import merge_reports, validate_and_clean_chunk
from schemas.domain import IngestionReport


def test_valid_rows_pass_through(raw_chunk):
    clean, report = validate_and_clean_chunk(raw_chunk)
    # row 1 ("12.9716"/"77.5946") and row 7 ("13.9999" lat -> out of geofence,
    # since lat_max default is 13.45) -- only row 1 and the well-formed ones survive.
    assert set(clean["id"]) <= set(raw_chunk["id"])
    assert report.total_rows_seen == len(raw_chunk)
    assert report.valid_rows == len(clean)
    assert report.rejected_rows == report.total_rows_seen - report.valid_rows


def test_missing_or_non_numeric_coordinates_rejected(raw_chunk):
    clean, report = validate_and_clean_chunk(raw_chunk)
    # row 4 (id="4") has latitude "abc"; row 5 (id="5") has latitude None
    assert "4" not in set(clean["id"])
    assert "5" not in set(clean["id"])
    assert report.rejection_reasons.get("missing_or_non_numeric_coordinates", 0) >= 2


def test_null_island_rejected(raw_chunk):
    clean, report = validate_and_clean_chunk(raw_chunk)
    # row 3 (id="3") has latitude "0.0"
    assert "3" not in set(clean["id"])
    assert report.rejection_reasons.get("null_island_coordinates", 0) >= 1


def test_outside_geofence_rejected(raw_chunk):
    clean, report = validate_and_clean_chunk(raw_chunk)
    # row 7 (id="7") has latitude "13.9999", outside the Bengaluru geofence
    assert "7" not in set(clean["id"])
    assert report.rejection_reasons.get("outside_service_geofence", 0) >= 1


def test_unparseable_timestamp_rejected():
    chunk = pd.DataFrame({
        "id": ["x1"], "latitude": ["12.97"], "longitude": ["77.59"],
        "vehicle_type": ["CAR"], "violation_type": ['["NO PARKING"]'],
        "junction_name": ["MG Road"], "police_station": ["Upparpet"],
        "created_datetime": ["not-a-real-date"],
    })
    clean, report = validate_and_clean_chunk(chunk)
    assert clean.empty
    assert report.rejection_reasons.get("unparseable_timestamp") == 1


def test_duplicate_id_keeps_first(raw_chunk):
    clean, report = validate_and_clean_chunk(raw_chunk)
    # row 1 and row 8 share id "1"; only one should survive
    assert (clean["id"] == "1").sum() == 1
    assert report.rejection_reasons.get("duplicate_id", 0) >= 1


def test_vehicle_type_normalized_and_weight_defaulted(raw_chunk):
    clean, _ = validate_and_clean_chunk(raw_chunk)
    row2 = clean.loc[clean["id"] == "2"].iloc[0]
    assert row2["vehicle_type"] == "SCOOTER"  # stripped + uppercased

    row6 = clean.loc[clean["id"] == "6"]
    if not row6.empty:
        # unmapped vehicle type falls back to default_vehicle_weight, never raises/drops
        assert row6.iloc[0]["vehicle_weight"] == 1


def test_violation_type_clean_parses_stringified_list(raw_chunk):
    clean, _ = validate_and_clean_chunk(raw_chunk)
    row2 = clean.loc[clean["id"] == "2"].iloc[0]
    assert row2["violation_type_clean"] == "WRONG PARKING, PARKING ON FOOTPATH"


def test_junction_and_station_nulls_get_safe_defaults(raw_chunk):
    clean, _ = validate_and_clean_chunk(raw_chunk)
    row2 = clean.loc[clean["id"] == "2"]
    if not row2.empty:
        assert row2.iloc[0]["junction_name"] == "No Junction"  # was None


def test_missing_required_column_raises():
    bad = pd.DataFrame({"id": ["1"], "latitude": ["12.9"]})  # missing most required cols
    with pytest.raises(ValueError, match="missing required columns"):
        validate_and_clean_chunk(bad)


def test_merge_reports_aggregates_counts_and_reasons():
    r1 = IngestionReport(total_rows_seen=10, valid_rows=8, rejected_rows=2,
                          rejection_rate=0.2, rejection_reasons={"null_island_coordinates": 2})
    r2 = IngestionReport(total_rows_seen=20, valid_rows=15, rejected_rows=5,
                          rejection_rate=0.25, rejection_reasons={"null_island_coordinates": 1, "duplicate_id": 4})
    merged = merge_reports([r1, r2])
    assert merged.total_rows_seen == 30
    assert merged.valid_rows == 23
    assert merged.rejected_rows == 7
    assert merged.rejection_reasons == {"null_island_coordinates": 3, "duplicate_id": 4}
    assert merged.rejection_rate == pytest.approx(7 / 30)


def test_quality_gate_threshold(settings):
    healthy = IngestionReport(total_rows_seen=100, valid_rows=90, rejected_rows=10, rejection_rate=0.10)
    unhealthy = IngestionReport(total_rows_seen=100, valid_rows=50, rejected_rows=50, rejection_rate=0.50)
    assert healthy.passed_quality_gate is True
    assert unhealthy.passed_quality_gate is False
