"""Shared fixtures. Tests never touch the real 298K-row CSV — every
dataframe here is small and hand-built so failures point at logic bugs,
not data quirks."""
from __future__ import annotations

import pandas as pd
import pytest

from config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """A fresh Settings instance (NOT the lru_cached get_settings()) so
    tests never leak state into each other via process-level caching."""
    return Settings()


@pytest.fixture
def raw_chunk() -> pd.DataFrame:
    """Mimics one chunk straight off the CSV: everything is a string
    (as dd.read_csv with dtype=str yields), including the columns that
    will become numeric/datetime after validation."""
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6", "7", "1"],
            "latitude": ["12.9716", "12.9352", "0.0", "abc", None, "12.9700", "13.9999", "12.9716"],
            "longitude": ["77.5946", "77.6245", "77.6000", "77.6000", "77.6000", "77.5900", "77.6000", "77.5946"],
            "vehicle_type": ["CAR", "scooter ", None, "BUS", "LORRY/GOODS VEHICLE", "TOTALLY_UNKNOWN_TYPE", "CAR", "CAR"],
            "violation_type": [
                '["NO PARKING"]', '["WRONG PARKING", "PARKING ON FOOTPATH"]', "NULL", '["NO PARKING"]',
                "['PARKING NEAR ROAD CROSSING']", "['WRONG PARKING']", '["NO PARKING"]', '["NO PARKING"]',
            ],
            "junction_name": ["MG Road", None, "NULL", "Silk Board", "  ", "Hebbal", "MG Road", "MG Road"],
            "police_station": ["Upparpet", "Upparpet", "Upparpet", "Whitefield", "Whitefield", "NULL", "Upparpet", "Upparpet"],
            "created_datetime": [
                "2024-01-15 08:30:00", "2024-01-15 09:15:00", "2024-01-15 10:00:00",
                "not-a-date", "2024-01-16 18:00:00", "2024-01-16 19:00:00",
                "2024-01-17 08:00:00", "2024-01-15 08:30:00",  # row 8 duplicates row 1's id
            ],
        }
    )


@pytest.fixture
def hotspot_df() -> pd.DataFrame:
    """A small post-Phase-3 dataframe: two real clusters plus noise,
    already carrying cluster_id, severity, vehicle_weight, rush_hour_factor
    — the contract add_impact_score() and summarize_hotspot_impact() expect."""
    return pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(8)],
            "cluster_id": ["A::1", "A::1", "A::1", "B::2", "B::2", "NOISE", "NOISE", "A::1"],
            "latitude": [12.97, 12.971, 12.969, 12.93, 12.931, 12.80, 12.81, 12.972],
            "longitude": [77.59, 77.591, 77.589, 77.62, 77.621, 77.50, 77.51, 77.592],
            "junction_name": ["MG Road"] * 3 + ["Silk Board"] * 2 + ["Elsewhere"] * 2 + ["MG Road"],
            "police_station": ["Upparpet"] * 3 + ["Whitefield"] * 2 + ["Other"] * 2 + ["Upparpet"],
            "violation_type_clean": ["NO PARKING", "WRONG PARKING", "PARKING NEAR ROAD CROSSING",
                                      "NO PARKING", "NO PARKING", "WRONG PARKING", "NO PARKING", "NO PARKING"],
            "severity": [1.5, 1.0, 3.0, 1.5, 1.5, 1.0, 1.5, 1.5],
            "vehicle_weight": [2, 1, 2, 4, 1, 1, 2, 2],
            "rush_hour_factor": [1.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.5],
        }
    )
