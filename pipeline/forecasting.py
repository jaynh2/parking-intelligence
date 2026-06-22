"""
Phase 5 — Predictive Hotspot Forecasting.

XGBoost regressor predicting hourly violation volume per
(cluster, day-of-week, hour). The prototype fed raw cluster IDs into the
model as if they were ordinal integers — cluster "482" is not
"more" than cluster "12", and the original IDs were never stable across
reruns anyway (HDBSCAN reassigns label numbers each run). Production fix:
cluster_id is label-encoded once and the encoder is serialized alongside
the model, so inference always uses the exact mapping the model was
trained on, and an unseen cluster_id at inference time is a clear,
explicit validation error rather than silently wrong output.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config.settings import Settings, get_settings
from pipeline.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_NAMES = ["cluster_id_encoded", "day_of_week_num", "hour"]


def compute_hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-partition rollup: violation counts per (date, cluster, dow, hour).
    Small (bounded by #dates x #clusters x 24) — safe to concatenate
    across all partitions, unlike the row-level data it's derived from."""
    hotspot_df = df[df["cluster_id"] != "NOISE"]
    if hotspot_df.empty:
        return pd.DataFrame(columns=["date", "cluster_id", "day_of_week_num", "hour", "hourly_violation_count"])
    return (
        hotspot_df.groupby(["date", "cluster_id", "day_of_week_num", "hour"])
        .size()
        .rename("hourly_violation_count")
        .reset_index()
    )


@dataclass
class ForecastArtifact:
    model: xgb.XGBRegressor
    encoder: LabelEncoder
    mae: float
    n_train: int
    n_test: int
    feature_importance: pd.DataFrame
    feature_names: list[str]


def train_forecast_model(historical_trends: pd.DataFrame, settings: Settings | None = None) -> ForecastArtifact:
    settings = settings or get_settings()
    if historical_trends.empty or historical_trends["cluster_id"].nunique() < 2:
        raise ValueError("Not enough hotspot history to train a forecasting model")

    historical_trends = historical_trends.copy()
    encoder = LabelEncoder()
    historical_trends["cluster_id_encoded"] = encoder.fit_transform(historical_trends["cluster_id"])

    X = historical_trends[FEATURE_NAMES]
    y = historical_trends["hourly_violation_count"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.model_test_size, random_state=settings.model_random_state
    )

    logger.info("Phase 5: training XGBoost on %d rows (%d held out)", len(X_train), len(X_test))
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=settings.xgb_n_estimators,
        learning_rate=settings.xgb_learning_rate,
        max_depth=settings.xgb_max_depth,
        random_state=settings.model_random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, predictions))

    raw_importance = model.get_booster().get_score(importance_type="weight")
    # XGBoost only reports features it actually split on; backfill zeros so
    # every declared feature always appears in the artifact.
    importance_df = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": [raw_importance.get(f, 0.0) for f in FEATURE_NAMES]}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info("Phase 5 complete: MAE=%.2f vehicles/hour over %d held-out samples", mae, len(y_test))
    return ForecastArtifact(
        model=model,
        encoder=encoder,
        mae=mae,
        n_train=len(X_train),
        n_test=len(X_test),
        feature_importance=importance_df,
        feature_names=FEATURE_NAMES,
    )
