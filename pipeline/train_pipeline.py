"""
Training pipeline orchestrator — Phases 1 through 6, end to end.

This is the ONLY place that writes to the model registry (artifacts/).
The inference layer never imports anything from here except schemas — it only
reads serialized artifacts. That boundary means you can re-run this script on
a new data drop, on a schedule, in a separate container/job, without touching
or restarting the serving layer. The API picks up the new model on its next
cache refresh (see inference/artifact_store.py).

Usage:
    python -m pipeline.train_pipeline
    python -m pipeline.train_pipeline --csv-path data/raw/violations.csv
    python -m pipeline.train_pipeline --force-promote   # skip MAE gate
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from config.settings import get_settings
from pipeline import enforcement_engine, forecasting, geospatial, heatmap, impact_scoring, temporal
from pipeline.data_gateway import get_gateway
from pipeline.logging_config import configure_logging, get_logger
from pipeline.storage import ParquetPartitionWriter, list_partitions, partition_key, read_partition

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Training lineage
# ------------------------------------------------------------------ #
def _collect_lineage(csv_path: Path | None) -> dict:
    """Snapshot of everything that can affect model reproducibility:
    input data hash, library versions, git commit, Python version."""
    lineage: dict = {"python_version": sys.version.split()[0]}

    # SHA-256 of the input CSV — detects silent data changes between runs.
    if csv_path and Path(csv_path).exists():
        sha = hashlib.sha256()
        with open(csv_path, "rb") as f:
            for chunk in iter(lambda: f.read(65_536), b""):
                sha.update(chunk)
        lineage["input_csv_sha256"] = sha.hexdigest()
        lineage["input_csv_path"] = str(csv_path)
    else:
        lineage["input_csv_sha256"] = "unavailable"

    # Key library versions — a version bump here explains MAE changes.
    libs = ["pandas", "numpy", "xgboost", "sklearn", "hdbscan", "dask", "joblib", "pyarrow", "pydantic"]
    versions: dict[str, str] = {}
    for lib in libs:
        try:
            mod = importlib.import_module(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    lineage["library_versions"] = versions

    # Git commit — ties the run to exact source code.
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        lineage["git_commit"] = commit + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        lineage["git_commit"] = "unknown"

    return lineage


# ------------------------------------------------------------------ #
# Model promotion gate
# ------------------------------------------------------------------ #
def _read_previous_metrics(settings) -> dict | None:
    """Return metrics.json from the currently-promoted run, or None if this
    is the first run or the pointer is missing/corrupt."""
    if not settings.latest_pointer.exists():
        return None
    try:
        pointer = json.loads(settings.latest_pointer.read_text())
        prev = Path(pointer["run_dir"]) / "metrics.json"
        return json.loads(prev.read_text()) if prev.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _check_promotion_gate(
    new_mae: float, prev_metrics: dict | None, settings, force: bool = False
) -> tuple[bool, str]:
    """Return (should_promote, human_readable_reason).

    Blocks promotion when MAE regresses beyond settings.max_mae_regression_pct
    relative to the previous run. All artifacts are already written to disk;
    this only controls whether latest.json is updated.
    """
    if force:
        return True, "force-promote flag set — gate bypassed"

    if not settings.require_promotion_gate:
        return True, "promotion gate disabled (PCI_REQUIRE_PROMOTION_GATE=false)"

    if prev_metrics is None:
        return True, "no baseline found — first run, promoting unconditionally"

    prev_mae = prev_metrics.get("forecast_mae")
    if prev_mae is None:
        return True, "previous run has no forecast_mae — promoting unconditionally"

    ceiling = prev_mae * (1.0 + settings.max_mae_regression_pct / 100.0)
    if new_mae > ceiling:
        return False, (
            f"REJECTED — MAE regressed: {new_mae:.4f} > {ceiling:.4f} "
            f"(prev={prev_mae:.4f}, tolerance={settings.max_mae_regression_pct:.1f}%). "
            "Re-run with --force-promote to override."
        )

    delta_pct = (prev_mae - new_mae) / prev_mae * 100.0
    direction = f"+{delta_pct:.1f}% better" if delta_pct > 0 else f"{abs(delta_pct):.1f}% worse (within tolerance)"
    return True, f"passed (prev={prev_mae:.4f} → new={new_mae:.4f}, {direction})"


# ------------------------------------------------------------------ #
# Main pipeline
# ------------------------------------------------------------------ #
def run_pipeline(csv_path: str | None = None, force_promote: bool = False) -> Path:
    settings = get_settings()
    configure_logging(settings.log_level)

    if csv_path:
        settings.csv_path = Path(csv_path)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = settings.artifact_root / run_id
    clean_data_root = run_dir / "clean_data"
    assignments_root = run_dir / "cluster_assignments"
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 70)
    logger.info("STARTING TRAINING RUN %s", run_id)
    logger.info("=" * 70)

    # ---------------------------------------------------------------- #
    # Lineage snapshot — captured before we touch anything.
    # ---------------------------------------------------------------- #
    lineage = _collect_lineage(settings.csv_path)
    logger.info("Lineage: CSV SHA256=%s, git=%s", lineage.get("input_csv_sha256", "?")[:12], lineage.get("git_commit", "?")[:12])

    # ---------------------------------------------------------------- #
    # Read the previous run's metrics BEFORE training so the gate has a
    # baseline even if something fails partway through.
    # ---------------------------------------------------------------- #
    prev_metrics = _read_previous_metrics(settings)

    # ---------------------------------------------------------------- #
    # Ingestion
    # ---------------------------------------------------------------- #
    gateway = get_gateway(settings)
    with ParquetPartitionWriter(clean_data_root, settings.cluster_partition_column) as writer:
        for chunk in gateway.iter_clean_chunks():
            writer.write_chunk(chunk)
    ingestion_report = gateway.ingestion_report()
    logger.info(
        "Ingestion: %d/%d rows valid (%.1f%% rejected). Reasons: %s",
        ingestion_report.valid_rows, ingestion_report.total_rows_seen,
        ingestion_report.rejection_rate * 100, ingestion_report.rejection_reasons,
    )
    if not ingestion_report.passed_quality_gate:
        raise RuntimeError(
            f"Data quality gate failed: {ingestion_report.rejection_rate:.1%} rejected "
            f"(threshold {settings.max_row_rejection_rate:.1%}). Aborting."
        )

    # ---------------------------------------------------------------- #
    # Phase 1: partitioned, parallel HDBSCAN clustering.
    # ---------------------------------------------------------------- #
    geospatial.run_partitioned_clustering(clean_data_root, assignments_root, settings)

    # ---------------------------------------------------------------- #
    # Global time bounds for Phase 6 recency/growth calculation.
    # ---------------------------------------------------------------- #
    min_time, max_time = temporal.compute_global_time_bounds(clean_data_root)
    midpoint_time = min_time + (max_time - min_time) / 2
    logger.info("Dataset time span: %s -> %s", min_time, max_time)

    # ---------------------------------------------------------------- #
    # Phases 3+4+6: one partition resident in memory at a time.
    # ---------------------------------------------------------------- #
    hotspot_summaries, hourly_counts_parts, recency_growth_parts, heatmap_samples = [], [], [], []
    partitions = list_partitions(clean_data_root)
    per_partition_sample_budget = max(1, settings.heatmap_max_points // max(1, len(partitions)))

    for partition_dir in partitions:
        df = read_partition(partition_dir)
        if df.empty:
            continue

        assign_path = (
            assignments_root
            / f"{settings.cluster_partition_column}={partition_key(partition_dir)}"
            / "assignments.parquet"
        )
        assignments = pd.read_parquet(assign_path) if assign_path.exists() else pd.DataFrame(columns=["id", "cluster_id"])
        df = df.merge(assignments, on="id", how="left")
        df["cluster_id"] = df["cluster_id"].fillna(geospatial.NOISE_LABEL)

        df = temporal.add_temporal_features(df)
        df = temporal.add_rush_hour_factor(df)
        df = impact_scoring.add_severity(df, settings)
        df = impact_scoring.add_impact_score(df)

        hotspot_summaries.append(impact_scoring.summarize_hotspot_impact(df))
        hourly_counts_parts.append(forecasting.compute_hourly_counts(df))
        recency_growth_parts.append(
            enforcement_engine.compute_cluster_recency_and_growth(df, max_time, midpoint_time, settings)
        )
        sample = df[["latitude", "longitude"]]
        if len(sample) > per_partition_sample_budget:
            sample = sample.sample(per_partition_sample_budget, random_state=42)
        heatmap_samples.append(sample)

    hotspot_impact_summary = pd.concat(hotspot_summaries, ignore_index=True) if hotspot_summaries else pd.DataFrame()
    historical_trends = pd.concat(hourly_counts_parts, ignore_index=True) if hourly_counts_parts else pd.DataFrame()
    recency_growth = pd.concat(recency_growth_parts, ignore_index=True) if recency_growth_parts else pd.DataFrame()
    heatmap_sample = pd.concat(heatmap_samples, ignore_index=True) if heatmap_samples else pd.DataFrame()

    # ---------------------------------------------------------------- #
    # Phase 5: forecasting model.
    # ---------------------------------------------------------------- #
    forecast_artifact = forecasting.train_forecast_model(historical_trends, settings)

    # ---------------------------------------------------------------- #
    # Phase 6: priority leaderboard.
    # ---------------------------------------------------------------- #
    leaderboard = enforcement_engine.build_leaderboard(hotspot_impact_summary, recency_growth, settings)

    # ---------------------------------------------------------------- #
    # Phase 2: heatmap.
    # ---------------------------------------------------------------- #
    heatmap.generate_heatmap(heatmap_sample, leaderboard, run_dir / "heatmap.html", settings)

    # ---------------------------------------------------------------- #
    # Model promotion gate — evaluated before writing latest.json.
    # ---------------------------------------------------------------- #
    should_promote, gate_reason = _check_promotion_gate(
        forecast_artifact.mae, prev_metrics, settings, force=force_promote
    )
    if should_promote:
        logger.info("Promotion gate: %s", gate_reason)
    else:
        logger.warning("Promotion gate: %s", gate_reason)

    # ---------------------------------------------------------------- #
    # Serialize artifacts.
    # ---------------------------------------------------------------- #
    joblib.dump(forecast_artifact.model, run_dir / "forecast_model.joblib")
    joblib.dump(forecast_artifact.encoder, run_dir / "cluster_encoder.joblib")
    leaderboard.to_parquet(run_dir / "leaderboard.parquet", index=False)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)
    forecast_artifact.feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)

    metrics = {
        "run_id": run_id,
        "model_version": settings.model_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": round(time.time() - t0, 1),
        "ingestion_report": ingestion_report.model_dump(mode="json"),
        "total_hotspots": (
            int(hotspot_impact_summary["cluster_id"].nunique()) if not hotspot_impact_summary.empty else 0
        ),
        "forecast_mae": forecast_artifact.mae,
        "forecast_n_train": forecast_artifact.n_train,
        "forecast_n_test": forecast_artifact.n_test,
        "feature_names": forecast_artifact.feature_names,
        "dataset_time_span": {"min": min_time.isoformat(), "max": max_time.isoformat()},
        "promotion_gate": {"promoted": should_promote, "reason": gate_reason},
        "lineage": lineage,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Remove large intermediates. Comment these two lines out to retain full
    # partition-level audit data (clean_data + cluster_assignments).
    shutil.rmtree(clean_data_root, ignore_errors=True)
    shutil.rmtree(assignments_root, ignore_errors=True)

    # ---------------------------------------------------------------- #
    # Update latest.json — only if the gate passed.
    # ---------------------------------------------------------------- #
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    if should_promote:
        settings.latest_pointer.write_text(
            json.dumps({"run_id": run_id, "run_dir": str(run_dir)}, indent=2)
        )
        logger.info("=" * 70)
        logger.info(
            "RUN %s PROMOTED in %.1fs — %d hotspots, MAE=%.4f, artifacts at %s",
            run_id, time.time() - t0, metrics["total_hotspots"], forecast_artifact.mae, run_dir,
        )
    else:
        logger.warning("=" * 70)
        logger.warning(
            "RUN %s COMPLETE but NOT PROMOTED — gate rejected. "
            "Artifacts saved at %s for manual inspection. Re-run with --force-promote to override.",
            run_id, run_dir,
        )
    logger.info("=" * 70)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full parking-congestion training pipeline (Phases 1-6)."
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Override PCI_CSV_PATH for this run.")
    parser.add_argument(
        "--force-promote", action="store_true",
        help="Bypass the MAE regression gate and always promote the new run to latest.",
    )
    args = parser.parse_args()
    run_pipeline(csv_path=args.csv_path, force_promote=args.force_promote)


if __name__ == "__main__":
    main()
