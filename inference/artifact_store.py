"""
Artifact store — the ONLY bridge between the offline training pipeline
and the online serving layer. It never imports anything from pipeline/;
it only reads files that pipeline/train_pipeline.py already wrote.

Loads are cached in-process and refreshed on a TTL (settings.cache_ttl_seconds)
so a new training run lands automatically on the next refresh window, with
zero downtime and zero coupling to how/where training ran (cron, separate
container, CI job — doesn't matter, it only has to update artifacts/latest.json).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from config.settings import Settings, get_settings
from pipeline.logging_config import get_logger

logger = get_logger(__name__)


class ArtifactLoadError(RuntimeError):
    """Raised when the model registry has no usable artifacts yet."""


@dataclass
class LoadedArtifacts:
    run_id: str
    run_dir: Path
    model: object
    encoder: object
    leaderboard: pd.DataFrame
    feature_importance: pd.DataFrame
    metrics: dict
    loaded_at: float


class ArtifactStore:
    """Thread-safe, TTL-cached singleton view of the latest training run."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._lock = threading.Lock()
        self._cached: LoadedArtifacts | None = None

    def _read_pointer(self) -> dict:
        pointer_path = self._settings.latest_pointer
        if not pointer_path.exists():
            raise ArtifactLoadError(
                f"No model registry pointer at {pointer_path}. "
                "Run `python -m pipeline.train_pipeline` at least once before starting the API."
            )
        try:
            return json.loads(pointer_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactLoadError(f"Corrupt registry pointer at {pointer_path}: {exc}") from exc

    def _load_from_disk(self) -> LoadedArtifacts:
        pointer = self._read_pointer()
        run_id = pointer["run_id"]
        run_dir = Path(pointer["run_dir"])

        required = ["forecast_model.joblib", "cluster_encoder.joblib", "leaderboard.parquet",
                    "feature_importance.csv", "metrics.json"]
        missing = [f for f in required if not (run_dir / f).exists()]
        if missing:
            raise ArtifactLoadError(f"Run {run_id} at {run_dir} is missing artifacts: {missing}")

        try:
            model = joblib.load(run_dir / "forecast_model.joblib")
            encoder = joblib.load(run_dir / "cluster_encoder.joblib")
            leaderboard = pd.read_parquet(run_dir / "leaderboard.parquet")
            feature_importance = pd.read_csv(run_dir / "feature_importance.csv")
            metrics = json.loads((run_dir / "metrics.json").read_text())
        except Exception as exc:  # noqa: BLE001 — surfacing as a typed load error is the point
            raise ArtifactLoadError(f"Failed to deserialize artifacts for run {run_id}: {exc}") from exc

        logger.info("Loaded artifacts for run %s (%d hotspots)", run_id, len(leaderboard))
        return LoadedArtifacts(
            run_id=run_id, run_dir=run_dir, model=model, encoder=encoder,
            leaderboard=leaderboard, feature_importance=feature_importance,
            metrics=metrics, loaded_at=time.time(),
        )

    def get(self, force_refresh: bool = False) -> LoadedArtifacts:
        """Returns the cached artifacts, transparently reloading if the TTL
        has expired or a refresh is forced. Falls back to a stale cache (with
        a warning) if a refresh attempt fails but a previous load succeeded —
        a temporarily broken pointer shouldn't take a healthy API down."""
        with self._lock:
            stale = (
                self._cached is None
                or force_refresh
                or (time.time() - self._cached.loaded_at) > self._settings.cache_ttl_seconds
            )
            if not stale:
                return self._cached

            try:
                self._cached = self._load_from_disk()
            except ArtifactLoadError:
                if self._cached is not None:
                    logger.warning("Artifact refresh failed; continuing to serve stale run %s", self._cached.run_id)
                    return self._cached
                raise
            return self._cached

    def is_ready(self) -> bool:
        try:
            self.get()
            return True
        except ArtifactLoadError:
            return False


_store: ArtifactStore | None = None
_store_lock = threading.Lock()


def get_artifact_store() -> ArtifactStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ArtifactStore()
    return _store
