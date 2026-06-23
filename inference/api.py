"""
Inference API — read-only HTTP surface over whatever the training pipeline
last produced. This process never trains anything; it only loads serialized
artifacts (see inference/artifact_store.py) and answers requests.

Security layers:
  - API key auth: X-API-Key header, required on all routes except /health.
    Empty PCI_API_KEYS → dev-mode bypass (see inference/auth.py).
  - Rate limiting: per-API-key buckets via slowapi (see inference/rate_limit.py).
  - CORS: restricted to PCI_CORS_ORIGINS (default: localhost:8501 only).

Run:
    uvicorn inference.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from starlette.responses import Response

from config.settings import get_settings
from inference import metrics as pci_metrics
from inference import service
from inference.artifact_store import ArtifactLoadError, get_artifact_store
from inference.auth import verify_api_key
from inference.rate_limit import limiter
from pipeline.logging_config import configure_logging, get_logger
from schemas.api import (
    FeatureImportance,
    ForecastRequest,
    ForecastResponse,
    HealthStatus,
    HotspotSummary,
    KPISummary,
    LeaderboardResponse,
)

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    if not settings.auth_enabled:
        logger.warning(
            "PCI_API_KEYS is not set — API key auth is DISABLED. "
            "This is expected in development; set PCI_API_KEYS before any public-facing deployment."
        )
    try:
        get_artifact_store().get()
        logger.info("Artifact cache warmed on startup.")
    except ArtifactLoadError as exc:
        logger.warning("No artifacts available at startup: %s", exc)
    yield


app = FastAPI(
    title="Parking Congestion Intelligence API",
    description="Read-only serving layer over the latest trained hotspot/forecast/leaderboard artifacts.",
    version=settings.model_version,
    lifespan=_lifespan,
    # Disable the interactive docs in production to reduce attack surface.
    # Remove these two lines to re-enable /docs and /redoc for internal environments.
    docs_url="/docs" if not settings.auth_enabled else None,
    redoc_url="/redoc" if not settings.auth_enabled else None,
)

# ---- Prometheus --------------------------------------------------------
if settings.metrics_enabled:
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False, tags=["meta"])

    @app.middleware("http")
    async def _refresh_model_metrics_on_scrape(request: Request, call_next) -> Response:
        """Update model-quality Prometheus gauges just before each /metrics scrape
        so Grafana always sees the current artifact state, not just startup values."""
        if request.url.path == "/metrics":
            try:
                artifacts = get_artifact_store().get()
                pci_metrics.update_from_metrics_dict(artifacts.metrics)
            except ArtifactLoadError:
                pass
        return await call_next(request)

# ---- Rate limiting -----------------------------------------------------
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ---- CORS ---------------------------------------------------------------
# Scoped to configured origins only. In production this should be the
# dashboard's actual deployed URL (e.g. https://dashboard.example.com).
# Default is localhost:8501 (local Streamlit dev server).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
    allow_credentials=False,
)


# --------------------------------------------------------------------------
# Public routes — no auth, no rate limit (health must work for LB probes)
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthStatus, tags=["meta"])
def health() -> HealthStatus:
    """Load-balancer / readiness probe. Always public, never rate-limited."""
    return service.health_check()


# --------------------------------------------------------------------------
# Protected routes — all require a valid API key (or dev-mode bypass)
# --------------------------------------------------------------------------

@app.get(
    "/api/v1/leaderboard",
    response_model=LeaderboardResponse,
    tags=["enforcement"],
    dependencies=[Security(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
def leaderboard(request: Request, limit: int | None = Query(default=None, ge=1, le=5000)) -> LeaderboardResponse:
    """Ranked hotspot leaderboard with dispatch status and recommendations."""
    try:
        return service.get_leaderboard(limit=limit)
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    "/api/v1/hotspots/{cluster_id}",
    response_model=HotspotSummary,
    tags=["enforcement"],
    dependencies=[Security(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
def hotspot(request: Request, cluster_id: str) -> HotspotSummary:
    """Detail for a single hotspot by cluster_id."""
    try:
        return service.get_hotspot(cluster_id)
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/api/v1/predict",
    response_model=ForecastResponse,
    tags=["forecasting"],
    dependencies=[Security(verify_api_key)],
)
@limiter.limit(settings.rate_limit_predict)
def predict(request: Request, body: ForecastRequest) -> ForecastResponse:
    """Predict hourly violation volume for a given cluster, hour, and day of week."""
    try:
        return service.predict(body)
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except service.UnknownClusterError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/api/v1/feature-importance",
    response_model=list[FeatureImportance],
    tags=["forecasting"],
    dependencies=[Security(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
def feature_importance(request: Request) -> list[FeatureImportance]:
    """XGBoost feature importance from the latest trained model."""
    try:
        return service.get_feature_importance()
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    "/api/v1/kpis",
    response_model=KPISummary,
    tags=["dashboard"],
    dependencies=[Security(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
def kpis(request: Request) -> KPISummary:
    """Aggregated KPI summary for the dashboard header cards."""
    try:
        return service.get_kpis()
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    "/api/v1/map",
    tags=["dashboard"],
    dependencies=[Security(verify_api_key)],
)
@limiter.limit(settings.rate_limit_map)
def map_html(request: Request) -> FileResponse:
    """Serve the Folium heatmap HTML for the latest training run."""
    try:
        artifacts = get_artifact_store().get()
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    heatmap_path = artifacts.run_dir / "heatmap.html"
    if not heatmap_path.exists():
        raise HTTPException(status_code=404, detail=f"No heatmap artifact for run {artifacts.run_id}")
    return FileResponse(
        heatmap_path,
        media_type="text/html",
        headers={"Cache-Control": "public, max-age=300"},  # 5-min browser cache for the 9MB file
    )
