"""
Centralized configuration. Every module (pipeline, API, dashboard) imports
from here instead of hardcoding paths, thresholds, or connection strings.

All values are overridable via environment variables / .env, so the same
code runs unmodified in dev, staging, and production (12-factor app).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PCI_", extra="ignore")

    # ------------------------------------------------------------------ #
    # Data backend
    # ------------------------------------------------------------------ #
    data_backend: Literal["dask", "postgis"] = "dask"

    csv_path: Path = Path("data/raw/violations.csv")
    chunk_size: int = 25_000

    postgis_dsn: str = Field(
        default="postgresql://user:password@localhost:5432/traffic_db",
        description="SQLAlchemy DSN for the PostGIS-backed gateway. Not used by default.",
    )
    postgis_table: str = "violations"

    # ------------------------------------------------------------------ #
    # Secrets backend
    # ------------------------------------------------------------------ #
    # "env"   : all sensitive values read from environment variables (default, dev-safe)
    # "aws"   : sensitive values loaded from AWS Secrets Manager at startup
    # "vault" : sensitive values loaded from HashiCorp Vault KV v2 at startup
    #
    # Switching backend is one env-var change (PCI_SECRETS_BACKEND=aws).
    # See config/secrets.py for the full contract and what each backend expects.
    secrets_backend: Literal["env", "aws", "vault"] = "env"
    aws_secret_name: str = "pci/prod-secrets"
    aws_region: str = "ap-south-1"
    vault_addr: str = "http://localhost:8200"
    vault_path: str = "secret/data/pci"

    # ------------------------------------------------------------------ #
    # Security / auth
    # ------------------------------------------------------------------ #
    # Comma-separated list of valid API keys.
    # Empty (default) → auth disabled, all endpoints public (dev mode only).
    # In production always set this: PCI_API_KEYS=key-abc123,key-xyz456
    api_keys: frozenset[str] = Field(
        default=frozenset(),
        description="Valid API keys. Empty = no auth (dev only). Comma-separated in env.",
    )

    # CORS — restrict to your dashboard's deployed origin(s) in production.
    # Default covers local Streamlit dev server only.
    cors_origins: list[str] = Field(
        default=["http://localhost:8501"],
        description="Allowed CORS origins. Comma-separated string in env.",
    )

    # Rate limits (slowapi / limits library notation: "N/period")
    # Applied per API key (or per IP if no key is present).
    rate_limit_default: str = "200/minute"    # read endpoints (leaderboard, kpis, …)
    rate_limit_predict: str = "60/minute"     # compute-bound predict endpoint
    rate_limit_map: str = "20/minute"         # large HTML response — extra conservative

    # ------------------------------------------------------------------ #
    # Geofence / data quality bounds (Bengaluru metro area, with margin)
    # ------------------------------------------------------------------ #
    lat_min: float = 12.70
    lat_max: float = 13.45
    lon_min: float = 77.30
    lon_max: float = 77.90
    max_row_rejection_rate: float = 0.25

    # ------------------------------------------------------------------ #
    # Clustering (Phase 1)
    # ------------------------------------------------------------------ #
    cluster_partition_column: str = "police_station"
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int | None = None

    # ------------------------------------------------------------------ #
    # Heatmap (Phase 2)
    # ------------------------------------------------------------------ #
    heatmap_max_points: int = 20_000
    heatmap_sample_seed: int = 42

    # ------------------------------------------------------------------ #
    # Impact scoring (Phase 4)
    # ------------------------------------------------------------------ #
    severity_mapping: dict[str, float] = {
        "PARKING NEAR ROAD CROSSING": 3.0,
        "PARKING ON FOOTPATH": 2.5,
        "PARKING NEAR FIRE HYDRANT": 2.5,
        "PARKING ON A MAIN ROAD": 2.0,
        "PARKING IN A MAIN ROAD": 2.0,
        "NO PARKING": 1.5,
        "WRONG PARKING": 1.0,
    }
    default_severity: float = 1.0

    vehicle_weight_mapping: dict[str, int] = {
        "MOTOR CYCLE": 1, "SCOOTER": 1, "MOPED": 1, "PASSENGER AUTO": 1,
        "GOODS AUTO": 1, "AUTO RICKSHAW": 1, "CAR": 2, "JEEP": 2, "VAN": 2,
        "MAXI-CAB": 2, "TEMPO": 2, "TRACTOR": 3, "MINI LORRY": 3, "LGV": 3,
        "TANKER": 4, "HGV": 4, "HEAVY GOODS VEHICLE": 4, "LORRY/GOODS VEHICLE": 4,
        "BUS": 4, "BUS (BMTC/KSRTC)": 4, "PRIVATE BUS": 4, "TOURIST BUS": 4,
        "FACTORY BUS": 4, "SCHOOL VEHICLE": 4,
    }
    default_vehicle_weight: int = 1

    # Path to the external business rules YAML (see config/business_rules.yaml).
    # When it exists, it overrides the Python defaults above at startup.
    # Domain experts can edit the YAML and trigger a re-train without touching code.
    business_rules_path: Path = Path("config/business_rules.yaml")

    # ------------------------------------------------------------------ #
    # Forecasting (Phase 5)
    # ------------------------------------------------------------------ #
    model_test_size: float = 0.2
    model_random_state: int = 42
    xgb_n_estimators: int = 200
    xgb_learning_rate: float = 0.08
    xgb_max_depth: int = 6

    # ------------------------------------------------------------------ #
    # Model promotion gate
    # ------------------------------------------------------------------ #
    # A new training run is only promoted to `artifacts/latest.json` if its
    # forecast MAE does not regress beyond this % relative to the previous run.
    # The run artifacts are always written to disk — only the latest.json
    # pointer update is blocked. Use --force-promote to override.
    # Set require_promotion_gate=false to always promote (useful for first
    # deployment or debugging — leave enabled in steady-state production).
    require_promotion_gate: bool = True
    max_mae_regression_pct: float = 10.0  # block if new MAE > prev MAE × (1 + this/100)

    # ------------------------------------------------------------------ #
    # Enforcement engine (Phase 6)
    # ------------------------------------------------------------------ #
    growth_rate_clip_min: float = 0.5
    growth_rate_clip_max: float = 3.0
    urgent_growth_threshold: float = 1.2
    high_priority_recency_threshold: float = 0.9

    # ------------------------------------------------------------------ #
    # Artifacts / model registry
    # ------------------------------------------------------------------ #
    artifact_root: Path = Path("artifacts")
    model_version: str = "v1"

    # ------------------------------------------------------------------ #
    # API / service
    # ------------------------------------------------------------------ #
    api_host: str = "0.0.0.0"  # nosec B104 — intentional for container/reverse-proxy deployments
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    cache_ttl_seconds: int = 60
    log_level: str = "INFO"
    # Expose /metrics endpoint for Prometheus scraping.
    # In production, restrict access via network policy or a separate internal port.
    metrics_enabled: bool = True

    # ------------------------------------------------------------------ #
    # Field validators
    # ------------------------------------------------------------------ #
    @field_validator("api_keys", mode="before")
    @classmethod
    def _parse_api_keys(cls, v) -> frozenset[str]:
        if isinstance(v, (set, frozenset)):
            return frozenset(str(k).strip() for k in v if k)
        if isinstance(v, str) and v.strip():
            return frozenset(k.strip() for k in v.split(",") if k.strip())
        return frozenset()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v) -> list[str]:
        if isinstance(v, list):
            return [o.strip() for o in v if o.strip()]
        if isinstance(v, str) and v.strip():
            return [o.strip() for o in v.split(",") if o.strip()]
        return ["http://localhost:8501"]

    # ------------------------------------------------------------------ #
    # Secrets-backend override
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _load_from_secrets_backend(self) -> "Settings":
        """If a non-env secrets backend is configured, override sensitive fields
        with values pulled from it. This runs once at construction time (inside
        the lru_cached get_settings()) so there's exactly one backend call per
        process lifetime."""
        if self.secrets_backend == "env":
            return self  # nothing to do; pydantic-settings already resolved env vars

        from config.secrets import SecretsLoader  # local import avoids circular dep at module load

        loader = SecretsLoader(
            backend=self.secrets_backend,
            aws_secret_name=self.aws_secret_name,
            aws_region=self.aws_region,
            vault_addr=self.vault_addr,
            vault_path=self.vault_path,
        )

        raw_keys = loader.get("api_keys")
        if raw_keys:
            self.api_keys = frozenset(k.strip() for k in raw_keys.split(",") if k.strip())

        raw_dsn = loader.get("postgis_dsn")
        if raw_dsn:
            self.postgis_dsn = raw_dsn

        return self

    @model_validator(mode="after")
    def _load_business_rules(self) -> "Settings":
        """If config/business_rules.yaml exists, override severity and vehicle-weight
        mappings with its values so domain experts can edit rules without code changes."""
        path = self.business_rules_path
        if not path.exists():
            return self
        try:
            import yaml  # lazy: only pipeline image carries pyyaml
            with open(path) as f:
                rules = yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001
            # A malformed YAML file is a warning, not a crash — fall back to defaults.
            import warnings
            warnings.warn(f"Could not load business rules from {path}: {exc}. Using defaults.", stacklevel=2)
            return self

        if "severity_mapping" in rules:
            self.severity_mapping = {str(k): float(v) for k, v in rules["severity_mapping"].items()}
        if "vehicle_weight_mapping" in rules:
            self.vehicle_weight_mapping = {str(k): int(v) for k, v in rules["vehicle_weight_mapping"].items()}
        return self

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #
    @property
    def latest_pointer(self) -> Path:
        return self.artifact_root / "latest.json"

    @property
    def auth_enabled(self) -> bool:
        """True when API key auth is active (api_keys is non-empty)."""
        return bool(self.api_keys)


@lru_cache
def get_settings() -> Settings:
    """Settings are read once per process and cached — cheap to call anywhere."""
    return Settings()
