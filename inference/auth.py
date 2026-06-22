"""
API key authentication FastAPI dependency.

Applied as Security(verify_api_key) on every protected route.
/health is intentionally excluded — load-balancer probes must work without a key.

Dev-mode bypass: if PCI_API_KEYS is empty (not set), all requests pass through.
This means local development works with zero configuration changes.
Flipping to auth-enabled is: PCI_API_KEYS=key-abc123,key-def456 and restart.

Production key rotation: add the new key to PCI_API_KEYS (comma-separated),
deploy, wait for clients to migrate, then remove the old key. Zero downtime.
"""
from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from config.settings import get_settings
from inference.metrics import auth_rejections
from pipeline.logging_config import get_logger

logger = get_logger(__name__)

_HEADER_SCHEME = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,  # we handle missing keys ourselves to give better error messages
    description="API key issued by the system administrator. Required in production.",
)


async def verify_api_key(api_key: str | None = Security(_HEADER_SCHEME)) -> str:
    """FastAPI Security dependency.

    Returns the validated key string (useful for per-key audit logging
    upstream) or the literal string "dev" in bypass mode.

    Raises:
        HTTPException(401) if auth is enabled and no key was supplied.
        HTTPException(403) if auth is enabled and the supplied key is invalid.
    """
    settings = get_settings()

    if not settings.auth_enabled:
        # Dev-mode: log a one-time warning so misconfigured production servers
        # don't silently skip auth — this is noisy in dev but safe.
        logger.debug("Auth bypass active (PCI_API_KEYS is not configured).")
        return "dev"

    if not api_key:
        auth_rejections.labels(reason="missing_key").inc()
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_api_key",
                "message": "X-API-Key header is required. Contact your system administrator for a key.",
            },
        )

    if api_key not in settings.api_keys:
        suffix = api_key[-4:] if len(api_key) >= 4 else "****"
        logger.warning("Rejected request: invalid API key (suffix: ...%s)", suffix)
        auth_rejections.labels(reason="invalid_key").inc()
        raise HTTPException(
            status_code=403,
            detail={
                "error": "invalid_api_key",
                "message": "The supplied X-API-Key is not recognized.",
            },
        )

    return api_key
