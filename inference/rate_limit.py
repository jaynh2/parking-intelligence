"""
Rate limiter setup using slowapi (built on the limits library).

Key design:
- Rate-limit buckets are keyed by X-API-Key header, NOT by IP.
  This means each API consumer gets an independent quota — one misbehaving
  client hitting their limit doesn't affect other key holders.
- For unauthenticated (dev-mode) traffic, buckets fall back to client IP.
- Limits are configurable per-endpoint via PCI_RATE_LIMIT_* settings.
- Backend is in-memory (single-process). For multi-replica deployments,
  swap to Redis: Limiter(key_func=..., storage_uri="redis://localhost:6379/0")
  — no other changes needed.

Limits applied (overridable in .env):
  PCI_RATE_LIMIT_DEFAULT = 200/minute   (leaderboard, kpis, feature-importance)
  PCI_RATE_LIMIT_PREDICT = 60/minute    (predict — model inference has real cost)
  PCI_RATE_LIMIT_MAP     = 20/minute    (map HTML — large response, cache on client)
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _rate_limit_key(request: Request) -> str:
    """Identify the caller: prefer API key over IP for fairer per-consumer limits."""
    key = request.headers.get("X-API-Key", "").strip()
    return key if key else get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
