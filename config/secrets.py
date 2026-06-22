"""
Secrets loader — abstracts WHERE sensitive values come from.

Default (PCI_SECRETS_BACKEND=env): pydantic-settings reads directly from
environment variables / .env file. Zero extra infrastructure.

Production swap-ins:
  PCI_SECRETS_BACKEND=aws   → AWS Secrets Manager (requires boto3 installed,
                              PCI_AWS_SECRET_NAME and PCI_AWS_REGION configured)
  PCI_SECRETS_BACKEND=vault → HashiCorp Vault KV v2 (requires hvac installed,
                              PCI_VAULT_ADDR and PCI_VAULT_PATH configured,
                              VAULT_TOKEN env var set by the platform)

Both boto3 and hvac are imported lazily: they're NOT in requirements-api.txt
or requirements-pipeline.txt so production images stay lean — install
whichever library your chosen backend needs alongside those requirements.

Switching is one env-var change (PCI_SECRETS_BACKEND=aws). No code changes.

AWS/Vault paths are documented and implemented but have not been exercised
against a live backend in this sandbox (no credentials available). The env
backend is fully tested.
"""
from __future__ import annotations

import json
import os
from typing import Literal

from pipeline.logging_config import get_logger

logger = get_logger(__name__)


class SecretsLoadError(RuntimeError):
    """Raised when a configured secrets backend is unreachable or misconfigured."""


class SecretsLoader:
    """Loads a single JSON-formatted secret blob and exposes its fields.

    The secret stored in AWS / Vault should be a JSON object whose keys
    match the sensitive PCI_ setting names without the prefix, e.g.:
        { "api_keys": "key-abc123,key-def456", "postgis_dsn": "postgresql://..." }
    """

    def __init__(
        self,
        backend: Literal["env", "aws", "vault"],
        aws_secret_name: str = "",
        aws_region: str = "ap-south-1",
        vault_addr: str = "http://localhost:8200",
        vault_path: str = "secret/data/pci",
    ) -> None:
        self._backend = backend
        self._aws_secret_name = aws_secret_name
        self._aws_region = aws_region
        self._vault_addr = vault_addr
        self._vault_path = vault_path
        self._cache: dict | None = None

    def _fetch_aws(self) -> dict:
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SecretsLoadError(
                "PCI_SECRETS_BACKEND=aws requires boto3. "
                "Install it alongside your other requirements: pip install boto3"
            ) from exc
        client = boto3.client("secretsmanager", region_name=self._aws_region)
        try:
            response = client.get_secret_value(SecretId=self._aws_secret_name)
            raw = response.get("SecretString") or response.get("SecretBinary", b"{}").decode()
            return json.loads(raw)
        except Exception as exc:
            raise SecretsLoadError(
                f"Failed to fetch secret '{self._aws_secret_name}' from AWS Secrets Manager "
                f"(region {self._aws_region}): {exc}"
            ) from exc

    def _fetch_vault(self) -> dict:
        try:
            import hvac  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SecretsLoadError(
                "PCI_SECRETS_BACKEND=vault requires hvac. "
                "Install it: pip install hvac"
            ) from exc
        token = os.environ.get("VAULT_TOKEN")
        if not token:
            raise SecretsLoadError("VAULT_TOKEN environment variable is required for PCI_SECRETS_BACKEND=vault.")
        client = hvac.Client(url=self._vault_addr, token=token)
        if not client.is_authenticated():
            raise SecretsLoadError(f"Vault client at {self._vault_addr} is not authenticated. Check VAULT_TOKEN.")
        try:
            secret = client.secrets.kv.v2.read_secret_version(path=self._vault_path.lstrip("secret/data/"))
            return secret["data"]["data"]
        except Exception as exc:
            raise SecretsLoadError(f"Failed to read Vault path '{self._vault_path}': {exc}") from exc

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if self._backend == "env":
            self._cache = {}  # env vars are handled natively by pydantic-settings
        elif self._backend == "aws":
            logger.info("Loading secrets from AWS Secrets Manager: %s", self._aws_secret_name)
            self._cache = self._fetch_aws()
        elif self._backend == "vault":
            logger.info("Loading secrets from HashiCorp Vault: %s", self._vault_path)
            self._cache = self._fetch_vault()
        else:
            self._cache = {}
        return self._cache

    def get(self, key: str, default: str = "") -> str:
        """Return the value of `key` from the secrets blob, or `default` if absent.
        For the 'env' backend this always returns `default` — env vars are
        already handled by pydantic-settings field resolution."""
        return self._load().get(key, default)
