"""Secret providers and outbound credential resolution (§8)."""

from __future__ import annotations

import os
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class SecretProvider(Protocol):
    """Resolve a named secret to its value (or None)."""

    def get(self, key: str) -> str | None: ...


class EnvProvider:
    """Read secrets from environment variables."""

    def get(self, key: str) -> str | None:
        return os.environ.get(key)


class KeyringProvider:
    """Read secrets from the OS keyring (optional dependency)."""

    def __init__(self, service: str = "agentkit") -> None:
        self.service = service

    def get(self, key: str) -> str | None:
        try:
            import keyring
        except ImportError:
            return None
        return keyring.get_password(self.service, key)


class VaultProvider:
    """Optional external secret manager (no infra required by core).

    Accepts a callable resolver so any backend can be plugged in without a hard
    dependency.
    """

    def __init__(self, resolver) -> None:
        self._resolver = resolver

    def get(self, key: str) -> str | None:
        return self._resolver(key)


class AuthConfig(BaseModel):
    """Outbound auth for a tool / MCP server / A2A peer."""

    type: Literal["none", "bearer", "api_key", "basic", "oauth2"] = "none"
    token: str | None = None
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    header: str | None = None  # custom header name for api_key
    # For bearer/api_key the value may be a secret *name* resolved via a provider.
    secret: str | None = None

    def headers(self) -> dict[str, str]:
        """Render outbound HTTP headers for this auth config."""
        if self.type == "bearer" and self.token:
            return {"Authorization": f"Bearer {self.token}"}
        if self.type == "api_key" and self.api_key:
            return {self.header or "X-Api-Key": self.api_key}
        if self.type == "basic" and self.username is not None:
            import base64

            raw = f"{self.username}:{self.password or ''}".encode()
            return {"Authorization": "Basic " + base64.b64encode(raw).decode()}
        return {}


class CredentialResolver:
    """Resolve outbound auth per tool/server/peer from a secret provider.

    Outbound tool auth is keyed by MCP server / A2A peer / tool name. The resolver
    fills in secret *values* from the provider given a config that names a secret.
    """

    def __init__(self, provider: SecretProvider | None = None) -> None:
        self.provider = provider or EnvProvider()
        self._configs: dict[str, AuthConfig] = {}

    def register(self, target: str, config: AuthConfig) -> None:
        self._configs[target] = config

    def resolve(self, target: str) -> AuthConfig | None:
        cfg = self._configs.get(target)
        if cfg is None:
            return None
        # If a secret name is given, materialize its value into the right field.
        if cfg.secret:
            value = self.provider.get(cfg.secret)
            if value is not None:
                if cfg.type == "bearer":
                    cfg = cfg.model_copy(update={"token": value})
                elif cfg.type == "api_key":
                    cfg = cfg.model_copy(update={"api_key": value})
        return cfg
