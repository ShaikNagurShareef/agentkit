"""Identity & secrets (§8).

Core needs no identity infrastructure: model and tool keys come from env/keyring.
This module separates inbound exposure auth from outbound tool auth and keeps
secrets out of traces.
"""

from .secrets import (
    AuthConfig,
    CredentialResolver,
    EnvProvider,
    KeyringProvider,
    SecretProvider,
    VaultProvider,
)

__all__ = [
    "SecretProvider",
    "EnvProvider",
    "KeyringProvider",
    "VaultProvider",
    "AuthConfig",
    "CredentialResolver",
]
