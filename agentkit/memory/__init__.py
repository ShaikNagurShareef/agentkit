"""Memory subsystem (§5): one store, three scopes, extraction strategies."""

from .base import MemoryConfig, MemoryRecord, MemoryStore, Scope, Summary
from .store import LocalMemoryStore, build_memory_store
from .strategies import build_summary, extract_preferences, extract_semantic

__all__ = [
    "MemoryConfig",
    "MemoryRecord",
    "MemoryStore",
    "Scope",
    "Summary",
    "LocalMemoryStore",
    "build_memory_store",
    "extract_semantic",
    "extract_preferences",
    "build_summary",
]
