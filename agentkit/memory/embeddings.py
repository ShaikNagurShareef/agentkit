"""Zero-dependency deterministic embedder for offline local memory search.

Hashes tokens into a fixed-dimension bag-of-words vector and L2-normalizes it, so
cosine similarity ranks by lexical overlap. No model download, no server — recall
works out of the box (NFR-1). Heavier embedders (sentence-transformers, provider
embeddings) can be swapped behind the same `embed`/`cosine` surface.
"""

from __future__ import annotations

import hashlib
import math
import re

DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def embed(text: str, dim: int = DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
