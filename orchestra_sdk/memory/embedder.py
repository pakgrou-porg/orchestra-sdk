"""
orchestra_sdk.memory.embedder
================================
Local embedding via Ollama's nomic-embed-text model.
Falls back gracefully if the embedding service is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 768  # nomic-embed-text output dimension


class EmbeddingError(Exception):
    pass


class LocalEmbedder:
    """
    Generates embeddings using a locally running Ollama instance.
    Endpoint: POST {base_url}/api/embeddings
    Model: nomic-embed-text (768 dimensions)
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = _EMBEDDING_DIM

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string. Returns a list of floats (768 dims).
        Raises EmbeddingError if the service is unavailable.
        """
        try:
            response = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if not embedding:
                raise EmbeddingError(f"No embedding in response: {data}")
            return embedding
        except httpx.ConnectError:
            raise EmbeddingError(
                f"Cannot connect to embedding service at {self.base_url}. "
                f"Is Ollama running? Try: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"Embedding API error: {e.response.status_code} {e.response.text}")
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts sequentially."""
        return [self.embed(t) for t in texts]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def is_available(self) -> bool:
        """Check if the embedding service is reachable."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
