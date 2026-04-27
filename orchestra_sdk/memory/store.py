"""
orchestra_sdk.memory.store
============================
Memory CRUD backed by Supabase pgvector.
Stores per-iteration memories with semantic search capability.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from ..config import MemoryConfig, SupabaseConfig
from .embedder import EmbeddingError, LocalEmbedder

logger = logging.getLogger(__name__)


class MemoryStoreError(Exception):
    pass


class MemoryRecord:
    def __init__(
        self,
        content: str,
        similarity: float,
        iteration: int,
        decision: str,
        created_at: str,
        metadata: Optional[dict] = None,
    ):
        self.content = content
        self.similarity = similarity
        self.iteration = iteration
        self.decision = decision
        self.created_at = created_at
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"<Memory iter={self.iteration} decision={self.decision} "
            f"sim={self.similarity:.3f}: {self.content[:60]}...>"
        )


class MemoryStore:
    """
    Stores and retrieves memories using Supabase pgvector.

    Each memory is a text description of an iteration outcome,
    embedded with nomic-embed-text and stored with a 768-dim vector.
    Semantic search uses cosine similarity via pgvector.
    """

    def __init__(
        self,
        supabase_config: SupabaseConfig,
        memory_config: MemoryConfig,
        session_name: str,
    ):
        self.supabase_config = supabase_config
        self.memory_config = memory_config
        self.session_name = session_name
        self.embedder = LocalEmbedder(
            base_url=memory_config.embedding_url,
            model=memory_config.embedding_model,
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from supabase import create_client
            self._client = create_client(
                self.supabase_config.get_url(),
                self.supabase_config.get_key(),
            )
        return self._client

    def add(
        self,
        content: str,
        iteration: int,
        decision: str,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Embed and store a memory. Returns the record ID or None on failure.
        Failures are logged but do not raise — memory is non-critical.
        """
        try:
            embedding = self.embedder.embed(content)
        except EmbeddingError as e:
            logger.warning(f"[MemoryStore] Embedding failed, skipping memory: {e}")
            return None

        record = {
            "session_name": self.session_name,
            "content": content,
            "embedding": embedding,
            "iteration": iteration,
            "decision": decision,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = (
                self._get_client()
                .table(self.supabase_config.memories_table)
                .insert(record)
                .execute()
            )
            if result.data:
                return result.data[0].get("id")
            return None
        except Exception as e:
            logger.error(f"[MemoryStore] Failed to store memory: {e}")
            return None

    def search(self, query: str) -> list[MemoryRecord]:
        """
        Semantic search over memories for this session.
        Returns top-K memories sorted by cosine similarity.
        Falls back to empty list if embedding or DB fails.
        """
        try:
            query_embedding = self.embedder.embed(query)
        except EmbeddingError as e:
            logger.warning(f"[MemoryStore] Search embedding failed: {e}")
            return []

        try:
            # Use Supabase RPC for pgvector cosine similarity search
            result = self._get_client().rpc(
                "search_conductor_memories",
                {
                    "query_embedding": query_embedding,
                    "session_name_filter": self.session_name,
                    "embedding_model_filter": self.memory_config.embedding_model,
                    "match_threshold": self.memory_config.similarity_threshold,
                    "match_count": self.memory_config.top_k,
                },
            ).execute()

            records = []
            for row in (result.data or []):
                records.append(
                    MemoryRecord(
                        content=row.get("content", ""),
                        similarity=row.get("similarity", 0.0),
                        iteration=row.get("iteration", 0),
                        decision=row.get("decision", "unknown"),
                        created_at=row.get("created_at", ""),
                        metadata=row.get("metadata", {}),
                    )
                )
            return records
        except Exception as e:
            logger.error(f"[MemoryStore] Memory search failed: {e}")
            return []

    def list_recent(self, limit: int = 10) -> list[dict]:
        """List recent memories without semantic search (for status display)."""
        try:
            result = (
                self._get_client()
                .table(self.supabase_config.memories_table)
                .select("id, content, iteration, decision, created_at, metadata")
                .eq("session_name", self.session_name)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"[MemoryStore] list_recent failed: {e}")
            return []
