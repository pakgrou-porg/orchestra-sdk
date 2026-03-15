"""
orchestra_sdk.tools.memory_tools
==================================
Tool wrappers for the memory store.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseTool
from ..memory.store import MemoryStore


class SearchMemories(BaseTool):
    name = "search_memories"
    description = (
        "Semantic search over past experiment memories for this session. "
        "Returns the most relevant memories ranked by similarity."
    )

    def __init__(self, store: MemoryStore):
        self.store = store

    def run(self, query: str) -> list[dict]:
        records = self.store.search(query)
        return [
            {
                "content": r.content,
                "similarity": round(r.similarity, 4),
                "iteration": r.iteration,
                "decision": r.decision,
                "created_at": r.created_at,
            }
            for r in records
        ]


class AddMemory(BaseTool):
    name = "add_memory"
    description = "Store a memory about this iteration's outcome for future retrieval."

    def __init__(self, store: MemoryStore):
        self.store = store

    def run(
        self,
        content: str,
        iteration: int,
        decision: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        record_id = self.store.add(
            content=content,
            iteration=iteration,
            decision=decision,
            metadata=metadata,
        )
        return {"id": record_id, "stored": record_id is not None}


class ListMemories(BaseTool):
    name = "list_memories"
    description = "List recent memories for this session (no semantic search)."

    def __init__(self, store: MemoryStore):
        self.store = store

    def run(self, limit: int = 10) -> list[dict]:
        return self.store.list_recent(limit=limit)
