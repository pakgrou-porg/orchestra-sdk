"""
orchestra_sdk.tools.base
========================
Abstract base class for all Conductor tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolError(Exception):
    """Base exception for all tool failures."""
    pass


class BaseTool(ABC):
    """
    Every tool must define:
      - name: str          — unique identifier used in the tool registry
      - description: str   — shown to the LLM in the system prompt
      - run(**kwargs)      — synchronous execution method
    """

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool. Raises ToolError on failure."""
        ...

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
