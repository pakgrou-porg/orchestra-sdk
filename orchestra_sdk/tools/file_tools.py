"""
orchestra_sdk.tools.file_tools
================================
File operations for the Conductor: read, edit (find/replace), list.
All paths are resolved relative to the session workspace directory.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..constants import CHARS_PER_TOKEN as _CHARS_PER_TOKEN
from .base import BaseTool, ToolError


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FileToolError(ToolError):
    pass


class EditFileError(FileToolError):
    """Raised when find/replace fails validation."""
    pass


# ---------------------------------------------------------------------------
# Token counting (approximate)
# Shared constant — also used in context.py. Both must stay in sync.
# ---------------------------------------------------------------------------


def _approx_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated: {len(text)} chars, ~{_approx_tokens(text)} tokens]"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ReadFile(BaseTool):
    name = "read_file"
    description = "Read a file from the workspace. Returns file contents as a string."

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_dir / p
        p = p.resolve()  # resolves symlinks — symlinks pointing outside workspace are caught below
        workspace_resolved = self.workspace_dir.resolve()
        # Security: both sides are resolved so symlinks crossing the workspace
        # boundary are detected correctly.
        try:
            p.relative_to(workspace_resolved)
        except ValueError:
            raise FileToolError(
                f"Path escape attempt: {path!r} resolves outside workspace"
            )
        return p

    def run(self, path: str, max_tokens: Optional[int] = None) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if not p.is_file():
            raise FileToolError(f"Not a file: {p}")
        content = p.read_text(encoding="utf-8", errors="replace")
        if max_tokens:
            content = _truncate_to_tokens(content, max_tokens)
        return content


class EditFile(BaseTool):
    name = "edit_file"
    description = (
        "Apply a find/replace edit to a file in the workspace. "
        "By default, validates that exactly one match exists before writing."
    )

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_dir / p
        p = p.resolve()  # resolves symlinks
        workspace_resolved = self.workspace_dir.resolve()
        try:
            p.relative_to(workspace_resolved)
        except ValueError:
            raise FileToolError(f"Path escape attempt: {path!r}")
        return p

    def run(
        self,
        path: str,
        find: str,
        replace: str,
        validate_single_match: bool = True,
    ) -> dict:
        """
        Apply find/replace to a file.

        Returns:
            {"matches_found": int, "applied": bool, "path": str}

        Raises:
            EditFileError if validate_single_match=True and matches != 1.
            FileNotFoundError if the file does not exist.
        """
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        original = p.read_text(encoding="utf-8")
        matches = original.count(find)

        if validate_single_match:
            if matches == 0:
                raise EditFileError(
                    f"Edit failed: find string not found in {path!r}.\n"
                    f"Find string was:\n{find!r}\n"
                    "Check that the find string exactly matches the file content, "
                    "including whitespace and indentation."
                )
            if matches > 1:
                raise EditFileError(
                    f"Edit failed: find string matched {matches} locations in {path!r}. "
                    "The find string must be unique. Make it more specific."
                )

        if matches == 0:
            return {"matches_found": 0, "applied": False, "path": str(p)}

        new_content = original.replace(find, replace, 1 if validate_single_match else -1)
        p.write_text(new_content, encoding="utf-8")

        return {
            "matches_found": matches,
            "applied": True,
            "path": str(p),
            "lines_changed": abs(new_content.count("\n") - original.count("\n")),
        }


class ListFiles(BaseTool):
    name = "list_files"
    description = "List files in the workspace directory, optionally filtered by glob pattern."

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    def run(
        self,
        path: Optional[str] = None,
        pattern: str = "**/*",
        max_results: int = 100,
    ) -> list[dict]:
        base = self.workspace_dir
        if path:
            base = (base / path).resolve()

        results = []
        for p in sorted(base.glob(pattern))[:max_results]:
            if p.is_file():
                stat = p.stat()
                results.append(
                    {
                        "path": str(p.relative_to(self.workspace_dir)),
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    }
                )
        return results


class WriteFile(BaseTool):
    name = "write_file"
    description = "Write content to a file in the workspace (creates if not exists)."

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_dir / p
        p = p.resolve()  # resolves symlinks
        workspace_resolved = self.workspace_dir.resolve()
        try:
            p.relative_to(workspace_resolved)
        except ValueError:
            raise FileToolError(f"Path escape attempt: {path!r}")
        return p

    def run(self, path: str, content: str) -> dict:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p), "size_bytes": len(content.encode("utf-8"))}
