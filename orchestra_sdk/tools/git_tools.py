"""
orchestra_sdk.tools.git_tools
==============================
Git operations for the Conductor: log, commit, reset, diff.
All operations are scoped to the session workspace directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import git
from git import GitCommandError, InvalidGitRepositoryError, Repo

from .base import BaseTool, ToolError


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GitToolError(ToolError):
    pass


class GitCommitError(GitToolError):
    pass


class GitResetError(GitToolError):
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CommitRecord:
    sha: str
    short_sha: str
    message: str
    timestamp: datetime
    diff_summary: str  # abbreviated diff (first 500 chars)


# ---------------------------------------------------------------------------
# Helper: get or init repo
# ---------------------------------------------------------------------------


def _get_repo(workspace_dir: Path) -> Repo:
    """Return the git Repo for the workspace, initializing if needed."""
    try:
        return Repo(workspace_dir, search_parent_directories=False)
    except InvalidGitRepositoryError:
        repo = Repo.init(workspace_dir)
        # Create initial commit so HEAD exists
        (workspace_dir / ".orchestra_session").write_text("# Orchestra session\n")
        repo.index.add([".orchestra_session"])
        repo.index.commit("chore: initialize orchestra session workspace")
        return repo


def _ensure_branch(repo: Repo, branch: str) -> None:
    """Create and checkout the session branch if it doesn't exist."""
    if branch not in [b.name for b in repo.branches]:
        repo.create_head(branch)
    repo.heads[branch].checkout()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class GitLog(BaseTool):
    name = "git_log"
    description = "Return the last N commits on the session branch with abbreviated diffs"

    def __init__(self, workspace_dir: Path, branch: str):
        self.workspace_dir = workspace_dir
        self.branch = branch

    def run(self, n: int = 10) -> list[CommitRecord]:
        repo = _get_repo(self.workspace_dir)
        _ensure_branch(repo, self.branch)
        records: list[CommitRecord] = []
        try:
            commits = list(repo.iter_commits(self.branch, max_count=n))
        except GitCommandError as e:
            raise GitToolError(f"git log failed: {e}") from e

        for commit in commits:
            # Get abbreviated diff
            try:
                if commit.parents:
                    diff_text = repo.git.diff(
                        commit.parents[0].hexsha, commit.hexsha, "--stat"
                    )
                else:
                    diff_text = "(initial commit)"
            except GitCommandError:
                diff_text = "(diff unavailable)"

            records.append(
                CommitRecord(
                    sha=commit.hexsha,
                    short_sha=commit.hexsha[:8],
                    message=commit.message.strip(),
                    timestamp=datetime.fromtimestamp(commit.committed_date),
                    diff_summary=diff_text[:500],
                )
            )
        return records


class GitCommit(BaseTool):
    name = "git_commit"
    description = "Stage all changes to the workspace and create a commit"

    def __init__(self, workspace_dir: Path, branch: str):
        self.workspace_dir = workspace_dir
        self.branch = branch

    def run(self, message: str) -> dict:
        repo = _get_repo(self.workspace_dir)
        _ensure_branch(repo, self.branch)

        # Stage all changes
        repo.git.add(A=True)

        # Check if there's anything to commit.
        # repo.index.diff("HEAD") raises BadName on a brand-new repo that has no
        # HEAD yet (i.e. before the very first commit). In that case we fall back
        # to checking the staged diff against an empty tree, which is always
        # non-empty if any files were added.
        try:
            has_staged = bool(repo.index.diff("HEAD"))
        except Exception:
            # No HEAD yet — any staged file counts as a change
            has_staged = bool(repo.index.entries)

        if not has_staged and not repo.untracked_files:
            raise GitCommitError(
                "Nothing to commit — working tree is clean. "
                "The edit_file tool may not have made any changes."
            )

        try:
            commit = repo.index.commit(message)
        except GitCommandError as e:
            raise GitCommitError(f"git commit failed: {e}") from e

        return {
            "sha": commit.hexsha,
            "short_sha": commit.hexsha[:8],
            "timestamp": datetime.fromtimestamp(commit.committed_date).isoformat(),
            "message": message,
        }


class GitReset(BaseTool):
    name = "git_reset"
    description = "Hard reset the workspace to a specific commit SHA"

    def __init__(self, workspace_dir: Path, branch: str):
        self.workspace_dir = workspace_dir
        self.branch = branch

    def run(self, sha: str) -> dict:
        repo = _get_repo(self.workspace_dir)
        _ensure_branch(repo, self.branch)

        # Verify the SHA exists
        try:
            commit = repo.commit(sha)
        except (git.BadName, ValueError) as e:
            raise GitResetError(f"Commit SHA not found: {sha!r}") from e

        try:
            repo.git.reset("--hard", sha)
        except GitCommandError as e:
            raise GitResetError(f"git reset failed: {e}") from e

        return {
            "success": True,
            "current_sha": repo.head.commit.hexsha,
            "reset_to": sha,
        }


class GitDiff(BaseTool):
    name = "git_diff"
    description = "Show the diff between two commits, or HEAD vs working tree"

    def __init__(self, workspace_dir: Path, branch: str):
        self.workspace_dir = workspace_dir
        self.branch = branch

    def run(
        self,
        from_sha: Optional[str] = None,
        to_sha: Optional[str] = None,
        max_chars: int = 3000,
    ) -> str:
        repo = _get_repo(self.workspace_dir)

        try:
            if from_sha and to_sha:
                diff = repo.git.diff(from_sha, to_sha)
            elif from_sha:
                diff = repo.git.diff(from_sha)
            else:
                diff = repo.git.diff("HEAD")
        except GitCommandError as e:
            raise GitToolError(f"git diff failed: {e}") from e

        if len(diff) > max_chars:
            diff = diff[:max_chars] + f"\n... (truncated, {len(diff)} chars total)"
        return diff


# ---------------------------------------------------------------------------
# Convenience: GitManager bundles all tools for a session
# ---------------------------------------------------------------------------


class GitManager:
    """Convenience wrapper that bundles all git tools for a session."""

    def __init__(self, workspace_dir: Path, branch: str):
        self.workspace_dir = workspace_dir
        self.branch = branch
        self._repo: Optional[Repo] = None
        self.log = GitLog(workspace_dir, branch)
        self.commit = GitCommit(workspace_dir, branch)
        self.reset = GitReset(workspace_dir, branch)
        self.diff = GitDiff(workspace_dir, branch)

    def initialize(self) -> None:
        """Ensure workspace directory and git repo exist."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._repo = _get_repo(self.workspace_dir)
        _ensure_branch(self._repo, self.branch)

    def current_sha(self) -> str:
        repo = _get_repo(self.workspace_dir)
        return repo.head.commit.hexsha

    def find_last_keep_sha(self, keep_marker: str = "[KEEP]") -> Optional[str]:
        """
        Walk the git log backwards to find the last commit marked as KEEP.
        Returns None if no KEEP commit exists (use initial commit instead).
        """
        repo = _get_repo(self.workspace_dir)
        for commit in repo.iter_commits(self.branch):
            if keep_marker in commit.message:
                return commit.hexsha
        # Fall back to the first commit
        commits = list(repo.iter_commits(self.branch))
        if commits:
            return commits[-1].hexsha
        return None
