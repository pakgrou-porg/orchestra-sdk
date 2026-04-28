"""
orchestra_sdk.context
======================
Context window assembly for the Conductor.
Assembles the prompt from multiple sources with token budget enforcement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import ConductorConfig
from .constants import CHARS_PER_TOKEN as _CHARS_PER_TOKEN
from .llm import Message
from .memory.store import MemoryRecord
from .tools.git_tools import CommitRecord

logger = logging.getLogger(__name__)


def _count_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _truncate(text: str, max_tokens: int, label: str = "") -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    note = f"\n... [{label} truncated: {_count_tokens(text)} tokens total, showing {max_tokens}]"
    return truncated + note


# ---------------------------------------------------------------------------
# Hypothesis schema (Pydantic) — defined here for import by loop.py
# ---------------------------------------------------------------------------


class HypothesisEdit(BaseModel):
    find: str = Field(description="Exact string to find in train.py (must be unique)")
    replace: str = Field(description="Replacement string")


class Hypothesis(BaseModel):
    hypothesis: str = Field(description="One-sentence description of the proposed change")
    change_type: Literal[
        "hyperparameter", "architecture", "data", "optimizer", "scheduler", "regularization", "other"
    ] = Field(description="Category of the change")
    expected_effect: str = Field(
        description="Expected effect on the target metric (e.g., 'reduce val_loss by ~0.05')"
    )
    risk: Literal["low", "medium", "high"] = Field(
        description="Risk level: low=safe tweak, medium=structural change, high=experimental"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence that this change will improve the metric (0.0–1.0)"
    )
    memory_note: str = Field(
        description="Brief note to store as a memory about this hypothesis (1-2 sentences)"
    )
    edit: HypothesisEdit = Field(
        description="The exact find/replace edit to apply to train.py"
    )


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are the Conductor in the Orchestra framework — an autonomous research agent \
that iteratively improves LLM fine-tuning configurations.

Your role: propose ONE testable hypothesis per iteration that may improve the \
target metric. You will edit train.py using a precise find/replace operation.

Rules:
1. The 'find' string MUST appear EXACTLY ONCE in train.py (whitespace-sensitive).
2. Propose only one change per iteration — compound changes are harder to attribute.
3. Learn from past failures: if a similar hypothesis was tried and discarded, try something different.
4. Be conservative: prefer low-risk changes unless you have strong evidence for higher-risk ones.
5. Output ONLY valid JSON matching the Hypothesis schema. No prose, no markdown.
"""


def assemble_context(
    config: ConductorConfig,
    program_text: str,
    train_script_text: str,
    git_log: list[CommitRecord],
    metric_history: list[dict],
    memories: list[MemoryRecord],
    baseline_metric: Optional[float],
    iteration: int,
) -> list[Message]:
    """
    Assemble the full context for the LLM hypothesis call.
    Enforces the token budget from config.llm.context_budget_tokens.
    Truncates sections in priority order: git log, metric history, memories, train.py.
    """
    budget = config.llm.context_budget_tokens
    used = _count_tokens(SYSTEM_PROMPT)

    # --- Fixed sections (always included) ---
    iteration_header = (
        f"ITERATION: {iteration}\n"
        f"SESSION: {config.session.name}\n"
        f"TARGET METRIC: {config.session.target_metric}\n"
        f"CURRENT BASELINE: {f'{baseline_metric:.4f}' if baseline_metric is not None else 'N/A (first iteration)'}\n"
        f"KEEP THRESHOLD: Δ ≤ {config.session.keep_threshold}\n"
    )
    used += _count_tokens(iteration_header)

    # --- Program (high priority — always include) ---
    program_budget = min(1500, budget - used - 3000)  # reserve 3000 for rest
    program_section = _truncate(program_text, program_budget, "program")
    used += _count_tokens(program_section)

    # --- Train script (high priority) ---
    train_budget = min(2000, budget - used - 2000)
    train_section = _truncate(train_script_text, train_budget, "train.py")
    used += _count_tokens(train_section)

    # --- Metric history (medium priority) ---
    remaining = budget - used
    history_text = _format_metric_history(metric_history)
    history_budget = min(remaining // 3, 1500)
    history_section = _truncate(history_text, history_budget, "metric history")
    used += _count_tokens(history_section)

    # --- Git log (lower priority) ---
    remaining = budget - used
    git_text = _format_git_log(git_log)
    git_budget = min(remaining // 2, 1000)
    git_section = _truncate(git_text, git_budget, "git log")
    used += _count_tokens(git_section)

    # --- Memories (lower priority) ---
    remaining = budget - used
    memory_text = _format_memories(memories)
    memory_budget = remaining - 100
    if memory_budget > 0:
        memory_section = _truncate(memory_text, memory_budget, "memories")
    else:
        memory_section = "(memories omitted — context budget exhausted)"

    logger.debug(f"[Context] Assembled ~{used} tokens for iteration {iteration}")

    # Build the user message
    user_content = f"""
{iteration_header}

## RESEARCH PROGRAM
{program_section}

## CURRENT train.py
```python
{train_section}
```

## METRIC HISTORY (last {len(metric_history)} iterations)
{history_section}

## RECENT GIT LOG
{git_section}

## RELEVANT MEMORIES
{memory_section}

---
Propose the next hypothesis. Output JSON only.
""".strip()

    return [
        Message.system(SYSTEM_PROMPT),
        Message.user(user_content),
    ]


def _format_metric_history(history: list[dict]) -> str:
    if not history:
        return "(no history yet — this is the first iteration)"
    lines = ["iter | hypothesis (truncated)                          | metric   | Δ        | decision"]
    lines.append("-" * 95)
    for exp in history:
        hyp = (exp.get("hypothesis", "")[:45] + "...") if len(exp.get("hypothesis", "")) > 45 else exp.get("hypothesis", "")
        metric = exp.get("target_metric", 0)
        delta = exp.get("delta", 0)
        decision = exp.get("decision", "?")
        lines.append(
            f"{exp.get('iteration', '?'):4} | {hyp:<48} | {metric:8.4f} | {delta:+8.4f} | {decision}"
        )
    return "\n".join(lines)


def _format_git_log(log: list[CommitRecord]) -> str:
    if not log:
        return "(no commits yet)"
    lines = []
    for record in log[:8]:  # cap at 8 commits
        lines.append(f"[{record.short_sha}] {record.timestamp.strftime('%Y-%m-%d %H:%M')} — {record.message}")
        if record.diff_summary:
            lines.append(f"  {record.diff_summary[:200]}")
    return "\n".join(lines)


def _format_memories(memories: list[MemoryRecord]) -> str:
    if not memories:
        return "(no relevant memories)"
    lines = []
    for m in memories:
        lines.append(
            f"[iter {m.iteration}, {m.decision}, sim={m.similarity:.2f}] {m.content}"
        )
    return "\n".join(lines)
