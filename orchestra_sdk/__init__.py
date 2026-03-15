"""
orchestra_sdk
=============
Autonomous LLM-driven research loop for fine-tuning local language models.

Public API:
    ConductorLoop   — main loop class
    ConductorConfig — configuration schema
    Hypothesis      — LLM output schema

Usage:
    from orchestra_sdk import ConductorLoop, ConductorConfig
    import asyncio

    config = ConductorConfig.from_yaml("conductor_config.yaml")
    loop = ConductorLoop(config)
    asyncio.run(loop.run())
"""

from .config import ConductorConfig
from .context import Hypothesis, HypothesisEdit
from .loop import ConductorLoop, Decision, IterationResult

__version__ = "0.1.0a1"
__all__ = [
    "ConductorConfig",
    "ConductorLoop",
    "Decision",
    "Hypothesis",
    "HypothesisEdit",
    "IterationResult",
]
