"""
orchestra_sdk.config
====================
Pydantic v2 schema for conductor_config.yaml.
Validates all fields on load; fails fast with clear error messages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class SessionConfig(BaseModel):
    name: str = Field(description="Unique session name, used as git branch suffix")
    dataset_id: str = Field(description="Supabase datasets.name to target")
    workspace_dir: str = Field(description="Absolute or ~ path to workspace directory")
    branch: str = Field(description="Git branch for this session")
    max_iterations: int = Field(default=50, ge=1, le=500)
    target_metric: str = Field(default="val_loss", description="Metric key in results.json to minimize")
    target_value: Optional[float] = Field(default=None, description="Stop if metric reaches this value")
    keep_threshold: float = Field(
        default=-0.005,
        description="Keep commit if delta <= this (negative = improvement)",
    )

    @field_validator("keep_threshold")
    @classmethod
    def threshold_must_be_negative(cls, v: float) -> float:
        if v >= 0:
            raise ValueError(
                f"keep_threshold must be negative (got {v}). "
                "A positive threshold would accept regressions."
            )
        return v

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_dir).expanduser().resolve()


class LLMConfig(BaseModel):
    provider: Literal["openrouter", "anthropic", "openai", "lmstudio", "custom"] = "openrouter"
    model: str = Field(description="Model ID as accepted by the provider's API")
    api_key_env: str = Field(
        default="OPENROUTER_API_KEY",
        description="Name of the environment variable holding the API key",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Override base URL (required for lmstudio/custom)",
    )
    max_tokens: int = Field(default=4096, ge=256, le=32768)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    context_budget_tokens: int = Field(
        default=8000,
        ge=1000,
        le=128000,
        description="Max tokens for assembled context window",
    )

    @model_validator(mode="after")
    def validate_base_url_for_local(self) -> "LLMConfig":
        if self.provider in ("lmstudio", "custom") and not self.base_url:
            raise ValueError(
                f"provider='{self.provider}' requires base_url to be set"
            )
        return self

    def get_api_key(self) -> Optional[str]:
        """Resolve API key from environment. Returns None for local providers."""
        if self.provider in ("lmstudio",):
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise EnvironmentError(
                f"Environment variable '{self.api_key_env}' is not set. "
                f"Required for provider='{self.provider}'."
            )
        return key

    def get_base_url(self) -> str:
        """Return the effective base URL for this provider."""
        defaults = {
            "openrouter": "https://openrouter.ai/api/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "openai": "https://api.openai.com/v1",
        }
        return self.base_url or defaults.get(self.provider, "")


class RunnerConfig(BaseModel):
    type: Literal["docker", "k8s"] = "docker"
    image: str = Field(default="orchestra-musician:latest")
    gpu_device: str = Field(
        default="0",
        description="GPU device index, 'all', or 'none'",
    )
    timeout_seconds: int = Field(default=3600, ge=60, le=86400)
    # K8s-specific
    node_selector: dict[str, str] = Field(default_factory=dict)
    namespace: str = Field(default="orchestra")
    k8s_job_ttl_seconds: int = Field(default=300)


class MemoryConfig(BaseModel):
    embedding_model: str = Field(default="nomic-embed-text")
    embedding_url: str = Field(default="http://localhost:11434")
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    enabled: bool = Field(default=True)


class SupabaseConfig(BaseModel):
    url_env: str = Field(default="SUPABASE_URL")
    key_env: str = Field(default="SUPABASE_ANON_KEY")
    session_table: str = Field(default="conductor_sessions")
    experiments_table: str = Field(default="conductor_experiments")
    memories_table: str = Field(default="conductor_memories")

    def get_url(self) -> str:
        url = os.environ.get(self.url_env)
        if not url:
            raise EnvironmentError(f"Environment variable '{self.url_env}' is not set.")
        return url

    def get_key(self) -> str:
        key = os.environ.get(self.key_env)
        if not key:
            raise EnvironmentError(f"Environment variable '{self.key_env}' is not set.")
        return key


class ProgramConfig(BaseModel):
    path: str = Field(default="program.md", description="Path relative to workspace_dir")
    train_script: str = Field(default="train.py")
    eval_script: str = Field(default="evaluate.py")
    results_file: str = Field(default="results.json")
    datasets_dir: str = Field(default="~/.orchestra/datasets")

    @property
    def datasets_path(self) -> Path:
        return Path(self.datasets_dir).expanduser().resolve()


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class ConductorConfig(BaseModel):
    session: SessionConfig
    llm: LLMConfig
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    supabase: SupabaseConfig = Field(default_factory=SupabaseConfig)
    program: ProgramConfig = Field(default_factory=ProgramConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConductorConfig":
        """Load and validate a conductor_config.yaml file."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def validate_environment(self) -> list[str]:
        """
        Check that all required environment variables are set.
        Returns a list of error messages (empty = all good).
        """
        errors: list[str] = []
        # LLM key (skip for local providers)
        if self.llm.provider not in ("lmstudio",):
            if not os.environ.get(self.llm.api_key_env):
                errors.append(
                    f"LLM API key not set: export {self.llm.api_key_env}=<your-key>"
                )
        # Supabase
        if not os.environ.get(self.supabase.url_env):
            errors.append(f"Supabase URL not set: export {self.supabase.url_env}=<url>")
        if not os.environ.get(self.supabase.key_env):
            errors.append(f"Supabase key not set: export {self.supabase.key_env}=<key>")
        return errors

    def summary(self) -> str:
        """Return a human-readable summary for --dry-run output."""
        lines = [
            f"Session:      {self.session.name}",
            f"Dataset:      {self.session.dataset_id}",
            f"Workspace:    {self.session.workspace_path}",
            f"Branch:       {self.session.branch}",
            f"Max iters:    {self.session.max_iterations}",
            f"Target:       {self.session.target_metric} ≤ {self.session.target_value or 'N/A'}",
            f"Keep if Δ ≤:  {self.session.keep_threshold}",
            f"LLM:          {self.llm.provider} / {self.llm.model}",
            f"Runner:       {self.runner.type} / {self.runner.image}",
            f"GPU:          {self.runner.gpu_device}",
            f"Memory:       {'enabled' if self.memory.enabled else 'disabled'} (top-{self.memory.top_k})",
        ]
        return "\n".join(lines)
