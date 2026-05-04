from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


def find_default_prompt_a_path(repo_root: Path) -> Path:
    prompt_dir = repo_root / "prompts"
    matches = sorted(prompt_dir.glob("Prompt_A_*.md"))
    if not matches:
        return prompt_dir / "Prompt_A_候选生成.md"
    return matches[0]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    remote_model: str
    api_key_env: str
    max_tokens: int
    temperature: float
    top_p: float
    extra_payload: dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "deepseek": ModelConfig(
        key="deepseek",
        remote_model="deepseek-ai/deepseek-v3.2",
        api_key_env="SCIQUERY_DEEPSEEK_API_KEY",
        max_tokens=2048,
        temperature=0.9,
        top_p=0.95,
        extra_payload={},
    ),
    "nemotron": ModelConfig(
        key="nemotron",
        remote_model="nvidia/nemotron-3-super-120b-a12b",
        api_key_env="SCIQUERY_NEMOTRON_API_KEY",
        max_tokens=3072,
        temperature=0.2,
        top_p=0.9,
        extra_payload={},
    ),
    "glm5": ModelConfig(
        key="glm5",
        remote_model="z-ai/glm5",
        api_key_env="SCIQUERY_GLM5_API_KEY",
        max_tokens=768,
        temperature=0.0,
        top_p=1.0,
        extra_payload={},
    ),
    "qwen122": ModelConfig(
        key="qwen122",
        remote_model="qwen/qwen3.5-122b-a10b",
        api_key_env="SCIQUERY_QWEN122_API_KEY",
        max_tokens=768,
        temperature=0.0,
        top_p=0.95,
        extra_payload={},
    ),
    "qwen397": ModelConfig(
        key="qwen397",
        remote_model="qwen/qwen3.5-397b-a17b",
        api_key_env="SCIQUERY_QWEN397_API_KEY",
        max_tokens=768,
        temperature=0.0,
        top_p=0.95,
        extra_payload={
            "top_k": 20,
            "presence_penalty": 0,
            "repetition_penalty": 1,
        },
    ),
}


@dataclass
class PipelineConfig:
    repo_root: Path
    mode: str = "live"
    artifacts_root: Path | None = None
    prompt_a_path: Path | None = None
    candidate_runs: int = 3
    candidate_temperatures: tuple[float, ...] = (0.7, 0.9, 1.1)
    max_candidates_per_call: int = 2
    noise_min_chars: int = 40
    chunk_min_chars: int = 180
    chunk_target_max_chars: int = 1000
    chunk_hard_max_chars: int = 1300
    cluster_threshold: float = 0.82
    cluster_batch_size: int = 50
    merge_group_size: int = 10
    merge_skip_prior_invalid_groups: bool = True
    request_timeout_seconds: int = 240
    retry_count: int = 2
    retry_backoff_seconds: int = 8
    resume_enabled: bool = True
    heartbeat_chunk_interval: int = 5
    heartbeat_score_interval: int = 5

    def __post_init__(self) -> None:
        if self.artifacts_root is None:
            self.artifacts_root = self.repo_root / "output"
        if self.prompt_a_path is None:
            self.prompt_a_path = find_default_prompt_a_path(self.repo_root)
        self.artifacts_root = Path(self.artifacts_root)
        self.prompt_a_path = Path(self.prompt_a_path)
        self.noise_min_chars = max(1, int(self.noise_min_chars))
        self.chunk_min_chars = max(self.noise_min_chars, int(self.chunk_min_chars))
        self.chunk_target_max_chars = max(self.chunk_min_chars, int(self.chunk_target_max_chars))
        self.chunk_hard_max_chars = max(self.chunk_target_max_chars, int(self.chunk_hard_max_chars))
        self.retry_backoff_seconds = max(1, int(self.retry_backoff_seconds))
        self.heartbeat_chunk_interval = max(1, int(self.heartbeat_chunk_interval))
        self.heartbeat_score_interval = max(1, int(self.heartbeat_score_interval))


JUDGE_MODEL_KEYS = ("glm5", "qwen122", "qwen397")
