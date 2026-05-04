from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    paper_id: str
    chunk_id: str
    section_path: list[str]
    chunk_index: int
    chunk_type: str
    priority: str
    char_count: int
    chunk_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(**payload)


@dataclass
class Candidate:
    candidate_id: str
    paper_id: str
    chunk_id: str
    generator_model: str
    run_index: int
    temperature: float
    topic: str
    normalized_topic: str
    derivation_logic: str
    implementation_steps: list[str]
    source_type: str
    source_chunk_id: str
    evidence_section_path: list[str]
    raw_markdown_block: str
    raw_output: str
    valid_flag: bool
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Candidate":
        return cls(**payload)


@dataclass
class TopicCluster:
    cluster_id: str
    paper_id: str
    candidate_ids: list[str]
    cluster_size: int
    representative_topic: str
    representative_logic: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopicCluster":
        return cls(**payload)


@dataclass
class MergedTopic:
    topic_id: str
    paper_id: str
    merge_model: str
    title: str
    normalized_title: str
    merged_derivation_logic: str
    implementation_steps: list[str]
    source_candidate_ids: list[str]
    source_chunk_ids: list[str]
    evidence_quotes: list[str]
    merge_notes: str
    raw_output: str
    source_chunk_contexts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MergedTopic":
        return cls(**payload)


@dataclass
class TopicScore:
    topic_id: str
    judge_model: str
    specificity: float
    feasibility: float
    evidence_alignment: float
    novelty: float
    value: float
    overall: float
    comment: str
    raw_output: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopicScore":
        return cls(**payload)


@dataclass
class TopicAggregate:
    topic_id: str
    final_specificity: float
    final_feasibility: float
    final_evidence_alignment: float
    final_novelty: float
    final_value: float
    final_overall: float
    overall_spread: float
    evidence_spread: float
    status: str
    veto_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TopicAggregate":
        return cls(**payload)


@dataclass
class LLMCallResult:
    content: str
    raw_response: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LLMCallResult":
        return cls(**payload)
