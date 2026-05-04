from __future__ import annotations

import re
from typing import Iterable

from .models import Candidate, Chunk


TOPIC_HEADER_RE = re.compile(
    r"^(?:#{2,6})\s*(?:[^\w\u4e00-\u9fff]*)?"
    r"(?:\u8bfe\u9898|topic)\s*"
    r"(?:[\[\(\uFF08]?\s*[0-9A-Za-z\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+\s*[\]\)\uFF09]?)?"
    r"\s*[\uFF1A:\-]\s*(?P<title>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
LOGIC_LABEL_RE = re.compile(
    r"^\s*[-*]?\s*(?:\*\*|__)?"
    r"(?:\u63a8\u6f14\u903b\u8f91|\u7814\u7a76\u903b\u8f91|\u903b\u8f91\u8bf4\u660e|derivation logic|rationale)"
    r"(?:\*\*|__)?\s*[\uFF1A:]\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
STEP_SECTION_LABEL_RE = re.compile(
    r"^\s*[-*]?\s*(?:\*\*|__)?"
    r"(?:\u5177\u4f53\u5b9e\u65bd\u6b65\u9aa4|\u5b9e\u65bd\u6b65\u9aa4|\u7814\u7a76\u6b65\u9aa4|implementation steps|steps)"
    r"(?:\*\*|__)?\s*[\uFF1A:]?\s*$",
    re.IGNORECASE,
)
STEP_RE = re.compile(
    r"^\s*(?P<index>\d+)[\.\)\u3001\uFF09]\s*"
    r"(?:(?:\*\*|__)?[^:*\uFF1A]+(?:\*\*|__)?\s*[\uFF1A:]\s*)?"
    r"(?P<text>.+?)\s*$"
)


def normalize_topic(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
    return normalized


def infer_source_type(chunk: Chunk) -> str:
    section_text = " ".join(chunk.section_path).lower()
    chunk_text = chunk.chunk_text.lower()
    if any(token in section_text for token in ("future", "outlook", "limitation")):
        return "explicit_future_work"
    if any(token in chunk_text for token in ("future work", "limitation", "open question")):
        return "explicit_future_work"
    if chunk.priority == "high":
        return "high_value_inference"
    if chunk.priority == "medium":
        return "medium_value_inference"
    return "general_inference"


def split_topic_blocks(raw_text: str) -> list[str]:
    starts = list(TOPIC_HEADER_RE.finditer(raw_text))
    if not starts:
        return []

    blocks: list[str] = []
    for index, match in enumerate(starts):
        start = match.start()
        end = starts[index + 1].start() if index + 1 < len(starts) else len(raw_text)
        block = raw_text[start:end].strip()
        if block:
            blocks.append(block)
    return blocks


def validate_candidate(topic: str, logic: str, steps: list[str]) -> list[str]:
    issues: list[str] = []
    if not topic.strip():
        issues.append("missing_topic")
    if not logic.strip():
        issues.append("missing_derivation_logic")
    if len(steps) != 3:
        issues.append("invalid_step_count")
    if re.search(r"\bchunk_\d+\b", topic, re.IGNORECASE):
        issues.append("placeholder_topic_reference")

    vague_markers = {
        "\u5f85\u8865\u5145",
        "\u7565",
        "\u540c\u4e0a",
        "\u6682\u65e0",
        "n/a",
        "tbd",
        "future work",
    }
    if any(step.strip().lower() in vague_markers for step in steps):
        issues.append("vague_steps")
    return issues


def parse_prompt_a_response(
    raw_text: str,
    chunk: Chunk,
    run_index: int,
    temperature: float,
    candidate_start_index: int,
) -> tuple[list[Candidate], int, dict[str, int | bool]]:
    candidates: list[Candidate] = []
    next_index = candidate_start_index
    blocks = split_topic_blocks(raw_text)

    for block in blocks:
        topic = _extract_topic_from_block(block)
        logic = _extract_logic(block)
        steps = _extract_steps(block)
        notes = validate_candidate(topic, logic, steps)

        candidate = Candidate(
            candidate_id=f"cand_{next_index:06d}",
            paper_id=chunk.paper_id,
            chunk_id=chunk.chunk_id,
            generator_model="deepseek-v3.2",
            run_index=run_index,
            temperature=temperature,
            topic=topic,
            normalized_topic=normalize_topic(topic),
            derivation_logic=logic,
            implementation_steps=steps,
            source_type=infer_source_type(chunk),
            source_chunk_id=chunk.chunk_id,
            evidence_section_path=chunk.section_path.copy(),
            raw_markdown_block=block,
            raw_output=raw_text,
            valid_flag=not notes,
            validation_notes=notes,
        )
        candidates.append(candidate)
        next_index += 1

    diagnostics = {
        "block_count": len(blocks),
        "parsed_count": len(candidates),
        "empty_parse": bool(raw_text.strip()) and len(candidates) == 0,
    }
    return candidates, next_index, diagnostics


def candidate_dicts(candidates: Iterable[Candidate]) -> list[dict]:
    return [candidate.to_dict() for candidate in candidates]


def _extract_topic_from_block(block: str) -> str:
    match = TOPIC_HEADER_RE.search(block)
    return match.group("title").strip() if match else ""


def _extract_logic(block: str) -> str:
    for line in block.splitlines():
        match = LOGIC_LABEL_RE.match(line)
        if match:
            return _clean_inline_text(match.group("value"))
    return ""


def _extract_steps(block: str) -> list[str]:
    lines = [line.rstrip() for line in block.splitlines()]
    step_section_index = next(
        (index for index, line in enumerate(lines) if STEP_SECTION_LABEL_RE.match(line)),
        None,
    )
    numbered_lines = (
        lines[step_section_index + 1 :] if step_section_index is not None else lines
    )

    steps = _extract_numbered_steps(numbered_lines)
    if not steps and step_section_index is not None:
        steps = _extract_numbered_steps(lines)
    return steps


def _extract_numbered_steps(lines: list[str]) -> list[str]:
    steps: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*]\s*", "", line)
        match = STEP_RE.match(line)
        if not match:
            continue
        steps.append(_clean_inline_text(match.group("text")))
    return steps


def _clean_inline_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -")
