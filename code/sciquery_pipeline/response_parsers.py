from __future__ import annotations

import json
import re
from statistics import median
from typing import Any

from .config import JUDGE_MODEL_KEYS
from .json_utils import extract_json_payload
from .models import MergedTopic, TopicAggregate, TopicScore


def _clean_steps(steps: list[str]) -> list[str]:
    return [step.strip() for step in steps if isinstance(step, str) and step.strip()]


def _unique_nonempty(items: list[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        text = item.strip() if isinstance(item, str) else ""
        if not text or text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [text]
            return _coerce_string_list(parsed)
        return [text]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _coerce_merge_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        items = payload.get("merged_topics", [])
    else:
        items = payload

    if isinstance(items, list):
        return items
    if isinstance(items, dict):
        return [items]
    if isinstance(items, str):
        text = items.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [text]
            return _coerce_merge_items(parsed)
        return [text]
    return []


def _coerce_merge_item(item: Any, index: int) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                return parsed[0]
        return {
            "topic_id": f"topic_{index:04d}",
            "title": text,
            "normalized_title": text,
            "merged_derivation_logic": "",
            "implementation_steps": [],
            "source_candidate_ids": [],
            "source_chunk_ids": [],
            "evidence_quotes": [],
            "merge_notes": "coerced_from_string_item",
        }
    return None


def _resolve_source_chunk_contexts(
    item: dict,
    available_source_chunks: list[dict],
) -> list[dict]:
    if not available_source_chunks:
        return []

    by_candidate_id: dict[str, dict] = {}
    by_chunk_id: dict[str, list[dict]] = {}
    for source_chunk in available_source_chunks:
        candidate_id = str(source_chunk.get("candidate_id", "")).strip()
        chunk_id = str(source_chunk.get("source_chunk_id", "")).strip()
        if candidate_id and candidate_id not in by_candidate_id:
            by_candidate_id[candidate_id] = source_chunk
        if chunk_id:
            by_chunk_id.setdefault(chunk_id, []).append(source_chunk)

    resolved: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()

    for candidate_id in item.get("source_candidate_ids", []):
        source_chunk = by_candidate_id.get(str(candidate_id).strip())
        if source_chunk is None:
            continue
        key = (
            str(source_chunk.get("candidate_id", "")).strip(),
            str(source_chunk.get("source_chunk_id", "")).strip(),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        resolved.append(dict(source_chunk))

    for chunk_id in item.get("source_chunk_ids", []):
        for source_chunk in by_chunk_id.get(str(chunk_id).strip(), []):
            key = (
                str(source_chunk.get("candidate_id", "")).strip(),
                str(source_chunk.get("source_chunk_id", "")).strip(),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            resolved.append(dict(source_chunk))

    return resolved


def parse_merged_topics(
    raw_text: str,
    paper_id: str,
    merge_model: str,
    available_source_chunks: list[dict] | None = None,
) -> list[MergedTopic]:
    payload = extract_json_payload(raw_text)
    items = _coerce_merge_items(payload)

    merged_topics: list[MergedTopic] = []
    source_chunks = available_source_chunks or []
    for index, raw_item in enumerate(items, start=1):
        item = _coerce_merge_item(raw_item, index)
        if item is None:
            continue
        title = str(item.get("title", "")).strip()
        normalized_title = str(item.get("normalized_title", "")).strip() or title
        merged_topics.append(
            MergedTopic(
                topic_id=str(item.get("topic_id") or f"topic_{index:04d}"),
                paper_id=str(item.get("paper_id") or paper_id),
                merge_model=merge_model,
                title=title,
                normalized_title=normalized_title,
                merged_derivation_logic=str(item.get("merged_derivation_logic", "")).strip(),
                implementation_steps=_clean_steps(_coerce_string_list(item.get("implementation_steps", []))),
                source_candidate_ids=_unique_nonempty(_coerce_string_list(item.get("source_candidate_ids", []))),
                source_chunk_ids=_unique_nonempty(_coerce_string_list(item.get("source_chunk_ids", []))),
                evidence_quotes=_unique_nonempty(_coerce_string_list(item.get("evidence_quotes", []))),
                merge_notes=str(item.get("merge_notes", "")).strip(),
                raw_output=raw_text,
                source_chunk_contexts=_resolve_source_chunk_contexts(item, source_chunks),
            )
        )
    return merged_topics


def parse_score(raw_text: str, topic_id: str, judge_model: str) -> TopicScore:
    text = raw_text.strip()
    kv_patterns = {
        "topic_id": re.compile(r"(?im)^\s*TOPIC_ID\s*:\s*(.*?)\s*$"),
        "specificity": re.compile(r"(?im)^\s*SPECIFICITY\s*:\s*(.*?)\s*$"),
        "feasibility": re.compile(r"(?im)^\s*FEASIBILITY\s*:\s*(.*?)\s*$"),
        "evidence_alignment": re.compile(r"(?im)^\s*EVIDENCE_ALIGNMENT\s*:\s*(.*?)\s*$"),
        "novelty": re.compile(r"(?im)^\s*NOVELTY\s*:\s*(.*?)\s*$"),
        "value": re.compile(r"(?im)^\s*VALUE\s*:\s*(.*?)\s*$"),
        "comment": re.compile(r"(?im)^\s*COMMENT\s*:\s*(.*?)\s*$"),
    }

    kv_matches = {key: pattern.search(text) for key, pattern in kv_patterns.items()}
    if all(match is not None for match in kv_matches.values()):
        parsed_topic_id = kv_matches["topic_id"].group(1).strip() or topic_id
        specificity = round(float(kv_matches["specificity"].group(1).strip()), 1)
        feasibility = round(float(kv_matches["feasibility"].group(1).strip()), 1)
        evidence_alignment = round(float(kv_matches["evidence_alignment"].group(1).strip()), 1)
        novelty = round(float(kv_matches["novelty"].group(1).strip()), 1)
        value = round(float(kv_matches["value"].group(1).strip()), 1)
        comment = kv_matches["comment"].group(1).strip()
    else:
        payload = extract_json_payload(raw_text)
        if "scores" in payload:
            scores = payload["scores"]
        else:
            scores = payload

        parsed_topic_id = str(payload.get("topic_id") or topic_id)
        specificity = round(float(scores.get("specificity", payload.get("specificity", 0.0))), 1)
        feasibility = round(float(scores.get("feasibility", payload.get("feasibility", 0.0))), 1)
        evidence_alignment = round(
            float(scores.get("evidence_alignment", payload.get("evidence_alignment", 0.0))),
            1,
        )
        novelty = round(float(scores.get("novelty", payload.get("novelty", 0.0))), 1)
        value = round(float(scores.get("value", payload.get("value", 0.0))), 1)
        comment = str(payload.get("comment", "")).strip()

    overall = (
        0.25 * specificity
        + 0.25 * feasibility
        + 0.20 * evidence_alignment
        + 0.15 * novelty
        + 0.15 * value
    )
    return TopicScore(
        topic_id=parsed_topic_id,
        judge_model=judge_model,
        specificity=specificity,
        feasibility=feasibility,
        evidence_alignment=evidence_alignment,
        novelty=novelty,
        value=value,
        overall=round(float(overall), 2),
        comment=comment,
        raw_output=raw_text,
    )


def aggregate_scores(
    merged_topics: list[MergedTopic],
    scores: list[TopicScore],
) -> list[TopicAggregate]:
    score_map: dict[str, list[TopicScore]] = {}
    for score in scores:
        score_map.setdefault(score.topic_id, []).append(score)

    aggregates: list[TopicAggregate] = []
    expected_judge_count = len(JUDGE_MODEL_KEYS)
    for topic in merged_topics:
        topic_scores = score_map.get(topic.topic_id, [])
        if len(topic_scores) < expected_judge_count:
            continue

        specificities = [score.specificity for score in topic_scores]
        feasibilities = [score.feasibility for score in topic_scores]
        evidence_scores = [score.evidence_alignment for score in topic_scores]
        novelties = [score.novelty for score in topic_scores]
        values = [score.value for score in topic_scores]
        overalls = [score.overall for score in topic_scores]

        veto_reason = None
        if sum(score.evidence_alignment < 3.0 for score in topic_scores) >= 2:
            veto_reason = "low_evidence_alignment"
        elif sum(score.feasibility < 3.0 for score in topic_scores) >= 2:
            veto_reason = "low_feasibility"
        elif len(topic.evidence_quotes) == 0:
            veto_reason = "missing_evidence_quotes"
        elif len(topic.implementation_steps) != 3:
            veto_reason = "invalid_step_count"

        final_overall = round(float(median(overalls)), 2)
        overall_spread = round(max(overalls) - min(overalls), 2)
        evidence_spread = round(max(evidence_scores) - min(evidence_scores), 2)

        status = "rejected"
        if veto_reason is None:
            if (
                final_overall >= 4.2
                and median(evidence_scores) >= 4.0
                and median(feasibilities) >= 4.0
                and overall_spread <= 0.8
            ):
                status = "accepted"
            elif final_overall >= 3.5 or overall_spread > 0.8 or evidence_spread > 1.0:
                status = "review"

        aggregates.append(
            TopicAggregate(
                topic_id=topic.topic_id,
                final_specificity=round(float(median(specificities)), 2),
                final_feasibility=round(float(median(feasibilities)), 2),
                final_evidence_alignment=round(float(median(evidence_scores)), 2),
                final_novelty=round(float(median(novelties)), 2),
                final_value=round(float(median(values)), 2),
                final_overall=final_overall,
                overall_spread=overall_spread,
                evidence_spread=evidence_spread,
                status=status,
                veto_reason=veto_reason,
            )
        )
    return aggregates
