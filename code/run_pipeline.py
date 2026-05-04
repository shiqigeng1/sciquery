from __future__ import annotations

import re
import sys
from pathlib import Path

from sciquery_pipeline.cli import main as cli_main
from sciquery_pipeline.config import PipelineConfig, find_default_prompt_a_path
from sciquery_pipeline.files import ensure_dir, read_json, read_jsonl, write_json
from sciquery_pipeline.models import Candidate, Chunk, MergedTopic, TopicCluster, TopicScore
from sciquery_pipeline.pipeline import PipelineRunner
from sciquery_pipeline.progress import ConsoleProgressReporter
from sciquery_pipeline.response_parsers import aggregate_scores


CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent

# Edit parameters here for the no-argument workflow:
MODE = "live"
INPUT_PATH = REPO_ROOT / "input" / "md" / "review_xiaolingcui.md"
OUTPUT_ROOT = REPO_ROOT / "output"
PROMPT_A_PATH = find_default_prompt_a_path(REPO_ROOT)
PAPER_ID = "review_xiaolingcui_live_02"
PARTIAL_TOPICS_PATH = OUTPUT_ROOT / PAPER_ID / "08_partial_topics_round3.json"
CANDIDATE_RUNS = 3
NOISE_MIN_CHARS = 40
CHUNK_MIN_CHARS = 180
CHUNK_TARGET_MAX_CHARS = 1000
CHUNK_HARD_MAX_CHARS = 1300
CLUSTER_THRESHOLD = 0.82
CLUSTER_BATCH_SIZE = 50
MERGE_GROUP_SIZE = 3
MERGE_SKIP_PRIOR_INVALID_GROUPS = True
RESUME_ENABLED = True
HEARTBEAT_CHUNK_INTERVAL = 5
HEARTBEAT_SCORE_INTERVAL = 5
REQUEST_TIMEOUT_SECONDS = 600
RETRY_BACKOFF_SECONDS = 8


def build_embedded_config() -> PipelineConfig:
    return PipelineConfig(
        repo_root=REPO_ROOT,
        mode=MODE,
        artifacts_root=OUTPUT_ROOT,
        prompt_a_path=PROMPT_A_PATH,
        candidate_runs=CANDIDATE_RUNS,
        noise_min_chars=NOISE_MIN_CHARS,
        chunk_min_chars=CHUNK_MIN_CHARS,
        chunk_target_max_chars=CHUNK_TARGET_MAX_CHARS,
        chunk_hard_max_chars=CHUNK_HARD_MAX_CHARS,
        cluster_threshold=CLUSTER_THRESHOLD,
        cluster_batch_size=CLUSTER_BATCH_SIZE,
        merge_group_size=MERGE_GROUP_SIZE,
        merge_skip_prior_invalid_groups=MERGE_SKIP_PRIOR_INVALID_GROUPS,
        request_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        retry_backoff_seconds=RETRY_BACKOFF_SECONDS,
        resume_enabled=RESUME_ENABLED,
        heartbeat_chunk_interval=HEARTBEAT_CHUNK_INTERVAL,
        heartbeat_score_interval=HEARTBEAT_SCORE_INTERVAL,
    )


def run_with_embedded_config() -> None:
    config = build_embedded_config()
    runner = PipelineRunner(config)
    runner.run(input_path=INPUT_PATH, paper_id=PAPER_ID or INPUT_PATH.stem)


def _build_final_topics_payload(
    merged_topics: list[MergedTopic],
    scores: list[TopicScore],
) -> list[dict]:
    aggregates = aggregate_scores(merged_topics, scores)
    aggregates_by_id = {aggregate.topic_id: aggregate for aggregate in aggregates}
    scores_by_topic: dict[str, list[TopicScore]] = {}
    for score in scores:
        scores_by_topic.setdefault(score.topic_id, []).append(score)

    final_topics: list[dict] = []
    for topic in merged_topics:
        aggregate = aggregates_by_id.get(topic.topic_id)
        if aggregate is None or aggregate.status == "rejected":
            continue
        final_topics.append(
            {
                "topic_id": topic.topic_id,
                "title": topic.title,
                "merged_derivation_logic": topic.merged_derivation_logic,
                "implementation_steps": topic.implementation_steps,
                "evidence_quotes": topic.evidence_quotes,
                "source_chunk_ids": topic.source_chunk_ids,
                "judges": [score.to_dict() for score in scores_by_topic.get(topic.topic_id, [])],
                "aggregate_score": aggregate.to_dict(),
                "status": aggregate.status,
            }
        )
    return final_topics


def _build_unique_partial_topic_id(
    *,
    source_round: object,
    source_group: object,
    original_topic_id: str,
    fallback_index: int,
) -> str:
    round_part = str(source_round).strip() or "x"
    group_part = str(source_group).strip() or "x"
    topic_part = original_topic_id.strip() or f"topic_{fallback_index:04d}"
    return f"r{round_part}_g{group_part}_{topic_part}"


def _strip_partial_topic_prefix(topic_id: str) -> str:
    text = topic_id.strip()
    pattern = re.compile(r"^(?:r[^_]+_g[^_]+_)+")
    return pattern.sub("", text)


def _normalize_partial_topics_file(partial_topics_path: Path) -> dict:
    payload = read_json(partial_topics_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid partial topics payload: {partial_topics_path}")

    changed = False
    seen_ids: dict[str, int] = {}
    for index, item in enumerate(payload.get("topics") or [], start=1):
        if not isinstance(item, dict):
            continue
        topic_payload = item.get("topic")
        if not isinstance(topic_payload, dict):
            continue
        original_topic_id = _strip_partial_topic_prefix(
            str(
            item.get("original_topic_id") or topic_payload.get("topic_id", "")
            )
        ).strip()
        unique_topic_id = _build_unique_partial_topic_id(
            source_round=item.get("source_round"),
            source_group=item.get("source_group"),
            original_topic_id=original_topic_id,
            fallback_index=index,
        )
        occurrence = seen_ids.get(unique_topic_id, 0) + 1
        seen_ids[unique_topic_id] = occurrence
        if occurrence > 1:
            unique_topic_id = f"{unique_topic_id}__{occurrence}"
        if original_topic_id != unique_topic_id:
            item["original_topic_id"] = original_topic_id
            topic_payload["topic_id"] = unique_topic_id
            changed = True

    if changed:
        write_json(partial_topics_path, payload)
    return payload


def _load_partial_merged_topics(
    partial_topics_path: Path,
) -> tuple[dict, list[MergedTopic], dict[str, dict]]:
    payload = _normalize_partial_topics_file(partial_topics_path)

    merged_topics: list[MergedTopic] = []
    source_meta_by_topic_id: dict[str, dict] = {}
    for index, item in enumerate(payload.get("topics") or [], start=1):
        if not isinstance(item, dict):
            continue
        topic_payload = item.get("topic") if isinstance(item.get("topic"), dict) else item
        if not isinstance(topic_payload, dict):
            continue
        merged_topic = MergedTopic.from_dict(topic_payload)
        if not merged_topic.topic_id:
            merged_topic.topic_id = f"partial_topic_{index:04d}"
        merged_topics.append(merged_topic)
        source_meta_by_topic_id[merged_topic.topic_id] = {
            "source_round": item.get("source_round"),
            "source_group": item.get("source_group"),
            "source_group_file": item.get("source_group_file"),
            "original_topic_id": item.get("original_topic_id"),
        }
    return payload, merged_topics, source_meta_by_topic_id


def run_from_merge_with_embedded_config() -> None:
    config = build_embedded_config()
    runner = PipelineRunner(config)
    progress = ConsoleProgressReporter()
    paper_id = PAPER_ID or INPUT_PATH.stem
    artifacts_dir = ensure_dir(config.artifacts_root / paper_id)
    merge_rounds_dir = ensure_dir(artifacts_dir / "05_merge_rounds")

    chunks_path = artifacts_dir / "01_chunks.jsonl"
    deduped_path = artifacts_dir / "03_candidates_deduped.jsonl"
    clusters_path = artifacts_dir / "04_clusters.jsonl"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunk artifact: {chunks_path}")
    if not deduped_path.exists():
        raise FileNotFoundError(f"Missing deduped candidate artifact: {deduped_path}")
    if not clusters_path.exists():
        raise FileNotFoundError(f"Missing cluster artifact: {clusters_path}")

    chunks = [Chunk.from_dict(row) for row in read_jsonl(chunks_path)]
    deduped_candidates = [Candidate.from_dict(row) for row in read_jsonl(deduped_path)]
    clusters = [TopicCluster.from_dict(row) for row in read_jsonl(clusters_path)]

    progress.stage(4, 6, "Nemotron 归并", f"{len(clusters)} clusters")
    candidates_by_id = {
        candidate.candidate_id: candidate.to_dict() for candidate in deduped_candidates
    }
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    merged_topics, _ = runner._prepare_merged_topics(
        paper_id=paper_id,
        clusters=clusters,
        candidates_by_id=candidates_by_id,
        chunks_by_id=chunks_by_id,
        artifacts_dir=artifacts_dir,
        merge_rounds_dir=merge_rounds_dir,
        progress=progress,
    )

    progress.stage(5, 6, "三模型评分", f"{len(merged_topics)} topics")
    scores, _ = runner._prepare_scores(
        merged_topics=merged_topics,
        artifacts_dir=artifacts_dir,
        progress=progress,
    )

    progress.stage(6, 6, "聚合与落盘")
    final_topics = _build_final_topics_payload(merged_topics, scores)

    output_path = artifacts_dir / "08_final_topics.json"
    write_json(
        output_path,
        {
            "paper_id": paper_id,
            "final_topics": final_topics,
        },
    )
    progress.info(f"最终保留 {len(final_topics)} 个 topic")
    progress.info(f"输出文件: {output_path}")


def run_from_partial_score_with_embedded_config() -> None:
    config = build_embedded_config()
    runner = PipelineRunner(config)
    progress = ConsoleProgressReporter()
    artifacts_dir = ensure_dir(config.artifacts_root / (PAPER_ID or INPUT_PATH.stem))
    partial_topics_path = PARTIAL_TOPICS_PATH

    if not partial_topics_path.exists():
        raise FileNotFoundError(f"Missing partial topics artifact: {partial_topics_path}")

    partial_payload, merged_topics, source_meta_by_topic_id = _load_partial_merged_topics(partial_topics_path)
    paper_id = str(partial_payload.get("paper_id") or PAPER_ID or INPUT_PATH.stem)
    partial_source = str(partial_payload.get("source") or partial_topics_path.name)
    partial_round = partial_payload.get("round")

    progress.stage(5, 6, "三模型评分", f"{len(merged_topics)} topics from {partial_source}")
    scores, _ = runner._prepare_scores(
        merged_topics=merged_topics,
        artifacts_dir=artifacts_dir,
        progress=progress,
    )

    progress.stage(6, 6, "聚合与落盘")
    final_topics = _build_final_topics_payload(merged_topics, scores)
    for topic in final_topics:
        topic.update(source_meta_by_topic_id.get(str(topic.get("topic_id")), {}))

    output_path = artifacts_dir / "08_partial_topics_scored.json"
    write_json(
        output_path,
        {
            "paper_id": paper_id,
            "partial_source_path": str(partial_topics_path),
            "partial_source": partial_source,
            "partial_round": partial_round,
            "input_topic_count": len(merged_topics),
            "final_topics": final_topics,
        },
    )
    progress.info(f"基于 partial topics 保留 {len(final_topics)} 个 topic")
    progress.info(f"输出文件: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "from-merge":
        run_from_merge_with_embedded_config()
    elif len(sys.argv) > 1 and sys.argv[1] == "from-partial-score":
        run_from_partial_score_with_embedded_config()
    elif len(sys.argv) > 1:
        cli_main(CODE_ROOT)
    else:
        run_with_embedded_config()
