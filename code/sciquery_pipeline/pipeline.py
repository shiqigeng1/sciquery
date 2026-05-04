from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .candidate_parser import parse_prompt_a_response
from .clustering import cluster_candidates
from .config import JUDGE_MODEL_KEYS, PipelineConfig
from .dedupe import dedupe_candidates
from .files import (
    append_jsonl,
    ensure_dir,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .llm import build_client
from .markdown_processing import load_markdown_chunks
from .models import Candidate, Chunk, MergedTopic, TopicCluster, TopicScore
from .progress import ConsoleProgressReporter
from .prompting import (
    build_judge_prompt,
    build_merge_prompt,
    cluster_to_merge_item,
    load_prompt_template,
    merged_topic_to_merge_item,
    render_prompt_a,
)
from .response_parsers import aggregate_scores, parse_merged_topics, parse_score


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.client = build_client(config)

    def run(self, input_path: Path, paper_id: str) -> None:
        progress = ConsoleProgressReporter()
        artifacts_dir = ensure_dir(self.config.artifacts_root / paper_id)
        cluster_rounds_dir = ensure_dir(artifacts_dir / "04_clusters_rounds")
        merge_rounds_dir = ensure_dir(artifacts_dir / "05_merge_rounds")

        prompt_a_template = load_prompt_template(self.config.prompt_a_path)
        prompt_a_hash = _sha256_text(prompt_a_template)
        candidate_run_count = len(self.config.candidate_temperatures[: self.config.candidate_runs])
        run_started_at = datetime.now().isoformat(timespec="seconds")
        resume_status: dict[str, dict[str, int | str | bool]] = {}
        counts = {
            "chunks": 0,
            "raw_candidate_calls": 0,
            "candidate_failures": 0,
            "parsed_candidates": 0,
            "deduped_candidates": 0,
            "clusters": 0,
            "merged_topics": 0,
            "scores": 0,
            "score_failures": 0,
            "final_topics": 0,
        }
        latest_stage_progress: dict[str, Any] | None = None
        current_stage = "starting"
        self._write_run_state(
            artifacts_dir=artifacts_dir,
            paper_id=paper_id,
            input_path=input_path,
            started_at=run_started_at,
            status="running",
            current_stage=current_stage,
            resume_status=resume_status,
            counts=counts,
        )

        def candidate_state_callback(stage_progress: dict[str, Any]) -> None:
            nonlocal latest_stage_progress
            latest_stage_progress = dict(stage_progress)
            self._write_run_state(
                artifacts_dir=artifacts_dir,
                paper_id=paper_id,
                input_path=input_path,
                started_at=run_started_at,
                status="running",
                current_stage=current_stage,
                resume_status=resume_status,
                counts={
                    **counts,
                    "chunks": len(chunks),
                    "raw_candidate_calls": int(latest_stage_progress["completed_calls"]),
                    "candidate_failures": int(latest_stage_progress.get("failed_calls", 0)),
                    "parsed_candidates": int(latest_stage_progress["parsed_candidates"]),
                },
                stage_progress=latest_stage_progress,
            )

        try:
            current_stage = "chunks"
            progress.stage(1, 6, "Markdown 切块", str(input_path))
            chunks, chunk_status = self._prepare_chunks(
                input_path=input_path,
                paper_id=paper_id,
                artifacts_dir=artifacts_dir,
                progress=progress,
            )
            resume_status["chunks"] = chunk_status
            counts["chunks"] = len(chunks)
            self._write_run_state(
                artifacts_dir=artifacts_dir,
                paper_id=paper_id,
                input_path=input_path,
                started_at=run_started_at,
                status="running",
                current_stage=current_stage,
                resume_status=resume_status,
                counts=counts,
            )

            current_stage = "candidates"
            progress.stage(
                2,
                6,
                "Prompt A 候选生成",
                f"{len(chunks)} chunks x {candidate_run_count} runs",
            )
            raw_candidate_rows, parsed_candidates, candidate_status = self._prepare_candidates(
                paper_id=paper_id,
                chunks=chunks,
                prompt_a_template=prompt_a_template,
                prompt_a_hash=prompt_a_hash,
                artifacts_dir=artifacts_dir,
                progress=progress,
                state_callback=candidate_state_callback,
            )
            resume_status["candidates"] = candidate_status
            counts["raw_candidate_calls"] = len(raw_candidate_rows)
            counts["candidate_failures"] = int(candidate_status.get("failed_items", 0))
            counts["parsed_candidates"] = len(parsed_candidates)
            latest_stage_progress = {
                "completed_chunks": len(chunks),
                "total_chunks": len(chunks),
                "completed_calls": len(raw_candidate_rows),
                "total_calls": len(chunks) * candidate_run_count,
                "failed_calls": counts["candidate_failures"],
                "parsed_candidates": len(parsed_candidates),
                "heartbeat_chunk_interval": self.config.heartbeat_chunk_interval,
                "last_completed_chunk_id": chunks[-1].chunk_id if chunks else None,
            }
            self._write_run_state(
                artifacts_dir=artifacts_dir,
                paper_id=paper_id,
                input_path=input_path,
                started_at=run_started_at,
                status="running",
                current_stage=current_stage,
                resume_status=resume_status,
                counts=counts,
                stage_progress=latest_stage_progress,
            )

            current_stage = "clusters"
            progress.stage(3, 6, "去重与聚类")
            deduped_candidates, clusters, cluster_status = self._prepare_clusters(
                paper_id=paper_id,
                parsed_candidates=parsed_candidates,
                artifacts_dir=artifacts_dir,
                cluster_rounds_dir=cluster_rounds_dir,
                progress=progress,
            )
            resume_status["clusters"] = cluster_status
            counts["deduped_candidates"] = len(deduped_candidates)
            counts["clusters"] = len(clusters)
            self._write_run_state(
                artifacts_dir=artifacts_dir,
                paper_id=paper_id,
                input_path=input_path,
                started_at=run_started_at,
                status="running",
                current_stage=current_stage,
                resume_status=resume_status,
                counts=counts,
            )

            candidates_by_id = {candidate.candidate_id: candidate.to_dict() for candidate in deduped_candidates}
            chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            current_stage = "merged_topics"
            progress.stage(4, 6, "Nemotron 归并")
            merged_topics, merge_status = self._prepare_merged_topics(
                paper_id=paper_id,
                clusters=clusters,
                candidates_by_id=candidates_by_id,
                chunks_by_id=chunks_by_id,
                artifacts_dir=artifacts_dir,
                merge_rounds_dir=merge_rounds_dir,
                progress=progress,
            )
            resume_status["merged_topics"] = merge_status
            counts["merged_topics"] = len(merged_topics)
            self._write_run_state(
                artifacts_dir=artifacts_dir,
                paper_id=paper_id,
                input_path=input_path,
                started_at=run_started_at,
                status="running",
                current_stage=current_stage,
                resume_status=resume_status,
                counts=counts,
            )

            def score_state_callback(stage_progress: dict[str, Any]) -> None:
                nonlocal latest_stage_progress
                latest_stage_progress = dict(stage_progress)
                self._write_run_state(
                    artifacts_dir=artifacts_dir,
                    paper_id=paper_id,
                    input_path=input_path,
                    started_at=run_started_at,
                    status="running",
                    current_stage=current_stage,
                    resume_status=resume_status,
                    counts={
                        **counts,
                        "scores": int(latest_stage_progress["completed_calls"]),
                        "score_failures": int(latest_stage_progress.get("failed_calls", 0)),
                    },
                    stage_progress=latest_stage_progress,
                )

            current_stage = "scores"
            progress.stage(5, 6, "三模型评分")
            scores, score_status = self._prepare_scores(
                merged_topics=merged_topics,
                artifacts_dir=artifacts_dir,
                progress=progress,
                state_callback=score_state_callback,
            )
            resume_status["scores"] = score_status
            counts["scores"] = len(scores)
            counts["score_failures"] = int(score_status.get("failed_items", 0))
            latest_stage_progress = {
                "completed_topics": len(merged_topics),
                "total_topics": len(merged_topics),
                "completed_calls": len(scores),
                "total_calls": len(merged_topics) * len(JUDGE_MODEL_KEYS),
                "failed_calls": counts["score_failures"],
                "heartbeat_score_interval": self.config.heartbeat_score_interval,
                "last_completed_topic_id": merged_topics[-1].topic_id if merged_topics else None,
            }
            self._write_run_state(
                artifacts_dir=artifacts_dir,
                paper_id=paper_id,
                input_path=input_path,
                started_at=run_started_at,
                status="running",
                current_stage=current_stage,
                resume_status=resume_status,
                counts=counts,
                stage_progress=latest_stage_progress,
            )

            current_stage = "finalizing"
            progress.stage(6, 6, "聚合与落盘")
            aggregates = aggregate_scores(merged_topics, scores)
            aggregates_by_id = {aggregate.topic_id: aggregate for aggregate in aggregates}
            scores_by_topic: dict[str, list[TopicScore]] = {}
            for score in scores:
                scores_by_topic.setdefault(score.topic_id, []).append(score)

            final_topics = []
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

            write_json(
                artifacts_dir / "08_final_topics.json",
                {
                    "paper_id": paper_id,
                    "final_topics": final_topics,
                },
            )
            counts["final_topics"] = len(final_topics)
            progress.info(f"最终保留 {len(final_topics)} 个 topic")
            progress.info(f"最终输出 {artifacts_dir / '08_final_topics.json'}")

            self._write_run_state(
                artifacts_dir=artifacts_dir,
                paper_id=paper_id,
                input_path=input_path,
                started_at=run_started_at,
                status="completed",
                current_stage="completed",
                resume_status=resume_status,
                counts=counts,
            )
        except Exception as error:
            self._write_run_state(
                artifacts_dir=artifacts_dir,
                paper_id=paper_id,
                input_path=input_path,
                started_at=run_started_at,
                status="failed",
                current_stage=current_stage,
                resume_status=resume_status,
                counts=counts,
                stage_progress=latest_stage_progress,
                error_message=str(error),
            )
            raise

    def _prepare_chunks(
        self,
        *,
        input_path: Path,
        paper_id: str,
        artifacts_dir: Path,
        progress: ConsoleProgressReporter,
    ) -> tuple[list[Chunk], dict[str, int | str | bool]]:
        chunks_path = artifacts_dir / "01_chunks.jsonl"
        meta_path = artifacts_dir / "01_chunks.meta.json"
        expected_meta = {
            "stage": "chunks",
            "paper_id": paper_id,
            "input_path": str(input_path.resolve()),
            "noise_min_chars": self.config.noise_min_chars,
            "chunk_min_chars": self.config.chunk_min_chars,
            "chunk_target_max_chars": self.config.chunk_target_max_chars,
            "chunk_hard_max_chars": self.config.chunk_hard_max_chars,
        }

        if self._can_resume([chunks_path], meta_path, expected_meta):
            chunks = [Chunk.from_dict(row) for row in read_jsonl(chunks_path)]
            progress.info(f"检测到 01_chunks.jsonl，直接复用 {len(chunks)} 个 chunk")
            return chunks, {"mode": "reused", "reused_items": len(chunks), "new_items": 0}

        chunks = load_markdown_chunks(input_path, paper_id, self.config)
        write_jsonl(chunks_path, [chunk.to_dict() for chunk in chunks])
        write_json(meta_path, expected_meta)
        progress.info(f"生成 {len(chunks)} 个 chunk")
        return chunks, {"mode": "generated", "reused_items": 0, "new_items": len(chunks)}

    def _prepare_candidates(
        self,
        *,
        paper_id: str,
        chunks: list[Chunk],
        prompt_a_template: str,
        prompt_a_hash: str,
        artifacts_dir: Path,
        progress: ConsoleProgressReporter,
        state_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[dict], list[Candidate], dict[str, int | str | bool]]:
        raw_path = artifacts_dir / "02_candidates_raw.jsonl"
        failed_path = artifacts_dir / "02_candidates_failed.jsonl"
        parse_failed_path = artifacts_dir / "02_candidates_parse_failed.jsonl"
        meta_path = artifacts_dir / "02_candidates_raw.meta.json"
        temperatures = self.config.candidate_temperatures[: self.config.candidate_runs]
        total_candidate_calls = len(chunks) * len(temperatures)
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        total_chunks = len(chunks)

        expected_meta = {
            "stage": "candidates",
            "paper_id": paper_id,
            "prompt_a_path": str(self.config.prompt_a_path.resolve()),
            "prompt_a_sha256": prompt_a_hash,
            "candidate_runs": self.config.candidate_runs,
            "candidate_temperatures": list(temperatures),
            "chunk_count": len(chunks),
            "chunks_sha256": _fingerprint_chunks(chunks),
        }

        reused_calls = 0
        new_calls = 0
        failed_calls = 0
        raw_candidate_rows: list[dict] = []
        parsed_candidates: list[Candidate] = []
        candidate_counter = 1
        completed_keys: set[tuple[str, int]] = set()

        if self._can_resume([raw_path], meta_path, expected_meta):
            raw_candidate_rows = read_jsonl(raw_path)
            raw_candidate_rows, duplicate_count = _dedupe_rows_by_key(
                raw_candidate_rows,
                lambda row: (str(row.get("chunk_id", "")), int(row.get("run_index", 0))),
            )
            if duplicate_count:
                write_jsonl(raw_path, raw_candidate_rows)
                progress.info(f"清理 02_candidates_raw.jsonl 中的 {duplicate_count} 条重复记录")
            (
                raw_candidate_rows,
                parsed_candidates,
                candidate_counter,
                invalid_rows,
            ) = self._restore_candidates_from_rows(raw_candidate_rows, chunk_map)
            if invalid_rows:
                write_jsonl(raw_path, raw_candidate_rows)
                for invalid_row in invalid_rows:
                    append_jsonl(parse_failed_path, invalid_row)
                progress.info(
                    f"Removed {len(invalid_rows)} stale candidate rows that no longer parse; they will be retried"
                )
            completed_keys = {
                (str(row["chunk_id"]), int(row["run_index"]))
                for row in raw_candidate_rows
                if str(row.get("chunk_id", "")) in chunk_map
            }
            reused_calls = len(completed_keys)
            progress.info(f"恢复 {reused_calls}/{total_candidate_calls} 次 DeepSeek 调用")
        else:
            write_json(meta_path, expected_meta)
            write_jsonl(raw_path, [])
            if failed_path.exists():
                failed_path.unlink()
            if parse_failed_path.exists():
                parse_failed_path.unlink()

        completed_chunk_ids = _completed_chunk_ids(chunks, completed_keys, len(temperatures))
        completed_chunk_count = len(completed_chunk_ids)

        if state_callback is not None:
            state_callback(
                {
                    "completed_chunks": completed_chunk_count,
                    "total_chunks": total_chunks,
                    "completed_calls": len(completed_keys),
                    "total_calls": total_candidate_calls,
                    "failed_calls": failed_calls,
                    "parsed_candidates": len(parsed_candidates),
                    "heartbeat_chunk_interval": self.config.heartbeat_chunk_interval,
                    "last_completed_chunk_id": None,
                }
            )

        candidate_call_index = reused_calls
        for chunk in chunks:
            for run_index, temperature in enumerate(temperatures, start=1):
                key = (chunk.chunk_id, run_index)
                if key in completed_keys:
                    continue
                candidate_call_index += 1
                new_calls += 1
                progress.update(
                    "DeepSeek 调用",
                    candidate_call_index,
                    total_candidate_calls,
                    f"{chunk.chunk_id} run {run_index}/{len(temperatures)}",
                )
                prompt = render_prompt_a(prompt_a_template, chunk)
                try:
                    call = self._complete_with_retry(
                        "deepseek",
                        prompt,
                        temperature=temperature,
                    )
                except Exception as error:
                    failed_calls += 1
                    append_jsonl(
                        failed_path,
                        {
                            "paper_id": paper_id,
                            "chunk_id": chunk.chunk_id,
                            "run_index": run_index,
                            "temperature": temperature,
                            "model_key": "deepseek",
                            "error_type": type(error).__name__,
                            "error_message": _summarize_error(error),
                            "failed_at": datetime.now().isoformat(timespec="seconds"),
                        },
                    )
                    progress.info(
                        f"DeepSeek failed {chunk.chunk_id} run {run_index}/{len(temperatures)} | {_summarize_error(error)}"
                    )
                    if state_callback is not None:
                        state_callback(
                            {
                                "completed_chunks": completed_chunk_count,
                                "total_chunks": total_chunks,
                                "completed_calls": len(completed_keys),
                                "total_calls": total_candidate_calls,
                                "failed_calls": failed_calls,
                                "parsed_candidates": len(parsed_candidates),
                                "heartbeat_chunk_interval": self.config.heartbeat_chunk_interval,
                                "last_completed_chunk_id": None,
                                "last_failed_chunk_id": chunk.chunk_id,
                                "last_error": _summarize_error(error),
                            }
                        )
                    continue
                row = {
                    "paper_id": paper_id,
                    "chunk_id": chunk.chunk_id,
                    "run_index": run_index,
                    "temperature": temperature,
                    "prompt": prompt,
                    "raw_output": call.content,
                    "usage": call.usage,
                }
                try:
                    parsed, candidate_counter, parse_diagnostics = parse_prompt_a_response(
                        call.content,
                        chunk,
                        run_index,
                        temperature,
                        candidate_counter,
                    )
                except Exception as error:
                    failed_calls += 1
                    append_jsonl(
                        parse_failed_path,
                        _candidate_parse_failure_row(
                            paper_id=paper_id,
                            chunk_id=chunk.chunk_id,
                            run_index=run_index,
                            temperature=temperature,
                            raw_output=call.content,
                            reason="parser_exception",
                            error_message=_summarize_error(error),
                        ),
                    )
                    progress.info(
                        f"DeepSeek parse failed {chunk.chunk_id} run {run_index}/{len(temperatures)} | {_summarize_error(error)}"
                    )
                    if state_callback is not None:
                        state_callback(
                            {
                                "completed_chunks": completed_chunk_count,
                                "total_chunks": total_chunks,
                                "completed_calls": len(completed_keys),
                                "total_calls": total_candidate_calls,
                                "failed_calls": failed_calls,
                                "parsed_candidates": len(parsed_candidates),
                                "heartbeat_chunk_interval": self.config.heartbeat_chunk_interval,
                                "last_completed_chunk_id": None,
                                "last_failed_chunk_id": chunk.chunk_id,
                                "last_error": _summarize_error(error),
                            }
                        )
                    continue

                if parse_diagnostics["empty_parse"]:
                    failed_calls += 1
                    append_jsonl(
                        parse_failed_path,
                        _candidate_parse_failure_row(
                            paper_id=paper_id,
                            chunk_id=chunk.chunk_id,
                            run_index=run_index,
                            temperature=temperature,
                            raw_output=call.content,
                            reason="empty_parse",
                            block_count=int(parse_diagnostics["block_count"]),
                        ),
                    )
                    progress.info(
                        f"DeepSeek parse empty {chunk.chunk_id} run {run_index}/{len(temperatures)} | no topic block matched"
                    )
                    if state_callback is not None:
                        state_callback(
                            {
                                "completed_chunks": completed_chunk_count,
                                "total_chunks": total_chunks,
                                "completed_calls": len(completed_keys),
                                "total_calls": total_candidate_calls,
                                "failed_calls": failed_calls,
                                "parsed_candidates": len(parsed_candidates),
                                "heartbeat_chunk_interval": self.config.heartbeat_chunk_interval,
                                "last_completed_chunk_id": None,
                                "last_failed_chunk_id": chunk.chunk_id,
                                "last_error": "empty_parse",
                            }
                        )
                    continue

                raw_candidate_rows.append(row)
                append_jsonl(raw_path, row)
                completed_keys.add(key)
                parsed_candidates.extend(parsed)

            if chunk.chunk_id not in completed_chunk_ids and all(
                (chunk.chunk_id, run_index) in completed_keys
                for run_index in range(1, len(temperatures) + 1)
            ):
                completed_chunk_ids.add(chunk.chunk_id)
                completed_chunk_count += 1
                if state_callback is not None and (
                    completed_chunk_count % self.config.heartbeat_chunk_interval == 0
                    or completed_chunk_count == total_chunks
                ):
                    state_callback(
                        {
                            "completed_chunks": completed_chunk_count,
                            "total_chunks": total_chunks,
                            "completed_calls": len(completed_keys),
                            "total_calls": total_candidate_calls,
                            "failed_calls": failed_calls,
                            "parsed_candidates": len(parsed_candidates),
                            "heartbeat_chunk_interval": self.config.heartbeat_chunk_interval,
                            "last_completed_chunk_id": chunk.chunk_id,
                        }
                    )

        progress.info(
            f"累计完成 {len(raw_candidate_rows)} 次调用，解析得到 {len(parsed_candidates)} 条原始候选"
        )
        progress.info(
            f"DeepSeek completed {len(raw_candidate_rows)} successful parsed calls, {failed_calls} failed calls, {len(parsed_candidates)} parsed candidates"
        )
        mode = "generated"
        if reused_calls and new_calls:
            mode = "resumed"
        elif reused_calls and not new_calls:
            mode = "reused"
        return raw_candidate_rows, parsed_candidates, {
            "mode": mode,
            "reused_items": reused_calls,
            "new_items": new_calls,
            "failed_items": failed_calls,
        }

    def _prepare_clusters(
        self,
        *,
        paper_id: str,
        parsed_candidates: list[Candidate],
        artifacts_dir: Path,
        cluster_rounds_dir: Path,
        progress: ConsoleProgressReporter,
    ) -> tuple[list[Candidate], list[TopicCluster], dict[str, int | str | bool]]:
        deduped_path = artifacts_dir / "03_candidates_deduped.jsonl"
        deduped_meta_path = artifacts_dir / "03_candidates_deduped.meta.json"
        clusters_path = artifacts_dir / "04_clusters.jsonl"
        clusters_meta_path = artifacts_dir / "04_clusters.meta.json"

        parsed_candidate_signature = _fingerprint_candidates(parsed_candidates)
        deduped_meta = {
            "stage": "deduped_candidates",
            "paper_id": paper_id,
            "parsed_candidate_count": len(parsed_candidates),
            "parsed_candidates_sha256": parsed_candidate_signature,
        }

        if self._can_resume([deduped_path], deduped_meta_path, deduped_meta):
            deduped_candidates = [Candidate.from_dict(row) for row in read_jsonl(deduped_path)]
            deduped_mode = "reused"
            progress.info(f"检测到 03_candidates_deduped.jsonl，直接复用 {len(deduped_candidates)} 条候选")
        else:
            deduped_candidates = dedupe_candidates(parsed_candidates)
            write_jsonl(deduped_path, [candidate.to_dict() for candidate in deduped_candidates])
            write_json(deduped_meta_path, deduped_meta)
            deduped_mode = "generated"
            progress.info(f"去重后保留 {len(deduped_candidates)} 条候选")

        cluster_meta = {
            "stage": "clusters",
            "paper_id": paper_id,
            "deduped_candidate_count": len(deduped_candidates),
            "deduped_candidates_sha256": _fingerprint_candidates(deduped_candidates),
            "cluster_threshold": self.config.cluster_threshold,
            "cluster_batch_size": self.config.cluster_batch_size,
            "cluster_algorithm_version": 2,
        }

        if self._can_resume([clusters_path], clusters_meta_path, cluster_meta):
            clusters = [TopicCluster.from_dict(row) for row in read_jsonl(clusters_path)]
            cluster_mode = "reused"
            progress.info(f"检测到 04_clusters.jsonl，直接复用 {len(clusters)} 个 cluster")
        else:
            self._clear_stage_json_files(cluster_rounds_dir)
            clusters, cluster_rounds = cluster_candidates(
                deduped_candidates,
                threshold=self.config.cluster_threshold,
                batch_size=self.config.cluster_batch_size,
            )
            write_jsonl(clusters_path, [cluster.to_dict() for cluster in clusters])
            write_json(clusters_meta_path, cluster_meta)
            for round_info in cluster_rounds:
                write_json(
                    cluster_rounds_dir / f"round_{round_info['round_index']:02d}.json",
                    round_info,
                )
            cluster_mode = "generated"
            progress.info(f"聚类得到 {len(clusters)} 个 cluster，共 {len(cluster_rounds)} 轮")

        return deduped_candidates, clusters, {
            "mode": "reused" if deduped_mode == "reused" and cluster_mode == "reused" else "generated",
            "reused_items": len(deduped_candidates) if deduped_mode == "reused" else 0,
            "new_items": len(deduped_candidates) if deduped_mode != "reused" else 0,
            "reused_clusters": len(clusters) if cluster_mode == "reused" else 0,
            "new_clusters": len(clusters) if cluster_mode != "reused" else 0,
        }

    def _prepare_merged_topics(
        self,
        *,
        paper_id: str,
        clusters: list[TopicCluster],
        candidates_by_id: dict[str, dict],
        chunks_by_id: dict[str, Chunk],
        artifacts_dir: Path,
        merge_rounds_dir: Path,
        progress: ConsoleProgressReporter,
    ) -> tuple[list[MergedTopic], dict[str, int | str | bool]]:
        merged_path = artifacts_dir / "06_merged_topics.json"
        meta_path = artifacts_dir / "06_merged_topics.meta.json"
        partial_meta_path = merge_rounds_dir / "_resume_meta.json"
        expected_meta = {
            "stage": "merged_topics",
            "paper_id": paper_id,
            "cluster_count": len(clusters),
            "clusters_sha256": _fingerprint_clusters(clusters),
            "merge_group_size": self.config.merge_group_size,
            "merge_source_context_version": 2,
        }

        if self._can_resume([merged_path], meta_path, expected_meta):
            merged_topics = [MergedTopic.from_dict(item) for item in read_json(merged_path)]
            progress.info(f"检测到 06_merged_topics.json，直接复用 {len(merged_topics)} 个 topic")
            return merged_topics, {"mode": "reused", "reused_items": len(merged_topics), "new_items": 0}

        partial_resume_enabled = self._meta_matches(partial_meta_path, expected_meta)
        if not partial_resume_enabled:
            legacy_group_paths = sorted(merge_rounds_dir.glob("round_*_group_*.json"))
            if self.config.resume_enabled and legacy_group_paths:
                partial_resume_enabled = True
                progress.info(
                    f"Detected {len(legacy_group_paths)} legacy merge group files; attempting compatible per-group reuse"
                )
            else:
                self._clear_stage_json_files(merge_rounds_dir)
        write_json(partial_meta_path, expected_meta)
        merged_topics = self._merge_topics(
            paper_id=paper_id,
            clusters=clusters,
            candidates_by_id=candidates_by_id,
            chunks_by_id=chunks_by_id,
            merge_rounds_dir=merge_rounds_dir,
            progress=progress,
            allow_legacy_round_reuse=partial_resume_enabled,
        )
        write_json(merged_path, [topic.to_dict() for topic in merged_topics])
        write_json(meta_path, expected_meta)
        progress.info(f"归并后得到 {len(merged_topics)} 个 topic")
        return merged_topics, {
            "mode": "resumed" if partial_resume_enabled else "generated",
            "reused_items": 0,
            "new_items": len(merged_topics),
        }

    def _prepare_scores(
        self,
        *,
        merged_topics: list[MergedTopic],
        artifacts_dir: Path,
        progress: ConsoleProgressReporter,
        state_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[TopicScore], dict[str, int | str | bool]]:
        scores_path = artifacts_dir / "07_topic_scores.jsonl"
        failed_path = artifacts_dir / "07_topic_scores_failed.jsonl"
        meta_path = artifacts_dir / "07_topic_scores.meta.json"
        expected_meta = {
            "stage": "topic_scores",
            "topic_count": len(merged_topics),
            "topics_sha256": _fingerprint_topics(merged_topics),
            "judge_model_keys": list(JUDGE_MODEL_KEYS),
        }

        total_score_calls = len(merged_topics) * len(JUDGE_MODEL_KEYS)
        reused_scores = 0
        new_scores = 0
        failed_scores = 0
        scores: list[TopicScore] = []
        completed_keys: set[tuple[str, str]] = set()

        if self._can_resume([scores_path], meta_path, expected_meta):
            score_rows = read_jsonl(scores_path)
            score_rows, duplicate_count = _dedupe_rows_by_key(
                score_rows,
                lambda row: (str(row.get("topic_id", "")), str(row.get("judge_model", ""))),
            )
            if duplicate_count:
                write_jsonl(scores_path, score_rows)
                progress.info(f"清理 07_topic_scores.jsonl 中的 {duplicate_count} 条重复记录")
            scores = [TopicScore.from_dict(row) for row in score_rows]
            completed_keys = {(score.topic_id, score.judge_model) for score in scores}
            reused_scores = len(completed_keys)
            progress.info(f"恢复 {reused_scores}/{total_score_calls} 条评分记录")
        else:
            write_json(meta_path, expected_meta)
            write_jsonl(scores_path, [])
            if failed_path.exists():
                failed_path.unlink()

        total_topics = len(merged_topics)
        completed_topic_ids = _completed_topic_ids(merged_topics, completed_keys, len(JUDGE_MODEL_KEYS))
        completed_topic_count = len(completed_topic_ids)

        if state_callback is not None:
            state_callback(
                {
                    "completed_topics": completed_topic_count,
                    "total_topics": total_topics,
                    "completed_calls": len(completed_keys),
                    "total_calls": total_score_calls,
                    "failed_calls": failed_scores,
                    "heartbeat_score_interval": self.config.heartbeat_score_interval,
                    "last_completed_topic_id": None,
                }
            )

        score_call_index = reused_scores
        with ThreadPoolExecutor(max_workers=len(JUDGE_MODEL_KEYS)) as executor:
            for topic in merged_topics:
                future_map = {}
                for judge_model_key in JUDGE_MODEL_KEYS:
                    key = (topic.topic_id, judge_model_key)
                    if key in completed_keys:
                        continue
                    score_call_index += 1
                    new_scores += 1
                    progress.update(
                        "三模型评分",
                        score_call_index,
                        total_score_calls,
                        f"{topic.topic_id} / {judge_model_key}",
                    )
                    prompt = build_judge_prompt(topic, judge_model_key)
                    future = executor.submit(self._complete_with_retry, judge_model_key, prompt)
                    future_map[future] = judge_model_key

                for future in as_completed(future_map):
                    judge_model_key = future_map[future]
                    key = (topic.topic_id, judge_model_key)
                    try:
                        call = future.result()
                    except Exception as error:
                        failed_scores += 1
                        append_jsonl(
                            failed_path,
                            {
                                "topic_id": topic.topic_id,
                                "judge_model": judge_model_key,
                                "model_key": judge_model_key,
                                "error_type": type(error).__name__,
                                "error_message": _summarize_error(error),
                                "failed_at": datetime.now().isoformat(timespec="seconds"),
                            },
                        )
                        progress.info(
                            f"Judge failed {topic.topic_id} / {judge_model_key} | {_summarize_error(error)}"
                        )
                        if state_callback is not None:
                            state_callback(
                                {
                                    "completed_topics": completed_topic_count,
                                    "total_topics": total_topics,
                                    "completed_calls": len(completed_keys),
                                    "total_calls": total_score_calls,
                                    "failed_calls": failed_scores,
                                    "heartbeat_score_interval": self.config.heartbeat_score_interval,
                                    "last_completed_topic_id": None,
                                    "last_failed_topic_id": topic.topic_id,
                                    "last_failed_judge_model": judge_model_key,
                                    "last_error": _summarize_error(error),
                                }
                            )
                        continue
                    try:
                        score = parse_score(call.content, topic.topic_id, judge_model_key)
                    except Exception as error:
                        failed_scores += 1
                        append_jsonl(
                            failed_path,
                            {
                                "topic_id": topic.topic_id,
                                "judge_model": judge_model_key,
                                "model_key": judge_model_key,
                                "error_type": type(error).__name__,
                                "error_message": _summarize_error(error),
                                "failed_at": datetime.now().isoformat(timespec="seconds"),
                                "failure_stage": "parse_score",
                            },
                        )
                        progress.info(
                            f"Judge parse failed {topic.topic_id} / {judge_model_key} | {_summarize_error(error)}"
                        )
                        if state_callback is not None:
                            state_callback(
                                {
                                    "completed_topics": completed_topic_count,
                                    "total_topics": total_topics,
                                    "completed_calls": len(completed_keys),
                                    "total_calls": total_score_calls,
                                    "failed_calls": failed_scores,
                                    "heartbeat_score_interval": self.config.heartbeat_score_interval,
                                    "last_completed_topic_id": None,
                                    "last_failed_topic_id": topic.topic_id,
                                    "last_failed_judge_model": judge_model_key,
                                    "last_error": _summarize_error(error),
                                }
                            )
                        continue
                    scores.append(score)
                    append_jsonl(scores_path, score.to_dict())
                    completed_keys.add(key)

                if topic.topic_id not in completed_topic_ids and all(
                    (topic.topic_id, judge_model_key) in completed_keys for judge_model_key in JUDGE_MODEL_KEYS
                ):
                    completed_topic_ids.add(topic.topic_id)
                    completed_topic_count += 1
                    if state_callback is not None and (
                        completed_topic_count % self.config.heartbeat_score_interval == 0
                        or completed_topic_count == total_topics
                    ):
                        state_callback(
                            {
                                "completed_topics": completed_topic_count,
                                "total_topics": total_topics,
                                "completed_calls": len(completed_keys),
                                "total_calls": total_score_calls,
                                "failed_calls": failed_scores,
                                "heartbeat_score_interval": self.config.heartbeat_score_interval,
                                "last_completed_topic_id": topic.topic_id,
                            }
                        )

        progress.info(f"累计写入 {len(scores)} 条评分记录")
        progress.info(f"Judges completed {len(scores)} successful scores, {failed_scores} failed scores")
        mode = "generated"
        if reused_scores and new_scores:
            mode = "resumed"
        elif reused_scores and not new_scores:
            mode = "reused"
        return scores, {
            "mode": mode,
            "reused_items": reused_scores,
            "new_items": new_scores,
            "failed_items": failed_scores,
        }

    def _restore_candidates_from_rows(
        self,
        rows: list[dict],
        chunks_by_id: dict[str, Chunk],
    ) -> tuple[list[dict], list[Candidate], int, list[dict]]:
        valid_rows: list[dict] = []
        parsed_candidates: list[Candidate] = []
        invalid_rows: list[dict] = []
        candidate_counter = 1
        for row in rows:
            chunk_id = str(row.get("chunk_id", ""))
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            try:
                parsed, candidate_counter, parse_diagnostics = parse_prompt_a_response(
                    str(row.get("raw_output", "")),
                    chunk,
                    int(row.get("run_index", 0)),
                    float(row.get("temperature", 0.0)),
                    candidate_counter,
                )
            except Exception as error:
                invalid_rows.append(
                    _candidate_parse_failure_row(
                        paper_id=str(row.get("paper_id", chunk.paper_id)),
                        chunk_id=chunk.chunk_id,
                        run_index=int(row.get("run_index", 0)),
                        temperature=float(row.get("temperature", 0.0)),
                        raw_output=str(row.get("raw_output", "")),
                        reason="parser_exception",
                        error_message=_summarize_error(error),
                    )
                )
                continue
            if parse_diagnostics["empty_parse"]:
                invalid_rows.append(
                    _candidate_parse_failure_row(
                        paper_id=str(row.get("paper_id", chunk.paper_id)),
                        chunk_id=chunk.chunk_id,
                        run_index=int(row.get("run_index", 0)),
                        temperature=float(row.get("temperature", 0.0)),
                        raw_output=str(row.get("raw_output", "")),
                        reason="empty_parse",
                        block_count=int(parse_diagnostics["block_count"]),
                    )
                )
                continue
            valid_rows.append(row)
            parsed_candidates.extend(parsed)
        return valid_rows, parsed_candidates, candidate_counter, invalid_rows

    def _complete_with_retry(
        self,
        model_key: str,
        prompt: str,
        *,
        temperature: float | None = None,
    ):
        last_error: Exception | None = None
        total_attempts = self.config.retry_count + 1
        for attempt_index in range(total_attempts):
            try:
                return self.client.complete(model_key, prompt, temperature=temperature)
            except Exception as error:  # pragma: no cover - retry path
                last_error = error
                if attempt_index >= total_attempts - 1:
                    break
                time.sleep(self.config.retry_backoff_seconds * (attempt_index + 1))
        assert last_error is not None
        raise last_error

    def _merge_topics(
        self,
        *,
        paper_id: str,
        clusters: list[TopicCluster],
        candidates_by_id: dict[str, dict],
        chunks_by_id: dict[str, Chunk],
        merge_rounds_dir: Path,
        progress: ConsoleProgressReporter,
        allow_legacy_round_reuse: bool = False,
    ) -> list[MergedTopic]:
        merge_items = [
            cluster_to_merge_item(cluster, candidates_by_id, chunks_by_id) for cluster in clusters
        ]
        current_topics: list[MergedTopic] = []
        round_index = 1

        while True:
            groups = list(_batched(merge_items, self.config.merge_group_size))
            partial_topics: list[MergedTopic] = []
            reused_group_count = 0
            executed_group_count = 0
            skipped_group_count = 0
            resume_frontier = _existing_merge_group_frontier(merge_rounds_dir, round_index)
            progress.info(f"归并 round {round_index}，共 {len(groups)} 组")
            for group_index, group in enumerate(groups, start=1):
                progress.update(
                    f"Nemotron round {round_index}",
                    group_index,
                    len(groups),
                    f"group {group_index}/{len(groups)}",
                )
                group_path = merge_rounds_dir / f"round_{round_index:02d}_group_{group_index:02d}.json"
                group_signature = _fingerprint_merge_items(group)
                cached_topics = self._load_cached_merge_group(
                    group_path=group_path,
                    paper_id=paper_id,
                    input_group_size=len(group),
                    group_signature=group_signature,
                    allow_legacy_round_reuse=allow_legacy_round_reuse,
                    available_source_chunks=_flatten_source_chunks(group),
                )
                if cached_topics is not None:
                    partial_topics.extend(cached_topics)
                    reused_group_count += 1
                    continue
                if (
                    self.config.merge_skip_prior_invalid_groups
                    and resume_frontier > 0
                    and group_index <= resume_frontier
                ):
                    skipped_group_count += 1
                    self._mark_merge_group_skipped(
                        group_path=group_path,
                        paper_id=paper_id,
                        round_index=round_index,
                        group_index=group_index,
                        input_group_size=len(group),
                        group_signature=group_signature,
                        reason="historical_invalid_before_frontier",
                    )
                    progress.info(
                        f"Nemotron skipped historical invalid group round {round_index} group {group_index}"
                    )
                    continue
                prompt = build_merge_prompt(paper_id, list(group))
                request_error: str | None = None
                try:
                    call = self._complete_with_retry("nemotron", prompt)
                except Exception as error:
                    call = None
                    request_error = _summarize_error(error)
                parse_error: str | None = None
                available_source_chunks = _flatten_source_chunks(group)
                try:
                    if call is None:
                        parsed_topics = []
                    else:
                        parsed_topics = parse_merged_topics(
                            call.content,
                            paper_id,
                            "nemotron",
                            available_source_chunks=available_source_chunks,
                        )
                    if not parsed_topics and request_error is None:
                        parse_error = "empty_parse"
                except Exception as error:
                    parsed_topics = []
                    parse_error = _summarize_error(error)

                if parsed_topics:
                    partial_topics.extend(parsed_topics)
                executed_group_count += 1
                group_payload = {
                    "paper_id": paper_id,
                    "round_index": round_index,
                    "group_index": group_index,
                    "input_group_size": len(group),
                    "group_signature": group_signature,
                    "raw_output": "" if call is None else call.content,
                    "parsed_topics": [topic.to_dict() for topic in parsed_topics],
                }
                if request_error is not None:
                    group_payload["request_error"] = request_error
                    group_payload["request_failed_at"] = datetime.now().isoformat(timespec="seconds")
                    progress.info(
                        f"Nemotron request failed round {round_index} group {group_index} | {request_error}"
                    )
                if parse_error is not None:
                    group_payload["parse_error"] = parse_error
                    group_payload["parse_failed_at"] = datetime.now().isoformat(timespec="seconds")
                    progress.info(
                        f"Nemotron parse failed round {round_index} group {group_index} | {parse_error}"
                    )
                write_json(group_path, group_payload)

            if reused_group_count:
                progress.info(
                    f"Nemotron round {round_index} reused {reused_group_count} groups and executed {executed_group_count} new groups"
                )
            if skipped_group_count:
                progress.info(
                    f"Nemotron round {round_index} skipped {skipped_group_count} historical invalid groups"
                )
            current_topics = partial_topics
            if len(groups) <= 1:
                break

            next_items = [merged_topic_to_merge_item(topic) for topic in partial_topics]
            if len(next_items) >= len(merge_items):
                break
            merge_items = next_items
            round_index += 1

        return current_topics

    def _write_run_state(
        self,
        *,
        artifacts_dir: Path,
        paper_id: str,
        input_path: Path,
        started_at: str,
        status: str,
        current_stage: str,
        resume_status: dict[str, dict[str, int | str | bool]],
        counts: dict[str, int],
        stage_progress: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        payload: dict[str, Any] = {
            "paper_id": paper_id,
            "input_path": str(input_path),
            "mode": self.config.mode,
            "status": status,
            "current_stage": current_stage,
            "started_at": started_at,
            "last_updated_at": now,
            "resume_enabled": self.config.resume_enabled,
            "resume": resume_status,
            "counts": dict(counts),
        }
        if status != "running":
            payload["finished_at"] = now
            payload["generated_at"] = now
        if stage_progress is not None:
            payload["stage_progress"] = stage_progress
        if error_message:
            payload["error"] = error_message
        write_json(artifacts_dir / "run_meta.json", payload)
        write_json(artifacts_dir / "run_heartbeat.json", payload)

    def _can_resume(
        self,
        data_paths: list[Path],
        meta_path: Path,
        expected_meta: dict,
    ) -> bool:
        if not self.config.resume_enabled:
            return False
        if not meta_path.exists():
            return False
        if any(not path.exists() for path in data_paths):
            return False
        try:
            existing_meta = read_json(meta_path)
        except Exception:
            return False
        return existing_meta == expected_meta

    def _meta_matches(self, meta_path: Path, expected_meta: dict) -> bool:
        if not self.config.resume_enabled:
            return False
        if not meta_path.exists():
            return False
        try:
            existing_meta = read_json(meta_path)
        except Exception:
            return False
        return existing_meta == expected_meta

    def _load_cached_merge_group(
        self,
        *,
        group_path: Path,
        paper_id: str,
        input_group_size: int,
        group_signature: str,
        allow_legacy_round_reuse: bool,
        available_source_chunks: list[dict] | None = None,
    ) -> list[MergedTopic] | None:
        if not self.config.resume_enabled or not group_path.exists():
            return None
        try:
            payload = read_json(group_path)
        except Exception:
            return None

        if str(payload.get("paper_id", "")) != paper_id:
            return None
        if int(payload.get("input_group_size", -1)) != input_group_size:
            return None

        stored_signature = str(payload.get("group_signature", "")).strip()
        if stored_signature:
            if stored_signature != group_signature:
                return None
        elif not allow_legacy_round_reuse:
            return None

        parsed_rows = payload.get("parsed_topics")
        if isinstance(parsed_rows, list) and parsed_rows:
            try:
                return [MergedTopic.from_dict(row) for row in parsed_rows]
            except Exception:
                return None

        raw_output = str(payload.get("raw_output", "")).strip()
        if not raw_output:
            return None
        try:
            reparsed_topics = parse_merged_topics(
                raw_output,
                paper_id,
                "nemotron",
                available_source_chunks=available_source_chunks or [],
            )
        except Exception:
            return None
        if not reparsed_topics:
            return None

        payload["parsed_topics"] = [topic.to_dict() for topic in reparsed_topics]
        if not payload.get("group_signature"):
            payload["group_signature"] = group_signature
        write_json(group_path, payload)
        return reparsed_topics

    def _mark_merge_group_skipped(
        self,
        *,
        group_path: Path,
        paper_id: str,
        round_index: int,
        group_index: int,
        input_group_size: int,
        group_signature: str,
        reason: str,
    ) -> None:
        payload: dict[str, Any]
        if group_path.exists():
            try:
                payload = read_json(group_path)
            except Exception:
                payload = {}
        else:
            payload = {}
        payload.update(
            {
                "paper_id": paper_id,
                "round_index": round_index,
                "group_index": group_index,
                "input_group_size": input_group_size,
                "group_signature": group_signature,
                "skipped": True,
                "skip_reason": reason,
                "skipped_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        payload.setdefault("raw_output", "")
        payload.setdefault("parsed_topics", [])
        write_json(group_path, payload)

    def _clear_stage_json_files(self, directory: Path) -> None:
        for path in directory.glob("*.json"):
            path.unlink()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fingerprint_lines(lines: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for line in lines:
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _fingerprint_chunks(chunks: list[Chunk]) -> str:
    return _fingerprint_lines(
        f"{chunk.chunk_id}|{' > '.join(chunk.section_path)}|{chunk.char_count}|{chunk.chunk_text}"
        for chunk in chunks
    )


def _fingerprint_candidates(candidates: list[Candidate]) -> str:
    return _fingerprint_lines(
        "|".join(
            [
                candidate.candidate_id,
                candidate.chunk_id,
                candidate.normalized_topic,
                candidate.derivation_logic,
                " / ".join(candidate.implementation_steps),
            ]
        )
        for candidate in candidates
    )


def _fingerprint_clusters(clusters: list[TopicCluster]) -> str:
    return _fingerprint_lines(
        "|".join(
            [
                cluster.cluster_id,
                ",".join(cluster.candidate_ids),
                cluster.representative_topic,
                cluster.representative_logic,
            ]
        )
        for cluster in clusters
    )


def _fingerprint_topics(topics: list[MergedTopic]) -> str:
    return _fingerprint_lines(
        "|".join(
            [
                topic.topic_id,
                topic.title,
                topic.normalized_title,
                topic.merged_derivation_logic,
                " / ".join(topic.implementation_steps),
                " / ".join(topic.evidence_quotes),
                ",".join(topic.source_chunk_ids),
            ]
        )
        for topic in topics
    )


def _fingerprint_merge_items(items: list[dict[str, Any]]) -> str:
    return _fingerprint_lines(
        json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items
    )


def _dedupe_rows_by_key(
    rows: list[dict],
    key_fn,
) -> tuple[list[dict], int]:
    seen: set[tuple[str, int] | tuple[str, str]] = set()
    unique_rows: list[dict] = []
    duplicate_count = 0
    for row in rows:
        key = key_fn(row)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows, duplicate_count


def _completed_chunk_ids(
    chunks: list[Chunk],
    completed_keys: set[tuple[str, int]],
    run_count: int,
) -> set[str]:
    completed_ids: set[str] = set()
    for chunk in chunks:
        if all((chunk.chunk_id, run_index) in completed_keys for run_index in range(1, run_count + 1)):
            completed_ids.add(chunk.chunk_id)
    return completed_ids


def _completed_topic_ids(
    topics: list[MergedTopic],
    completed_keys: set[tuple[str, str]],
    judge_count: int,
) -> set[str]:
    if judge_count != len(JUDGE_MODEL_KEYS):
        expected_judges = JUDGE_MODEL_KEYS[:judge_count]
    else:
        expected_judges = JUDGE_MODEL_KEYS

    completed_ids: set[str] = set()
    for topic in topics:
        if all((topic.topic_id, judge_model_key) in completed_keys for judge_model_key in expected_judges):
            completed_ids.add(topic.topic_id)
    return completed_ids


def _batched(items: list[dict], size: int):
    iterator = iter(items)
    while True:
        group = list(itertools.islice(iterator, size))
        if not group:
            return
        yield group


def _existing_merge_group_frontier(directory: Path, round_index: int) -> int:
    max_index = 0
    pattern = f"round_{round_index:02d}_group_*.json"
    for path in directory.glob(pattern):
        stem = path.stem
        try:
            group_index = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if group_index > max_index:
            max_index = group_index
    return max_index


def _flatten_source_chunks(merge_items: list[dict]) -> list[dict]:
    source_chunks: list[dict] = []
    for merge_item in merge_items:
        for source_chunk in merge_item.get("source_chunks", []):
            source_chunks.append(dict(source_chunk))
    return source_chunks


def _candidate_parse_failure_row(
    *,
    paper_id: str,
    chunk_id: str,
    run_index: int,
    temperature: float,
    raw_output: str,
    reason: str,
    error_message: str | None = None,
    block_count: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "paper_id": paper_id,
        "chunk_id": chunk_id,
        "run_index": run_index,
        "temperature": temperature,
        "failure_kind": reason,
        "failed_at": datetime.now().isoformat(timespec="seconds"),
        "raw_output_excerpt": raw_output[:800],
        "raw_output": raw_output,
    }
    if error_message:
        payload["error_message"] = error_message
    if block_count is not None:
        payload["parser_block_count"] = block_count
    return payload


def _summarize_error(error: Exception, max_length: int = 160) -> str:
    text = " ".join(str(error).split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
