from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig, find_default_prompt_a_path
from .pipeline import PipelineRunner


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SciQuery full-scan pipeline.")
    parser.add_argument("--input", required=True, help="Path to the input markdown file.")
    parser.add_argument("--paper-id", help="Paper identifier. Defaults to markdown stem.")
    parser.add_argument(
        "--mode",
        choices=("live", "mock"),
        default="live",
        help="Use live NVIDIA APIs or local mock responses.",
    )
    parser.add_argument(
        "--output-root",
        "--artifacts-root",
        dest="artifacts_root",
        default=str(repo_root / "output"),
        help="Pipeline output root directory.",
    )
    parser.add_argument(
        "--prompt-a",
        default=str(find_default_prompt_a_path(repo_root)),
        help="Prompt A template path.",
    )
    parser.add_argument(
        "--candidate-runs",
        type=int,
        default=3,
        help="How many Prompt A calls to run per chunk.",
    )
    parser.add_argument(
        "--noise-min-chars",
        type=int,
        default=40,
        help="Paragraphs shorter than this are treated as noise and dropped.",
    )
    parser.add_argument(
        "--chunk-min-chars",
        type=int,
        default=180,
        help="Chunks shorter than this are merged with neighboring chunks when possible.",
    )
    parser.add_argument(
        "--chunk-target-max-chars",
        type=int,
        default=1000,
        help="Preferred maximum chunk length before sentence-based splitting.",
    )
    parser.add_argument(
        "--chunk-hard-max-chars",
        type=int,
        default=1300,
        help="Absolute maximum chunk length before fallback soft/hard splitting.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for clustering.",
    )
    parser.add_argument(
        "--cluster-batch-size",
        type=int,
        default=50,
        help="Batch size for multi-round clustering.",
    )
    parser.add_argument(
        "--merge-group-size",
        type=int,
        default=10,
        help="How many cluster items to send per Nemotron merge call.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=240,
        help="Read timeout for each NVIDIA API request.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=int,
        default=8,
        help="Base sleep interval between retry attempts.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable stage-level resume and regenerate outputs from scratch.",
    )
    parser.add_argument(
        "--heartbeat-score-interval",
        type=int,
        default=5,
        help="Write run_meta/heartbeat every N completed topics during judge scoring.",
    )
    parser.add_argument(
        "--heartbeat-chunk-interval",
        type=int,
        default=5,
        help="Write run_meta/heartbeat every N completed chunks during candidate generation.",
    )
    return parser


def main(code_root: Path) -> None:
    repo_root = code_root.parent
    parser = build_parser(repo_root)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    paper_id = args.paper_id or input_path.stem

    config = PipelineConfig(
        repo_root=repo_root,
        mode=args.mode,
        artifacts_root=Path(args.artifacts_root).resolve(),
        prompt_a_path=Path(args.prompt_a).resolve(),
        candidate_runs=args.candidate_runs,
        noise_min_chars=args.noise_min_chars,
        chunk_min_chars=args.chunk_min_chars,
        chunk_target_max_chars=args.chunk_target_max_chars,
        chunk_hard_max_chars=args.chunk_hard_max_chars,
        cluster_threshold=args.cluster_threshold,
        cluster_batch_size=args.cluster_batch_size,
        merge_group_size=args.merge_group_size,
        request_timeout_seconds=args.request_timeout_seconds,
        retry_backoff_seconds=args.retry_backoff_seconds,
        resume_enabled=not args.no_resume,
        heartbeat_chunk_interval=args.heartbeat_chunk_interval,
        heartbeat_score_interval=args.heartbeat_score_interval,
    )

    runner = PipelineRunner(config)
    runner.run(input_path=input_path, paper_id=paper_id)
