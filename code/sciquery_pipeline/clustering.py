from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .models import Candidate, TopicCluster


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class _ClusterSeed:
    member_ids: list[str]
    topic: str
    logic: str


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    total = float(len(tokens)) or 1.0
    for token in list(counts):
        counts[token] /= total
    return counts


def _idf(documents: list[list[str]]) -> dict[str, float]:
    total_docs = len(documents) or 1
    doc_frequency: dict[str, int] = {}
    for document in documents:
        for token in set(document):
            doc_frequency[token] = doc_frequency.get(token, 0) + 1
    return {
        token: math.log((1 + total_docs) / (1 + frequency)) + 1.0
        for token, frequency in doc_frequency.items()
    }


def _vectorize(documents: list[str]) -> list[dict[str, float]]:
    tokenized = [_tokenize(document) for document in documents]
    idf = _idf(tokenized)
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        tf = _tf(tokens)
        vector = {token: value * idf[token] for token, value in tf.items()}
        vectors.append(vector)
    return vectors


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    numerator = sum(left[token] * right.get(token, 0.0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _cluster_batch(
    seeds: list[_ClusterSeed],
    threshold: float,
) -> list[_ClusterSeed]:
    if not seeds:
        return []

    documents = [f"{seed.topic}\n{seed.logic}" for seed in seeds]
    vectors = _vectorize(documents)
    parent = list(range(len(seeds)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            if _cosine_similarity(vectors[left], vectors[right]) >= threshold:
                union(left, right)

    grouped: dict[int, list[_ClusterSeed]] = {}
    for index, seed in enumerate(seeds):
        grouped.setdefault(find(index), []).append(seed)

    clustered: list[_ClusterSeed] = []
    for group in grouped.values():
        member_ids: list[str] = []
        representative_topic = max(group, key=lambda item: len(item.topic)).topic
        representative_logic = max(group, key=lambda item: len(item.logic)).logic
        for seed in group:
            member_ids.extend(seed.member_ids)
        clustered.append(
            _ClusterSeed(
                member_ids=member_ids,
                topic=representative_topic,
                logic=representative_logic,
            )
        )
    return clustered


def cluster_candidates(
    candidates: list[Candidate],
    threshold: float,
    batch_size: int,
) -> tuple[list[TopicCluster], list[dict]]:
    if not candidates:
        return [], []

    seeds = [
        _ClusterSeed(
            member_ids=[candidate.candidate_id],
            topic=candidate.topic,
            logic=candidate.derivation_logic,
        )
        for candidate in candidates
    ]

    rounds: list[dict] = []
    round_index = 1

    while True:
        input_count = len(seeds)
        if input_count == 0:
            break

        if input_count <= batch_size:
            clustered = _cluster_batch(seeds, threshold)
            rounds.append(
                {
                    "round_index": round_index,
                    "mode": "global",
                    "input_count": input_count,
                    "output_count": len(clustered),
                    "clusters": [
                        {
                            "candidate_ids": seed.member_ids,
                            "representative_topic": seed.topic,
                            "representative_logic": seed.logic,
                        }
                        for seed in clustered
                    ],
                }
            )
            if len(clustered) == input_count:
                seeds = clustered
                break
            seeds = clustered
            round_index += 1
            continue

        next_seeds: list[_ClusterSeed] = []
        batch_records: list[dict] = []
        for batch_start in range(0, input_count, batch_size):
            batch = seeds[batch_start : batch_start + batch_size]
            clustered = _cluster_batch(batch, threshold)
            next_seeds.extend(clustered)
            batch_records.append(
                {
                    "batch_start": batch_start,
                    "input_count": len(batch),
                    "output_count": len(clustered),
                    "clusters": [
                        {
                            "candidate_ids": seed.member_ids,
                            "representative_topic": seed.topic,
                            "representative_logic": seed.logic,
                        }
                        for seed in clustered
                    ],
                }
            )

        rounds.append(
            {
                "round_index": round_index,
                "mode": "batched",
                "input_count": input_count,
                "output_count": len(next_seeds),
                "batches": batch_records,
            }
        )
        if len(next_seeds) == input_count:
            globally_clustered = _cluster_batch(next_seeds, threshold)
            rounds.append(
                {
                    "round_index": round_index + 1,
                    "mode": "global_after_batched_stall",
                    "input_count": len(next_seeds),
                    "output_count": len(globally_clustered),
                    "clusters": [
                        {
                            "candidate_ids": seed.member_ids,
                            "representative_topic": seed.topic,
                            "representative_logic": seed.logic,
                        }
                        for seed in globally_clustered
                    ],
                }
            )
            seeds = globally_clustered
            if len(globally_clustered) == input_count:
                break
            round_index += 2
            continue
        seeds = next_seeds
        round_index += 1

    clusters = [
        TopicCluster(
            cluster_id=f"cluster_{index:04d}",
            paper_id=candidates[0].paper_id,
            candidate_ids=seed.member_ids,
            cluster_size=len(seed.member_ids),
            representative_topic=seed.topic,
            representative_logic=seed.logic,
        )
        for index, seed in enumerate(seeds, start=1)
    ]
    return clusters, rounds
