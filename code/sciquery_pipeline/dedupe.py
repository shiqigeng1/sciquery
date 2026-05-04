from __future__ import annotations

from .models import Candidate


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Candidate] = []

    for candidate in candidates:
        if not candidate.valid_flag:
            continue
        key = (
            candidate.chunk_id,
            candidate.normalized_topic,
            candidate.derivation_logic.strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    return deduped
