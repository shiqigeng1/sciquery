from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .models import Chunk


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
SENTENCE_SPLIT_RE = re.compile(
    r"(?<!Fig\.)(?<!Ref\.)(?<!Eq\.)(?<!Sec\.)(?<!Dr\.)(?<!Mr\.)(?<!Ms\.)(?<=[.!?])\s+"
)
SOFT_BREAK_RE = re.compile(
    r",\s+|;\s+|:\s+|\s+(?:which|where|while|when|that|and|but|or)\s+",
    re.IGNORECASE,
)

PUNCT_TRANSLATION = str.maketrans(
    {
        chr(8216): "'",
        chr(8217): "'",
        chr(8220): '"',
        chr(8221): '"',
        chr(8211): "-",
        chr(8212): "-",
        chr(160): " ",
    }
)

IGNORABLE_HEADINGS = {
    "review article",
    "open access",
    "springer",
    "acknowledgements",
    "funding",
    "data availability",
    "declarations",
    "competing interests",
    "authors contributions",
    "author contributions",
    "publisher s note",
}

BACK_MATTER_TOKENS = (
    "acknowledgements",
    "funding",
    "data availability",
    "declarations",
    "competing interests",
    "author contributions",
    "authors contributions",
    "publisher s note",
)


def normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(PUNCT_TRANSLATION)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def compact_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_for_match(text)).strip()


def infer_priority(section_path: list[str]) -> str:
    lowered = compact_for_match(" ".join(section_path))
    if any(
        token in lowered
        for token in (
            "summary",
            "outlook",
            "discussion",
            "conclusion",
            "abstract",
            "limitation",
            "future work",
        )
    ):
        return "high"
    if any(token in lowered for token in ("results", "methods", "method", "analysis")):
        return "medium"
    return "low"


def is_reference_section(section_path: list[str]) -> bool:
    lowered = compact_for_match(" ".join(section_path))
    return any(token in lowered for token in ("references", "bibliography", "citation"))


def is_ignorable_heading(title: str) -> bool:
    lowered = compact_for_match(title)
    if not lowered:
        return True
    if lowered in IGNORABLE_HEADINGS:
        return True
    if lowered.startswith("received "):
        return True
    return False


def is_image_or_caption_paragraph(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    non_image_lines = [line for line in lines if not IMAGE_LINE_RE.match(line)]
    if not non_image_lines:
        return True

    first_text_line = normalize_for_match(non_image_lines[0]).lstrip("* ").strip()
    if (
        first_text_line.startswith("fig.")
        or first_text_line.startswith("fig ")
        or first_text_line.startswith("figure ")
    ):
        return True

    if all(_is_panel_label(line) for line in non_image_lines) and len(non_image_lines) <= 2:
        return True
    return False


def is_correspondence_block(text: str) -> bool:
    normalized = normalize_for_match(text)
    if normalized.startswith("*correspondence:") or normalized.startswith("correspondence:"):
        return True
    if "@" in text and any(
        token in normalized for token in ("institute", "university", "laboratory", "college")
    ):
        return True
    return False


def is_license_block(text: str) -> bool:
    normalized = compact_for_match(text)
    return any(
        token in normalized
        for token in (
            "open access this article is licensed under",
            "creative commons attribution",
            "copyright holder",
            "copy of this licence",
            "copy of this license",
            "the author s 2024",
            "the author s",
        )
    )


def is_metadata_block(text: str, section_path: list[str]) -> bool:
    normalized = compact_for_match(text)
    section_normalized = compact_for_match(" ".join(section_path))

    if any(token in section_normalized for token in BACK_MATTER_TOKENS):
        return True
    if normalized.startswith("the work is supported by") or normalized.startswith(
        "this work is supported by"
    ):
        return True
    if "supporting information is available" in normalized:
        return True
    if "have no competing interests" in normalized or "no competing interests" in normalized:
        return True
    if "springer nature remains neutral with regard to jurisdictional claims" in normalized:
        return True
    if "supervised the project" in normalized and "wrote the manuscript" in normalized:
        return True
    if "received" in normalized and "accepted" in normalized and re.search(r"\b\d{4}\b", normalized):
        return True
    return False


def is_noise_paragraph(text: str, section_path: list[str], noise_min_chars: int) -> bool:
    stripped = re.sub(r"\s+", " ", text).strip()
    if not stripped:
        return True
    if is_reference_section(section_path):
        return True
    if is_image_or_caption_paragraph(text):
        return True
    if is_correspondence_block(stripped):
        return True
    if is_license_block(stripped):
        return True
    if is_metadata_block(stripped, section_path):
        return True

    token_count = len(re.findall(r"[A-Za-z]+|\d+", stripped))
    punctuation_count = len(re.findall(r"[,.;:()\[\]-]", stripped))

    if len(stripped) < noise_min_chars:
        return True
    if token_count > 0 and punctuation_count > token_count * 2 and re.search(r"\b\d{4}\b", stripped):
        return True
    return False


def normalize_paragraph(lines: list[str]) -> str:
    joined = "\n".join(line.rstrip() for line in lines).strip()
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined


def split_long_paragraph(text: str, target_max: int, hard_max: int) -> list[str]:
    cleaned = text.strip()
    if len(cleaned) <= target_max:
        return [cleaned]

    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(cleaned) if sentence.strip()]
    if len(sentences) <= 1:
        return _split_overlong_text(cleaned, target_max, hard_max)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        parts = _split_overlong_text(sentence, target_max, hard_max)
        for part in parts:
            projected = current_len + len(part) + (1 if current else 0)
            if current and projected > target_max:
                chunks.append(" ".join(current).strip())
                current = [part]
                current_len = len(part)
            else:
                current.append(part)
                current_len = projected

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def parse_markdown_chunks(
    markdown_text: str,
    paper_id: str,
    config: PipelineConfig,
) -> list[Chunk]:
    heading_stack: list[tuple[int, str]] = []
    paragraph_buffer: list[str] = []
    segment_rows: list[dict[str, Any]] = []

    def current_section_path() -> list[str]:
        return [title for _, title in heading_stack]

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        text = normalize_paragraph(paragraph_buffer)
        paragraph_buffer.clear()

        section_path = current_section_path()
        if is_noise_paragraph(text, section_path, config.noise_min_chars):
            return

        split_rows = [
            {
                "section_path": section_path.copy(),
                "chunk_text": split_text.strip(),
            }
            for split_text in split_long_paragraph(
                text,
                config.chunk_target_max_chars,
                config.chunk_hard_max_chars,
            )
            if split_text.strip()
        ]
        segment_rows.extend(split_rows)

    for raw_line in markdown_text.splitlines():
        stripped_line = raw_line.strip()
        heading_match = HEADING_RE.match(stripped_line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            if is_ignorable_heading(title):
                continue
            heading_stack[:] = [
                (existing_level, existing_title)
                for existing_level, existing_title in heading_stack
                if existing_level < level
            ]
            heading_stack.append((level, title))
            continue

        if not stripped_line:
            flush_paragraph()
            continue

        paragraph_buffer.append(raw_line)

    flush_paragraph()

    merged_rows = _merge_short_segments(
        segment_rows,
        min_chars=config.chunk_min_chars,
        hard_max=config.chunk_hard_max_chars,
    )

    chunks: list[Chunk] = []
    for chunk_index, row in enumerate(merged_rows, start=1):
        chunk_text = str(row["chunk_text"]).strip()
        section_path = list(row["section_path"])
        chunks.append(
            Chunk(
                paper_id=paper_id,
                chunk_id=f"chunk_{chunk_index:04d}",
                section_path=section_path,
                chunk_index=chunk_index,
                chunk_type="paragraph",
                priority=infer_priority(section_path),
                char_count=len(chunk_text),
                chunk_text=chunk_text,
            )
        )
    return chunks


def load_markdown_chunks(
    markdown_path: Path,
    paper_id: str,
    config: PipelineConfig,
) -> list[Chunk]:
    markdown_text = markdown_path.read_text(encoding="utf-8")
    return parse_markdown_chunks(markdown_text, paper_id, config)


def _split_overlong_text(text: str, target_max: int, hard_max: int) -> list[str]:
    remaining = text.strip()
    if not remaining:
        return []

    pieces: list[str] = []
    while len(remaining) > hard_max:
        split_at = _find_best_split_position(remaining, target_max, hard_max)
        head = remaining[:split_at].strip()
        if not head:
            break
        pieces.append(head)
        remaining = _trim_remaining_text(remaining[split_at:])

    while len(remaining) > target_max:
        split_at = _find_best_split_position(remaining, target_max, hard_max)
        if split_at >= len(remaining):
            break
        head = remaining[:split_at].strip()
        if not head:
            break
        pieces.append(head)
        remaining = _trim_remaining_text(remaining[split_at:])

    if remaining:
        pieces.append(remaining)
    return pieces


def _find_best_split_position(text: str, target_max: int, hard_max: int) -> int:
    lower_bound = max(1, min(target_max // 2, target_max - 200))
    upper_bound = min(len(text), hard_max)
    candidate_positions = [
        match.end()
        for match in SOFT_BREAK_RE.finditer(text[:upper_bound])
        if lower_bound <= match.end() <= upper_bound
    ]
    if candidate_positions:
        return min(candidate_positions, key=lambda position: abs(position - target_max))

    whitespace_candidates = [
        match.start()
        for match in re.finditer(r"\s+", text[:upper_bound])
        if lower_bound <= match.start() <= upper_bound
    ]
    if whitespace_candidates:
        return min(whitespace_candidates, key=lambda position: abs(position - target_max))

    return upper_bound


def _trim_remaining_text(text: str) -> str:
    return text.lstrip(" \t\r\n,;:")


def _merge_short_segments(
    segment_rows: list[dict[str, Any]],
    *,
    min_chars: int,
    hard_max: int,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in segment_rows]
    if min_chars <= 0:
        return rows

    changed = True
    while changed:
        changed = False
        merged_rows: list[dict[str, Any]] = []
        index = 0
        while index < len(rows):
            current = dict(rows[index])
            current_text = str(current["chunk_text"]).strip()

            if len(current_text) < min_chars:
                if (
                    merged_rows
                    and merged_rows[-1]["section_path"] == current["section_path"]
                    and len(str(merged_rows[-1]["chunk_text"])) + 2 + len(current_text) <= hard_max
                ):
                    merged_rows[-1]["chunk_text"] = (
                        str(merged_rows[-1]["chunk_text"]).rstrip() + "\n\n" + current_text
                    )
                    changed = True
                    index += 1
                    continue

                if (
                    index + 1 < len(rows)
                    and rows[index + 1]["section_path"] == current["section_path"]
                    and len(current_text) + 2 + len(str(rows[index + 1]["chunk_text"])) <= hard_max
                ):
                    current["chunk_text"] = current_text + "\n\n" + str(rows[index + 1]["chunk_text"]).strip()
                    merged_rows.append(current)
                    changed = True
                    index += 2
                    continue

            merged_rows.append(current)
            index += 1
        rows = merged_rows

    return rows


def _is_panel_label(text: str) -> bool:
    lowered = compact_for_match(text)
    if not lowered:
        return True
    if lowered in {"high low", "low high"}:
        return True
    return bool(re.fullmatch(r"[a-z]\d?", lowered))
