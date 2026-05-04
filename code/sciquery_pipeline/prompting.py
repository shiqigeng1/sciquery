from __future__ import annotations

import json
from pathlib import Path

from .models import Chunk, MergedTopic, TopicCluster


JUDGE_PROFILES = {
    "glm5": {
        "focus": "重点严格评估证据一致性、推演是否紧贴原文、课题是否被过度延伸。",
        "extra_rules": [
            "如果 evidence_quotes 无法直接支持课题核心动作、对象或约束，evidence_alignment 不得高于 2.5。",
            "如果 merged_derivation_logic 引入了输入里没有出现的关键变量、机制或结论，evidence_alignment 不得高于 2.5。",
            "不能因为课题看起来合理就补足证据缺口；缺证据时必须严格扣分。",
        ],
    },
    "qwen122": {
        "focus": "重点严格评估课题是否足够具体、步骤是否可执行、结构是否完整。",
        "extra_rules": [
            "如果标题不是明确动作 + 研究对象 + 限定条件的短语，specificity 不得高于 2.5。",
            "如果 implementation_steps 出现空泛表达，如“进一步研究”“开展实验”“深入分析”，且没有对象、方法或产出，feasibility 不得高于 2.5。",
            "如果三步之间缺少连贯的执行顺序，specificity 和 feasibility 都应明显下调。",
        ],
    },
    "qwen397": {
        "focus": "重点严格评估课题的新颖性、研究价值、是否值得保留。",
        "extra_rules": [
            "如果课题只是把原文已有结论换一种说法，novelty 不得高于 3.0。",
            "如果课题缺少明确的新变量、新场景、新约束或新验证目标，novelty 不得高于 3.0。",
            "如果课题即使做成也难以带来理论、方法或应用层面的明显增量，value 不得高于 3.0。",
        ],
    },
}


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt_a(template: str, chunk: Chunk) -> str:
    section_path = " > ".join(chunk.section_path) if chunk.section_path else "(root)"
    return (
        template.replace("{{paper_id}}", chunk.paper_id)
        .replace("{{section_path}}", section_path)
        .replace("{{chunk_id}}", chunk.chunk_id)
        .replace("{{review_chunk}}", chunk.chunk_text)
    )


def build_merge_prompt(
    paper_id: str,
    merge_items: list[dict],
) -> str:
    input_payload = {
        "paper_id": paper_id,
        "merge_items": merge_items,
    }
    schema = {
        "merged_topics": [
            {
                "topic_id": "topic_xxxx",
                "paper_id": paper_id,
                "title": "具体明确动作的短语",
                "normalized_title": "规范化标题",
                "merged_derivation_logic": "合并后的推演逻辑",
                "implementation_steps": ["步骤1", "步骤2", "步骤3"],
                "source_candidate_ids": ["cand_0001"],
                "source_chunk_ids": ["chunk_0001"],
                "evidence_quotes": ["证据片段1"],
                "merge_notes": "归并说明",
            }
        ]
    }
    output_contract = {
        "rules": [
            "只能输出一个 JSON 对象。",
            "禁止输出 markdown、代码块、解释文字、前缀、后缀、道歉、说明或思考过程。",
            "顶层字段只能是 merged_topics。",
            "merged_topics 必须是数组。",
            "没有可保留结果时必须返回 {\"merged_topics\": []}。",
            "每个 merged topic 必须完整包含 schema 中全部字段，不得缺字段，不得新增字段。",
            "implementation_steps 必须严格为 3 条字符串。",
            "source_candidate_ids、source_chunk_ids、evidence_quotes 必须是数组。",
            "evidence_quotes 必须直接摘自 source_chunk_text，不得改写、概括或编造。",
        ]
    }

    return f"""任务：对同一篇论文的候选研究课题做保守归并，输出可直接进入评分阶段的 merged topics。

角色要求：
- 充当保守型学术归并器。
- 宁可拆分，不要误并。
- 不允许补写输入中不存在的变量、机制、结论或证据。

归并规则：
1. 只有当多个候选在“核心动作、研究对象、限定条件、验证目标”四项都高度一致时，才允许合并。
2. 只要存在显著差异，如变量不同、场景不同、对象不同、方法不同、验证目标不同，就必须保留为不同课题。
3. title 必须是具体明确的动作短语，不能写成泛主题、宽泛方向或空泛名词。
4. merged_derivation_logic 必须说明该课题弥补了什么缺口、推广了什么条件或迁移了什么场景。
5. implementation_steps 必须严格为 3 步，且每一步都必须是可执行动作，禁止空泛表述。
6. source_candidate_ids 和 source_chunk_ids 必须保留全部来源，去重但不得遗漏。
7. evidence_quotes 必须直接摘自 source_chunk_text，保留原句或原句片段。
8. 如果输入候选质量差，可以在 merge_notes 中标注，但不能借机虚构更完整的课题。
9. 无法确定是否属于同一课题时，默认不要合并。

输出协议：
1. 只能输出一个 JSON 对象。
2. 禁止输出 markdown、代码块、解释文字、前后缀文本、思考过程。
3. 顶层字段固定为 merged_topics。
4. 如果没有可保留结果，返回 {{"merged_topics": []}}。
5. 不得缺字段，不得新增字段，不得改变字段类型。

输出约束：
{json.dumps(output_contract, ensure_ascii=False, indent=2)}

输出 schema：
{json.dumps(schema, ensure_ascii=False, indent=2)}

INPUT_JSON:
{json.dumps(input_payload, ensure_ascii=False, indent=2)}
"""


def build_judge_prompt(topic: MergedTopic, judge_model_key: str) -> str:
    input_payload = {
        "topic_id": topic.topic_id,
        "paper_id": topic.paper_id,
        "title": topic.title,
        "merged_derivation_logic": topic.merged_derivation_logic,
        "implementation_steps": topic.implementation_steps,
        "source_chunk_ids": topic.source_chunk_ids,
        "evidence_quotes": topic.evidence_quotes,
    }
    output_example = "\n".join(
        [
            f"TOPIC_ID: {topic.topic_id}",
            "SPECIFICITY: 4.0",
            "FEASIBILITY: 4.0",
            "EVIDENCE_ALIGNMENT: 4.0",
            "NOVELTY: 4.0",
            "VALUE: 4.0",
            "COMMENT: 简短评语",
        ]
    )
    profile = JUDGE_PROFILES[judge_model_key]
    extra_rules = "\n".join(f"- {rule}" for rule in profile["extra_rules"])

    return f"""任务：对以下研究课题进行严格五维评分。

评分维度：
- specificity：具体性
- feasibility：可操作性
- evidence_alignment：证据一致性
- novelty：新颖性
- value：研究价值

评分总原则：
1. 只能基于输入中的 title、merged_derivation_logic、implementation_steps、evidence_quotes 评分。
2. 不允许用外部常识补证据、补步骤、补创新点。
3. 评分必须偏保守；信息不足时，优先下调分数而不是脑补。
4. 每个维度范围为 1.0 到 5.0，保留一位小数。
5. overall 必须严格使用以下公式：
   0.25 * specificity +
   0.25 * feasibility +
   0.20 * evidence_alignment +
   0.15 * novelty +
   0.15 * value
6. comment 必须是一句短评，指出最关键优点或缺陷。

分数锚点：
- 5.0：证据充分、定义清楚、步骤可直接执行、几乎没有明显短板。
- 3.0：基本成立，但存在明显缺口，如边界不清、步骤偏空、证据不足或创新有限。
- 1.0：高度空泛、缺少证据、难以执行或几乎没有研究增量。

统一扣分规则：
- implementation_steps 不是严格 3 步时，feasibility 不得高于 2.5。
- evidence_quotes 为空时，evidence_alignment 不得高于 2.0。
- title 不是具体动作短语而是宽泛主题时，specificity 不得高于 2.5。
- merged_derivation_logic 只是重复 title，没有说明缺口、推广或迁移逻辑时，specificity 和 novelty 都应下调。

该评审器关注重点：
- {profile["focus"]}

该评审器的额外硬规则：
{extra_rules}

输出协议：
1. 只能输出严格 7 行纯文本，顺序固定，不允许增减行数。
2. 每一行都必须采用 `字段名: 值` 的形式。
3. 字段名必须严格使用以下大写键名：
   `TOPIC_ID`
   `SPECIFICITY`
   `FEASIBILITY`
   `EVIDENCE_ALIGNMENT`
   `NOVELTY`
   `VALUE`
   `COMMENT`
4. 禁止输出 JSON、markdown、代码块、解释文字、前后缀文本、思考过程。
5. 5 个分数字段必须是 `1.0` 到 `5.0` 之间的数字，保留 1 位小数。
6. `TOPIC_ID` 必须原样返回。
7. `COMMENT` 必须是单行字符串。
8. 不要输出 `OVERALL`，该值由本地程序计算。

输出示例：
{output_example}

INPUT_JSON:
{json.dumps(input_payload, ensure_ascii=False, indent=2)}
"""


def cluster_to_merge_item(
    cluster: TopicCluster,
    candidates_by_id: dict[str, dict],
    chunks_by_id: dict[str, Chunk],
) -> dict:
    source_chunks: list[dict] = []
    for candidate_id in cluster.candidate_ids:
        candidate = candidates_by_id[candidate_id]
        chunk = chunks_by_id[candidate["chunk_id"]]
        source_chunks.append(
            {
                "candidate_id": candidate["candidate_id"],
                "topic": candidate["topic"],
                "derivation_logic": candidate["derivation_logic"],
                "implementation_steps": candidate["implementation_steps"],
                "source_chunk_id": chunk.chunk_id,
                "section_path": chunk.section_path,
                "source_chunk_text": _compress_source_chunk_text(chunk.chunk_text),
            }
        )

    return {
        "cluster_id": cluster.cluster_id,
        "representative_topic": cluster.representative_topic,
        "representative_logic": cluster.representative_logic,
        "candidate_ids": cluster.candidate_ids,
        "cluster_size": cluster.cluster_size,
        "source_chunks": source_chunks,
    }


def merged_topic_to_merge_item(topic: MergedTopic) -> dict:
    if topic.source_chunk_contexts:
        source_chunks = [
            {
                "candidate_id": str(source_chunk.get("candidate_id", "")).strip(),
                "topic": str(source_chunk.get("topic", topic.title)).strip() or topic.title,
                "derivation_logic": str(
                    source_chunk.get("derivation_logic", topic.merged_derivation_logic)
                ).strip()
                or topic.merged_derivation_logic,
                "implementation_steps": list(
                    source_chunk.get("implementation_steps", topic.implementation_steps)
                ),
                "source_chunk_id": str(source_chunk.get("source_chunk_id", "")).strip(),
                "section_path": list(source_chunk.get("section_path", [])),
                "source_chunk_text": _compress_source_chunk_text(
                    str(source_chunk.get("source_chunk_text", ""))
                ),
            }
            for source_chunk in topic.source_chunk_contexts
            if str(source_chunk.get("source_chunk_id", "")).strip()
        ]
    else:
        source_chunks = [
            {
                "candidate_id": candidate_id,
                "topic": topic.title,
                "derivation_logic": topic.merged_derivation_logic,
                "implementation_steps": topic.implementation_steps,
                "source_chunk_id": chunk_id,
                "section_path": [],
                "source_chunk_text": " ".join(topic.evidence_quotes),
            }
            for candidate_id, chunk_id in zip(
                topic.source_candidate_ids,
                topic.source_chunk_ids or ["chunk_unknown"] * len(topic.source_candidate_ids),
            )
        ]

    return {
        "cluster_id": topic.topic_id,
        "representative_topic": topic.title,
        "representative_logic": topic.merged_derivation_logic,
        "candidate_ids": topic.source_candidate_ids,
        "cluster_size": len(topic.source_candidate_ids),
        "source_chunks": source_chunks,
    }


def _compress_source_chunk_text(text: str, max_chars: int = 1800) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    separator = " ... "
    head_chars = int(max_chars * 0.65)
    tail_chars = max_chars - head_chars - len(separator)
    if tail_chars <= 0:
        return normalized[:max_chars]
    return f"{normalized[:head_chars].rstrip()}{separator}{normalized[-tail_chars:].lstrip()}"
