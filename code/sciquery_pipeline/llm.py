from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from .api_keys import MODEL_API_KEYS
from .config import MODEL_REGISTRY, NVIDIA_ENDPOINT, PipelineConfig
from .models import LLMCallResult


class LiveNvidiaClient:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def complete(
        self,
        model_key: str,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        model_config = MODEL_REGISTRY[model_key]
        api_key = os.getenv(model_config.api_key_env) or MODEL_API_KEYS.get(model_key, "")
        if not api_key:
            raise RuntimeError(
                f"Missing API key for model {model_key}. Set env var {model_config.api_key_env} or edit sciquery_pipeline/api_keys.py."
            )

        payload: dict[str, Any] = {
            "model": model_config.remote_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": max_tokens or model_config.max_tokens,
            "temperature": temperature if temperature is not None else model_config.temperature,
            "top_p": model_config.top_p,
        }
        payload.update(model_config.extra_payload)

        response = requests.post(
            NVIDIA_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            json=payload,
            timeout=(30, self.config.request_timeout_seconds),
        )
        response.raise_for_status()
        raw_response = response.json()
        content = _extract_message_text(raw_response)
        usage = raw_response.get("usage", {})
        return LLMCallResult(content=content, raw_response=raw_response, usage=usage)


class MockNvidiaClient:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def complete(
        self,
        model_key: str,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCallResult:
        if model_key == "deepseek":
            content = self._mock_prompt_a(prompt)
        elif model_key == "nemotron":
            content = self._mock_merge(prompt)
        else:
            content = self._mock_score(prompt, model_key)
        return LLMCallResult(content=content, raw_response={"mock": True}, usage={})

    def _mock_prompt_a(self, prompt: str) -> str:
        chunk_id_match = re.search(r"分块编号：(.+)", prompt)
        chunk_id = chunk_id_match.group(1).strip() if chunk_id_match else "chunk_mock"
        excerpt_match = re.search(r"\n\n(.+)$", prompt, re.DOTALL)
        excerpt = excerpt_match.group(1).strip()[:60] if excerpt_match else "原文片段"
        return f"""### 📌 课题 1：引入可控参数扰动分析{chunk_id}相关模型稳定性

- **推演逻辑**：原文提到“{excerpt}”，说明现有框架仍有变量扩展空间，可进一步分析参数扰动对模型稳定性的影响。
- **具体实施步骤**：
  1. **第一步（准备/建模/设计）**：定义核心状态变量、可控扰动项和比较基线，建立最小化分析框架。
  2. **第二步（执行/推导/验证）**：在不同扰动强度下推导关键量变化，并对照基线模型计算差异。
  3. **第三步（分析/对比）**：比较不同扰动设置下的结果稳定性，判断是否存在可检验的新预测。

### 📌 课题 2：将{chunk_id}中的结论推广到含额外边界条件的场景

- **推演逻辑**：原文段落给出了受限条件下的结论，但尚未讨论边界条件变化时结论是否成立，因此可进行场景迁移。
- **具体实施步骤**：
  1. **第一步（准备/建模/设计）**：明确原始条件与新增边界条件的差异，重写模型约束与变量定义。
  2. **第二步（执行/推导/验证）**：在新增边界条件下重复核心推导，比较关键表达式与原结论的一致性。
  3. **第三步（分析/对比）**：统计不同边界条件下的偏差范围，识别原结论失效或保持成立的条件区间。
"""

    def _mock_merge(self, prompt: str) -> str:
        payload = _extract_input_json(prompt)
        merge_items = payload.get("merge_items", [])
        merged_topics = []
        for index, item in enumerate(merge_items, start=1):
            source_chunks = item.get("source_chunks", [])
            first_chunk = source_chunks[0] if source_chunks else {}
            evidence_text = first_chunk.get("source_chunk_text", "")[:80]
            merged_topics.append(
                {
                    "topic_id": f"topic_{index:04d}",
                    "paper_id": payload.get("paper_id", "paper_mock"),
                    "title": item.get("representative_topic", f"课题 {index}"),
                    "normalized_title": item.get("representative_topic", f"课题 {index}").lower(),
                    "merged_derivation_logic": item.get("representative_logic", ""),
                    "implementation_steps": (
                        first_chunk.get("implementation_steps")
                        or ["定义变量与约束", "执行模型推导与验证", "分析结果与基线对比"]
                    )[:3],
                    "source_candidate_ids": item.get("candidate_ids", []),
                    "source_chunk_ids": [
                        chunk.get("source_chunk_id", "chunk_unknown") for chunk in source_chunks
                    ],
                    "evidence_quotes": [evidence_text] if evidence_text else [],
                    "merge_notes": "mock merge",
                }
            )
        return json.dumps({"merged_topics": merged_topics}, ensure_ascii=False, indent=2)

    def _mock_score(self, prompt: str, model_key: str) -> str:
        payload = _extract_input_json(prompt)
        title = payload.get("title", "")
        evidence_count = len(payload.get("evidence_quotes", []))
        base = 3.6
        if model_key == "glm5":
            specificity = 4.0
            feasibility = 3.9
            evidence_alignment = 4.4 if evidence_count else 2.8
            novelty = 3.5
            value = 3.9
        elif model_key == "qwen122":
            specificity = 4.3 if len(title) > 8 else 3.6
            feasibility = 4.2
            evidence_alignment = 4.0
            novelty = 3.4
            value = 3.8
        else:
            specificity = 3.8
            feasibility = 3.7
            evidence_alignment = 3.9
            novelty = 4.4
            value = 4.3

        overall = round(
            0.25 * specificity
            + 0.25 * feasibility
            + 0.20 * evidence_alignment
            + 0.15 * novelty
            + 0.15 * value,
            2,
        )
        return json.dumps(
            {
                "topic_id": payload.get("topic_id", "topic_mock"),
                "specificity": specificity,
                "feasibility": feasibility,
                "evidence_alignment": evidence_alignment,
                "novelty": novelty,
                "value": value,
                "overall": max(base, overall),
                "comment": f"{model_key} mock score",
            },
            ensure_ascii=False,
            indent=2,
        )


def build_client(config: PipelineConfig) -> LiveNvidiaClient | MockNvidiaClient:
    if config.mode == "mock":
        return MockNvidiaClient(config)
    return LiveNvidiaClient(config)


def _extract_message_text(raw_response: dict[str, Any]) -> str:
    choices = raw_response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _extract_input_json(prompt: str) -> dict[str, Any]:
    marker = "INPUT_JSON:"
    _, _, tail = prompt.partition(marker)
    tail = tail.strip()
    if not tail:
        return {}
    return json.loads(tail)
