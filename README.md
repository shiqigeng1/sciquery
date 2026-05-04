# SciQuery

SciQuery 会把输入论文切成 chunk，经 LLM 生成候选课题，再进行去重、聚类、合并、三模型评分，最终输出可保留的研究课题列表。

## 环境要求

- Python 3.10 或更高版本。
- 推荐在项目根目录运行命令：`C:\science\sciquery`。
- `input/` 和 `output/` 是本地数据目录，已经在 `.gitignore` 中排除，不会被 Git 跟踪。

安装依赖：

```powershell
cd C:\science\sciquery
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 中的 `requests` 是主流水线依赖；`openai` 供 `llm/` 目录下的独立 NVIDIA/OpenAI-compatible 示例脚本使用。

## API Key

`--mode live` 会调用 NVIDIA API，需要设置以下环境变量：

```powershell
$env:SCIQUERY_DEEPSEEK_API_KEY = "your_deepseek_key"
$env:SCIQUERY_NEMOTRON_API_KEY = "your_nemotron_key"
$env:SCIQUERY_GLM5_API_KEY = "your_glm5_key"
$env:SCIQUERY_QWEN122_API_KEY = "your_qwen122_key"
$env:SCIQUERY_QWEN397_API_KEY = "your_qwen397_key"
```

`--mode mock` 不需要 API key，适合检查流程是否能跑通。

## 推荐运行方式

使用命令行显式传参：

```powershell
cd C:\science\sciquery
python code\run_pipeline.py --input input\md\review_xiaolingcui.md --paper-id review_xiaolingcui_live_01 --mode live
```

用内置样例和 mock 模式做快速检查：

```powershell
python code\run_pipeline.py --input code\sample_review.md --paper-id sample_review_mock --mode mock --output-root output
```

查看所有命令行参数：

```powershell
python code\run_pipeline.py --help
```

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input` | 必填 | 输入 Markdown 文件路径。 |
| `--paper-id` | 输入文件名 stem | 输出目录名和结果中的论文标识。 |
| `--mode` | `live` | `live` 调用 NVIDIA API；`mock` 使用本地模拟响应。 |
| `--output-root`, `--artifacts-root` | `<repo>\output` | 流水线输出根目录。两个参数名等价。 |
| `--prompt-a` | `prompts/Prompt_A_*.md` 中排序后的第一个文件 | 候选课题生成提示词模板。 |
| `--candidate-runs` | `3` | 每个 chunk 调用 DeepSeek 生成候选的次数。实际使用 `candidate_temperatures` 的前 N 个温度。 |
| `--noise-min-chars` | `40` | 短于该长度的段落视为噪声并丢弃。 |
| `--chunk-min-chars` | `180` | 短于该长度的 chunk 会尽量与同 section 相邻 chunk 合并。 |
| `--chunk-target-max-chars` | `1000` | 优先按句子或软分隔切分的目标最大长度。 |
| `--chunk-hard-max-chars` | `1300` | 强制切分的硬上限。 |
| `--cluster-threshold` | `0.82` | TF-IDF 余弦相似度聚类阈值，越高越保守。 |
| `--cluster-batch-size` | `50` | 多轮聚类时每批处理的候选数量。 |
| `--merge-group-size` | `10` | 每次 Nemotron 合并调用输入的 cluster 数量。 |
| `--request-timeout-seconds` | `240` | 每次 NVIDIA API 请求的读取超时时间。 |
| `--retry-backoff-seconds` | `8` | 重试间隔基数。第 1 次重试等待 8 秒，第 2 次等待 16 秒。 |
| `--no-resume` | 默认启用 resume | 关闭阶段级断点续跑，重新生成阶段输出。 |
| `--heartbeat-score-interval` | `5` | 评分阶段每完成 N 个 topic 写入一次 `run_meta.json` 和 `run_heartbeat.json`。 |
| `--heartbeat-chunk-interval` | `5` | 候选生成阶段每完成 N 个 chunk 写入一次 `run_meta.json` 和 `run_heartbeat.json`。 |

## 代码配置参数

以下参数定义在 `code/sciquery_pipeline/config.py` 的 `PipelineConfig`，其中一部分已通过 CLI 暴露：

| 参数 | 默认值 | 是否有 CLI | 说明 |
| --- | --- | --- | --- |
| `repo_root` | 无 | 间接 | 项目根目录。命令行入口会把 `code` 的父目录作为项目根目录。 |
| `mode` | `live` | 是 | 运行模式。 |
| `artifacts_root` | `<repo>\output` | 是 | 输出根目录。 |
| `prompt_a_path` | 自动查找 `prompts/Prompt_A_*.md` | 是 | Prompt A 模板路径。 |
| `candidate_runs` | `3` | 是 | 每个 chunk 候选生成调用次数。 |
| `candidate_temperatures` | `(0.7, 0.9, 1.1)` | 否 | 候选生成温度列表；`candidate_runs` 只取前 N 个。 |
| `max_candidates_per_call` | `2` | 否 | 配置项已定义，当前主流程没有直接读取。 |
| `noise_min_chars` | `40` | 是 | Markdown 噪声段落过滤阈值。 |
| `chunk_min_chars` | `180` | 是 | 最小 chunk 长度。 |
| `chunk_target_max_chars` | `1000` | 是 | 目标 chunk 长度上限。 |
| `chunk_hard_max_chars` | `1300` | 是 | 硬 chunk 长度上限。 |
| `cluster_threshold` | `0.82` | 是 | 聚类相似度阈值。 |
| `cluster_batch_size` | `50` | 是 | 聚类批大小。 |
| `merge_group_size` | `10` | 是 | 合并阶段每组 cluster 数。 |
| `merge_skip_prior_invalid_groups` | `True` | 否 | resume 时跳过历史上已经失败且位于 frontier 之前的合并组。 |
| `request_timeout_seconds` | `240` | 是 | API 读取超时。 |
| `retry_count` | `2` | 否 | 每个 LLM 调用最多重试次数。总尝试次数为 `retry_count + 1`。 |
| `retry_backoff_seconds` | `8` | 是 | 重试等待基数。 |
| `resume_enabled` | `True` | 是，反向参数 `--no-resume` | 是否复用已有阶段产物。 |
| `heartbeat_chunk_interval` | `5` | 是 | 候选生成阶段 heartbeat 间隔。 |
| `heartbeat_score_interval` | `5` | 是 | 评分阶段 heartbeat 间隔。 |

模型注册参数定义在同一文件的 `MODEL_REGISTRY`：

| 模型键 | 远端模型 | API key 环境变量 | `max_tokens` | `temperature` | `top_p` |
| --- | --- | --- | --- | --- | --- |
| `deepseek` | `deepseek-ai/deepseek-v3.2` | `SCIQUERY_DEEPSEEK_API_KEY` | `2048` | `0.9` | `0.95` |
| `nemotron` | `nvidia/nemotron-3-super-120b-a12b` | `SCIQUERY_NEMOTRON_API_KEY` | `3072` | `0.2` | `0.9` |
| `glm5` | `z-ai/glm5` | `SCIQUERY_GLM5_API_KEY` | `768` | `0.0` | `1.0` |
| `qwen122` | `qwen/qwen3.5-122b-a10b` | `SCIQUERY_QWEN122_API_KEY` | `768` | `0.0` | `0.95` |
| `qwen397` | `qwen/qwen3.5-397b-a17b` | `SCIQUERY_QWEN397_API_KEY` | `768` | `0.0` | `0.95` |

`qwen397` 额外携带 `top_k=20`、`presence_penalty=0`、`repetition_penalty=1`。

## 无参运行方式

直接运行 `python code\run_pipeline.py` 会使用 `code/run_pipeline.py` 顶部的硬编码参数。这适合个人本地固定任务，不适合作为通用入口。

| 变量 | 当前值 | 说明 |
| --- | --- | --- |
| `MODE` | `live` | 运行模式。 |
| `INPUT_PATH` | `input/md/review_xiaolingcui.md` | 默认输入文件。 |
| `OUTPUT_ROOT` | `output` | 默认输出根目录。 |
| `PROMPT_A_PATH` | 自动查找 Prompt A | 候选生成提示词模板。 |
| `PAPER_ID` | `review_xiaolingcui_live_02` | 默认输出目录名。 |
| `PARTIAL_TOPICS_PATH` | `output/<PAPER_ID>/08_partial_topics_round3.json` | `from-partial-score` 使用的 partial topics 文件。 |
| `CANDIDATE_RUNS` | `3` | 每个 chunk 候选生成调用次数。 |
| `NOISE_MIN_CHARS` | `40` | 噪声段落过滤阈值。 |
| `CHUNK_MIN_CHARS` | `180` | 最小 chunk 长度。 |
| `CHUNK_TARGET_MAX_CHARS` | `1000` | 目标 chunk 长度上限。 |
| `CHUNK_HARD_MAX_CHARS` | `1300` | 硬 chunk 长度上限。 |
| `CLUSTER_THRESHOLD` | `0.82` | 聚类阈值。 |
| `CLUSTER_BATCH_SIZE` | `50` | 聚类批大小。 |
| `MERGE_GROUP_SIZE` | `3` | 无参入口的合并组大小；注意 CLI 默认是 `10`。 |
| `MERGE_SKIP_PRIOR_INVALID_GROUPS` | `True` | resume 时是否跳过历史无效合并组。 |
| `RESUME_ENABLED` | `True` | 是否启用断点续跑。 |
| `HEARTBEAT_CHUNK_INTERVAL` | `5` | 候选生成 heartbeat 间隔。 |
| `HEARTBEAT_SCORE_INTERVAL` | `5` | 评分 heartbeat 间隔。 |
| `REQUEST_TIMEOUT_SECONDS` | `600` | 无参入口的 API 读取超时；注意 CLI 默认是 `240`。 |
| `RETRY_BACKOFF_SECONDS` | `8` | 重试等待基数。 |

辅助命令：

```powershell
python code\run_pipeline.py from-merge
python code\run_pipeline.py from-partial-score
```

- `from-merge`：复用 `01_chunks.jsonl`、`03_candidates_deduped.jsonl`、`04_clusters.jsonl`，从合并阶段继续。
- `from-partial-score`：读取 `PARTIAL_TOPICS_PATH`，只对 partial topics 做评分和聚合。该流程会在需要时规范化 partial topic id，并可能写回 `PARTIAL_TOPICS_PATH`。

## 输出文件

每次运行会写入 `output/<paper-id>/`：

| 文件 | 说明 |
| --- | --- |
| `01_chunks.jsonl` | Markdown 切块结果。 |
| `02_candidates_raw.jsonl` | DeepSeek 原始候选生成调用结果。 |
| `02_candidates_failed.jsonl` | 候选生成请求失败记录。 |
| `02_candidates_parse_failed.jsonl` | 候选解析失败记录。 |
| `03_candidates_deduped.jsonl` | 去重后的候选课题。 |
| `04_clusters.jsonl` | 聚类结果。 |
| `04_clusters_rounds/` | 聚类轮次明细。 |
| `05_merge_rounds/` | Nemotron 合并分组明细和 resume 元数据。 |
| `06_merged_topics.json` | 合并后的 topic。 |
| `07_topic_scores.jsonl` | 三模型评分结果。 |
| `07_topic_scores_failed.jsonl` | 评分请求或解析失败记录。 |
| `08_final_topics.json` | 最终聚合后的保留 topic。 |
| `run_meta.json` | 当前运行状态和最终状态。 |
| `run_heartbeat.json` | 运行中 heartbeat 状态。 |

各阶段还会写入对应的 `*.meta.json`。resume 只有在元数据与当前输入、参数、上游产物指纹完全一致时才会复用阶段结果。

## 常见问题

- 缺少 API key：确认已设置对应 `SCIQUERY_*_API_KEY` 环境变量，或使用 `--mode mock`。
- 想重新生成所有结果：加 `--no-resume`，或换一个新的 `--paper-id`。
- 输入文件找不到：使用相对于项目根目录的路径，例如 `input\md\xxx.md`，或传入绝对路径。
- 最终 topic 为空：检查 `02_candidates_parse_failed.jsonl`、`05_merge_rounds/`、`07_topic_scores_failed.jsonl`，判断是模型返回格式不合规、合并阶段为空，还是评分聚合被拒绝。
