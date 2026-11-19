# Skill 驱动的中文客服 Agent

Python 3.11 / FastAPI / LangGraph / FastMCP / SQLite / Chroma。订单、物流、退款和工单使用虚构种子数据；退款仅表示申请受理。单 Agent，无多 Agent 架构。

项目使用根目录 `.venv`，已完成 Agent Harness、本地 BGE 检索与 DeepSeek 真实 Agent 验证。用户配置并授权后，`deepseek-flash` 真实工作流 10/10、安全集 4/4；全量 119 项测试通过。真实 Judge 尚未启用。完整结果与首轮失败修复见 [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) 的 DeepSeek 真实验收。

## Windows 安装（venv）

在仓库根目录 PowerShell 执行。已有 `.venv` 时直接同步依赖。

```powershell
uv python install 3.11
uv venv --python 3.11 .venv
uv sync --frozen --group dev
.\.venv\Scripts\Activate.ps1
if (!(Test-Path .env)) { Copy-Item .env.example .env }
python -m alembic upgrade head
python -m scripts.init_db
```

`scripts.init_db` is restricted to `APP_ENV=local`, `DEMO_MODE=true`, and the project-local `data/app.db`. It loads the 20-row simulated product fixture by inserting missing IDs only; existing product rows (including inventory changes), tenant b's catalog, and order records are preserved. The same command is safe to rerun.

不激活时可把 `python` 换成 `.\.venv\Scripts\python.exe`。Linux 使用 `source .venv/bin/activate`。`uv.lock` 保存实际解析版本，安装不要主动升级锁文件。

`.env` 中配置 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`。两个进程的 `MCP_SERVICE_SECRET` 必须相同。默认值只用于本机演示。不要提交真实密钥。

## 显式下载和校验本地模型

以下命令显式下载和校验 BGE 模型。API、测试和索引构建不会自动下载权重。

```powershell
python -m scripts.download_models --all
python -m scripts.verify_models
python -m scripts.build_kb
```

如果 Xet/整文件传输无进展，可先诊断，再使用带断点续传的 HTTP 分段下载：

```powershell
python -m scripts.download_models --all --diagnose --disable-xet
python -m scripts.download_models --all --http-range --disable-xet --workers 16 --timeout-seconds 3600
```

脚本只下载推理需要的配置、tokenizer 和一套 PyTorch 权重，避免重复下载 ONNX。分段记录来自固定 commit，完成后校验上游文件哈希，再原子替换目标文件；不完整文件不会写入成功 manifest。网络中断可执行同一命令续传。可选 `--source hf-mirror` / `--source modelscope` 使用公开替代下载端点，仍以 Hugging Face 固定 revision 的哈希为准，不向替代站点传递 HF token。端点说明：[HF-Mirror](https://hf-mirror.com/)、[ModelScope BGE-M3](https://www.modelscope.cn/models/BAAI/bge-m3)。本次通过官方源 HTTP 分段续传成功下载两个模型，并校验完整权重哈希。

需要固定修订时：`python -m scripts.download_models --model embedding --revision <commit-sha>`；reranker 同理。完整下载可能耗时较长且占用数 GB。下载记录包含 commit SHA、时间及文件 SHA256。`verify_models` 校验文件完整性并执行本地 embedding / reranker 推理。显存不足可将 `.env` 的 `MODEL_DEVICE=cpu` 后重启；本次使用 PyTorch CPU 构建。CPU 运行可在启动前设置 `$env:OMP_NUM_THREADS='8'; $env:MKL_NUM_THREADS='8'`，避免线程过量。

## 启动和演示

两个已激活 `.venv` 的终端分别执行：

```powershell
python -m app.mcp
```

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

访问 [Swagger](http://127.0.0.1:8000/docs)。在 demo-login 输入 `{"demo_user_key":"user_a"}`，将响应中的 token 填入请求的 `Authorization: Bearer <token>` header。可选用户：`user_a`（租户 a）、`user_b`（租户 b）、`admin`（租户 a）。

聊天默认 `mode=real`，缺少 API 配置返回 503。显式 `mode=deterministic` 使用规则测试路由，适合离线业务演示；不会自动用 Fake 模型构建正式知识索引。无本地知识索引时知识咨询返回 503，其余订单与退款功能正常。

```powershell
python -m scripts.demo
```

完整脚本要求知识索引就绪，展示政策引用、订单列表、第二单物流、退款预览、插入产品问题、确认、重复确认、退款进度与人工工单。种子 O1001 送达两天、O1002 已发货、O1003 已支付且待发货、O1004 超过七天，O2001 属于另一租户。退款有实际副作用，seed 幂等，不会删除已受理记录。全新演示需将 `SQLITE_URL`、`CHECKPOINT_DB_PATH`、`CHROMA_PATH` 和 `BM25_INDEX_PATH` 指向新目录，迁移、初始化并重建该数据库的索引；两个服务使用相同配置。详细命令见 [演示步骤](docs/demo_walkthrough.md)。

退款的 `confirm_nonce` 仅在预览响应出现一次，客户端须保留直到确认；数据库和 checkpoint 不保存明文。凭据十五分钟有效，过期需重新申请。不要在普通聊天里发送 nonce。查询 workflow 不能重新获取 nonce。

## 测试和评测

```powershell
python -m pytest -q -m "not real"
python -m scripts.run_eval --dataset datasets/workflow.jsonl --mode deterministic --agent-release agent-v1
python -m scripts.run_experiments
```

每个评测 case 使用独立 SQLite fixture 和固定 UTC 时钟；不会修改演示业务库。确定性模式显式使用 FakeRouter / FakeModels、真实 FastMCP SDK 与真实后端约束，且不调用 Judge。小型随机本地权重测试验证 SentenceTransformers 离线 API，不代表 BGE 测试。`models` 标记的 BGE 测试缺权重时明确跳过；`real` 标记单独控制真实 LLM，本次授权后已执行退款、转人工和知识引用三项真实 LLM 测试。OpenAI-compatible Mock HTTP 测试不访问真实服务。离线 CI 使用 `-m "not real"`，完整配置后可执行 `python -m pytest -q`。

真实检索与 Agent 评测：

```powershell
python -m scripts.run_eval --dataset datasets/retrieval.jsonl --mode rag-real --retrieval-mode hybrid_rerank
python -m scripts.run_experiments --rag-real-only
python -m scripts.run_eval --dataset datasets/workflow.jsonl --mode agent-real
python -m scripts.run_experiments --real
python -m scripts.compare_runs --baseline <run-id> --candidate <run-id>
```

`--rag-real-only` 仅使用本地 BGE，先预热，再在同一知识快照与 gold 查询上运行四种检索模式，不调用主 LLM 或 Judge。本次已运行 `agent-real` 的 workflow/safety 数据集及真实 HTTP 演示；`--real` 包含额外 Skill 效果消融，尚未运行。不要把未列出的实验视为已完成。

`--retrieval-mode` 支持 bm25 / dense / hybrid / hybrid_rerank；`--skill-mode none` 禁用动态 Skill 工具，`selected` 为按需加载，旧 `all` 参数兼容但不再全量注入。后端权限不变。数据集 SHA、case 覆盖、fixture 版本和时钟一致才允许比较。数据集有未知断言或 fixture 时直接报错。输出在 `data/eval_runs/<run-id>/report.json|md|html`，工具证据来自持久化网关 Trace；检索 case 保存完整 top-k、版本、模型来源和阶段耗时。注入的测试模型明确标记为 INJECTED_NOT_VERIFIED。

Judge 默认 NOT_EVALUATED。配置 `JUDGE_*` 后，可对保留回答的已有 run 单独复核；输出独立 judge_report.json，不修改原始确定性报告或 Trace：

```powershell
python -m scripts.run_eval --mode judge --run-id <run-id>
```

本次没有执行真实 Judge。其网络失败测试与协议测试通过本地 Mock 完成。`uv sync --frozen` 会可编辑安装项目，因此规范中的 `python scripts/run_eval.py ...` 形式也能从仓库根目录执行。

### 边界订单 Fixture

50 笔边界订单只会进入每个评测用例的临时 SQLite 数据库，不会进入默认演示库。加载顺序是基础用户/租户、20 个商品 Fixture、所选订单场景及其明细/物流/已有退款。选择器支持单个订单、同一场景组和 `orders_boundary:all`；例如 `orders_boundary:O3041` 或 `orders_boundary:delivered_exactly_7d`。评测主体可用 `actor_user_id` 指定为 `user_a`、`admin` 或 `user_b`。

Oracle 只供确定性评测使用，不进入 Agent 提示或 MCP 工具结果。运行全部 50 条固定时钟边界检查：

```powershell
python -m scripts.run_order_boundary_eval
```

报告写入 `data/eval_runs/`，逐案保存 Oracle、实际结果、预期结果和失败检查。Agent 评测 JSONL 可以按用例填写 `fixture` 选择器；未指定时仍使用原始 `user_a_delivered_refundable` 基线。

主 LLM 费用仅在 API 返回完整 token usage 且配置 `LLM_INPUT_COST_PER_MILLION` / `LLM_OUTPUT_COST_PER_MILLION` 时按 Decimal 计算，币种由 `LLM_COST_CURRENCY` 标注。未提供 usage 或单价时为 NOT_EVALUATED，不假定免费，也不把 Judge 的费用混入主模型费用。

人工审核后才可晋升失败案例：

```powershell
python -m scripts.run_eval --review-run <run-id> --case-id <case-id> --reviewer <name> --reason "人工核对理由"
python -m scripts.run_eval --promote-failures <run-id> --confirmed-only
python -m scripts.run_eval --dataset datasets/regression.jsonl
```

审核文件独立保存，不改写原 trace；重复 case_id 不重复写入。故障注入实验默认保留 `PENDING_HUMAN_REVIEW`，没有伪造人工审核记录。测试中的 reviewer 仅属于隔离测试 fixture。

本次本地 BGE 消融（10 条 gold 查询、30 篇模拟知识）：BM25 / dense / hybrid / hybrid_rerank 的 Recall@5 分别为 0.9 / 1.0 / 1.0 / 1.0，MRR@10 分别为 0.7875 / 0.9 / 0.8667 / 1.0。结果仅说明这组演示样本；完整 top-k、失败案例和耗时保存在评测报告。

## Agent Harness 增量层

chat、SSE 和退款确认统一经 Harness 调用 LangGraph。普通聊天由主模型原生 Tool Calling 驱动：提议 → 受控执行 → 观察结果 → 再决策或结束。不运行独立意图分类器；Skill 正文仅在模型调用 `load_skill` 时按需读取。主模型需支持 OpenAI 兼容的原生 `tools` 协议；退款授权与幂等仍由原 Business + SQLite 执行。

```powershell
python -m scripts.run_agent_loop_eval
python -m scripts.run_agent_loop_eval --dataset datasets/workflow.jsonl --mode agent-real
```

对照脚本比较禁用动态 Skill 与按需加载两组；`--baseline-report` 可比对相同数据、fixture、知识快照与模型模式的旧运行。Loop 能处理组合任务，但有工具的回复通常多一次模型观察，不能假定比固定工作流更省调用。

升级已有库并验证（不下载模型、不访问真实 LLM）：

```powershell
python -m alembic upgrade head
python -m scripts.harness_smoke
python -m scripts.context_experiment
python -m pytest -q tests/unit/harness tests/integration/harness
python -m pytest -q -m "not real"
```

`.env.example` 追加 HARNESS_ENABLED、HARNESS_TRACE_ENABLED、MODEL_CONTEXT_WINDOW、CONTEXT_*、TOOL_*、MODEL_MAX_RETRIES 等配置。Harness 默认开启；设 false 会暂停 Agent 执行并返回 CONFIG_ERROR，不恢复绕过控制的调用路径。窗口配置在启动时校验，部署时需匹配实际主模型。

chat.request_id 与新增可选 confirm.request_id 是客户端请求幂等键；同键同内容返回原响应或处理中状态，改内容返回 409。响应追加 invocation_id，可在 `GET /api/v1/invocations/{id}` 查询本人状态。预览 nonce 仍只交付一次，重放响应不包含明文，客户端需保留首次凭据。

Context Manager 支持 none / L0+L1 / L0+L1+L2；保留归档原文、可信业务状态和证据 ID。当前 L2 为明确标注的结构化事实降级摘要，未调用真实模型生成语义摘要。所有 Token 预算估算和 Mock/Fake 实验均单独标注。

运行使用 Uvicorn 单 worker。进程内图线程锁不是分布式锁；写请求超时/断连保持原业务操作标识并对账，不等同于业务已取消。详细错误策略、持久化和限制见 [Harness 运行说明](docs/harness_runtime.md)。

进一步说明：[架构](docs/architecture.md)、[评测](docs/evaluation.md)、[演示步骤](docs/demo_walkthrough.md)、[原逐项验收](docs/acceptance_audit.md)、[Harness 集成图](docs/harness_integration_map.md)。

## 跨会话长期记忆

先执行 `python -m alembic upgrade head`。功能默认可用，但**每位用户默认不自动保存**；在 Swagger 的「长期记忆」分组启用 `auto_write_enabled`，或手动添加记忆。`retrieval_enabled` 独立控制召回；关闭保存不会清除已存条目。用户可查看、编辑、删除自己的条目及查看修订。

演示时用已登录用户在会话 A 说「请记住我买耳机优先关注降噪」，然后等待 `GET /api/v1/memories` 出现该条目。在**全新会话 B**问「推荐一款耳机」；偏好由记忆提供，商品和价格由原 `search_products` 获取。没有主 LLM API 也可手动新增偏好并以 `mode=deterministic` 演示召回；线上自动提取仍走现有 Model Gateway。

```powershell
python -m pytest -q tests/test_memory_schema.py tests/test_memory_storage.py tests/test_memory_batches.py tests/test_memory_integration.py tests/test_memory_evaluation.py
python -m scripts.run_memory_eval --ablation memory-off,memory-on,extraction-per-turn,trigger-only,trigger-batch
python -m scripts.rebuild_memory_index --dry-run
```

设置、会话结束触发、删除范围和索引维护见 [长期记忆说明](docs/memory.md)。测试与实际指标见 [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md)。

## RAG v3 模拟知识 Fixture

本包资料为虚构模拟内容。50 篇 Markdown 文档进入本地开发知识库；Gold Query、证据映射和负例只用于离线评测，不写入 Agent 提示或 MCP 工具结果。原有知识文档保留。

```powershell
.venv/Scripts/python.exe -m scripts.import_rag_v3_bundle
.venv/Scripts/python.exe -m scripts.import_rag_v3_bundle --sync-static-product-attributes
.venv/Scripts/python.exe -m scripts.build_kb
.venv/Scripts/python.exe -m scripts.run_rag_v3_experiments
```

导入器可重复执行；相同文件跳过，目标已有不同内容时停止。静态属性同步只补齐数据库中与 Fixture 已有值不冲突的缺失键，不改价格、库存、租户 B 商品或订单；只应对本地开发 SQLite 执行。评测脚本另建隔离 SQLite、Chroma 和 BM25 索引，按 `user_a` / `user_b` 运行 52 条正例，并对 8 条无答案/权限负例做独立检索审查。不会把扩展负例伪造成非空 `gold_chunks`。当前 `RETRIEVAL_RERANK_K=12`；k=8、12、16 的对照和逐案变化见 [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md)。

当前本地开发活动快照为 123 个 chunk（原知识与 50 篇 v3 合并），由本地 BGE-M3 构建。四模式实验结果位于 [基准实验](data/eval_runs/rag-v3-20261119T052417Z-0df36dba/experiment_summary.json)，k=12 复测位于 [rerank 调整实验](data/eval_runs/rag-v3-20261119T060308Z-870c490d/experiment_summary.json)，k=16 的逐案比较位于 [k=16 对照结果](data/eval_runs/rag-v3-20261119T060308Z-870c490d/rerank_k16_comparison.json)。评测是合成数据的检索实验，不代表生产问答质量或服务级别。

## 商品图文检索

访问 [本地图文检索工作台](http://127.0.0.1:8000/multimodal) 上传、预览、检索和删除图片，也可在 [Swagger](http://127.0.0.1:8000/docs) 调试。沿用现有 Bearer 登录和会话。空图库返回空列表，文字客服可正常启动。

### 下载权重与导入商品图

在项目根目录执行。图片导入及重建期间请停止使用同一个 Chroma 路径的 API 进程。

```powershell
python -m alembic upgrade head
python scripts/download_chinese_clip.py
python scripts/import_product_images.py --product-id headphones-a --file ./my-headphones.png --caption "黑色头戴式无线耳机"
python scripts/rebuild_multimodal_index.py --target all --dry-run
```

`--file` 可重复，用于同一商品的多张图。商品 ID 必须已存在，导入工具从商品表读取租户。相同商品与相同规范化图片会复用已登记的图片 ID。通过 `--replace-id <image_id>` 替换旧图，或由所属租户管理员调用图片删除 API。知识库已构建时可加 `--doc-id warranty` 将短描述关联到可见知识文档，Agent 可以沿着候选商品继续用原 BGE RAG 检索并引用知识块。

权重默认位于 `models/chinese-clip/`，下载脚本在该目录记录模型 commit，重复执行使用已锁定版本。切换权重时用 `--revision <commit> --local-dir <new-directory>` 下载，再设置 `CHINESE_CLIP_MODEL_PATH`，全量重建两个图文集合。显存不足时可设置 `CHINESE_CLIP_DEVICE=cpu`。

```powershell
python scripts/rebuild_multimodal_index.py --target all
python scripts/rebuild_multimodal_index.py --target all --retry
```

导入先保存 SQLite 和文件，再执行索引作业。可用 `--queue-only` 延迟编码。失败或中断后的索引作业可通过 `--retry` 收敛。重建仅读取 SQLite 有效记录，处理范围限于 `mm_product_images_v1` 和 `mm_visual_texts_v1`。索引 manifest 位于 `CHROMA_PATH/multimodal_index_manifest.json`。图片储存在 `MULTIMODAL_IMAGE_PATH` 下，由鉴权 API 读取。私有上传默认一天过期，过期即拒绝读取，重建或重试命令同时清理过期文件。

### 上传与查询

| 路由 | 请求内容 |
| --- | --- |
| `POST /api/v1/images` | multipart 表单 `file`，可选 `conversation_id` |
| `GET /api/v1/images/{image_id}` | Bearer 鉴权后读取规范化 JPEG |
| `DELETE /api/v1/images/{image_id}` | 上传者或所属租户的商品图库管理员 |
| `POST /api/v1/multimodal/search/by-image` | `image_id`、可选 `top_k` |
| `POST /api/v1/multimodal/search/text-by-image` | `image_id`、可选 `top_k` |
| `POST /api/v1/multimodal/search/by-text` | `query`、可选 `top_k` 和 `filters` |
| `POST /api/v1/multimodal/search/by-image-and-text` | `image_id`、`query`、可选 `top_k` 和 `filters` |

过滤器支持 `max_price_cents`、`category`、`in_stock`。例如 500 元以内传入 `"filters":{"max_price_cents":50000}`。检索使用 512 维余弦相似度排序，再用 SQLite 校验商品和过滤条件。商品卡片的名称、价格和库存来自商品表。相似度用于候选排序，型号需要结合实际规格确认。

```json
{
  "conversation_id": "<existing-conversation-id>",
  "message": "找类似的商品，价格不超过 500 元",
  "image_ids": ["<uploaded-image-id>"],
  "allow_external_image_send": false
}
```

原 `/api/v1/chat` 和 SSE 可接收上述附件字段。图片在上传指定会话或首次聊天时绑定。结果可能包含 `cards`，其中的 `image_url` 需要当前 Bearer Token。独立检索 API 只依赖本地 CLIP 和索引，聊天决策继续使用配置的模型。

### 外部 VLM

确认目标端点同时支持 Chat Completions 的 `image_url` data URI 和原生工具调用后，在本地 `.env` 设置：

```dotenv
LLM_BASE_URL=<provider-endpoint>
LLM_API_KEY=<provider-key>
LLM_MODEL=<text-chat-model>
VLM_MODEL=<vision-tool-calling-model>
MULTIMODAL_EXTERNAL_VLM_IMAGES_ENABLED=true
VLM_PROVIDER_TRANSPORT=data_uri
VLM_SUPPORTS_VISION=true
VLM_SUPPORTS_TOOL_CALLING=true
```

当前 VLM 使用同一组端点和凭据，通过 `VLM_MODEL` 选择视觉模型。服务端开启后，每次聊天还须明确传入 `allow_external_image_send: true`，或勾选工作台的本次授权。网关读取已授权图片，缩放并编码为供应商接收的 JPEG data URI。现有供应商仅支持文字时，请保留本地检索或配置支持该请求格式的端点。部署前运行下方的真实 VLM 冒烟以核实能力开关。

### 验收和排错

```powershell
python scripts/generate_multimodal_test_assets.py --output-dir tests/fixtures/multimodal
python -m pytest tests/multimodal -q -rs
python scripts/multimodal_smoke.py --run-local-model
# 配置了兼容供应商并同意外发生成的工程图片后，再运行
python scripts/multimodal_smoke.py --run-external-vlm
```

工程测试使用少量 Pillow 生成图、确定性测试编码器和 HTTP Mock。真实本地冒烟使用官方模型与 Chroma，输出向量形状、索引数量和耗时。当前没有标注商品图集，因此无语义检索准确率。

从调试页、`X-Trace-ID`（图片与检索）、聊天响应或错误体取得 Trace ID，用原 Bearer Token 请求 `GET /api/v1/traces/{trace_id}`。本地结构化事件亦位于检查点同级 `traces/`。已登记商品图但索引未建好时检查 `INDEX_NOT_READY` 和索引作业。缺少权重时根据 `MULTIMODAL_MODEL_NOT_READY` 提示执行下载。

适配依据见 [Chinese-CLIP 模型仓库](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/tree/36e679e65c2a2fead755ae21162091293ad37834) 和 [Chat Completions 请求格式](https://platform.openai.com/docs/api-reference/chat/create)。
