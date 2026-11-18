# Agent Harness Runtime 补充开发规格（Codex 增量开发指令）

> **文档版本**：1.0  · **适用项目**：Skill 驱动的智能客服 Agent 与自动化评测平台  
> **依赖现有文档**：`DEVELOPMENT_SPEC.md`、`AGENTS.md`  
> **执行性质**：**增量补充，不是重新搭建项目，也不是覆盖原开发规格**。  
> **语言与环境**：Python 3.11，单机本地开发，FastAPI + LangGraph + FastMCP + Chroma + SQLite。

---

## 0. Codex 必读：任务、优先级与交付定义

请先阅读当前仓库中的 `AGENTS.md`、`DEVELOPMENT_SPEC.md` 和现有实现，再按此文件添加 **独立、可测试的 Harness Runtime 层**。本补充规格专门规定 Harness 的职责和接口；原规格的业务范围、数据约束、安全边界、工具名字、退款规则、测试 T01–T20、模型下载方案、评测闭环等保持有效。

### 0.1 优先级和冲突处理

1. **安全与业务不变量最高优先级**：服务端订单归属校验、退款确认、业务状态及幂等唯一性必须由现有 Business Service + SQLite 事务执行，任何 Harness、Prompt、Skill、LLM 输出都不得绕过。
2. **原规格保留的约定不得更改**：普通会话 `thread_id="conv:{conversation_id}"`；需等待确认的退款工作流 `thread_id="wf:{workflow_id}"`；LangGraph `AsyncSqliteSaver` 存储于 `data/checkpoints.db`；7 个原有 FastMCP 工具名与返回规范保持一致。
3. **本文件只新增/细化 Harness 相关部分**。如需修改旧接口，先做兼容适配与数据库迁移，并同步更新调用方与测试；不得静默废弃原 API。
4. **职责单一**：LangGraph 是状态图及 Checkpointer；Harness 控制模型/工具调用、请求预算、生命周期、上下文与观测；Business Service 负责最终业务决定；Evaluation 使用结构化 Trace 和真实终态。
5. **先检查现有代码**。发现原模块已经能满足契约时，抽象/复用/迁移，不重复实现第二套“网关”“追踪”“状态机”。不必为了目录结构制造空文件。

### 0.2 不能做的事

- 不引入 Java、Spring Boot、Elasticsearch、Milvus、Kafka、Kubernetes、Celery、额外的多 Agent 平台或分布式调度系统。
- 不自研替代 LangGraph 的图调度器、Checkpointer、MCP 协议或大模型推理服务。
- 不修改本地 Embedding 与 Reranker 选型：`BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3`，继续从**已下载的本地路径**加载；Harness 不负责下载权重。
- 不因 HTTP/SSE 客户端断连而自动回滚或再次提交已经开始的业务写操作。
- 不把 FakeLLM 的测试表现当作真实模型指标；不编造测试、Token 费用或评测增益。
- 不把 MCP Tool 的原始异常、密钥、签名 token 或完整敏感字段送给用户或 LLM。

### 0.3 完工条件

- 从 `POST /api/v1/chat`、流式接口和退款确认恢复入口触发的 Agent 执行，统一经过 Harness，**不存在绕开 Harness 直接调用真实 LLM 或 MCP 的图节点**。
- Model Gateway、Tool Executor、Context Manager、Run Controller、Policy Guard、Recovery、Telemetry 均有**实际功能与离线测试**。
- 退款确认、未知执行结果、跨会话隔离、不重复提交等 T01–T20 不退化；新增 H01–H25 Harness 测试通过（见第 15 节）。
- 本地 Trace + Evaluation 记录 `invocation_id`、错误分类、尝试次数、压缩元数据、模型/工具耗时和预算使用，支持定位失败及回归对比。
- 更新 `README.md`、`docs/architecture.md`、`docs/evaluation.md`（如存在）和 `IMPLEMENTATION_REPORT.md`，说明实现状态、命令、真实测试结果及限制。不得把仅有函数签名的模块标为完成。

---

## 1. 系统定位与准确边界

Harness 不是另一个 Agent，而是 Agent 的**运行时控制层（Runtime Infrastructure）**。

```text
FastAPI (身份认证 / chat / chat-stream / confirm)
          │    已认证主体、会话、请求 ID
          ▼
    Harness Run Controller
      ├─ Invocation 管理 / 全局预算 / 并发控制 / 取消
      ├─ ExecutionContext (可信主体、deadline、版本、trace)
      └─ 交给既有 LangGraph StateGraph
               │
               ├─ Intent Router → Skill Selector/Loader
               ├─ Agent Decision → Model Gateway
               ├─ RAG → Chroma + BM25 + RRF + Local Reranker
               ├─ MCP Read Tools → Tool Executor → Policy Guard → MCP Gateway/Client
               └─ Refund Workflow → 确认/中断/恢复 → 受控 Tool Executor

横切：Context Manager / Recovery Policy / Telemetry Recorder
      │
      └─ 本地 Trace / Evaluators / Regression / Langfuse Adapter(可选)

可信业务事实：Business Service + SQLite transaction
可恢复图状态：LangGraph AsyncSqliteSaver + SQLite checkpoint
```

**哪些层持有什么状态**：

| 状态/对象 | 唯一或主要权威 | 备注 |
|---|---|---|
| 用户/租户/角色和会话归属 | API Auth + SQLite | 不信任模型提供的 user_id、tenant_id |
| 订单、售后资格、退款确认与提交状态 | Business Service + SQLite | 最终业务真相；事务和幂等 |
| 图执行位置、待恢复 interrupt | LangGraph Checkpointer | 不是订单业务真相 |
| 当前一次 API 执行的状态、限额、错误与耗时 | Harness Invocation/ExecutionContext | 请求级生命周期 |
| 聊天历史原文 | 现有 messages 表 | 摘要不删除原文 |
| 历史摘要 | Context Manager 持久化摘要记录 | 摘要不能改变业务权威值 |
| Skill 目录/Skill 白名单 | 现有 Skill Manager + 可信注册表 | Skill 文本没有授权能力 |
| 工具真实调用 | 现有 MCP Client/Gateway | Harness Tool Executor 是统一调用包装层，不再建一个 MCP Server |
| Trace/Event | 现有 `observability.recorder` | Harness 只输出统一事件，避免双写两套真相 |
| 评测 batch 记录 | Evaluation Engine | 与单次 invocation 区分 |

---

## 2. 目录与模块集成方式

在**已有项目**基础上新增或整合以下文件：

```text
app/
  harness/
    __init__.py
    contracts.py            # Pydantic / dataclass 公共契约与错误类型
    runtime.py              # HarnessFacade，连接 FastAPI 与原有图
    run_controller.py       # invocation 生命周期、deadline、线程互斥
    execution_context.py   # 可信上下文，不序列化密钥
    model_gateway.py       # 统一 LLM 请求、重试、usage、模型适配
    tool_executor.py       # MCP 工具执行、结构校验、重试/结果归一化
    context_manager.py     # ContextWindow、压缩、摘要缓存、Token 预算
    policy_guard.py         # step/工具白名单/写权限/风险前置检查
    recovery.py             # 错误分类、退避、UNKNOWN_OUTCOME 处理
    telemetry.py            # 向既有 recorder 发结构化事件

app/agent/graph.py          # 只做必要的依赖注入/节点调用改造
app/agent/nodes/            # 使用 injected Harness Services
app/mcp/gateway.py          # 保留现有 MCP 传输和可信鉴权逻辑
app/observability/recorder.py  # 复用为 Trace 唯一持久出口
app/evaluation/             # 消费新增结构化事件，无需重写整套引擎
app/db/migrations/          # 按第 4 节增加最少数据结构

scripts/harness_smoke.py    # 新增，离线 FakeLLM + FakeMCP 测试流程
tests/unit/harness/
tests/integration/harness/
```

允许将薄封装合并到相邻模块，但**接口与行为必须保留**。例如若原 `app/agent/executor.py` 已管理节点执行，应复用而不是实现两套 executor。

### 2.1 依赖要求

- 使用原 `uv.lock` 中兼容的版本；新增依赖必须锁定并进行最小 smoke test。
- `asyncio`、`contextlib`、`dataclasses`、`uuid`、`time` 等优先用标准库。
- OpenAI-compatible 主模型可由 `AsyncOpenAI` 或已存在的 LangChain 模型适配器调用，**只允许一个真实模型调用入口**。
- LangGraph API 按仓库实际锁定版本使用；`ainvoke`/`astream`、`interrupt()`、`Command(resume=...)` 和 `AsyncSqliteSaver` 需要可运行验证。
- 不额外引入模型代理服务、Redis 分布式锁或专门的任务队列。

---

## 3. 核心类型与跨模块契约（优先实现）

设计以下**稳定的接口语义**，不要求照抄全部代码内部实现。所有契约需经 Pydantic/类型检查，统一字段名，默认拒绝非法状态。

### 3.1 ExecutionContext（可信信息）

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str
    role: str

@dataclass(frozen=True)
class ExecutionContext:
    invocation_id: str            # 本次 API chat/confirm 调用唯一 ID
    trace_id: str
    conversation_id: str
    graph_thread_id: str          # conv:{id} 或 wf:{id}
    principal: Principal          # 只能由 API Auth 构造
    agent_release_id: str
    deadline_monotonic: float     # 当前进程内单调时钟；不要持久化用于重启恢复
    evaluation_case_id: str | None = None
    evaluation_run_id: str | None = None
    workflow_id: str | None = None
    parent_invocation_id: str | None = None
    mode: Literal['real', 'deterministic', 'rag-real', 'agent-real'] = 'real'
```

规定：

- `ExecutionContext` **由 FastAPI 认证层创建、显式依赖注入**。Agent 的自然语言、Skill 文件和工具返回文本不能覆盖任何 Principal 字段。
- API 访问不得由客户端直接控制 `graph_thread_id`。由服务端经过会话/工作流归属校验后生成，防止不同用户猜测 thread ID 后窃取 checkpoint。
- `deadline_monotonic` 每个 invocation 新建；跨进程恢复时产生**新** invocation，不能重用旧进程的 monotonic 数值。
- `evaluation_run_id` 是已有评测 batch 的 `run_id`；**不要**把它与单次请求的 `invocation_id` 混为一谈。
- `LLM_API_KEY`、`MCP_SERVICE_SECRET`、内部签名 token 不属于可序列化 ExecutionContext，也不进入 State / Trace / Prompt。

### 3.2 标准响应与错误

建议公共类型：

```python
from enum import StrEnum
from pydantic import BaseModel, Field

class HarnessErrorCode(StrEnum):
    CONFIG_ERROR = 'CONFIG_ERROR'
    MODEL_TIMEOUT = 'MODEL_TIMEOUT'
    MODEL_RATE_LIMIT = 'MODEL_RATE_LIMIT'
    MODEL_UPSTREAM_ERROR = 'MODEL_UPSTREAM_ERROR'
    MODEL_BAD_OUTPUT = 'MODEL_BAD_OUTPUT'
    TOOL_TIMEOUT = 'TOOL_TIMEOUT'
    TOOL_UPSTREAM_ERROR = 'TOOL_UPSTREAM_ERROR'
    TOOL_BAD_RESULT = 'TOOL_BAD_RESULT'
    POLICY_DENIED = 'POLICY_DENIED'
    FORBIDDEN = 'FORBIDDEN'
    CONTEXT_BUDGET_EXCEEDED = 'CONTEXT_BUDGET_EXCEEDED'
    STEP_LIMIT_EXCEEDED = 'STEP_LIMIT_EXCEEDED'
    REPEATED_NO_PROGRESS = 'REPEATED_NO_PROGRESS'
    RUN_TIMEOUT = 'RUN_TIMEOUT'
    RUN_CANCELLED = 'RUN_CANCELLED'
    UNKNOWN_OUTCOME = 'UNKNOWN_OUTCOME'
    INTERNAL_ERROR = 'INTERNAL_ERROR'

class HarnessError(BaseModel):
    code: HarnessErrorCode
    message: str                 # 面向系统的脱敏摘要
    retryable: bool = False
    source: str | None = None     # model/tool/context/runtime
    details: dict = Field(default_factory=dict)   # 禁止令牌、原始敏感数据
```

**区分三种失败概念**：

1. **模型/工具建议错误**：Agent 决策错误，例如试图跳过确认；记录 `policy.denied`，评测可归因 `WORKFLOW_VIOLATION`，实际业务没有被改动。
2. **技术执行失败**：模型超时、MCP 不可用、格式错误；可根据策略尝试重试或安全降级。
3. **业务已提交但调用方未知结果**：写工具超时，不能判定业务失败，更不能用新的 idempotency key 再提交。

### 3.3 规范化模型输出与工具输出

```python
class ModelUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    source: str = 'provider'     # provider / estimated / unavailable

class ModelResult(BaseModel):
    text: str | None = None
    structured: dict | None = None
    model_id: str
    usage: ModelUsage
    attempts: int
    latency_ms: int
    finish_reason: str | None = None

class ToolExecutionResult(BaseModel):
    tool_name: str
    ok: bool
    data: dict | list | str | None = None
    error: dict | None = None    # code / message / retryable
    request_id: str
    attempts: int
    latency_ms: int
    outcome: str                 # SUCCESS / KNOWN_FAILURE / UNKNOWN_OUTCOME
```

`ToolExecutionResult` 是 Harness 内部通用形式，**不改变原 MCP 返回** `{"ok":..., "data":..., "error":..., "request_id":...}`。外层新增计时、尝试次数、outcome。

---

## 4. Invocation 生命周期与持久化

### 4.1 明确一次 Invocation 的定义

一次 `POST /api/v1/chat`、一次 `POST /api/v1/chat/stream` 或一次 `POST /api/v1/workflows/{id}/confirm` 调用 = **一个 invocation**。一次用户退款业务可能跨越初始调用和后续确认两个 invocation；两者通过同一个 `workflow_id` / `graph_thread_id` 关联。

**单次 invocation 状态机**：

```text
CREATED -> RUNNING -> SUCCEEDED
                   -> INTERRUPTED   (等待用户下一次独立 invocation)
                   -> FAILED
                   -> TIMED_OUT
                   -> CANCELLED      (只针对可安全停止的请求处理)
                   -> UNKNOWN_OUTCOME (某业务写操作状态暂时不可判定)
```

`INTERRUPTED` 不表示退款工作流终结；业务 `workflow_instances` 独立保持 `WAITING_CONFIRMATION`。初次 invocation 标记 `INTERRUPTED` 后不再复用这个 invocation ID；用户确认时创建新的 invocation，填写 `parent_invocation_id`，**沿用原 `wf:{workflow_id}`** 继续 LangGraph。

不能新增第二套 `workflow.status`。现有业务状态还是 `WAITING_CONFIRMATION`、`CONFIRMED`、`SUBMITTING`、`REQUEST_ACCEPTED` 等。

### 4.2 最小数据库变更

使用 SQLAlchemy + Alembic：

**新增 `agent_invocations` 表**（推荐字段）：

- `invocation_id` TEXT PRIMARY KEY
- `trace_id`, `conversation_id`, `graph_thread_id`, `workflow_id`(nullable)
- `tenant_id`, `user_id`, `parent_invocation_id`(nullable)
- `client_request_id`(nullable), `request_body_hash`(nullable)
- `response_message_id`(nullable)、`response_json`(nullable，建议保存已脱敏的最终 API 响应或可重建它的稳定引用)
- `status`，`error_code`(nullable)，`agent_release_id`
- `started_at`, `finished_at`(nullable)，`deadline_at_utc`(仅用于审计；不用于重启后计算超时)
- `model_calls`, `tool_calls`, `prompt_tokens`, `completion_tokens`(nullable)
- `created_at`, `updated_at`

索引：`(tenant_id, user_id, conversation_id, created_at)`；`trace_id`；`graph_thread_id`。若 `client_request_id` 非空，约束同一租户、用户、会话下的请求去重；相同 key + 不同 body hash 返回 `409`。可用 SQLite 部分唯一索引，或保持请求 ID 唯一且明确规定重试语义。

**可选新增 `context_summaries` 表**（若原项目没有同等结构）：

- `id`, `tenant_id`, `user_id`, `conversation_id`, `graph_thread_id`
- `covers_through_message_id`, `summary_text`, `summary_version`, `source_hash`, `created_at`

摘要必须绑定用户/租户/图线程；**不能跨用户复用同一个缓存键**。

事件优先写入现有 `observability.recorder` 的 JSONL + SQLite 索引，而不是另起一套 `harness_events` 真相库。若现有数据库有事件表，则扩展 Schema 并迁移。

### 4.3 入口幂等与并发

- `chat` 原接口已有可选 `request_id`；视为**客户端请求幂等键**，与服务端生成 `invocation_id`、MCP 调用 `request_id`、退款 `operation_key` 是不同概念。不同键绝不能互相替代。
- 同 `client_request_id` + 同 payload 重试：终态返回先前持久化的响应（或从 `response_message_id` 重建）；尚 `RUNNING` 时返回可查询的处理中状态，不创建第二次图执行或业务写；同 key + 不同 payload 返回 409。
- 同一 `graph_thread_id` 的两次 LangGraph 执行不可并发驱动同一个 checkpoint。单进程本地使用 `asyncio.Lock` 的按 thread 注册表（带清理逻辑）；线程互斥仅负责**图执行串行化**，业务正确性仍靠 SQLite 约束/条件更新。
- 本地默认 Uvicorn **单 worker**；若多进程运行，同进程锁不足以提供跨进程互斥。测试必须证明数据库层写幂等在并发情况下仍成立；不要对外声称多 worker 线性一致。
- 等待用户确认时不能持续持有线程锁。`interrupt` 返回后释放锁，后续确认 invocation 再获取锁恢复该工作流。
- 请求锁不要跨 LLM 调用长期阻塞其他**不同**图线程；不同会话允许并发。

### 4.4 超时与取消边界

- Agent 总执行预算保留原 `AGENT_TIMEOUT_SECONDS=90`，每次 invocation 创建独立 deadline。
- `remaining_seconds = deadline_monotonic - time.monotonic()`，所有模型/工具调用先计算剩余预算，实际单次超时采用 `min(组件超时, remaining_seconds)`，若预算耗尽直接终止。
- **超时不是强事务回滚**。不能把本地 `asyncio` 取消等同于 MCP 远端业务已取消。
- 对可取消的只读步骤，收到客户端取消/断连可以停止后续处理并标记 `CANCELLED`；对于已经开始的受控写操作，必须先确保操作标识及状态已持久化，再通过不依赖客户端连接存活的受控服务逻辑完成或对账。**仅使用 `asyncio.shield` 不等于持久可靠执行**；进程崩溃后须凭原 `operation_key` 在下一次显式查询/恢复时核对结果。无法确认时进入 `UNKNOWN_OUTCOME`，不得直接声称“写入已取消”。
- 退款业务的 `/cancel` 只取消仍处于 `WAITING_CONFIRMATION` 的未提交流程；一旦 `CONFIRMED/SUBMITTING/REQUEST_ACCEPTED` 不能由普通运行取消逻辑回滚。

---

## 5. Run Controller：Harness 的统一执行入口

### 5.1 建议接口

```python
from typing import Protocol

class HarnessRuntime(Protocol):
    async def invoke_chat(self, *, principal, conversation_id: str,
                          message: str, client_request_id: str | None = None,
                          evaluation: dict | None = None) -> dict: ...

    async def stream_chat(self, *, principal, conversation_id: str,
                          message: str, client_request_id: str | None = None): ...

    async def resume_workflow(self, *, principal, workflow_id: str,
                              confirmation_payload: dict,
                              client_request_id: str | None = None) -> dict: ...
```

**API handler 只做**：身份认证、入参校验、会话归属校验、调用 Harness、包装原响应结构。不要把模型客户端、MCP 客户端或 LangGraph 重试循环复制进 API handler。

`runtime.py` 对接原有的图实例，`run_controller.py` 管理生命周期、图线程锁和 deadline。图节点通过显式依赖拿到 `ModelGateway`、`ToolExecutor`、`ContextManager`、`Telemetry`。

### 5.2 执行伪代码（不是框架 API 的强制语法）

```python
async def invoke_chat(...):
    # 1. Auth 由 API 完成；Runtime 复核 conversation 归属
    # 2. 使用 client_request_id 做请求级幂等检查
    # 3. 创建 invocation + immutable ExecutionContext + trace
    # 4. 获取 conv:{conversation_id} 的线程执行锁
    # 5. 注入依赖，调用已编译 LangGraph 的 ainvoke/astream
    # 6. 识别完成、interrupt、失败、取消和未知结果
    # 7. 使用单一 Telemetry 记录事件并落库最终 invocation 状态
    # 8. 按原 API schema 返回最终结果 / pending_confirmation
```

**强调**：对于可能发生 `interrupt` 的图，不能只检查普通字典中的 `final_response` 就宣布成功。须根据实际 LangGraph 锁定版本规范识别中断事件与持久化 checkpoint，并映射为 `INTERRUPTED` 与 `pending_confirmation`。

### 5.3 最大步数与无进展检测

- 默认沿用 `AGENT_MAX_STEPS=8`。定义“步”的统计口径：每次 `agent_decide` 决策轮次计 1；内层模型重试、RAG 子阶段和工具传输重试不增加 Agent step，但单独记录 attempts。
- 在各次决策后检查 `step_count`；超过限制进入 `STEP_LIMIT_EXCEEDED`，只能澄清、失败安全回复或建人工工单。
- 记录最近 N 次行动签名：`(tool_name, canonical_args_hash, workflow_state_version)`。若同动作反复调用且没有新的业务状态、证据或用户输入，判为 `REPEATED_NO_PROGRESS`；不要仅因为用户连续查询同一订单就直接阻断合法操作。
- 自动循环修复只能在同 invocation 预算内发生，不进行无限重试或后台自动唤醒。

---

## 6. Model Gateway：唯一模型调用入口

### 6.1 功能和接口

覆盖**所有使用主 LLM 的任务**：意图路由、Agent 决策、Query Rewrite、历史摘要生成、最终回复、可选 LLM Judge（Judge 最好使用独立配置，但经过同一适配基础能力）。

```python
class ModelGateway(Protocol):
    async def complete(self, *, ctx: ExecutionContext,
                       messages: list[dict], purpose: str,
                       response_schema: type | None = None,
                       temperature: float | None = None) -> ModelResult: ...
```

可使用 OpenAI `AsyncOpenAI` 封装，与原 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 兼容。若已有 LangChain Model 接口，则保留原上层模型契约、实现 Gateway Adapter，不让图节点和 Gateway 各调用一次模型。

- 优先使用 API 返回的 usage；未提供时可记录 `source='estimated'` 的估算值，但**不得标成 provider 精确统计**；估算也不可可靠时用 `unavailable` + null。
- 可配置 temperature、top_p、max_output_tokens，但必须适配提供方的能力；不默认假设所有兼容 API 都支持 response_format/JSON Schema。
- 当要求结构化输出时：提供方支持则使用原生结构化模式；不支持则使用显式的 JSON 指令 + 严格 Pydantic 校验。结构验证失败可在剩余预算内做**最多 1 次纠错调用**，仍失败返回 `MODEL_BAD_OUTPUT`，不能直接把不合法字段当可信决策。
- `confidence` 由模型产生时只是**模型自报分数**，不等同于校准概率。路由阈值必须根据评测集确定。
- 真实 API 未配置时，`real` 模式返回清晰 `CONFIG_ERROR`；`deterministic` 模式显式依赖 FakeModelGateway，不偷换真实回答。
- 模型重试采用**单层统一策略**：若由 Gateway 控制重试，应设置底层 SDK `max_retries=0` 或关闭重复重试，否则会意外倍增调用。

### 6.2 模型重试矩阵

| 故障 | 自动重试 | 上限 | 说明 |
|---|---|---|---|
| 429 / 临时 5xx / 连接异常 | 可以 | 最多额外 2 次 | 指数退避 + 小幅 jitter；不能超过总 deadline |
| 单次读/生成超时 | 可以，视剩余预算 | 最多额外 1 次 | 不对模型生成结果视为已完成业务写 |
| 401/403、配置缺失 | 不可以 | 0 | 明确错误，防止无限重试 |
| 无效 Schema/解析错误 | 最多 1 次纠错 | 1 | 严格 Pydantic 校验，保留失败证据 |
| 用户取消、总 deadline 耗尽 | 不可以 | 0 | 终止剩余模型调用 |

- `MODEL_TIMEOUT` 和 `RUN_TIMEOUT` 必须区分。
- 预算同时约束尝试次数、延迟和允许的 Token 使用量；不能无限制“纠错—重试—纠错”。
- 对敏感用户数据默认不持久化模型请求/响应全文；Trace 记录 purpose、模型名称、耗时、usage、错误码、脱敏内容 hash。

---

## 7. Tool Executor：所有 MCP 调用统一控制

### 7.1 双层边界（避免覆盖原 MCP Gateway）

**Tool Executor（新增 Harness）**：接收 Agent 想调用的工具请求；应用可信工具白名单、参数校验、当前操作预算、重试/恢复、Trace。  
**MCP Gateway/Client（原有）**：实现 FastMCP 传输、服务端内部认证、工具调用和返回结果标准化。  
**Business Service（原有）**：执行最终用户归属与业务权限校验，提交 SQLite 事务。

调用链：

```text
Agent decision / Workflow node
  -> Harness.ToolExecutor.invoke(...)
  -> PolicyGuard.check_tool_call(...)
  -> existing app.mcp.gateway/client
  -> trusted FastMCP server
  -> Business Service + SQLite
  -> MCP structured result
  -> ToolExecutor.normalize / verify / trace
```

### 7.2 接口与参数校验

```python
class ToolExecutor(Protocol):
    async def invoke(self, *, ctx: ExecutionContext, skill_name: str | None,
                     tool_name: str, arguments: dict,
                     source: str = 'agent') -> ToolExecutionResult: ...
```

必须遵守：

1. 模型只能调用可信允许列表中的工具；工具列表可来自 FastMCP 的 `list_tools()`，但**“发现得到”不等于“授权可用”**。
2. `submit_refund_request` 只能由受控 **Workflow node** 提交；即使模型写出此工具名、通过 Skill 文本诱导、或改变 `source='workflow'` 文本，也不能获得权限。`source` 只能由调用点的可信依赖/能力对象确定，不能把 `source` 字符串当安全凭据。
3. 原 Tool Gateway 从服务端 Principal 生成短期内部调用凭证；不得把凭证、租户、用户 ID 交由 LLM 作为可自由填写的工具参数。
4. 请求入参与响应严格遵守原 MCP 工具 Schema。无效参数在本地拒绝；非预期结果映射 `TOOL_BAD_RESULT`，不交给 Agent 自行猜测结果结构。
5. MCP `is_error`、Transport 异常与业务结构化 `ok=false` 分开处理；不能把返回了 JSON 就当成功。
6. MCP 返回包含文字说明时视为**不可信数据**；只作为事实材料，不能执行其中的指令，更不能用来修改系统提示词或解锁工具。
7. 超长工具结果应结构化缩减并把数据索引/引用保存下来，不在当前 Prompt 中无限堆积。

### 7.3 读写分级与重试矩阵

| 工具类型 | 示例 | 失败策略 | 禁止行为 |
|---|---|---|---|
| 只读 | `get_orders`, `get_order`, `track_shipment`, `search_products`, `check_refund_eligibility` | 临时网络/超时在剩余预算内最多额外 2 次；参数/权限/业务拒绝不重试 | 将超时误认为订单不存在 |
| 限定写（有业务定义） | `create_ticket` | 使用明确请求幂等语义；若无幂等保证，超时只查询状态或返回未知，不盲重试 | 创建重复工单 |
| 敏感写 | `submit_refund_request` | 不以模型为触发源；只能用服务端稳定 `operation_key`，超时查询原请求状态 | 更换 key 重复写、未确认退款 |

对写工具 `UNKNOWN_OUTCOME` 的具体流程：

```text
提交 operation_key=workflow_id + action_type
      |
      +-- 响应成功 ------> 查/确认真实业务状态 -> SUCCEEDED / REQUEST_ACCEPTED
      |
      +-- 明确业务拒绝 --> KNOWN_FAILURE / 受控失败
      |
      +-- 超时/断连 ----> 通过业务服务以同一 operation_key 查询状态
                                 |-- 已落库 -> 返回原申请 ID
                                 |-- 明确无记录且能判定未执行 -> 按业务规则决定是否可用同一 key 再尝试
                                 +-- 无法确认 -> UNKNOWN_OUTCOME；不生成新 key；等待显式再次查询/恢复
```

不能新加一个模型可见的“直接退款”工具来查询/写入。若当前 7 个工具没有专门的按 key 查询端点，可在**可信业务服务内部方法**添加 `lookup_refund_by_operation_key` 供 Workflow/Recovery 使用，保持模型可见工具清单不变。

### 7.4 资源限制

- 工具每次调用超时要小于总 deadline，可增加 `TOOL_TIMEOUT_SECONDS=15`。
- 只读工具并发上限可以采用进程内 `asyncio.Semaphore`，如 `TOOL_MAX_CONCURRENCY=8`；不是分布式限流保证。
- 对外部瞬态故障可做有限退避；暂时不引入复杂的熔断器框架。
- 降级时选择明确的“不确定/请稍后重试/创建工单”结果；不得伪造查询成功或退款已受理。

---

## 8. Context Manager：真正的上下文工程

### 8.1 上下文来源及优先级

输入不仅包括聊天历史，还包括：

1. System/Policy 安全约束（可信系统配置）；
2. 当前用户消息、已认证主体授权范围；
3. 当前任务的业务工作流状态（从 DB 重新读取，不能被摘要替代）；
4. 当前 Skill 的 `SKILL.md` 指令及 hash/version（从白名单目录按需加载）；
5. 当前需要的 RAG 证据及 `chunk_id`/知识版本；
6. 最近对话消息；
7. 已生成的历史摘要；
8. 精简后的历史工具结果（含必要引用）。

**压缩优先级原则**：先删除冗余工具正文与旧无用消息，再压缩低优先级历史；保留当前业务状态、确认信息、近期真实用户意图、关键证据 ID、系统规则和有效 Skill 约束。不能把“用户已经同意退款”这种说法从摘要中升级为后端确认状态。

### 8.2 ContextBudget 与组装接口

```python
class ContextBudget(BaseModel):
    model_context_window: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    max_input_tokens: int

class ContextBuildResult(BaseModel):
    messages: list[dict]
    token_estimate: int
    compression_applied: bool
    compression_level: str
    preserved_refs: list[str]
    summary_id: str | None = None
```

建议入口：

```python
async def build_context(*, ctx, task_state, recent_messages,
                        selected_skill, evidence, tool_results,
                        budget) -> ContextBuildResult:
    ...
```

新增配置（**示例默认值，必须允许覆盖**）：

```env
MODEL_CONTEXT_WINDOW=16384
CONTEXT_RESERVED_OUTPUT_TOKENS=2048
CONTEXT_SAFETY_MARGIN_TOKENS=1024
CONTEXT_MAX_INPUT_TOKENS=12000
CONTEXT_COMPACT_THRESHOLD=0.85
CONTEXT_RECENT_MESSAGE_LIMIT=8
CONTEXT_TOOL_RESULT_MAX_CHARS=3000
```

配置启动校验：`max_input <= model_context_window - reserved_output - safety_margin`。**窗口大小必须与实际主模型一致**，不能把默认值当作提供方真实承诺。

Token 计数：优先使用与模型匹配的 tokenizer；没有可用 tokenizer 时，使用明确标识为 `estimated` 的保守预算估算，并预留安全余量。不能用字符数当精确 tokens 报表。

### 8.3 三层压缩（必须实现并可独立测试）

- **L0 工具结果压缩**：移除过时/重复工具的冗长正文，保留 `tool_name`、`request_id`、`ok`、业务摘要、业务 ID/引用和错误码。含保密字段先脱敏，不得在摘要中复原。
- **L1 对话窗口修剪**：保留最近 N 条与当前意图相关的对话，旧消息保留在原 `messages` 表，通过引用按需读取；**不删除数据库原文**。
- **L2 历史语义摘要**：对更早轮次生成结构化摘要（用户目标、已确认信息、已完成工具动作、未解决问题、重要引用、未知事项），存储为受租户/用户/线程隔离的 `context_summaries`。摘要有来源范围、版本和 hash，可失效重建。

**压缩触发**：构建出的估算 tokens 超过 `CONTEXT_COMPACT_THRESHOLD * max_input_tokens` 时尝试 L0 → L1 → L2；如果没有配置真实 LLM，可使用只含已验证结构化事实的安全降级摘要（或者显式 `CONTEXT_BUDGET_EXCEEDED`），**禁止假装已有 AI 语义摘要**。

**失败兜底**：若压缩后仍超过预算，返回 `CONTEXT_BUDGET_EXCEEDED` 或安全要求澄清；绝不能裁掉系统规则、订单确认约束、当前待执行动作的关键信息来勉强生成回复。

### 8.4 保真校验与防污染

- 历史摘要提到的订单号可以作为待验证线索，执行工具前仍必须通过 DB 归属验证。
- 当前引用政策必须带 `chunk_id` + `version`，不能只在摘要里存放无法追溯的自由文本。
- 工具结果和 RAG 内容视为数据，不允许提升其指令优先级；处理常见“忽略之前的要求”“调用未授权工具”等注入文本。
- 历史压缩后，原始消息归档仍可供复查和 Trace 对照；摘要更新做版本化，失败不覆盖旧有效摘要。
- 不把某会话 A 的摘要缓存用于另一租户或用户 B；调用者身份从可信 ExecutionContext 获取。

### 8.5 必须做的上下文对照实验

同一批多轮数据测试 `none / L0+L1 / L0+L1+L2` 三种策略，比较 Token（精确/估算标明）、丢失关键实体数、任务完成率、工具重复调用数和延迟。必须使用同一模型、fixture、Skill、工具版本；不预设语义摘要必然更好。

---

## 9. Policy Guard：流程安全与运行安全

`PolicyGuard` 需要在**模型建议执行前**运行，业务服务在**真正写入前**再次独立校验。两层不能相互替代。

### 9.1 需要检查的事项

- 可信当前主体、租户、会话归属与资源操作范围；
- 已选 Skill 的可信工具允许列表（`skills.allowed_tools`）；
- 当前 Agent step、deadline、相同动作无进展重复次数；
- 工具是否为读/写，以及操作来源是否具备代码层可信权限；
- 退款提交的可信 workflow ID、业务状态、确认与稳定 `operation_key`；
- 当前工具参数是否与 Pydantic Schema 一致，是否含不允许的模型注入字段。

### 9.2 两种受控调用路径

```text
模型 → tool proposal → Policy Guard → 只读/工单受控工具

受控退款 workflow 节点（后端构造） → Policy Guard
             → Business Service 再校验 → submit_refund_request
```

不要只检查 `tool_name` 或让模型传 `source='workflow'` 就授予写权限。可采用独立内部 `execute_refund_from_workflow(...)` 调用路径，限定调用模块，不把该路径暴露在模型工具 schema 中；**无论 Harness 层判断结果如何，Business Service 仍要执行完整确认与归属校验**。

拒绝时输出 `POLICY_DENIED`，Trace 写 `policy.denied`，Evaluation 应能区分**违规行动尝试**与**实际非法业务副作用**。

---

## 10. Recovery：统一错误处理与恢复

### 10.1 错误分类

| 分类 | 场景 | 应对 |
|---|---|---|
| `TRANSIENT_READ_FAILURE` | 查询工具网络暂时超时/502 | 同 invocation 限次重试，保留 deadline |
| `TRANSIENT_MODEL_FAILURE` | 主模型 429/临时 5xx | 限次退避后返回明确失败 |
| `PERMANENT_VALIDATION_ERROR` | Schema 不合法或参数缺失 | 不循环重试业务写；澄清或终止 |
| `PERMISSION_DENIED` | 跨用户、越权 | 不重试；不泄露目标是否存在 |
| `BUSINESS_RULE_DENIED` | 退款不符合条件 | 按后端原因解释/人工 |
| `WRITTEN_BUT_RESPONSE_UNKNOWN` | 写工具超时，但可能已成功 | 同 key 查询对账，不重复提交 |
| `GRAPH_INTERRUPTED` | 等待用户确认 | 正常挂起，非系统错误 |
| `CONTEXT_EXHAUSTED` | 压缩也无法容纳输入 | 安全告知/要求澄清，保留状态 |
| `PROCESS_RESTART` | checkpoint 存在，进程已重启 | 新 invocation + 原 graph_thread_id 恢复 |

### 10.2 恢复策略的约束

- 只读故障重试与模型重试**各自计数**，但共同消耗剩余 request deadline。
- 写操作的结果未知时，先查持久化业务终态；查不到不代表没有执行，必须根据原服务能否证明“未执行”决定下一步。
- LangGraph 的节点/任务可能因重试、恢复或 interrupt 再执行，因此任何有副作用的节点要么在业务事务里幂等，要么在受控工作流中放在 `interrupt()` 后独立节点，**且必须重新做 DB 条件更新**。
- 不要自动在应用启动时批量扫描并提交等待确认的退款；仅提供显式恢复（确认/状态查询）接口。
- `UNKNOWN_OUTCOME` 不可在评测中算成功或失败退款，应独立统计并通过后续对账确定实际结局。

### 10.3 异常对用户的映射

- 模型临时故障：说明当前暂时无法提供智能回答，可重试；不声称业务已完成。
- 查询工具失败：说明查询暂不可用，不虚构订单状态。
- 退款写调用结果未知：明确是**“申请状态暂未确认”**，允许查询操作状态；不能说“退款失败，所以可以重新提交”。
- 安全拒绝：不输出内部密钥、权限表、其他用户订单存在性或调用堆栈。

---

## 11. Telemetry：统一事件与 Trace

### 11.1 单一数据出口

Harness 只通过 `app.observability.recorder` 写持久 Trace；可以增量补充 `app.harness.telemetry` 作为事件发射 Adapter。Langfuse 仍为**可选、可失败**的观测镜像，不是业务依赖。离线测试不得依赖 Langfuse。

所有事件采用一致 envelope：

```json
{
  "schema_version": "1.1",
  "event_id": "evt-uuid",
  "trace_id": "trace-uuid",
  "invocation_id": "inv-uuid",
  "evaluation_run_id": null,
  "evaluation_case_id": null,
  "conversation_id": "conv-001",
  "graph_thread_id": "conv:conv-001",
  "parent_span_id": null,
  "span_id": "span-uuid",
  "ts": "2026-11-18T10:00:00Z",
  "type": "model.completed",
  "status": "ok",
  "duration_ms": 1200,
  "attributes": {
    "purpose": "intent_route",
    "model_id": "configured-model",
    "attempts": 1,
    "input_tokens": 220,
    "output_tokens": 40,
    "usage_source": "provider"
  }
}
```

时间戳仅为数据格式示例，实际从 UTC 时钟生成。不得把敏感原始工具入参和授权 token 放入 `attributes`。

### 11.2 最小事件目录

| 事件 | 触发点 | 必须字段 |
|---|---|---|
| `invocation.started` / `invocation.finished` | API 执行始末 | status, latency, error_code |
| `intent.classified` | Router | intent, method, model/version, duration |
| `skill.selected` / `skill.loaded` | Skill Manager | name, version, hash, length/tokens |
| `model.started` / `model.completed` / `model.failed` | Model Gateway | purpose, model, attempts, usage, error |
| `context.built` / `context.compacted` | Context Manager | before/after tokens, level, refs/summary ID |
| `rag.retrieved` / `rag.reranked` | RAG | query hash, chunk IDs, snapshot, counts, duration |
| `tool.started` / `tool.completed` / `tool.failed` | Tool Executor | tool, request_id, attempts, outcome, error |
| `policy.denied` | Policy Guard | rule_id, proposed_action, whether side effect happened |
| `workflow.transition` | Business/Workflow | workflow_id, previous, next, version |
| `workflow.interrupted` / `workflow.resumed` | LangGraph | workflow_id, graph_thread_id, interrupt kind |
| `run.step_limit` / `run.timeout` / `run.cancelled` | Run Controller | step_count, elapsed, remaining |
| `recovery.started` / `recovery.completed` | Recovery | error class, action, result |

**去重**：事件要有稳定唯一 `event_id`，对重试的同一操作可通过 `operation_id` + `attempt_no` 区分；不能把一个业务操作的 3 次传输尝试统计成 3 次成功退款。

### 11.3 与现有评测结果关联

- 原 `DEVELOPMENT_SPEC.md` 中的 Trace 基本字段继续保留：`agent_release_id`、`model_id`、`prompt_version`、`skill_version/hash`、`embedding_revision`、`reranker_revision`、`kb_snapshot_id`、`business_outcome`。
- 添加：`invocation_id`、`parent_invocation_id`、`graph_thread_id`、`model_attempts`、`tool_attempts`、`context_before_tokens`、`context_after_tokens`、`compression_level`、`error_class`、`unknown_outcome`。
- 现有 `run_id` 继续表示评测 batch；新运行记录使用 `invocation_id`，避免污染 Evaluation API 与数据集格式。
- 对普通用户展示的 SSE `status/token/tool/interrupt/done/error` 事件，不包含内部密钥、System Prompt、工具原始参数或跨租户 Trace 内容。

---

## 12. 与 LangGraph、Skill、RAG、MCP 的具体改造点

### 12.1 LangGraph 节点如何接入

- 原 `load_state` 读取会话与权威业务状态，不在 node 内创建独立数据库连接/HTTP Client 的无界重试。
- `route_intent`, `agent_decide`, `final_answer` 调用 `ModelGateway.complete()`；其结构化输出走统一校验。
- `skill_select`、`skill_load` 复用现有 Skill Manager，但调用 `Telemetry` 发选择和加载事件。
- `RAG` 调用原 Chroma + BM25 + RRF + 本地 reranker；在完成后发 `rag.retrieved/reranked`；不要把 RAG 包装成偷偷调用外网模型的工具。
- `MCP read` 节点调用 `ToolExecutor.invoke()`，不直接 `Client.call_tool()`。
- `refund_workflow` 使用原有的独立 `wf:{workflow_id}` 图和 DB 状态机；确认后**在单独后续节点**通过受控内部调用执行敏感写。
- `observe/final_answer` 在模型前调用 `ContextManager.build_context()`；生成对话不得把纯摘要中的陈述当业务成功事实。

### 12.2 处理退款等待期间切换话题（必须保持）

```text
聊天 invocation #1: 用户申请退款
  conv: 会话图识别意图 -> 创建 wf-001 -> wf:wf-001 进入 WAITING_CONFIRMATION
  invocation #1 -> INTERRUPTED / response.pending_confirmation

聊天 invocation #2: 用户询问商品颜色
  conv:conv-001 回答商品问题（通过 Product Skill/RAG）
  wf:wf-001 checkpoint 完全不变

确认 invocation #3: POST /workflows/wf-001/confirm
  验证本人/nonce/原始订单/金额/确认状态
  使用同一个 wf:wf-001 以 Command(resume=...) 恢复
  受控 workflow 节点对同一 operation_key 提交/读取真实退款申请
  invocation #3 -> SUCCEEDED 或 UNKNOWN_OUTCOME
```

**技术注意**：普通聊天图 `conv:{conversation_id}` 的顶层运行会结束，而退款 workflow `wf:{workflow_id}` 的 graph invocation 被中断。不要为了维持等待而无限持有模型请求、线程锁或 Python Task。

### 12.3 可插拔意图路由与 Harness 的关系

沿用原 7 个顶层意图和 LLM 结构化分类基线。若原项目已有 Hybrid Router，可在 Router 里保留策略开关并统一记录 `intent.classified`，**不要让 Harness 再写第二个分类器**。Embedding Router 如需密集向量，相同本地 BGE-M3 模型可复用，但不能改变 Chroma 存储与本地加载要求。低置信度路由由原 Router 处理。

---

## 13. 配置变更及资源规划

保留原 `.env.example` 全部条目，追加：

```env
# Harness（默认值仅适用于单机演示，允许覆盖）
HARNESS_ENABLED=true
MODEL_CONTEXT_WINDOW=16384
CONTEXT_RESERVED_OUTPUT_TOKENS=2048
CONTEXT_SAFETY_MARGIN_TOKENS=1024
CONTEXT_MAX_INPUT_TOKENS=12000
CONTEXT_COMPACT_THRESHOLD=0.85
CONTEXT_RECENT_MESSAGE_LIMIT=8
CONTEXT_TOOL_RESULT_MAX_CHARS=3000
TOOL_TIMEOUT_SECONDS=15
TOOL_MAX_CONCURRENCY=8
MODEL_MAX_RETRIES=2
TOOL_READ_MAX_RETRIES=2
HARNESS_TRACE_ENABLED=true
```

**禁止的默认行为**：

- 不在 API 启动期间调用 Hugging Face 下载模型。
- 不因开启 Harness 自动读取敏感对话全文写入第三方平台。
- 不因为某个可选 Langfuse 服务不可达就阻断主业务请求。
- 不设置无界内存事件队列、无界重试或无限 Agent 回合。

性能约束是**可配置的保护阈值**，不是实际跑出的业务 SLO。项目运行在单机，优先保证机制正确；用实测数据报告延迟和 Token，而非预先声称性能收益。

---

## 14. 验收演示与 Evaluation 闭环接入

### 14.1 一条真实执行轨迹必须包含

以“申请退款 O1001”为例：

```text
invocation.started
intent.classified                 refund
skill.selected / skill.loaded    refund-processing
context.built                    selected_skill + current state + history
model.completed                  purpose=agent_decide
policy.checked                  only read/check allowed
 tool.started/completed           check_refund_eligibility
workflow.transition             ELIGIBILITY_CHECKED -> WAITING_CONFIRMATION
workflow.interrupted            graph_thread_id=wf:{workflow_id}
invocation.finished             status=INTERRUPTED

[之后用户确认，产生新 invocation_id]
invocation.started
workflow.resumed                same wf:{workflow_id}
workflow.transition             CONFIRMED -> SUBMITTING
tool.started/completed          submit_refund_request (trusted workflow only)
workflow.transition             SUBMITTING -> REQUEST_ACCEPTED
invocation.finished             status=SUCCEEDED
```

`policy.checked` 为可选额外正向事件；若无，可通过 `tool.*` + `policy.denied` 记录保证可观测。图实际节点名称可与示意不同。

### 14.2 Harness 专属评测指标

| 指标 | 推荐计算方法 | 备注 |
|---|---|---|
| `Invocation Success Rate` | `SUCCEEDED` 次数 / 有有效终态的 invocation 数 | `INTERRUPTED`、`UNKNOWN_OUTCOME` 单独报，不静默混入失败 |
| `Model Retry Rate` | 触发至少一次重试的模型调用数 / 模型调用数 | 还要提供总 attempts |
| `Tool Error Rate` | 失败工具业务调用数 / 工具业务调用数 | 业务调用与传输 attempt 分开 |
| `Step Limit Hit Rate` | `STEP_LIMIT_EXCEEDED` invocation / 有效 invocation | 明确分母 |
| `Context Compression Ratio` | `(before_tokens - after_tokens) / before_tokens` | 仅 `before_tokens>0`，需注明 estimated/provider |
| `Unknown Outcome Count` | 业务写结果未知次数 | 不当作业务失败或成功 |
| `Unauthorized Proposal Count` | 模型提出的越权行动次数 | 与实际副作用独立统计 |
| `Unauthorized Side-effect Count` | 实际未授权写入次数 | 必须为 0（固定安全集）；不能靠 Judge 猜 |
| `End-to-end Task Pass Rate` | 原 Evaluator 基于工具轨迹+DB 终态 | 不能用模型回答代替 |

必须可在现有评测 CLI 运行后输出以上指标（不能采集的指标标注 `NOT_AVAILABLE`，但强制指标必须实现）。新增一个 `harness_eval` 或扩展原 `reporter.py`，**不要另建独立 Eval Engine**。

### 14.3 回归闭环示例

**复现问题**：模型建议在用户尚未确认时直接调用 `submit_refund_request`。  
**Harness 观测**：`policy.denied`，`rule_id=REFUND_CONFIRMATION_REQUIRED`，记录 proposed action；数据库退款请求数增量为 0。  
**Evaluator**：对“非法行动尝试”判定 `WORKFLOW_VIOLATION`；对“真实越权写入”判定 0。  
**修复**：修改可信工具暴露策略/Skill/Agent 节点，而不是删除后端安全校验。  
**回归**：同一 fixture 对比 baseline/candidate，测试非法建议是否减少，同时 T02/T04/T14 等后端硬约束仍通过。

---

## 15. 强制新增测试清单 H01–H25

下列测试要求**可以在无真实 LLM 密钥、无 Langfuse、未下载 Embedding/Reranker 的环境下由 Fake Gateway + Mock MCP 运行**。涉及真实模型时仅作为额外集成测试，不得跳过安全型离线验收。

| ID | 测试 | 关键断言 |
|---|---|---|
| H01 | FakeLLM 统一入口 | 意图/决策/回复不绕过 Gateway；调用次数可统计 |
| H02 | Gateway 未配置真实 API | `CONFIG_ERROR`，不生成伪回复；Fake 模式可运行 |
| H03 | 模型 429 → 恢复 | 有限重试成功，实际 attempts 精确且无 SDK 双重重试 |
| H04 | 模型 401 / 配置错误 | 不重试；错误分类正确 |
| H05 | 结构化输出非法 | Pydantic 拒绝；最多 1 次纠错；最终错误可追踪 |
| H06 | 全局 deadline 耗尽 | 不发起新的 LLM/MCP；状态 `TIMED_OUT` |
| H07 | Agent step 达到上限 | 不再运行工具；`STEP_LIMIT_EXCEEDED` 可评测 |
| H08 | 相同动作无进展循环 | 中止，无无限模型/工具调用 |
| H09 | 只读 MCP 第一次超时 | 有限次重试；总 deadline 不超时预算 |
| H10 | MCP 业务 `ok=false` | 正确归一化，不被当成功 |
| H11 | MCP 传输报错、`is_error=true` | 与业务失败区分，保留规范错误码 |
| H12 | 模型试图调用退款写工具 | Harness `POLICY_DENIED`，数据库 0 新退款 |
| H13 | 写工具超时但已落库 | 查询同一 operation_key 得原单，不产生新退款 |
| H14 | 退款待确认时切换问题 | 原 `wf:{workflow_id}` checkpoint 不丢、不隐式恢复 |
| H15 | Interrupt 节点重入 | 不重复提交任何有副作用操作 |
| H16 | Checkpointer 进程重启恢复 | 用新 invocation + 原 wf thread 继续，状态准确 |
| H17 | 同一 client_request_id 重试 | 同一请求无重复副作用；改 body 返回 409 |
| H18 | 同图线程并发请求 | 无 checkpoint 冲突；不同线程能并行 |
| H19 | SSE 客户端断连 | 已提交写事务不产生新的 idempotency key，不伪造取消 |
| H20 | 上下文 L0/L1/L2 压缩 | 超预算触发；关键业务状态、证据 ID 和规则不丢 |
| H21 | 摘要跨租户/用户读取 | 拒绝；不能复用错误缓存键 |
| H22 | 无真实 LLM 时上下文超限 | 明确 `CONTEXT_BUDGET_EXCEEDED` 或受控安全降级，不虚构摘要 |
| H23 | Trace 结构完整且脱敏 | invocation、模型/工具/压缩事件关联；没有密钥/原始 PII |
| H24 | Langfuse 不可用 | 本地 Trace、业务和离线评测正常工作 |
| H25 | Harness Regression 对比 | baseline/candidate 能发现 fail→pass、pass→fail 和新安全失败 BLOCKED |

**原有 T01–T20 必须全部继续运行**。对真实模型才能覆盖的用例，在报告中明确前置条件、是否执行及原因；不得把未运行标为通过。

### 15.1 最低可用 smoke test

新增 `scripts/harness_smoke.py`，完成以下流程：

1. 使用测试 DB / FakeLLM / 本地 Mock MCP Client 初始化所有 Harness 组件；
2. 运行普通咨询 → 产生 `invocation`、`model.*`、`invocation.finished` Trace；
3. 运行退款申请 → 得到 `WAITING_CONFIRMATION`，记录 `workflow.interrupted`；
4. 运行无关产品咨询 → 待确认工作流不变；
5. 使用测试专用有效 nonce 确认 → 数据库仅 1 条 `ACCEPTED` 申请；
6. 重复 confirm → 返回同一退款申请 ID；
7. 注入模型超时/工具超时 → 验证异常分类与恢复；
8. 运行回归 Evaluator → 输出带 invocation 关联和 attempts 的报告。

示例调用：

```bash
uv run python scripts/harness_smoke.py
uv run pytest -q tests/unit/harness tests/integration/harness
uv run pytest -q
uv run python scripts/run_eval.py --dataset datasets/workflow.jsonl --mode deterministic --agent-release agent-v1
```

若项目当前尚未实现相关模块，应先补充可运行功能并逐步执行；不能仅写测试通过截图。

---

## 16. 增量实施顺序（不要打断原项目 P0–P4）

| 步骤 | 工作 | 可验收交付 |
|---|---|---|
| H-P0 | 盘点现有 API/图/MCP/Trace/评测，设计适配层 | `docs/harness_integration_map.md`，标明复用/变更点 |
| H-P1 | 类型、ExecutionContext、Invocation、Run Controller | H01、H02、H06、H17、H18 基础通过 |
| H-P2 | Model Gateway + Tool Executor + Policy Guard + Recovery | H03–H13 通过；原业务 T01–T06 不退化 |
| H-P3 | Context Manager L0/L1/L2 + 摘要持久化 | H20–H22 通过；模型调用前上下文构建可追踪 |
| H-P4 | Telemetry 与原 Evaluation 关联、Trace、故障回归 | H23–H25 通过；能输出真实报告 |
| H-P5 | 完整集成与文档，跑 T01–T20 + H01–H25 | smoke test、实际测试结果、演示和限制说明 |

**按依赖逐步集成**。若原项目仅完成部分 P0–P4，可优先补契约与核心执行链，再补其他部分。Codex 必须检查现状，不得以本补充文档为由删除或推翻已实现的模块。

---

## 17. Codex 必须提交的文档与最终汇报

1. `docs/harness_integration_map.md`：旧模块 → 新 Harness 模块责任对照、实际改动清单、哪些地方复用原实现。
2. `docs/harness_runtime.md`：运行生命周期图、各层职责、错误/重试策略、上下文压缩策略、示例 Trace、单机局限性。
3. `README.md`：Harness 开关、配置、离线 smoke test、所有测试和评测命令。
4. `IMPLEMENTATION_REPORT.md`：新增 **Harness Completion** 节，列 H-P0–H-P5、H01–H25、T01–T20 实际结果、失败/跳过原因。
5. Schema/Alembic 迁移和 `uv.lock` 变更（如适用）；新增/修改文件列表。

### 禁止在总结中夸大的事项

- 单机 `asyncio.Lock` 不是分布式锁。
- HTTP timeout 并不代表业务写入失败。
- 历史摘要不能保证不丢全部对话细节；必须由可测保真策略与引用机制补足。
- 压缩比、Token 节省和性能提升必须来自真实执行统计。
- 仅有 FakeLLM 运行正确不能证明真实 Agent 的智能任务成功率。
- 本项目尚未上线，不要称具备经过生产验证的高并发 SLA、故障恢复时间或真实客服满意度。

---

## 18. 与旧规格的精确对照（确保不重做）

| 原开发规格章节 | Harness 补充内容 | 是否替代旧章节 |
|---|---|---|
| §5 配置 | 新增 Context/Tool/Model retry 的可配置项 | **仅追加** |
| §6 SQLite | 新增 Invocation/摘要持久化 | **仅追加** |
| §8 LangGraph | 运行入口、依赖注入、step/deadline、暂停/恢复监测 | 不替代 StateGraph |
| §9 Skill Manager | Skill 加载事件与动态上下文预算 | 不替代 Skill Selector/Loader |
| §10 MCP | Tool Executor 上层约束与统一结果 | 不替代 MCP Gateway/认证 |
| §11 退款 | 异常对账、invocation 生命周期、策略双重校验 | **不更改退款业务协议** |
| §12 RAG | 上下文组装、Trace、证据引用保留 | 不更改 Chroma/模型选型 |
| §13 API | Chat/SSE/confirm 入口内部委托 Harness | 不更改外部 JSON 契约 |
| §14 Trace | Event Schema 新增 Harness 字段 | 扩展既有 Recorder |
| §15–16 Evaluation | Harness 指标、故障归因、回归对照 | 扩展既有 Eval Engine |
| §17 T01–T20 | 新增 H01–H25 | **全部保留** |

---

## 19. 官方接口参考（实现时仍需核对锁定版本）

- LangGraph Interrupt / Resume：<https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangGraph Persistence / Checkpoint：<https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph Checkpointers：<https://docs.langchain.com/oss/python/langgraph/checkpointers>
- FastMCP Client 调用：<https://gofastmcp.com/docs/clients/tools>
- OpenAI Python 客户端及重试：<https://github.com/openai/openai-python>
- 原规格中的 Chroma、Agent Skills、BGE-M3、BGE-Reranker 官方链接继续适用。

> 不强制要求使用任何特定的 LangGraph 事件版本、FastMCP 传输辅助参数、OpenAI 结构化输出方法。Codex 应根据 `uv.lock` 中实际安装版本做 smoke test，采用**已验证兼容的 API**，不得复制可能过期的签名。

---

## 20. 直接给 Codex 的最终执行要求

> 请阅读仓库现有 `AGENTS.md` 和 `DEVELOPMENT_SPEC.md`，再阅读本 `HARNESS_ADDENDUM.md`。先检查现有代码，做一个 integration map；在不改动现有业务安全规则、MCP 工具名、Chroma + 本地 BGE 模型、LangGraph checkpoint/thread ID 约定及 Evaluation 数据契约的前提下，按 H-P0 → H-P5 增量实现 Harness。重点保证唯一 Model Gateway、统一 Tool Executor、ExecutionContext、Invocation 生命周期、Context L0/L1/L2 压缩、Policy Guard、有限恢复与统一 Trace。所有实现需有真实离线测试。运行 T01–T20 与 H01–H25、执行 `scripts/harness_smoke.py`，更新文档和 `IMPLEMENTATION_REPORT.md`。不使用空壳实现冒充完成；若某项被阻塞，准确记录原因，继续完成其他可实现部分。
