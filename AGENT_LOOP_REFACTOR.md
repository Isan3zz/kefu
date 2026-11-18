# 智能客服 Agent — Agent Loop 架构重构增量开发规格

> 文档版本：v1.0（2026-11-18）  
> 适用范围：已依据 `DEVELOPMENT_SPEC.md`、`HARNESS_ADDENDUM.md`、`MEMORY_ADDENDUM.md`（v1.1）实现或正在开发的 Python 智能客服项目  
> 执行者：Codex，**基于当前代码增量重构，不可重做整个项目**。

## 0. 目标与规范优先级

**目标**：将原 `route_intent -> skill_select -> skill_load -> domain-specific node -> answer` 固定路由式编排，升级为**单主 Agent、模型驱动的有界 Tool-Calling Agent Loop**。LLM 在每轮中自主决定调用哪个*已授权*工具、是否加载 Skill、是否需要 RAG/订单/物流/人工工单或结束。Harness 对模型、工具、上下文、预算、取消、恢复、Trace 实施控制。写入型业务仍由确定性 Workflow + Business Service 强制执行。

规范优先级（冲突时）：
1. 既有文件中的**安全/业务不变量**及接口契约：权限隔离、退款确认与服务端幂等、记忆授权、MCP 认证、证据约束、实际数据校验。
2. 本文件的 Agent 图结构、路由方式、Skill 激活机制、Agent Loop 状态和评测变更。
3. `MEMORY_ADDENDUM.md` v1.1 的记忆提取/检索规则；`HARNESS_ADDENDUM.md` 的 Harness 生命周期/统一调用接口。
4. `DEVELOPMENT_SPEC.md` 的其余模块、技术栈、接口、数据模型、数据集、验收项。

**明确替换的旧要求**：
- `DEVELOPMENT_SPEC.md` §8.2 中强制串行 `route_intent -> skill_select -> skill_load -> agent_decide` **废止**，改为本文件 §2 的核心图。
- §8.3 中“每条消息默认先走单独结构化 LLM 意图分类”**不再是执行前置条件**。保留意图分类器供实验、弱提示、特殊命令和 Trace，不允许分类器决定最终工具授权。
- 原四种 Skill、七个 MCP 工具、RAG、Memory、Harness、Evaluation **保持存在**；不把功能绕开或砍掉。
- 原普通会话 `conv:{conversation_id}` 和退款 `wf:{workflow_id}` 的隔离、`AsyncSqliteSaver`、`interrupt/Command(resume=...)`、`/confirm`、`/cancel` 必须继续有效。

**不要**：引入多 Agent、独立任务规划 LLM、自研图调度器、给 LLM 任意 Python/shell/SQL 执行权、以 Prompt 代替业务鉴权，或把“Agent Loop”写成 `while True: llm(...)` 而没有可持久化的执行边界。

## 1. 期望行为（功能准则）

输入一个请求，运行时可产生 0..N 个工具调用：
- `查询耳机的保修政策` → Agent 选择 `search_knowledge`（Chroma + BM25 + RRF + 本地 Reranker）→ 读取证据 → 回答并引用真实来源。
- `查订单 O1001 到哪了` → Agent 可直接用 `get_order`、`track_shipment`（不需要每次先调用分类 LLM），由后端按登录用户验证归属。
- `我想退 O1001，先看看能否退款` → 加载 `refund-processing` Skill（需要时）→ `check_refund_eligibility` → 解释资格，不自动提交。
- `帮我退 O1001` → 校验资格 → 通过**受控** `begin_refund_workflow` 创建退款预览/待确认工作流 → 返回 `pending_confirmation`，主循环停止本轮；后续**只能**通过受控 `/confirm` 恢复独立 `wf:{workflow_id}` 流程，最终的 `submit_refund_request` **不暴露给 LLM**。
- `你们售后政策是什么？顺便查下我的订单` → 一个 invocation 内可执行 RAG + 订单工具，多步自适应。
- `我不喜欢入耳式耳机` → 视用户 Memory 同意状态，当前答复后仅执行 `MemoryTrigger`；若命中候选，批量提取由 Memory Worker 在适当时机完成。不得把每条对话改成即时 LLM 记忆提取。
- 退款等待确认时的其他咨询 → 新的普通对话 invocation 不会隐式恢复退款工作流。

## 2. 主架构：模型循环 + 确定性边界

```text
FastAPI -> Harness.RunController (Principal / deadline / limits / trace)
       -> LangGraph StateGraph (thread_id=conv:{conversation_id})
START -> prepare_context [load chat / current state / memory search / skill metadata]
       -> agent_decide [ModelGateway, available tools, bounded context]
           |-- final text ----------> validate_final -> END
           |-- tool_calls ----------> execute_tools [PolicyGuard -> ToolExecutor]
                                     |-- load_skill / search_knowledge / MCP read / create_ticket
                                     |-- begin_refund_workflow (guarded entry)
                                     '-- tool results as ToolMessage
                                             |
                                    check_budget_and_progress
                                      |-- continue -> agent_decide
                                      |-- pending_confirmation -> finalize_pending -> END
                                      '-- limit/error -> safe_fallback -> END

独立退款子流程（不是 LLM 循环的一部分）：
begin_refund_workflow -> trusted refund graph (wf:{workflow_id})
  -> eligibility/revalidate -> create bound preview -> interrupt()
  -> user confirms via POST /workflows/{id}/confirm
  -> server validates nonce/actor/state transactionally -> Command(resume=...)
  -> submit_refund_request via trusted workflow-only path -> persist outcome
```

**核心原则**：LLM 只能**提议**执行工具；Harness 根据当前权限/风险/预算决定是否执行；业务服务是写操作最终裁决者。Tool 结果驱动下一轮 Agent 决策，不允许工具调用失败后由模型虚构成功。

**工具执行适配**：可用 LangGraph 自定义 `execute_tools` 节点（推荐，便于复用 Harness ToolExecutor），也可以在满足完全相同约束时使用 `ToolNode`；不得出现某个 Tool 绕过 `PolicyGuard` / `ToolExecutor` 的分支。固定节点不再决定领域流程。**Agent Loop 的主闭环必须真实存在**：`agent_decide -> execute_tools -> agent_decide`。

## 3. Agent State 与执行上下文

保留当前 `AgentState`，以迁移方式增补字段，不要重新建一份与原有 checkpoint 不兼容的状态。建议：

```python
from typing import Annotated, Any, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    current_skill: str | None
    loaded_skill_versions: dict[str, str]
    loaded_skill_hashes: dict[str, str]
    tool_observations: list[dict[str, Any]]  # 受预算限制的结构化摘要，不保存秘密
    last_tool_batch: list[dict[str, Any]]
    pending_workflow_id: str | None
    terminal_reason: str | None  # COMPLETED / NEEDS_INPUT / PENDING_APPROVAL / HANDOFF / LIMIT / ERROR
    iteration_count: int
    tool_call_count: int
    repeated_action_count: int
    final_response: str | None
```

- 既有状态的其他字段（`conversation_id`, `active_workflow_id`, `retrieved_chunks`, `trace_id` 等）原位保留，按现有 SQL/checkpoint 迁移策略处理；不要直接丢字段。
- 使用 `add_messages` 维护消息，工具结果必须是与模型 `tool_call_id` 对齐的 `ToolMessage`，不能仅以 Python 字典 append 导致 LLM 看不到工具结果。
- **敏感可信信息**：`tenant_id/user_id/role` 必须从 `ExecutionContext.principal` 获取；即使旧 State 存有同名字段，也不能作为授权依据。不要让 Tool 参数覆盖可信身份。
- `ExecutionContext` 保持请求级不可变，继续从 FastAPI 认证层构造；线程归属由服务端验证。
- 主循环状态与独立退款 workflow state 不要共享可随意覆盖的 `messages` 列表。两者通过可信 `workflow_id`/持久化映射关联。
- 进入该迁移版本前处理 checkpoint 的版本兼容：支持旧 checkpoint 的迁移/忽略新字段默认值；测试新旧会话启动。禁止直接删除用户已有 checkpoint/业务数据。

## 4. Model Gateway 的 Tool-Calling 接口

已有 `ModelGateway.complete(...)` 保持兼容；新增或扩展统一接口，使其支持**绑定合法工具 Schema 的模型调用**：

```python
class ModelGateway(Protocol):
    async def complete_with_tools(
        self, *, ctx, messages, tool_schemas: list[dict],
        tool_choice: str = "auto", max_output_tokens: int | None = None
    ) -> "ModelDecision": ...

class ModelDecision(BaseModel):
    content: str | None
    tool_calls: list["ProposedToolCall"]
    raw_message: object  # 适配为可加入 LangGraph messages 的 AIMessage
    usage: dict | None
    finish_reason: str | None
```

`ProposedToolCall = {id:str, name:str, arguments:dict}`；使用实际锁定 SDK 的 tool-call 格式正确映射为 `AIMessage`，不手写伪造 `tool_calls` 内容。错误 Schema、不可解析参数或未知工具名映射为可评测错误，不执行任意函数。

- 单次模型返回无工具调用且包含有效答复 → 进入 `validate_final`；无答复也无工具调用 → 失败/澄清，不允许无穷调用。
- 模型一次提出多个 tool calls：默认**顺序执行**，记录每个结果；本期不要引入并发写工具。某个调用出现风险或需确认时按稳定顺序停止本批剩余调用，并返回规范化的 `ToolMessage`/状态，不得遗漏已发出的 tool_call_id。
- 对 SDK/provider 不支持 Tool Calling 的主 LLM，启动前明确报 `MODEL_TOOL_CALLING_UNSUPPORTED` 或使用测试 Fake；**不要**把纯文本 JSON 猜测解析当真实工具调用能力。
- 支持 Provider 级超时和 429/5xx 有界重试；已产生副作用的 Tool 超时不属于可重试模型请求。
- LLM 的 `content` 中即使包含“我已退款”也不意味着业务完成。

## 5. Agent Tool Catalog 与 Skill 渐进披露

Agent 默认看到**精简且可信的工具目录**，不是所有后端 API：

| 工具名（模型可见） | 实际实现 | 风险/规则 |
|---|---|---|
| `load_skill` | Skill Manager 读取已注册 `SKILL.md` | 只读、按需加载、路径白名单 |
| `search_knowledge` | 现有 RAG Service | 只读、带来源 ID/版本 |
| `get_orders` | Harness -> MCP 业务工具 | 只读、当前用户订单列表 |
| `get_order` | Harness -> MCP 业务工具 | 只读、订单归属校验 |
| `track_shipment` | Harness -> MCP 业务工具 | 只读、服务端归属校验 |
| `search_products` | Harness -> MCP 业务工具 | 只读 |
| `check_refund_eligibility` | Harness -> MCP 业务工具 | 只读、归属校验 |
| `create_ticket` | Harness -> MCP 业务工具 | 受控低风险写、固定幂等键 |
| `begin_refund_workflow` | 内部受控工作流入口，非任意 MCP 写工具 | 预览/确认；不提交退款 |

`list_skills` 可实现为只读工具，但首次上下文中已经有所有 Skill 的**元数据**时不强制额外调用；加载正文必须明确调用 `load_skill(name)` 或由可信 Harness 在同等条件下执行自动加载。建议首版让模型调用 `load_skill` 以便 Trace 可观测。

**原七个 MCP Tools 不变**。模型可见的 `begin_refund_workflow` 是**安全的内部 Agent Tool Adapter**，不是新建一个可跳过确认的业务 MCP Tool；它调用原有工作流服务。`submit_refund_request` 永远不进入模型工具列表。即使模型构造名称或 Skill 指令诱导，也不能调用。

- 首次 Prompt 只注入 4 个 Skill 的 `name`、`description`，不放全量 SOP 正文。
- `load_skill` 校验白名单名称、路径、已启用状态、版本、文件 hash、最大文本长度；将加载的 Skill 指令返回给模型并记录 `skill.selected/skill.loaded`，在后续 Loop Context 中可复用。不得将 `SKILL.md` 当高于系统安全策略的指令。
- Agent 可以直接调用某些只读工具而不加载 Skill；若某个场景期望使用 Skill，Evaluation 独立评估 Skill 激活质量。**不强制所有问题每次都走 Skill**。
- 动态工具可见性依据可信执行上下文、模型能力和注册工具白名单；Skill 只能缩小适用范围或提供指导，不能提升权限。
- RAG 证据、Skill 内容、Tool 原始文本、Memory 内容都是数据/低信任文本，不能借工具结果修改系统指令或操作权限。

## 6. 关键节点逻辑与终止判定

### 6.1 `prepare_context`
- 从服务端确认会话归属、最新普通会话 checkpoint。
- 查询长期记忆（如果用户同意、且开关启用），最多 K 条并限定 token 配额；订单/物流实时事实永远不从长期记忆作为真相读取。
- 仅加载 Skill 元数据；注入可见工具 Schema；按 Harness ContextManager L0/L1/L2 策略构建上下文。
- 应避免“每次进入 decide 节点都重新检索长期记忆或重新调用意图分类”。

### 6.2 `agent_decide`
- 调用 `ModelGateway.complete_with_tools`；校验 tool calls 的 JSON Schema 和数量；将 `AIMessage` 加入 graph state。
- 不允许直接修改后端业务状态；不能把内部服务秘密写进提示词。
- 迭代次数及模型调用次数在 Harness 计量（配置默认 `AGENT_MAX_STEPS=8`，定义为每次模型决策的最多次数）；Tool 调用单独设预算，建议 `AGENT_MAX_TOOL_CALLS=12`；请求 deadline 继续默认 90s。

### 6.3 `execute_tools`
- 按 `AIMessage.tool_calls` 获取 `id/name/args`，对每项调用 Harness `ToolExecutor` 或受控适配器。**每个请求必须经过 PolicyGuard**，包括 `load_skill`、RAG 和内部工作流入口。
- Tool 返回以 `ToolMessage(tool_call_id=...)` 附加给 `messages`，内容为脱敏且长度受限的结构化结果，并记录完整非敏感 Trace 摘要。
- `search_knowledge` 的结果包含 `chunk_id, document_title, version, snippet`；`get_order` 等返回可信结构化结果且可被回复验证。
- 工具失败返回明确错误码并可给 Agent 一次纠错机会；安全拒绝不能由 Agent 重试升级权限；同一错误/参数反复无进展触发 `REPEATED_NO_PROGRESS`。

### 6.4 `check_budget_and_progress`
- 检查 `deadline`、max steps、max tool calls、重复动作签名（工具名+规范化参数 hash+工作流状态版本）。
- `begin_refund_workflow` 返回 `pending_confirmation` 时主 Loop 必须立即停止，输出受控预览和确认入口；不得进入下一轮让模型“代用户确认”。
- 其他情况工具执行成功/可恢复失败可回到 `agent_decide`；不可恢复失败安全结束，必要时建议人工工单。

### 6.5 `validate_final`
- 检查有无未匹配的 tool_call_id、是否声称未被证实的退款/查询结果、知识证据引用是否合法。
- 校验最终答复是否符合已验证业务状态；缺乏证据则返回“未能确认”或澄清，而不是生成事实。
- 结果应有 `terminal_reason`、工具数量、模型决策次数、Token 消耗（可得则实际、不可得则标估算）。

**终止条件**：正常答复、需要用户补充信息、待用户确认、人工转接、达到预算、超时、不可恢复失败、取消。每个终态可在 Trace 与 API 响应中追溯。

## 7. Refund Workflow：必须保留原安全模型

升级 Agent Loop **不允许**让 Agent 直接调用 `submit_refund_request`。推荐：
1. `begin_refund_workflow(order_id)` 只接受模型提供的业务对象标识，不接受 `user_id`、`tenant_id`、确认结果、最终金额、业务幂等键等敏感字段。
2. 受控 Adapter 读取 `ExecutionContext.principal`，由后端查询资格/归属/金额/政策；创建或复用服务端工作流实例，分配稳定 operation key，生成预览及确认 nonce hash。
3. 使用独立 `wf:{workflow_id}` graph thread 在 `interrupt()` 等待用户确认；注意 interrupt 前的写操作必须幂等/可恢复，而且恢复节点不能重做未保护的副作用。
4. 主 Agent Loop 获取 `pending_confirmation`，本轮以 `PENDING_APPROVAL` 结束，允许用户继续发其他普通问题。
5. `POST /api/v1/workflows/{id}/confirm` 通过可信用户身份 + nonce + 当前金额及状态的事务校验后，使用 `Command(resume=...)` 恢复*该 wf thread*，再由受控后端路径提交退款。
6. 重复 confirm、重复 begin、恢复中断、MCP 写请求超时都必须收敛到同一业务结果，不重复新增退款申请。
7. `cancel` 不得撤销已经受理的退款；不明状态先查原操作结果。

LangGraph 可能重新运行包含 `interrupt()` 的节点，因此严禁在该节点的 interrupt 前直接执行非幂等写操作。最终权限验证位于业务层，而不是 Skill/Prompt/Harness 的软约束。

## 8. Harness / Memory / RAG 的兼容性

### 8.1 Harness
- `RunController` 的 invocation 生命周期、并发锁、超时、取消、重新恢复全部保留；`ModelGateway` 新增 tool calling 能力；`ToolExecutor` 复用现有网关与错误分类。
- 每次模型决策和工具调用都记录 `run.step`、预算消耗；上下文压缩只压低信任旧内容，不得删当前请求、工具调用对应关系或等待确认状态。
- 若一次模型输出多个工具调用，要保证所有发出的 call 都产生关联的观察消息；如中途必须暂停，记录未执行 call 的 `SKIPPED_DUE_TO_PENDING_APPROVAL` 结果，不能暗中执行剩余写操作。

### 8.2 Memory
- `MemoryService.search()` 在 `prepare_context` 按授权加载相关长期记忆，不要求 LLM 每轮显式查询。
- `MemoryTrigger.consider_extraction()` 在已保存的正常**用户消息处理后**执行；若是安全失败或业务中断，也不能将模型工具响应伪装成用户消息来提取。
- 记忆批量提取触发规则、用户授权撤回、SQLite 权威存储 + Chroma 索引、Memory On/Off 评测 **完全遵循** `MEMORY_ADDENDUM.md` v1.1。
- 不能让模型直接通过自由工具绕过 MemoryService 的用户授权、来源证据和删除规则。

### 8.3 RAG
- RAG 作为一个模型可调用的 `search_knowledge(query, filters?)`，内部复用原 `Rewrite -> Chroma + BM25 -> RRF -> Local Reranker` 实现；对 Query Rewrite 不要必然每次增加一轮 LLM，使用已有策略开关。
- `filters` 不接受模型自由指定的其他租户 ID；后端结合可信 Principal 进行权限与版本过滤。
- 回答必须引用真实检索结果的 chunk_id / version；检索失败时 Agent 可以追问或说明未知，但不能编造检索依据。

## 9. 代码目录的增量调整

```text
app/agent/
  graph.py                 # 修改：主图 agent_decide ↔ execute_tools 循环
  state.py                 # 修改：消息 reducer + Loop 状态字段
  nodes/
    prepare_context.py     # 新增或复用已有 load_state
    decide.py              # 新增：统一模型决策入口
    execute_tools.py       # 新增：执行模型 tool calls（不绕过 Harness）
    finalize.py            # 新增：输出验证、pending、安全终止
  tool_catalog.py          # 新增：模型可见工具 Schema 与信任等级注册
  tool_adapters.py         # 新增：内部 Skill/RAG/Workflow 工具适配
  router.py                # 保留供 baseline/可选预分类/评测，不再强制为主图前置
app/harness/
  model_gateway.py         # 扩展 complete_with_tools
  tool_executor.py         # 复用统一执行/错误规范
  context_manager.py       # 适配 ToolMessage、Skill 文本与 token 预算
  policy_guard.py          # ToolCatalog + Trusted Principal + budgets
  telemetry.py             # 增加 loop.* 事件
app/skills/                # 原 4 个 Skill 保持
app/mcp/                   # 原 FastMCP 服务/工具名不变
app/rag/                   # 原 Chroma/BM25/Rerank 不变
app/memory/                # 原长期记忆 v1.1 不变
tests/unit/agent_loop/
tests/integration/agent_loop/
datasets/agent_loop.jsonl
```

Codex 必须**先检查仓库实际现有文件**并做最小变更；若有同等功能，复用而不是添加重复模块。

## 10. 推荐的 LangGraph 最小骨架（结构示意，按锁定版本对齐 API）

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)
builder.add_node("prepare_context", prepare_context)
builder.add_node("agent_decide", agent_decide)
builder.add_node("execute_tools", execute_tools)
builder.add_node("check_budget", check_budget_and_progress)
builder.add_node("validate_final", validate_final)
builder.add_node("safe_fallback", safe_fallback)

builder.add_edge(START, "prepare_context")
builder.add_edge("prepare_context", "agent_decide")
builder.add_conditional_edges(
    "agent_decide", route_after_decision,
    {"tools": "execute_tools", "final": "validate_final", "error": "safe_fallback"}
)
builder.add_edge("execute_tools", "check_budget")
builder.add_conditional_edges(
    "check_budget", route_after_tools,
    {"continue": "agent_decide", "pending": "validate_final", "stop": "safe_fallback"}
)
builder.add_edge("validate_final", END)
builder.add_edge("safe_fallback", END)

# graph = builder.compile(checkpointer=existing_async_sqlite_saver)
```

约束：
- `route_after_decision` 读取当前模型消息的真实 `tool_calls`，不是用固定 Intent 字符串决定跳转。
- `check_budget` 能把 `pending_confirmation` 与普通最终答复区分开。
- SDK 输出的 `AIMessage`、`ToolMessage` 由适配器正确转换。注意 `add_messages`、异步节点和 checkpoint 的版本差异；要用 smoke test 验证，而不是仅写能通过静态检查的代码。
- 如果工具返回 `pending_confirmation`，在最终响应包含真实 workflow ID、待确认订单和金额，但敏感 nonce 仅通过原有受控确认响应承载。
- `graph` compile/checkpointer 的生命周期与原 API 保持一致，不为每个请求随意创建一次新 checkpoint 数据库。

## 11. 结构化 Trace 与 Evaluation 修改

保留原 `observability.recorder`，新增事件：

| 事件 | 关键字段 |
|---|---|
| `loop.started` | invocation_id, graph_thread_id, release |
| `loop.decision` | iteration, tool_call_count, message_id, duration, model_version |
| `loop.tool_proposed` | tool_call_id, name, normalized_args_hash, source=model |
| `loop.tool_allowed` / `loop.tool_denied` | rule_id, risk_class, source, error_code |
| `loop.tool_observed` | tool_call_id, outcome, evidence_ids, tool_duration |
| `loop.skill_activated` | name, version, hash, load_cost |
| `loop.no_progress` | repeated_signature, count |
| `loop.terminated` | terminal_reason, iterations, tool_calls, token_usage |
| `loop.workflow_pending` | workflow_id, graph_thread_id=wf:... |

所有事件关联现有 `trace_id`、`invocation_id`、`agent_release_id`。禁止记录原始凭证、完整敏感数据或模型的隐私相关隐藏推理；`loop.decision` 记录**可见行为决策**而非链路无法得到的内部思维。

Evaluation 继续执行原 T01–T20、H01–H25、M01–M32，追加如下场景：
- **Loop Task Success**：成功完成多步任务的比例；分母为有明确终态的样本数。
- **Tool Selection / Argument Accuracy**：模型选择是否正确，参数是否符合权限/场景，最终业务状态是否正确。
- **Skill Activation Precision/Recall**：是否在需要时加载相应 Skill，非必需时是否滥用。
- **Redundant Tool Rate**：不必要的工具调用 / 全部工具调用；由 gold trace 或可核验规则判定。
- **No-Progress Rate**：循环重复但没有新增证据/状态的比例。
- **Avg / p95 Loop Iterations, Tool Calls, Latency, Tokens**：对照模型及策略版本。
- **Workflow Compliance**：敏感写工具被禁止、等待确认成功终止、二次确认与幂等正确。

保留原 `intent.jsonl`、`skill.jsonl` 和 router baseline；但真实主图可以完全不调用显式 Intent Router。若评测需要意图标签，可离线独立算分类准确率或从 Trace 派生行为标签，**不要伪称它是在线的显式分类结果**。

评测对照组：
A. 旧固定路由 Workflow（仅基线、可选开关 `ORCHESTRATOR_MODE=legacy`）。
B. Agent Loop + MCP/RAG（无动态 Skill，能力不变，安全边界一致）。
C. Agent Loop + 按需 Skill + Harness（Proposed）。

确保相同任务 fixture、相同 LLM、相同知识快照、相同受控业务规则；报告成功率、重复工具率、Token 消耗和 p95 延迟。不要预设 Agent Loop 比 Workflow 更高效。

## 12. 必做新增验收测试（L01–L25）

| ID | 测试 | 必须断言 |
|---|---|---|
| L01 | 单轮问候 | 不调用工具，直接返回，终止原因正确 |
| L02 | 自主工具选择 | 不经过强制 Router 仍能调用订单工具 |
| L03 | 多工具组合 | 一轮提问顺序查知识和订单，最终整合真实结果 |
| L04 | 模型 tool_call 解析 | AIMessage/ToolMessage ID 一一对应 |
| L05 | 非法 JSON 参数 | 拒绝执行并产生明确错误/可恢复行为 |
| L06 | 未知工具 | 统一拒绝，不能执行任意 Python 函数 |
| L07 | Skill metadata | 初始只有元数据，未全量加载正文 |
| L08 | Skill 动态加载 | 可 load_skill，版本/hash 与 Trace 一致 |
| L09 | Skill 路径穿越 | 非法名称与路径被拒绝 |
| L10 | Skill 越权诱导 | 恶意 SOP 不得开放写工具 |
| L11 | RAG 证据 | 只能引用实际 chunk_id/version |
| L12 | MCP 权限 | A 用户不能查询 B 用户订单 |
| L13 | 敏感提交不可见 | 模型工具 Schema 中无 submit_refund_request |
| L14 | 退款待确认 | begin 后进入真实 pending，退款新增数 0 |
| L15 | 退款确认恢复 | /confirm 后同 wf 线程恢复，新增申请恰好 1 |
| L16 | 重复确认/超时 | operation_key 稳定，不能重复退款 |
| L17 | 退款中换话题 | 普通 conv 能回复，wf pending 不受影响 |
| L18 | 循环步数耗尽 | 正确 LIMIT 终止，不死循环 |
| L19 | 重复无进展 | 相同参数无状态变化触发终止 |
| L20 | 多 Tool Call 一批 | 顺序执行，有唯一观察 ID，风险时停止剩余操作 |
| L21 | 工具失败 | Agent 只可合理纠错/告知，不虚构成功 |
| L22 | 记忆跨会话 | 授权用户记忆可检索且不覆盖实时订单 |
| L23 | 记忆无提取触发 | 普通查询不会每轮做 LLM memory extraction |
| L24 | Trace 一致性 | 每个工具提议有允许/拒绝、观察或明确终止记录 |
| L25 | 评测回归 | 同 fixture 可运行 deterministic 回归并输出新旧版本报告 |

测试要求：Fake Tool-Calling Model 能按顺序返回 AIMessage tool_calls；Mock MCP、固定 Chroma fixture、Fake Clock、Fake Memory Worker；无付费 API 时仍可运行所有确定性测试。至少另配 3 条真实 LLM tool-calling smoke case，未配置 API 时明确标记 `SKIPPED`，不能记录成 PASS。退款安全用例必须由后端和 SQLite 断言，而非只看模型输出。

## 13. 分阶段迁移计划（Codex 顺序执行）

**L-P0：现状检查、接口映射**
- 盘点现有 `graph.py/router.py/skill_manager/Harness/Memory/RAG/Tool Gateway`；编制修改矩阵、检查现有测试与真实依赖版本。
- 建立迁移前基线（T/H/M 测试可运行部分），记录 Git 状态，不删除文件。

**L-P1：Tool Catalog + ModelGateway Tool Calling**
- 建模型可见工具 Schema、受控 Adapter、FakeToolCallingModel；模型真实 Tool Call 协议与 ToolMessage 对齐；完成 L04–L10、L13。

**L-P2：LangGraph Agent Loop**
- 将固定 Route Graph 改为两个核心循环节点；保持 Harness 调用、Context、Budget、Trace；完成 L01–L03、L18–L21、L24。

**L-P3：Refund/Memory/RAG 接入**
- 用受控 Adapter 接入既有退款 workflow（不放开写工具）；验证 thread 分离、confirm 和 idempotency；继续复用 Chroma、BGE-M3、Reranker 和 MemoryTrigger；完成 L11–L17、L22–L23。

**L-P4：Evaluation / 回归 / 文档**
- 更新自动评测 schema、失败类别、指标；跑 L01–L25 以及原 T/H/M；新增 Agent Loop 对照实验脚本并生成真实报告（无模型则标 skip）。
- 更新 `IMPLEMENTATION_REPORT.md`：每项真实代码路径、命令、通过/失败/跳过、模型与工具版本、遗留风险；更新 README 与架构示意图。

## 14. Codex 完工定义

可验证的完成条件：
1. 普通会话的主控制图**确实有** `agent_decide -> execute_tools -> agent_decide`，模型 Tool Call 决定行动，而不是先进行固定意图路由才选执行分支。
2. 所有工具调用穿过 Harness/PolicyGuard；Skill 动态加载、MCP 业务工具、Chroma RAG、Memory 检索都可以在新架构下真实工作。
3. 退款仍需要独立的用户确认 API 和业务层事务/幂等；模型完全看不到直接提交退款工具。
4. 所有 Loop 步骤均有结构化 Trace，Evaluation 能判断工具轨迹和业务终态。
5. L01–L25、T01–T20、H01–H25、M01–M32 的结果如实记录；任何因依赖/模型缺失而未运行的检查必须明确标记 SKIPPED/BLOCKED。
6. 不删除现有核心能力，不重写固定的业务安全约束，不伪造测试通过或性能提升。

## 15. 当前参考文档（Codex 可在需要时核对锁定版本）

- LangGraph 工作流与 Agent：<https://docs.langchain.com/oss/python/langgraph/workflows-agents>
- LangGraph Interrupts：<https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangChain Agents / Tool Calling：<https://docs.langchain.com/oss/python/langchain/agents>
- Agent Skills 规范：<https://agentskills.io/specification>

**注意**：文档中的接口与图示是可实现的技术契约，但 LangGraph、LangChain、FastMCP 的真实 API 必须根据当前项目 lockfile 做 smoke test；不得抄写与已锁定版本不兼容的旧代码。
