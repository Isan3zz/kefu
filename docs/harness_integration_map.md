# Harness 增量集成图（H-P0）

基于 AGENTS.md、DEVELOPMENT_SPEC.md、HARNESS_ADDENDUM.md 完整阅读后的代码盘点，2026-11-18。原技术栈和 `.venv` 保留，无新增模型下载、Agent 平台或分布式服务。

| 现有模块 | 复用内容 | 必要改动 / 新增适配 |
|---|---|---|
| app/main.py | Auth、REST/SSE 契约、异常封装 | chat / stream / confirm 委托 HarnessFacade；透传 request_id；新增本人 invocation 查询 |
| app/agent.py | StateGraph 节点、conv:/wf: thread、AsyncSqliteSaver、interrupt/Command、业务回执核对 | 注入 Harness services；唯一 ModelGateway；统一 ToolExecutor；决策步数；移除全局锁与节点 SDK 请求 |
| app/skills.py + catalog | 原四技能发现/选择/懒加载、可信工具白名单 | 不重写；通过已有 Recorder 的适配事件关联 invocation |
| app/mcp.py | 七工具、FastMCP Client/Server、HMAC actor、原返回 schema | 增加单次传输选项供 Harness 接管唯一重试层，旧直接调用兼容；识别 is_error |
| app/business.py | 所有归属、确认、事务、资格和退款幂等 | 不改变规则；Recovery 复用 refund_outcome 按原 operation_key 对账 |
| app/rag.py | 本地 BGE、Chroma/BM25/RRF/rerank、知识治理与快照 | 不重写；节点补结构化检索/重排事件与证据上下文 |
| app/observability.py | JSONL + SQLite 唯一 Trace 出口、脱敏、evaluation_scope | 扩展 schema 1.1 envelope 与 Harness 元数据，保留旧事件供旧评测消费 |
| app/evaluation.py | 数据集、隔离 fixture、断言、报告/比较、review/promote | 消费新事件；添加 Harness 指标与硬门禁；Judge 复用统一 ModelGateway |
| app/config.py / .env.example | 全部原配置 | 追加且校验 Context / timeout / retry / concurrency 保护阈值 |
| app/db.py / migrations | 现有业务模型和历史迁移 | 0003 仅添加 agent_invocations、context_summaries 及去重/查询索引 |
| app/harness/（新增） | 复用以上服务，不再建图/网关/Trace 库 | 契约、可信 ExecutionContext、RunController/Facade、模型/工具执行、PolicyGuard、Recovery、ContextManager、Telemetry |
| scripts / tests（新增） | 原 T01–T20 与所有回归测试保留 | harness_smoke、上下文三组对照、H01–H25 和跨进程/断连/并发/故障测试 |

H-P0 基线输出 `data/harness-baseline-tests.log`。依次推进 H-P1（契约/调用生命周期）→ H-P2（统一调用/策略/恢复）→ H-P3（上下文）→ H-P4（观测/评测）→ H-P5（完整测试与文档）。真实 LLM 保持先前用户要求不调用；本地 Mock HTTP 和显式 Fake Gateway 验证适配，真实 BGE 权重沿用已下载目录。

安全权威仍为 Business + SQLite；摘要不是授权，source 字符串不是退款能力；invocation_id、client_request_id、MCP request_id、operation_key、evaluation run_id 分别保留各自语义。按 thread 的进程内锁只用于单 worker 图执行串行化，业务并发正确性由原数据库独立保证。
