# 长期记忆：架构、演示与维护

## 复用与边界

| 已有组件 | 增量接入 |
| --- | --- |
| FastAPI Bearer 认证、Harness `ExecutionContext.principal` | 记忆 API 只用认证身份；消息再查所属会话及 USER 角色。请求体、模型和 Skill 无法提交用户或租户字段。 |
| `DB.tx()`、SQLite、Alembic | `0004` 新增授权、记忆、修订、pending、job、候选操作表；未改业务表。归档和入队同事务，入队失败通过 savepoint 隔离。 |
| `RAG.model()` / `RAG.client()` | 同一个本地 BGE-M3 和 Chroma 客户端，独立 `customer_memories_v1` collection；显式传 1024 维向量，不启用 Chroma 默认 embedding。按需复用同一 BGE reranker。 |
| Harness Model Gateway / Context Manager | 一批进行一次成功的结构化提取；重试有上限。召回只为推荐回复提供低信任数据，最多 5 条 / 600 估算 tokens；不进入意图、工具计划或退款确认。 |
| LangGraph、Recorder、Evaluation | 只有取得实时目录结果的推荐动作才按需召回；无可用记忆就不调用模型筛选。`conv:`、`wf:` 线程约定不变。任务与聊天通过统一 Recorder 关联源 ID；原评测断言、版本比较、Trace JSONL 和 SQLite 索引继续复用。 |

```
归档 USER 消息 → 本地规则 (NOOP / REJECT / CANDIDATE / URGENT)
 → 按租户/用户/会话/授权代缓冲 → SQLite job + lease → Model Gateway
 → 核对完整原文/分句和真实来源 → CREATE / REINFORCE / UPDATE / SKIP
 → 原子提交记忆、修订、去重记录与 job 终态 → 分离的 Chroma 索引同步
```

默认 3 条合批，上限 6 条；空闲 120 秒、最长等待 300 秒、明确结束会话均可触发；`URGENT` 直接建批并唤醒 worker。5 秒轮询是调度间隔，不是端到端 SLA。`NOOP` 无 pending/job，也不调用提取模型。`MEMORY_JOB_MODE=sync` 只对已成批任务同步处理，适合测试，不对每轮调用模型。单 worker 崩溃后恢复持久任务；若外部模型成功但尚未提交 SQLite 就崩溃，可能重复调用，**写入效果幂等**，不承诺模型请求 exactly-once。

查询先按租户和用户过滤 Chroma，再核对 SQLite 归属、版本、状态、授权和过期时间；失败回退无记忆。新旧预算冲突时采用当前用户明确的预算，并按**实时目录**库存与价格筛选。记忆不能证明商品具有未核实特性；订单、物流、政策与退款资格仍由原业务和 RAG 链路判断。

## Swagger 演示

1. 启动原 API 与 MCP，执行 `python -m alembic upgrade head`；使用模拟用户登录并携带 Bearer token。
2. `PATCH /api/v1/memories/settings` 请求体 `{"auto_write_enabled":true}`。然后在会话 A 以 `mode=real` 发送「请记住我买耳机优先关注降噪」。回复只表示已收到请求，等待列表出现条目才算保存成功。
3. 离线可以替代步骤 2 的模型提取：`POST /api/v1/memories` 请求体 `{"memory_type":"preference","subject_key":"headphones.noise_cancellation","content":"我买耳机优先关注降噪"}`，再以 `mode=deterministic` 检索。这是手动写入，不声称离线模型提取了偏好。
4. 创建全新会话 B，问「推荐一款耳机」，检查 `search_products` 的实时结果和偏好参考提示。现有商品数据未提供降噪时不会声称支持。
5. `PATCH /api/v1/memories/{id}` 传 `content` 和 `expected_version`；旧版本返回 409。`GET /api/v1/memories/{id}/revisions` 可查版本。其他用户看到 404。
6. `DELETE /api/v1/memories/{id}` 删除单条；全部删除请求为 `DELETE /api/v1/memories`，需显式 JSON `{"confirm":"DELETE_ALL_MEMORIES"}`。另建会话，确认删除或暂停检索后不再提及旧偏好。

普通候选可在明确结束当前交流时调用 `POST /api/v1/memories/conversations/{conversation_id}/finish` 触发冲刷；不冻结会话、不重置退款流程，SSE 响应结束也不会当作会话关闭。

## 隐私和索引维护

- 保存授权和检索开关独立：撤回保存作废之前已排队/运行的工作，不删除既有记忆；暂停检索后不再注入，仍可查看和管理自己的数据。`MEMORY_ENABLED=false` 停止自动提取、召回和新增/编辑，保留设置管理和删除。
- 删除时 SQLite 先屏蔽旧条目，清除正文、来源摘录和修订；即使 Chroma 清理失败也不会重新召回。为了防止旧批次写回，单条删除和自然语言遗忘请求也保守取消当前用户全部旧批次；自然语言不直接删除历史条目。
- 这是应用层删除，不覆盖原聊天归档、已有回复、SQLite/Chroma 旧页和备份；不宣称物理擦除。生产级 DLP 不在范围内，保守规则不能识别所有敏感表达。已启用自动保存且使用外部主模型时，通过门禁的少量用户消息会发送给该模型供应商。
- 索引失败由 `index_pending` 留待 worker 补偿。升级模型或恢复索引时先停 API worker；用 `python -m scripts.rebuild_memory_index --dry-run` 查范围，去掉参数后重建**记忆 collection**，不重建知识库。日常运行不下载模型权重。
