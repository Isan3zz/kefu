# 规格逐项验收

日期：2026-11-18。依据 DEVELOPMENT_SPEC.md 与用户“先不跑真实 LLM”的范围调整。代码、离线功能和真实本地 BGE 验证已完成；真实主 LLM / Judge 效果未验收。最终 pytest：82 passed、1 deselected（真实 LLM）、无失败/跳过，见 [verification.log](verification.log)。

## 完工定义与交付

| 要求 | 实现及验收证据 |
|---|---|
| venv / Python 3.11 / 锁定依赖 | 根目录 .venv；Python 3.11.16；pyproject.toml + uv.lock；uv lock --check |
| 空库迁移、seed、启动 | migrations/ 固定历史迁移；新 data/demo-v2 数据库真实执行；README / demo_walkthrough |
| 单 Agent 安全业务闭环 | app/agent.py、business.py、mcp.py；真实 HTTP 完整 demo 与跨进程恢复测试 |
| 四个可发现且懒加载的 Skill | app/catalog 四目录；SkillManager.select/load/allowed_tools；test_skill_contract.py、T13/T14 |
| FastMCP 七个工具真注册/调用 | typed ToolResponse、StreamableHttpTransport、actor HMAC；test_integration.py、test_dataset_contract.py |
| 真实本地 BGE / Chroma / BM25 / RRF / rerank | model-manifest.json、model-verification.log、real-rag-ablation.json；1024 维、无外部连接 |
| 独立 Trace + 确定性评测 | app/observability.py、evaluation.py、evidence.py；真实 gateway 事件、业务终态、每 run JSONL 与 SQLite 索引 |
| Judge / 失败归因 / 复核 / 晋升 / 比较 | CLI 可用，Mock 协议及隔离复核闭环测试通过；实际 Judge 未调用，正式待审样本未晋升 |
| JSON / Markdown / HTML 报告 | data/eval_runs/<run-id>/report.*，实际生成；docs 中保留汇总 |
| Swagger / 脚本演示 | scripts/demo.py 完整 HTTP 流程成功；demo-terminal.log；未新增独立前端 |
| 原工作及敏感信息 | 原始文档与已有数据库保留；.env/models/data 被 Git 忽略；凭据脱敏、nonce 不存明文 |

## 用户场景

| 场景 | 证据 |
|---|---|
| S01 知识咨询 | 真实 HTTP demo 政策/商品 chunk 引用；真实 RAG 四模式每 query top-k |
| S02 本人订单 | HTTP demo；T01 / test_evidence_audit.py 越权读检查 |
| S03 多轮物流 | HTTP demo 第二单 O1002；test_conversation_context.py |
| S04 确定性退款资格 | business.py P-REFUND-v1；PAID 待发货、签收期限、金额与状态变动测试 |
| S05 预览→确认→受理 | HTTP demo + T02–T06；数据库实际回执校验与同 ID 重放 |
| S06 等待期间切换咨询 | HTTP demo 检查 WAITING_CONFIRMATION 未变；T07 |
| S07 人工工单 | HTTP demo OPEN 工单；历史问题摘要及原因测试 |
| S08 越权/超时/重复 | T01/T04/T06/T14/T19、签名 HTTP 测试、未知写入结果核对 |
| S09 失败→回归→版本对比 | 故障 baseline/candidate 报告；test_evaluation_loop.py 隔离复核/晋升/回归；实际样本待人工复核 |

## T01–T20

所有编号在 tests/test_acceptance.py 中有可执行断言；补充测试不替代基础断言。

| ID | 实际断言与补充证据 |
|---|---|
| T01 | 跨租户订单隐藏存在性；工具输出不泄漏；业务读取限定 tenant + user |
| T02 | 未确认只到 WAITING_CONFIRMATION，退款增量 0 |
| T03 | 有效确认：单条 ACCEPTED、workflow REQUEST_ACCEPTED |
| T04 | 重复与并发确认同一退款编号，无重复行 |
| T05 | 订单版本/金额/物流变化拒绝旧预览；整数分校验 |
| T06 | 工具已落库后响应丢失：原幂等键核对，不创建第二笔 |
| T07 | 产品咨询不隐式恢复退款图，不销毁等待 workflow |
| T08 | SQLite checkpoint 可恢复；test_integration.py 新解释器经真实 HTTP MCP 恢复 |
| T09 | 旧索引保留撤回条目也不可召回；test_knowledge_visibility.py 双路 ACL/版本过滤 |
| T10 | 缺模型清楚报错；本地微型 SDK 断网测试；真实 BGE verify_models 禁 socket 加载/推理通过 |
| T11 | Chroma 显式 1024 维写入与 query_embeddings，禁默认编码；真实 BGE 构建与检索已运行 |
| T12 | BM25/dense 稳定 chunk ID、RRF 去重、本地 rerank 排序；真实四模式报告 |
| T13 | metadata 扫描与正文懒加载、版本/hash；目录与 frontmatter 格式测试 |
| T14 | 恶意 Skill 不授权；规划动作白名单与 MCP 写入 guard；Mock LLM 注入测试 |
| T15 | 无 LLM 配置仍可离线运行，真实 chat 返回明确配置错误 |
| T16 | 无 Langfuse 离线 Evaluator，报告有效分母，工具/数据库终态可追溯 |
| T17 | fail→pass / pass→fail；新增安全失败 BLOCKED；数据/fixture 不一致拒绝比较 |
| T18 | 构建/指针写入/SQLite 提交失败都保持旧已提交快照；治理记录不半更新 |
| T19 | 其他人的会话、Trace、评测运行不可读取；内部 actor 无签名/过期/不匹配拒绝 |
| T20 | Judge 缺配置/网络失败 NOT_EVALUATED，不影响确定性结论 |

## 接口与关键文件

规格第 13 节的全部业务路由在 app/main.py，额外提供本人 Trace 和管理员按 run/trace 取证接口。TestClient 覆盖认证、HTTP 状态码、SSE、workflow、评测权限与错误封装；FastMCP 独立服务签名通过实际 HTTP 测试。Swagger 展示接口及请求字段。

架构压缩了规格目录中的薄模块，保持公开接口和职责：config/db/business/mcp/agent/skills/rag/observability/evaluation/dataset/evidence。没有用目录空壳代替实现。

## 清楚保留的未测范围

- 真实主 LLM 任务完成率、三组 Skill 的真实收益、真实 Judge 与供应商费用：遵照用户要求不执行，不用 Fake 结果充数。
- 正式 fault 样本尚待真实人工复核，regression.jsonl 为 0 行。CLI 和隔离测试已覆盖复核到回归的闭环。
- 实际 GPU 推理、可选 Langfuse、生产身份/支付/物流、独立美化 UI 未验收；非本次交付的必需项。
- 演示集小且未经过外部人工标注。指标不能对外宣传为生产准确率。Git 尚未初始化，release 中标明 UNVERSIONED，同时记录 app 源码/锁文件及内容 hash。
