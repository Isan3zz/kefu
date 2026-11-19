# 评测口径与本次结果

本文前半保留原开发阶段的实验快照；新增 Harness 结果、指标与上下文对照见文末和 IMPLEMENTATION_REPORT 的 Harness Completion。历史 run 工件没有改写。

评测从真实网关事件、持久化 LangGraph 中断和 SQLite 终态取证；不凭自然语言自报成功。每个 case 使用独立数据库、冻结 UTC 时钟 `2026-11-17T00:00:00+00:00` 和 `demo-commerce-v2` fixture。PAID 种子物流为待发货。数据集 schema 拒绝未知断言、重复 case_id、未知 fixture 和非法步骤。

当前数据集为 intent 12、skill 12、workflow 10、safety 4、retrieval 10、fault 1，共 49 条；不同分类集有意重复部分表达。regression 为 0，等待实际人工复核。数据来自项目内显式内容表及生成脚本，经规则与逐项代码核对，未经过外部人工标注复核；不能称为企业真实数据或大规模人工 gold。30 篇知识也是项目编写的模拟资料。

每个 run 在 `data/eval_runs/<run-id>/` 保存 `report.json`、`report.md`、`report.html`、`traces.jsonl` 和可检索的 `index.sqlite`。临时业务 fixture 删除后仍能按 run_id / trace_id 追溯。报告保留原 case、回答、工具调用/参数/顺序、引用 top-k、最终状态、失败分类和 release。管理员可经评测 API 查看；普通 Trace 仍按用户及租户隔离。

## 指标

- Intent、Skill、工具、工作流分别给出有效分子/分母。Skill 准确率只计需选 Skill 的样本，负例误触发单列。本次 skill 集正例为 11/11，负例误触发为 0/1。
- Recall@5 是每条 query 的 gold 集命中比例均值；MRR@10 是首个 relevant chunk 排名倒数均值。检索评测实际请求 top 10，聊天默认上下文取 4。
- 安全硬指标分别记录非法行动尝试、越权读/写副作用和重复退款。即使 case 没写安全标签，也审计实际工具数据与数据库。引用校验 chunk 来源、版本、可见性和有效期。
- `latency` 包含 fixture 开销；`agent_latency` 和 `retrieval_latency` 来自真实阶段事件，单位毫秒。少量样本 p95 不代表生产容量。
- token usage 来自供应商返回；缺 usage 或缺单价时成本为 `NOT_EVALUATED`，不记成零。费用按 Decimal 和配置币种计算，主模型与 Judge 分开。

## 确定性实验

`python -m scripts.run_experiments` 使用明确的 FakeRouter / FakePolicy / FakeModels，配合真实 MCP SDK 和业务规则。意图 12/12、Skill 12/12、工作流 10/10、安全 4/4，明细见 [offline-experiments.json](offline-experiments.json)。FakeModels 的 hash 向量和字符交集评分只检查基础设施，不代表 BGE 语义能力。

Skill 三组 none / all / selected 均为 12/12。实际载入指令的字符级 token 估算为 0 / 3641 / 902；不是供应商账单，也不能据此判断真实 LLM 任务收益。本地 OpenAI-compatible Mock 测试另行验证三组 SOP 确实进入不同规划请求，恶意 Skill 不能越权。

故障 baseline `run-96e3397480fa7eee9239162a` 非法提议 1 次，被后端拒绝，分类 `WORKFLOW_VIOLATION`；candidate `run-c9ccf825417dcc0fb0d1e57e` 受控路由抑制提议，尝试降为 0。两个版本非法退款副作用均为 0，case fail→pass；反向比较触发 BLOCKED。它是受控故障实验，不是实测的真实 LLM 提示优化。

## 真实本地检索消融

命令：`python -m scripts.run_experiments --rag-real-only`。本次 CPU、8 线程，两个本地 BGE 预热后，在同一 10 条查询与同一 30 块快照运行；不调用主 LLM 或 Judge。实测拦截外部 socket 连接（Windows asyncio 自用 loopback 允许）。

| 模式 | Recall@5 | MRR@10 | 检索 p50 / p95（ms） | Run ID |
|---|---:|---:|---:|---|
| bm25 | 9/10 = 0.9 | 7.875/10 = 0.7875 | 5.24 / 379.17 | run-7aa4451a97af3f6edc86f722 |
| dense | 10/10 = 1.0 | 9/10 = 0.9 | 127.47 / 489.44 | run-4643987707e1eb80a75be7d9 |
| hybrid | 10/10 = 1.0 | 8.6667/10 = 0.8667 | 144.70 / 148.08 | run-386128083c8c737a15973227 |
| hybrid_rerank | 10/10 = 1.0 | 10/10 = 1.0 | 638.99 / 1465.81 | run-dd730ea150e730d1d9db81c0 |

BM25 的 `retrieval-warranty` 未命中前五，报告保留失败。混合检索 MRR 低于 dense；重排增加计算时间，不能只呈现有利结果。该小样本不能推断通用性能。原始记录见 [real-rag-ablation.json](real-rag-ablation.json)，完整 case 和 top-k 见各 run 报告。

## 复核、回归及版本

Judge 独立配置，接收回答、引用、实际工具结果和业务终态，输出结构化辅助结论。默认及故障时为 `NOT_EVALUATED`，不替代权限和幂等断言。`--mode judge --run-id ...` 对已有回答生成独立 `judge_report.json`，不改原报告或 Trace。按用户要求，本次不调用真实 LLM/Judge，也未进行真实 LLM Skill 消融。

失败复核经 `--review-run` 记录 reviewer、时间、理由，再经 `--promote-failures ... --confirmed-only` 原子去重写入回归集。实际 fault 案例保持 `PENDING_HUMAN_REVIEW`。pytest 在隔离 fixture 中模拟复核→晋升→去重→回归；这些测试身份不充当真实人工批准。

比较要求数据集 SHA、case 覆盖、fixture 版本和时钟一致，报告每条变化及根因分布；任何新增安全失败都 BLOCKED。release 包含 app 源码 SHA、uv.lock SHA、路由/规划/回答 Prompt 版本、Skill hash、tool schema v2、policy、chunker 和模型 revision。当前原文件夹没有 Git 仓库，Git commit 如实为 `UNVERSIONED`；源码 hash 不等同于已建立 Git 发布流程。

## Harness 增量评测

原引擎增加 report.harness 与 harness_config，没有另建 Eval Engine。所有 Agent case 都经 Harness，记录 invocation 与 eval run/case 的独立标识、模型/工具 attempts、错误分类、Context 估算及压缩策略。工具业务集合/顺序使用校验后的 layer=harness 事件计一次，传输尝试保留在 layer=transport；越权审计仍检查原始上游响应，不能掩盖被 Guard 拦截的上游权限问题。

指标包括 invocation_success_rate/状态分布、model_retry_rate/model_attempts、tool_error_rate/tool_attempts、step_limit_hit_rate、estimated context_compression_ratio、unknown_outcome_count、unauthorized_proposal_count、unauthorized_side_effect_count 和 end_to_end_task_pass_rate；均有有效分母或 NOT_AVAILABLE。compare 增加 harness_deltas 与两组 Harness 配置。中断不是业务失败；未知写入不能充当成功或失败的退款真值。

未决 case 的 evaluation_status=UNKNOWN_OUTCOME、passed=null；相应退款完成断言为 passed=null，业务计数保留 observed snapshot 便于后续对账。任务/工作流通过率排除未决分母并单列数量；mandatory 未决阻止门禁通过。JSON/HTML/Markdown 均单独列未决，不晋升为已确认失败回归样本。已观察到的真实越权或重复写仍照常触发硬失败，不能被“未知”覆盖。

`scripts.harness_smoke` 使用显式 FakeModelGateway、Mock HTTP 模型协议和真实 FastMCP 进程内测试服务，输出 data/harness_smoke/result.json 及本地 Trace，包含退款等待→产品咨询→确认→重放，以及模型/工具超时恢复。它不依赖真实 LLM、Langfuse 或本地 BGE 权重，也不宣称真实模型任务质量。

`scripts.context_experiment` 使用新增 datasets/harness_context.jsonl 的 2 条固定多轮用例、同一 fixture/Skill/工具/显式 Fake 模型，对照 none / L0+L1 / L0+L1+L2。none 为 0/2（输入预算中止）；两种压缩均为 2/2，完成用例中的关键订单绑定缺失均为 0/2，none 未到达该检查而标 NOT_REACHED。重复工具调用均为 0。L0+L1 压缩比约 53.77%，L0+L1+L2 约 49.81%，均为同次装配前后的保守字节估算；L2 保存更多事实引用并增加开销，没有预设它更优。

none 提前停止，不能将它较少的累计输入估算直接解读为成本优势或与完整执行组计算“节省百分比”。没有供应商 tokenizer/usage/价格，本实验不报告真实 Token 费用或智能任务收益。完整计数、耗时、相同数据 SHA、每条 Trace 与版本见 [harness-context-experiment.json](harness-context-experiment.json)，最新故障对照见 [harness-offline-experiments.json](harness-offline-experiments.json)。

## DeepSeek 真实验收（后续用户授权）

2026-11-18 用户在本地配置密钥并授权后，已执行 deepseek-flash 真实测试，改变此前仅离线验收的范围。最终 workflow 10/10、safety 4/4，均 PASS；越权动作/写入和重复退款计数为 0。全量 pytest 119 项通过，无失败/跳过，含三项真实 LLM 测试及真实本地 BGE。

保留首轮 handoff 意图错误（workflow 9/10）和后续缺订单时越界动作被拒绝（safety 3/4）。分别通过 router-v2 意图定义、planner-v4 动作语义及按意图收紧的 Pydantic 输出 Schema 修复；知识引用为 grounded-answer-v2。后置业务权限和幂等不变。颜色可由实时商品目录佐证，演示不再把有效工具结果误判为缺少 RAG 引用。

最终 workflow 的 API usage 为输入 17477 / 输出 2196 tokens（26 次模型调用），safety 为输入 4825 / 输出 831（8 次）。这只是两个最终 run 的供应商计数，未配置单价所以费用 NOT_EVALUATED；不包含全部测试/演示的总账单。真实 Judge 和真实 Skill 收益消融未执行，结果只是固定小样本，不能据此声称生产准确率。

对照、完整报告和真实 HTTP/SSE 演示证据见 [DeepSeek 验收汇总](deepseek-validation/summary.json) 以及 IMPLEMENTATION_REPORT 的“DeepSeek 真实验收”。未知写入依然按 Harness 口径单列，不用 Judge 代替业务真值。

## 长期记忆评测

`python -m scripts.run_memory_eval --ablation memory-off,memory-on,extraction-per-turn,trigger-only,trigger-batch` 对每个 case / 变体创建独立临时 SQLite 与 Chroma。写入使用模拟归档消息和 fixture 时钟，查询通过新会话和原 Harness / FastMCP。人工标签 `expected_trigger` / `expected_memories` 分离，`fixture_memories` 仅是假模型输出。Per-turn 离线基线绕过普通候选筛选，不绕过授权、禁存和证据规则；生产不开放该开关。

报告保存在 `data/eval_runs/<run_id>/`，包含 Markdown/HTML/JSON、各变体的 Trace JSONL / SQLite 索引、发布和数据 SHA、成对指标差值及失败 case。提取调用率只使用写入阶段用户轮数为分母，按实际 Gateway attempts 计数。Flush 延迟为 fixture 时钟下从缓冲到写入的时间；检索/查询延迟另记实时时钟。Fake 的 Token 只作估算，不提供供应商用量或费用。

可加 `--embedding local` 验证真实本地模型；仍默认使用脚本化的提取 Gateway。`--mode agent-real` 单独显式启用配置的外部主模型，只对虚构数据使用。语义评估、成本和生产精度不能由 Fake 结果外推。实际运行结果和限制见 `IMPLEMENTATION_REPORT.md`。

## 复合任务评测

```powershell
.venv/Scripts/python.exe -m scripts.run_eval --dataset datasets/compound_tasks.jsonl --mode agent-real --agent-release compound-v2-cold
```

数据集包含六条单轮多目标任务和六条多轮任务。每条使用独立订单 fixture，要求关键工具、参数、必要的执行顺序、业务终态和用户可见回复均满足断言。对政策查询，检查回复中的目标事实片段与至少一条实际检索到的引用，原有引用有效性和权限审计也会执行。

- `end_to_end_task_pass_rate` 为全部断言通过的任务数除以已判定任务数。未决写入单列，按 Harness 口径排除出通过率分母。
- `avg_tool_calls_per_task` 为任务中各次调用的 Harness 工具计数之和，除以有计数记录的任务数。分母包含失败和零调用任务。计数包括实际发起的业务工具、本地检索、技能加载和确认后触发的提交。传输重试归入同一次工具调用，被拦截的提议不计数。报告中的 `tool_call_count` 可逐案核对。

上述命令在新进程启动时按需加载本地模型。复现预热口径时，在调用 `app.evaluation.run` 的同一进程先执行一次真实混合检索，预热调用不纳入任务计数。

```powershell
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
@'
import asyncio
from app.config import settings
from app.db import DB
from app.evaluation import run
from app.rag import RAG

rag = RAG(DB(settings.sqlite_url), config=settings)
try:
    evidence = rag.search('已送达订单的退款时限和确认要求', 'a', 'hybrid_rerank')
    assert evidence, 'Missing warm-up evidence'
finally:
    rag.close()
    rag.db.engine.dispose()
report = asyncio.run(run('datasets/compound_tasks.jsonl', 'agent-real', 'compound-v2-final'))
print(report['run_id'])
'@ | .venv/Scripts/python.exe -
```

正式报告和 Trace 保存在 `data/eval_runs/<run-id>/`。文本片段与引用断言覆盖预设事实，开放式语义质量需另行评估。
