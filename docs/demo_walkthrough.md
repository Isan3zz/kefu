# 本地三进程演示

完整演示使用真实本地 BGE 检索和显式 `deterministic` 路由，不调用主 LLM。先按 README 安装 `.venv` 和模型。下面创建新目录，避免覆盖已有退款记录。

在每个参与演示的 PowerShell 终端设置相同变量；重新演示时将 `demo-new` 换成尚未使用的目录名：

```powershell
.\.venv\Scripts\Activate.ps1
$env:SQLITE_URL='sqlite:///./data/demo-new/app.db'
$env:CHECKPOINT_DB_PATH='data/demo-new/checkpoints.db'
$env:CHROMA_PATH='data/demo-new/chroma'
$env:MEMORY_CHROMA_PATH='data/demo-new/memory_chroma'
$env:BM25_INDEX_PATH='data/demo-new/bm25'
$env:MULTIMODAL_IMAGE_PATH='data/demo-new/images'
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
```

先初始化并构建该数据库的知识快照（仅需一次）：

```powershell
python -m alembic upgrade head
python -m scripts.init_db
python -m scripts.build_kb
```

在三个终端分别执行 `python -m app.mcp`、`python -m app.retrieval_mcp` 和 `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`。访问 Swagger，或在另一个终端执行以下命令。

```powershell
python -m scripts.demo
```

脚本实际断言并展示：政策咨询有 chunk 引用；查询本人订单及第二单 O1002 物流；O1001 退款预览待确认；产品咨询仍有证据且退款保持等待；确认后申请 ACCEPTED；再次确认返回同一 ID；退款进度查询返回已受理申请；人工工单 OPEN。不会把“已受理”说成到账。

本次在 `data/demo-v2/` 的新业务库完成上述 HTTP 链路，真实终端记录见 [demo-terminal.log](demo-terminal.log)。数据库只有一条 O1001、29900 分的退款，编号 `refund-a3c5c2b30e3dc5c9f1753336`；workflow 为 REQUEST_ACCEPTED。后续工单 `ticket-1921c3f849d9989366a42cf0` 为 OPEN。验证服务完成后停止，目录与原有根目录业务库均保留。

跨租户读取、并发确认、工具写入后丢响应、跨解释器重启恢复、SSE 和评测 API 的更多行为由 [verification.log](verification.log) 对应测试覆盖。接口错误不会被 HTTP 200 掩盖。

评测演示执行 `python -m scripts.run_experiments`，查看 `data/experiment_summary.json` 及其 run 的 HTML 报告。故障基线的非法行动被后端拦截，候选抑制提议，业务副作用始终为零。正式回归晋升需要真实人工复核；未经复核的本次样本保持待审核。
