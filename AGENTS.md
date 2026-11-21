# Instructions for Codex: Customer-Service Agent

Read `DEVELOPMENT_SPEC.md` in full before making architectural decisions. It is the source of truth for product scope, system interfaces, model choices, persistence, security invariants, evaluation, test cases, milestones, and completion criteria.

## Non-negotiable technical choices

- Python 3.11; FastAPI; LangGraph; FastMCP; Chroma; SQLite; Agent Skills; pytest.
- Use MCP wrappers only for tools the main Agent can call. The main Agent starts refunds through `begin_refund_workflow`. The confirmed Refund Workflow calls Business directly for submission and verification.
- Local **BAAI/bge-m3** embeddings and **BAAI/bge-reranker-v2-m3** reranker. Their weights are downloaded via a separate explicit command and loaded from local paths only.
- Main LLM uses a configurable OpenAI-compatible API. Offline tests must run with fakes; do not disguise fake-model metrics as real-model results.
- Business permissions, refund confirmation and idempotency are enforced by deterministic backend rules. Skills and LLM prompts do not authorize sensitive operations.
- Build a real evaluation loop: full trace, deterministic assertions, optional Judge, failure categorization, regression dataset and version comparison.
- Do not introduce Java/Spring Boot, Milvus/Elasticsearch, Kafka, Kubernetes or a multi-agent architecture.

## Working method

1. Inspect existing repository and confirm environment assumptions without deleting existing work.
2. Follow milestones P0 → P1 → P2 → P3 → P4 in `DEVELOPMENT_SPEC.md`.
3. Implement real functionality and tests for each milestone before claiming it is complete.
4. Maintain `IMPLEMENTATION_REPORT.md` with completed items, exact commands, test results, local model revision IDs, evaluation results and known limitations.
5. Use library APIs valid for the locked dependency versions; keep `uv.lock` updated. Test LangGraph interrupt/resume, FastMCP tool calls, Chroma embeddings and SentenceTransformers local model loading with small smoke tests.
6. Run the mandatory tests T01–T20. If a real LLM API key or downloaded model is unavailable, clearly identify the blocked checks; complete all offline-capable checks instead.

Avoid stub-only modules, fabricated performance numbers, and silent background model downloads. The acceptance criteria in the development specification take precedence over implementation convenience.
