import os
from pathlib import Path
from typing import Literal
from pydantic import PositiveInt,Field,model_validator
from decimal import Decimal
from pydantic_settings import BaseSettings, SettingsConfigDict

# Runtime never downloads weights, even if a parent process opted into network mode.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: Literal["local", "test", "production"] = "local"
    demo_mode: bool = True
    agent_release_id: str = "local"
    sqlite_url: str = "sqlite:///./data/app.db"
    checkpoint_db_path: str = "data/checkpoints.db"
    chroma_path: str = "data/chroma"
    bm25_index_path: str = "data/bm25"
    model_device: Literal['auto','cuda','cpu'] = "auto"
    embedding_model_path: str = "models/embedding/bge-m3"
    reranker_model_path: str = "models/reranker/bge-reranker-v2-m3"
    embed_batch_size: PositiveInt = 8
    rerank_batch_size: PositiveInt = 8
    embed_max_seq_length: PositiveInt = 1024
    rerank_max_length: PositiveInt = 512
    retrieval_dense_k: int = 12
    retrieval_bm25_k: int = 12
    retrieval_rrf_k: int = 60
    retrieval_rerank_k: int = 12
    retrieval_context_k: int = 4
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: int = 45
    llm_input_cost_per_million: Decimal|None = Field(default=None,ge=0)
    llm_output_cost_per_million: Decimal|None = Field(default=None,ge=0)
    llm_cost_currency: str = 'USD'
    agent_max_steps: int = 8
    agent_max_tool_calls: PositiveInt = 12
    agent_timeout_seconds: int = 90
    mcp_base_url: str = "http://127.0.0.1:8001/mcp"
    mcp_service_secret: str = "CHANGE_ME_INTERNAL_ONLY"
    judge_enabled: bool = False
    judge_base_url: str = ""
    judge_api_key: str = ""
    judge_model: str = ""
    trace_content_enabled: bool = False
    harness_enabled: bool = True
    harness_trace_enabled: bool = True
    model_context_window: PositiveInt = 16384
    context_reserved_output_tokens: int = Field(default=2048,ge=0)
    context_safety_margin_tokens: int = Field(default=1024,ge=0)
    context_max_input_tokens: PositiveInt = 12000
    context_compact_threshold: float = Field(default=.85,gt=0,le=1)
    context_recent_message_limit: PositiveInt = 8
    context_tool_result_max_chars: PositiveInt = 3000
    context_strategy: Literal['none','L0+L1','L0+L1+L2'] = 'L0+L1+L2'
    tool_timeout_seconds: float = Field(default=15,gt=0)
    tool_max_concurrency: PositiveInt = 8
    model_max_retries: int = Field(default=2,ge=0,le=2)
    tool_read_max_retries: int = Field(default=2,ge=0,le=2)
    model_max_output_tokens: PositiveInt = 2048
    model_max_token_budget: PositiveInt = 48000
    model_json_mode: bool = False
    repeated_action_limit: int = Field(default=3,ge=2)
    memory_enabled: bool = True
    memory_opt_in_default: bool = False
    memory_top_k: int = Field(default=5, ge=1, le=5)
    memory_candidate_k: int = Field(default=15, ge=5, le=100)
    memory_token_budget: int = Field(default=600, ge=1, le=600)
    memory_collection_name: str = Field(default='customer_memories_v1', pattern=r'^customer_memories_[a-z0-9_]{1,43}$')
    memory_min_similarity: float = Field(default=.25, ge=-1, le=1)
    memory_rerank_enabled: bool = False
    memory_job_mode: Literal['async', 'sync'] = 'async'
    memory_extraction_model: str = ''
    memory_trigger_enabled: bool = True
    memory_batch_enabled: bool = True
    memory_batch_max_eligible_messages: int = Field(default=3, ge=1, le=6)
    memory_batch_max_messages: int = Field(default=6, ge=1, le=6)
    memory_batch_idle_seconds: PositiveInt = 120
    memory_batch_max_wait_seconds: PositiveInt = 300
    memory_batch_worker_poll_seconds: float = Field(default=5, gt=0)
    memory_extraction_max_candidates_per_message: int = Field(default=3, ge=1, le=4)
    memory_extraction_max_candidates_per_batch: int = Field(default=12, ge=1, le=24)
    @model_validator(mode='after')
    def context_capacity(self):
        if self.context_max_input_tokens>self.model_context_window-self.context_reserved_output_tokens-self.context_safety_margin_tokens:
            raise ValueError('CONTEXT_MAX_INPUT_TOKENS exceeds context window after reserves')
        if self.model_max_output_tokens>self.context_reserved_output_tokens:
            raise ValueError('MODEL_MAX_OUTPUT_TOKENS exceeds reserved output budget')
        if self.memory_batch_max_eligible_messages > self.memory_batch_max_messages:
            raise ValueError('Memory batch threshold cannot exceed its capacity')
        if self.memory_opt_in_default:
            raise ValueError('Long-term memory must default to explicit user opt-in')
        if self.memory_enabled and not (self.memory_trigger_enabled and self.memory_batch_enabled):
            raise ValueError('Unbatched memory extraction is only permitted in isolated offline evaluations')
        return self

settings = Settings()
ROOT = Path(__file__).resolve().parents[1]
