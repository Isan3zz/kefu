from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

PIPELINE_VERSION = 'memory-v1'
EXTRACTOR_VERSION = 'extractive-v1'
TRIGGER_VERSION = 'trigger-v1'
BATCH_POLICY_VERSION = 'batch-v1'
MemoryType = Literal['preference', 'stable_constraint', 'stable_fact']


class TriggerDecision(StrEnum):
    NOOP = 'NOOP'
    REJECT = 'REJECT'
    CANDIDATE = 'CANDIDATE'
    URGENT = 'URGENT'


@dataclass(frozen=True)
class TriggerDispatch:
    decision: TriggerDecision
    reason_code: str
    pending_message_id: str | None = None
    batch_job_id: str | None = None


@dataclass(frozen=True)
class MemoryPolicy:
    eligible_count: int = 3
    max_messages: int = 6
    idle_seconds: int = 120
    max_wait_seconds: int = 300
    candidates_per_message: int = 3
    candidates_per_batch: int = 12
    max_attempts: int = 3
    max_message_bytes: int = 950


class MemoryInput(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    memory_type: MemoryType
    subject_key: str = Field(max_length=100, pattern=r'^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$')
    content: str = Field(min_length=2, max_length=240)


class Candidate(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    type: MemoryType
    subject_key: str = Field(max_length=100, pattern=r'^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$')
    content: str = Field(min_length=2, max_length=240)
    source_message_id: str = Field(min_length=1, max_length=100)
    evidence_excerpt: str = Field(min_length=2, max_length=240)
    is_user_asserted: StrictBool


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    candidates: list[Candidate] = Field(max_length=24)


class MemoryReferences(BaseModel):
    model_config = ConfigDict(extra='forbid')
    memory_ids: list[str] = Field(max_length=5)


class RetrievedMemory(BaseModel):
    memory_id: str
    content: str
    memory_type: MemoryType
    subject_key: str
    source_message_id: str | None = None
    version: int
    score: float | None = None
    status: Literal['active', 'superseded', 'deleted'] = 'active'
