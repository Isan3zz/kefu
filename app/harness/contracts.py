from enum import StrEnum
from typing import Literal,Any
from pydantic import BaseModel,Field,ConfigDict,model_validator
from app.business import DomainError

class HarnessErrorCode(StrEnum):
    CONFIG_ERROR='CONFIG_ERROR'
    MODEL_TIMEOUT='MODEL_TIMEOUT'
    MODEL_RATE_LIMIT='MODEL_RATE_LIMIT'
    MODEL_UPSTREAM_ERROR='MODEL_UPSTREAM_ERROR'
    MODEL_BAD_OUTPUT='MODEL_BAD_OUTPUT'
    MODEL_TOOL_CALLING_UNSUPPORTED='MODEL_TOOL_CALLING_UNSUPPORTED'
    TOOL_TIMEOUT='TOOL_TIMEOUT'
    TOOL_UPSTREAM_ERROR='TOOL_UPSTREAM_ERROR'
    TOOL_BAD_RESULT='TOOL_BAD_RESULT'
    POLICY_DENIED='POLICY_DENIED'
    FORBIDDEN='FORBIDDEN'
    CONTEXT_BUDGET_EXCEEDED='CONTEXT_BUDGET_EXCEEDED'
    STEP_LIMIT_EXCEEDED='STEP_LIMIT_EXCEEDED'
    REPEATED_NO_PROGRESS='REPEATED_NO_PROGRESS'
    RUN_TIMEOUT='RUN_TIMEOUT'
    RUN_CANCELLED='RUN_CANCELLED'
    UNKNOWN_OUTCOME='UNKNOWN_OUTCOME'
    INTERNAL_ERROR='INTERNAL_ERROR'

class HarnessError(BaseModel):
    model_config=ConfigDict(extra='forbid')
    code:HarnessErrorCode
    message:str
    retryable:bool=False
    source:str|None=None
    details:dict=Field(default_factory=dict)

class HarnessFailure(DomainError):
    def __init__(self,code,message=None,*,source='runtime',retryable=False,details=None):
        self.error=HarnessError(code=code,message=message or '当前服务暂不可用，请重试或转人工',source=source,retryable=retryable,details=details or {})
        super().__init__(self.error.code.value,self.error.message)

class InvocationStatus(StrEnum):
    CREATED='CREATED'
    RUNNING='RUNNING'
    SUCCEEDED='SUCCEEDED'
    INTERRUPTED='INTERRUPTED'
    FAILED='FAILED'
    TIMED_OUT='TIMED_OUT'
    CANCELLED='CANCELLED'
    UNKNOWN_OUTCOME='UNKNOWN_OUTCOME'

class ModelUsage(BaseModel):
    input_tokens:int|None=Field(default=None,ge=0)
    output_tokens:int|None=Field(default=None,ge=0)
    total_tokens:int|None=Field(default=None,ge=0)
    source:Literal['provider','estimated','unavailable']='unavailable'

class ModelResult(BaseModel):
    text:str|None=None
    structured:dict|None=None
    model_id:str
    usage:ModelUsage=Field(default_factory=ModelUsage)
    attempts:int=Field(ge=1)
    latency_ms:float=Field(ge=0)
    finish_reason:str|None=None

class ToolExecutionResult(BaseModel):
    tool_name:str
    ok:bool
    data:dict|list|str|None=None
    error:dict|None=None
    request_id:str
    attempts:int=Field(ge=0)
    latency_ms:float=Field(ge=0)
    outcome:Literal['SUCCESS','KNOWN_FAILURE','UNKNOWN_OUTCOME']
    @model_validator(mode='after')
    def consistent(self):
        if self.ok!=(self.outcome=='SUCCESS') or self.ok==(self.error is not None): raise ValueError('Inconsistent tool outcome')
        if not self.ok and self.data is not None: raise ValueError('Failed tool cannot carry business data')
        return self
    def envelope(self):
        return self.model_dump(include={'ok','data','error','request_id'})

class ContextBudget(BaseModel):
    model_context_window:int=Field(gt=0)
    reserved_output_tokens:int=Field(ge=0)
    safety_margin_tokens:int=Field(ge=0)
    max_input_tokens:int=Field(gt=0)
    @model_validator(mode='after')
    def capacity(self):
        if self.max_input_tokens>self.model_context_window-self.reserved_output_tokens-self.safety_margin_tokens: raise ValueError('Invalid input context capacity')
        return self

class ContextBuildResult(BaseModel):
    messages:list[dict]
    token_estimate:int
    before_tokens:int
    estimate_source:Literal['estimated']='estimated'
    compression_applied:bool
    compression_level:str
    preserved_refs:list[str]
    summary_id:str|None=None
    memory_ids:list[str]=Field(default_factory=list)
