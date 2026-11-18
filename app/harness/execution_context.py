from dataclasses import dataclass,field
from typing import Literal
import time,hashlib,json
from app.harness.contracts import HarnessFailure

@dataclass(frozen=True)
class Principal:
    tenant_id:str
    user_id:str
    role:str='customer'

@dataclass(frozen=True)
class ExecutionContext:
    invocation_id:str
    trace_id:str
    conversation_id:str
    graph_thread_id:str
    principal:Principal
    agent_release_id:str
    deadline_monotonic:float
    evaluation_case_id:str|None=None
    evaluation_run_id:str|None=None
    workflow_id:str|None=None
    parent_invocation_id:str|None=None
    mode:Literal['real','deterministic','rag-real','agent-real']='real'
    def remaining(self):
        seconds=self.deadline_monotonic-time.monotonic()
        if seconds<=0: raise HarnessFailure('RUN_TIMEOUT','本次执行已超时，请查询原操作状态后重试')
        return seconds

@dataclass
class RunBudget:
    steps:int=0
    model_calls:int=0
    model_attempts:int=0
    tool_calls:int=0
    tool_attempts:int=0
    prompt_tokens:int=0
    completion_tokens:int=0
    provider_usage_complete:bool=True
    token_estimate_spent:int=0
    write_started:bool=False
    write_in_flight:bool=False
    actions:dict=field(default_factory=dict)
    def step(self,ctx,limit):
        ctx.remaining()
        if self.steps>=limit: raise HarnessFailure('STEP_LIMIT_EXCEEDED','本次执行已达到步骤上限，请补充信息或转人工')
        self.steps+=1
    def action(self,name,args,version,limit):
        key=(name,hashlib.sha256(json.dumps(args,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),str(version))
        self.actions[key]=self.actions.get(key,0)+1
        if self.actions[key]>=limit: raise HarnessFailure('REPEATED_NO_PROGRESS','重复查询没有产生新进展，请补充信息或转人工')
