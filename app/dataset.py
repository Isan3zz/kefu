"""Validated evaluation contracts; typos must not silently reduce assertions."""
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

EXPECTED_FIELDS = {
    'intent','skill','workflow_status','refund_request_count_delta','final_business_status',
    'error_code','required_tools','forbidden_tools','tool_order','tool_arguments',
    'submit_tool_count_max','gold_chunks','eligibility_reason','shipping_order_id',
    'ticket_count_delta','duplicate_refund_count','unauthorized_action_count',
    'terminal_reason','activated_skills','max_iterations',
}
FIXTURE_VERSION='demo-commerce-v2'
FIXTURE_TIME='2026-11-17T00:00:00+00:00'

class Step(BaseModel):
    model_config=ConfigDict(extra='forbid')
    type:Literal['chat','confirm','cancel','expect_interrupt','expect_state','illegal_submit','model_proposal']
    content:str|None=None
    kind:Literal['refund_confirmation']='refund_confirmation'
    workflow_status:str|None=None
    use_pending_nonce:bool=True
    @model_validator(mode='after')
    def required_values(self):
        if self.type=='chat' and not self.content: raise ValueError('Chat step requires content')
        if self.type=='expect_state' and not self.workflow_status: raise ValueError('State assertion requires workflow_status')
        if self.type=='confirm' and not self.use_pending_nonce: raise ValueError('Only bound pending nonce confirmation is supported')
        return self

class Message(BaseModel):
    model_config=ConfigDict(extra='forbid')
    role:Literal['user']
    content:str=Field(min_length=1)

class Case(BaseModel):
    model_config=ConfigDict(extra='forbid')
    schema_version:Literal['1.0']
    case_id:str=Field(pattern=r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')
    category:Literal['intent','skill','workflow','safety','retrieval','agent_loop']
    fixture:Literal['user_a_delivered_refundable']='user_a_delivered_refundable'
    expected:dict
    steps:list[Step]|None=None
    messages:list[Message]|None=None
    query:str|None=None
    tags:list[str]=[]
    review:dict|None=None
    @model_validator(mode='after')
    def contract(self):
        if not self.expected: raise ValueError('Expected assertions cannot be empty')
        unknown=set(self.expected)-EXPECTED_FIELDS
        if unknown: raise ValueError('Unknown expected labels: '+', '.join(sorted(unknown)))
        if self.steps and self.messages: raise ValueError('Choose steps or messages, not both')
        if self.category=='retrieval':
            if not self.query or not self.expected.get('gold_chunks'): raise ValueError('Retrieval case needs query and nonempty gold_chunks')
        elif not (self.steps or self.messages): raise ValueError('Business case needs steps or messages')
        for key in ('required_tools','forbidden_tools','tool_order','gold_chunks','activated_skills'):
            if key in self.expected and (not isinstance(self.expected[key],list) or not all(isinstance(v,str) for v in self.expected[key])):
                raise ValueError(f'{key} must be a list of strings')
        for key in ('refund_request_count_delta','submit_tool_count_max','ticket_count_delta','duplicate_refund_count','unauthorized_action_count','max_iterations'):
            if key in self.expected and (type(self.expected[key]) is not int or self.expected[key]<0): raise ValueError(f'{key} must be a nonnegative integer')
        if 'tool_arguments' in self.expected:
            values=self.expected['tool_arguments']
            if not isinstance(values,list) or not all(isinstance(v,dict) and set(v)=={'name','args'} and isinstance(v['name'],str) and isinstance(v['args'],dict) for v in values):
                raise ValueError('tool_arguments requires a list of {name, args} objects')
        return self

def load_dataset(path):
    raw=Path(path).read_bytes()
    cases=[Case.model_validate(json.loads(line)).model_dump(exclude_none=True) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not cases: raise ValueError('Dataset has no reviewed cases')
    ids=[c['case_id'] for c in cases]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate case_id')
    return raw,cases
