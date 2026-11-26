"""Validated evaluation contracts; typos must not silently reduce assertions."""
import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

EXPECTED_FIELDS = {
    'intent','skill','workflow_status','refund_request_count_delta','final_business_status',
    'error_code','required_tools','forbidden_tools','tool_order','tool_arguments',
    'submit_tool_count_max','gold_chunks','eligibility_reason','shipping_order_id',
    'ticket_count_delta','duplicate_refund_count','unauthorized_action_count',
    'terminal_reason','activated_skills','max_iterations',
}
FIXTURE_VERSION='demo-commerce-v2'
FIXTURE_TIME='2026-11-17T00:00:00+00:00'
AnswerFragment = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class RequiredAction(Contract):
    any_of: list[AnswerFragment] = Field(min_length=1)
    arguments: dict = Field(default_factory=dict)
    accepted_errors: list[AnswerFragment] = Field(default_factory=list)


class Dependency(Contract):
    before: AnswerFragment
    after: AnswerFragment


class ArgumentConstraint(Contract):
    name: AnswerFragment
    args: dict = Field(min_length=1)


class CatalogConstraint(Contract):
    min_price_cents: int = Field(default=0, ge=0)
    max_price_cents: int | None = Field(default=None, ge=0)
    min_stock: int = Field(default=0, ge=0)
    category: str | None = None
    attributes: dict = Field(default_factory=dict)
    min_recommendations: int = Field(default=1, ge=1)
    oracle_ids: list[str] | None = None

    @model_validator(mode='after')
    def price_range(self):
        if self.max_price_cents is not None and self.min_price_cents > self.max_price_cents:
            raise ValueError('Invalid catalog price range')
        return self


class OutcomeConstraints(Contract):
    catalog: CatalogConstraint | None = None
    refund_result: Literal['VERIFIED_SUCCESS', 'FAILED', 'UNKNOWN'] | None = None
    refund_order_id: str | None = None
    refund_amount_cents: int | None = Field(default=None, gt=0)
    read_only: bool = False
    order_id: str | None = None
    order_status: str | None = None
    shipment_statuses: list[str] | None = None
    eligibility: bool | None = None


class ResponseRequirements(Contract):
    contains: list[AnswerFragment] = Field(default_factory=list)
    contains_any: list[AnswerFragment] = Field(default_factory=list)
    excludes: list[AnswerFragment] = Field(default_factory=list)
    required_citations: list[AnswerFragment] = Field(default_factory=list)
    required_documents: list[AnswerFragment] = Field(default_factory=list)
    forbidden_documents: list[AnswerFragment] = Field(default_factory=list)
    require_price: bool = False
    require_stock: bool = False
    assume_no_candidates_text: list[AnswerFragment] = Field(default_factory=list)
    judge_required: bool = False
    attribute_claims: dict[str, dict[str, bool]] = Field(default_factory=dict)


class Provenance(Contract):
    source: AnswerFragment
    scenario_group: AnswerFragment
    review_status: Literal['AUTHOR_CHECKED', 'PENDING_REVIEW', 'HUMAN_REVIEWED']

class Step(BaseModel):
    model_config=ConfigDict(extra='forbid')
    type:Literal['chat','confirm','cancel','expect_interrupt','expect_state','illegal_submit','model_proposal']
    content:str|None=None
    kind:Literal['refund_confirmation']='refund_confirmation'
    workflow_status:str|None=None
    use_pending_nonce:bool=True
    answer_contains:list[AnswerFragment]=Field(default_factory=list)
    answer_contains_any:list[AnswerFragment]=Field(default_factory=list)
    citation_required:bool=False
    @model_validator(mode='after')
    def required_values(self):
        if self.type=='chat' and not self.content: raise ValueError('Chat step requires content')
        if self.type=='expect_state' and not self.workflow_status: raise ValueError('State assertion requires workflow_status')
        if self.type=='confirm' and not self.use_pending_nonce: raise ValueError('Only bound pending nonce confirmation is supported')
        if (self.answer_contains or self.answer_contains_any) and self.type not in ('chat','confirm'):
            raise ValueError('Answer assertions require chat or confirm')
        if self.citation_required and self.type!='chat': raise ValueError('Citation assertions require chat')
        return self

class Message(BaseModel):
    model_config=ConfigDict(extra='forbid')
    role:Literal['user']
    content:str=Field(min_length=1)

class Case(BaseModel):
    model_config=ConfigDict(extra='forbid')
    schema_version:Literal['1.0','2.0']
    case_id:str=Field(pattern=r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')
    category:Literal['intent','skill','workflow','safety','retrieval','agent_loop']
    fixture:str='user_a_delivered_refundable'
    actor_user_id:Literal['user_a','user_b','admin']|None=None
    expected:dict=Field(default_factory=dict)
    steps:list[Step]|None=None
    messages:list[Message]|None=None
    query:str|None=None
    tags:list[str]=[]
    review:dict|None=None
    scenario: str | None = None
    dataset_split: Literal['development', 'validation', 'holdout', 'regression', 'fault_injection'] | None = None
    difficulty: Literal['easy', 'medium', 'hard'] | None = None
    provenance: Provenance | None = None
    required_actions: list[RequiredAction] | None = None
    forbidden_actions: list[AnswerFragment] | None = None
    optional_actions: list[AnswerFragment] | None = None
    dependency_constraints: list[Dependency] | None = None
    argument_constraints: list[ArgumentConstraint] | None = None
    outcome_constraints: OutcomeConstraints | None = None
    response_requirements: ResponseRequirements | None = None
    @model_validator(mode='after')
    def contract(self):
        if self.schema_version == '1.0' and not self.expected:
            raise ValueError('Expected assertions cannot be empty')
        if self.schema_version == '2.0' and not all((self.scenario, self.dataset_split, self.difficulty, self.provenance)):
            raise ValueError('Version 2 requires scenario, split, difficulty and provenance')
        if self.schema_version == '2.0' and not any((self.expected, self.required_actions, self.forbidden_actions,
                self.dependency_constraints, self.argument_constraints,
                self.outcome_constraints and self.outcome_constraints.model_dump(exclude_defaults=True),
                self.response_requirements and self.response_requirements.model_dump(exclude_defaults=True))):
            raise ValueError('Version 2 requires at least one explicit assertion')
        if self.fixture not in ('user_a_delivered_refundable', 'holdout_commerce_v1') and not re.fullmatch(r'orders_boundary:(?:all|O30(?:0[1-9]|[1-4][0-9]|50)|[a-z][a-z0-9_]*)',self.fixture):
            raise ValueError('Unknown fixture selector: '+self.fixture)
        unknown=set(self.expected)-EXPECTED_FIELDS
        if unknown: raise ValueError('Unknown expected labels: '+', '.join(sorted(unknown)))
        if self.steps and self.messages: raise ValueError('Choose steps or messages, not both')
        if self.fixture == 'holdout_commerce_v1' and self.dataset_split not in ('holdout', 'regression'):
            raise ValueError('Holdout fixtures are restricted to holdout or explicitly reviewed regression cases')
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
    path=Path(path)
    raw=path.read_bytes()
    cases=[Case.model_validate(json.loads(line)).model_dump(exclude_none=True) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not cases: raise ValueError('Dataset has no reviewed cases')
    ids=[c['case_id'] for c in cases]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate case_id')
    default_split = 'regression' if path.name == 'regression.jsonl' else 'development'
    splits = {case.get('dataset_split', default_split) for case in cases}
    if len(splits) != 1:
        raise ValueError('A run must use a single dataset split')
    if splits == {'holdout'}:
        manifest_path = path.with_suffix('.manifest.json')
        if not manifest_path.is_file():
            raise ValueError('Holdout dataset requires a frozen manifest')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest.get('frozen') is not True or manifest.get('case_ids') != ids
                or manifest.get('sha256') != hashlib.sha256(raw).hexdigest()
                or manifest.get('scenario_groups') != sorted({case['provenance']['scenario_group'] for case in cases})):
            raise ValueError('Holdout dataset differs from the frozen manifest')
        if any(case['fixture'] == 'holdout_commerce_v1' for case in cases):
            fixture_path = Path(__file__).resolve().parents[1] / 'datasets/holdout/business_v1.json'
            if manifest.get('business_fixture_sha256') != hashlib.sha256(fixture_path.read_bytes()).hexdigest():
                raise ValueError('Holdout fixture differs from the frozen manifest')
        regression = path.parent.parent / 'regression.jsonl'
        if regression.is_file():
            retired = {json.loads(line)['case_id'] for line in regression.read_text(encoding='utf-8').splitlines() if line.strip()}
            if retired.intersection(ids):
                raise ValueError('Promoted holdout cases must be replaced in a new frozen dataset version')
    return raw,cases
