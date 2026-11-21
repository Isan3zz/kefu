from pydantic import BaseModel,ConfigDict,Field,StrictStr
from app.business import ProductSearch as Products
from app.db import conversations
from app.harness.contracts import HarnessFailure
from app.multimodal.retrieval import TOOL_ARGS

class Arguments(BaseModel): model_config=ConfigDict(extra='forbid',strict=True)
class ListOrders(Arguments): limit:int=Field(default=10,ge=1,le=20)
class Order(Arguments): order_id:StrictStr=Field(min_length=1,max_length=100)
class Ticket(Arguments): category:str=Field(min_length=1,max_length=100); summary:str=Field(min_length=1,max_length=10000); conversation_id:str
class SkillName(Arguments): name:str=Field(min_length=1,max_length=80)
class KnowledgeQuery(Arguments):
    query:str=Field(min_length=1,max_length=1000)
    product_id:str|None=Field(default=None,min_length=1,max_length=100)

INTERNAL_SCHEMAS={'load_skill':SkillName,'search_knowledge':KnowledgeQuery, **TOOL_ARGS}
SCHEMAS={'get_orders':ListOrders,'get_order':Order,'track_shipment':Order,'search_products':Products,'check_refund_eligibility':Order,'create_ticket':Ticket,'begin_refund_workflow':Order}
READ_TOOLS=frozenset(SCHEMAS)-{'create_ticket','begin_refund_workflow'}

class PolicyGuard:
    def __init__(self,business,skills,config): self.b,self.skills,self.config=business,skills,config
    def deny(self,telemetry,rule,tool):
        telemetry.emit('policy.denied',rule_id=rule,proposed_action=tool,side_effect_happened=False,status='blocked',error_code='POLICY_DENIED')
        raise HarnessFailure('POLICY_DENIED','当前操作未获授权，请通过受控业务流程处理',source='tool',details={'rule_id':rule})
    def check(self,ctx,budget,telemetry,skill_name,tool_name,arguments,*,internal=False):
        ctx.remaining()
        with self.b.db.tx() as c: self.b.owned(c,conversations,ctx.principal,ctx.conversation_id)
        if budget.tool_calls>=self.config.agent_max_tool_calls: raise HarnessFailure('STEP_LIMIT_EXCEEDED')
        schemas=INTERNAL_SCHEMAS if internal else SCHEMAS
        if tool_name not in schemas: self.deny(telemetry,'TOOL_NOT_ALLOWED',tool_name)
        if not internal and tool_name not in self.skills.allowed_tools(skill_name): self.deny(telemetry,'SKILL_TOOL_NOT_ALLOWED',tool_name)
        try: validated=schemas[tool_name].model_validate(arguments).model_dump(exclude_unset=True)
        except ValueError:
            from app.business import DomainError
            raise DomainError('INVALID_ARGUMENT','工具请求参数不符合定义') from None
        if tool_name=='create_ticket' and validated['conversation_id']!=ctx.conversation_id: self.deny(telemetry,'CONVERSATION_SCOPE',tool_name)
        if budget.steps>self.config.agent_max_steps: raise HarnessFailure('STEP_LIMIT_EXCEEDED')
        version=None
        if ctx.workflow_id: version=self.b.workflow(ctx.principal,ctx.workflow_id)['version']
        budget.action(tool_name,validated,version,self.config.repeated_action_limit)
        telemetry.emit('policy.checked',tool_name=tool_name,rule_id='TRUSTED_SCOPE_AND_TOOL_ALLOWLIST')
        return validated
