import asyncio,time
from pydantic import ValidationError
from app.business import DomainError
from app.db import uid
from app.mcp import RETRIEVAL_TOOLS, TRANSIENT_READ_ERRORS, RetrievalResponse, ToolResponse
from app.harness.contracts import HarnessFailure,ToolExecutionResult
from app.harness.recovery import backoff
from pydantic import BaseModel,ConfigDict,Field,StrictBool,StrictInt,StrictStr

class Payload(BaseModel): model_config=ConfigDict(extra='allow',strict=True)
class OwnedOrder(Payload):
    id:StrictStr
    tenant_id:StrictStr
    user_id:StrictStr
    status:StrictStr
    paid_amount_cents:StrictInt
class Shipment(Payload): order_id:StrictStr; status:StrictStr
class OrderItem(Payload):
    order_id: StrictStr
    product_id: StrictStr | None
    product_name: StrictStr | None = None
    quantity: StrictInt | None
class ShipmentSummary(Payload):
    order_id: StrictStr
    status: StrictStr | None
    carrier: StrictStr | None = None
class OrderDetail(OwnedOrder):
    items: list[OrderItem] = Field(default_factory=list)
    shipments: list[ShipmentSummary] = Field(default_factory=list)
class Product(Payload): id:StrictStr; stock:StrictInt; price_cents:StrictInt
class Eligibility(Payload): eligible:StrictBool; reason_code:StrictStr; amount_cents:StrictInt; policy_version:StrictStr; requires_human:StrictBool
class RefundPreview(Payload):
    workflow_id:StrictStr|None
    order_id:StrictStr
    status:StrictStr
    amount_cents:StrictInt|None=None
    confirm_nonce:StrictStr|None=None
    reason_code:StrictStr|None=None
class TicketReceipt(Payload): ticket_id:StrictStr; status:StrictStr
class KnowledgeChunk(Payload):
    chunk_id: StrictStr
    doc_id: StrictStr
    document_title: StrictStr
    content: StrictStr
    source: StrictStr
    version: StrictStr
    snapshot: StrictStr
class VisualProduct(Payload):
    image_id: StrictStr
    product_id: StrictStr
    product_name: StrictStr
    price_cents: StrictInt
    stock: StrictInt
    similarity_score: float
class VisualText(Payload):
    visual_text_id: StrictStr
    product_id: StrictStr | None
    doc_id: StrictStr | None
    content: StrictStr
    score: float


def knowledge_observation(chunks):
    return [{key:chunk[key] for key in ('chunk_id','doc_id','document_title','version')}
            | {'snippet':chunk['content'][:400]} for chunk in chunks]


def validate_payload(name,data,ctx,args):
    if name in RETRIEVAL_TOOLS:
        schema = KnowledgeChunk if name=='search_knowledge' else VisualText if name=='search_text_by_image' else VisualProduct
        if not isinstance(data,list):
            raise ValueError('Retrieval result must be a list')
        for item in data:
            schema.model_validate(item)
        return
    schema={'get_orders':OwnedOrder,'get_order':OrderDetail,'track_shipment':Shipment,'search_products':Product,'check_refund_eligibility':Eligibility,'begin_refund_workflow':RefundPreview,'create_ticket':TicketReceipt}[name]
    if name in ('get_orders','track_shipment','search_products'):
        if not isinstance(data,list): raise ValueError('Tool result must be a list')
        for item in data: schema.model_validate(item)
    else: schema.model_validate(data)
    if name in ('get_orders','get_order'):
        for item in data if isinstance(data,list) else [data]:
            if item['tenant_id']!=ctx.principal.tenant_id or item['user_id']!=ctx.principal.user_id: raise HarnessFailure('FORBIDDEN','工具返回数据不在当前权限范围',source='tool')
    if name=='search_products' and any(item.get('tenant_id')!=ctx.principal.tenant_id for item in data): raise HarnessFailure('FORBIDDEN','工具返回商品不在当前租户范围',source='tool')
    if name=='get_order':
        order_id = args['order_id']
        if data['id'] != order_id: raise ValueError('Order scope mismatch')
        if any(row['order_id'] != order_id for field in ('items', 'shipments') for row in data.get(field, [])):
            raise HarnessFailure('FORBIDDEN', '订单明细包含其他订单的数据', source='tool')
    if name=='track_shipment' and any(item['order_id']!=args['order_id'] for item in data): raise ValueError('Shipment scope mismatch')
    if name=='begin_refund_workflow' and data['order_id']!=args['order_id']: raise ValueError('Refund order scope mismatch')

class ToolExecutor:
    def __init__(self,gateway,guard,config,controller):
        self.gateway,self.guard,self.config,self.controller=gateway,guard,config,controller
        self.semaphore=asyncio.Semaphore(config.tool_max_concurrency)
    async def invoke(self,*,ctx,skill_name,tool_name,arguments,budget,telemetry,source='agent',proposal_id=None,
                     required_max_price=None,rewritten_query=None):
        # source is audit text, NEVER a capability.
        return await self._invoke(ctx,skill_name,tool_name,arguments,budget,telemetry,proposal_id=proposal_id,
                                  required_max_price=required_max_price,rewritten_query=rewritten_query)
    async def invoke_image_search(self, *, ctx, tool_name, arguments, budget, telemetry):
        return await self._invoke(ctx,'product-consultation',tool_name,arguments,budget,telemetry,direct_image=True)

    async def invoke_internal(self, *, ctx, tool_name, arguments, budget, telemetry, handler, proposal_id):
        start=time.perf_counter(); operation=uid('op-')
        args=self.guard.check(ctx,budget,telemetry,None,tool_name,arguments,internal=True,
                              tool_call_id=proposal_id,operation_id=operation)
        budget.tool_calls+=1; budget.tool_attempts+=1
        telemetry.emit('loop.tool_allowed',tool_call_id=proposal_id,name=tool_name,risk_class='read')
        try:
            async with asyncio.timeout(min(self.config.tool_timeout_seconds,ctx.remaining())):
                data=await handler(args)
            outcome=dict(ok=True,data=data,outcome='SUCCESS')
        except DomainError as error:
            outcome=dict(ok=False,error={'code':error.code,'message':error.message},outcome='KNOWN_FAILURE')
        except TimeoutError:
            outcome=dict(ok=False,error={'code':'TOOL_TIMEOUT','message':'本地工具超时'},outcome='KNOWN_FAILURE')
        return self.record_result(ToolExecutionResult(tool_name=tool_name,request_id=operation, attempts=1,
                                  latency_ms=(time.perf_counter()-start)*1000,**outcome),args,operation)
    def record_result(self,result,args,operation,**correlation):
        from app.observability import emit
        observed=result.envelope()
        if result.tool_name=='load_skill' and result.ok:
            observed['data']={key:value for key,value in result.data.items() if key!='guidance'}
        if result.tool_name=='search_knowledge' and result.ok:
            observed['data']=knowledge_observation(result.data)
        emit('tool',layer='harness',name=result.tool_name,args=args,operation_id=operation,request_id=result.request_id,result=observed,ok=result.ok,status='success' if result.ok else 'error',error_code=result.error['code'] if result.error else None,attempts=result.attempts,outcome=result.outcome,duration_ms=result.latency_ms,**correlation)
        return result
    async def _invoke(self,ctx,skill_name,name,args,budget,telemetry,proposal_id=None,
                      required_max_price=None,rewritten_query=None,direct_image=False):
        start=time.perf_counter(); operation=uid('op-'); attempt=0
        args=self.guard.check(ctx,budget,telemetry,skill_name,name,args,
                              tool_call_id=proposal_id,operation_id=operation,direct_image=direct_image)
        server = self.gateway.server_for(name)
        correlation = dict(server_id=server.server_id, tool_call_id=proposal_id)
        budget.tool_calls+=1
        write=self.gateway.tools[name].kind == 'write'
        if proposal_id is not None:
            telemetry.emit('loop.tool_allowed',tool_call_id=proposal_id,name=name,
                           risk_class='preview' if name=='begin_refund_workflow' else 'write' if write else 'read')
        telemetry.emit('tool.started',tool_name=name,operation_id=operation,**correlation)
        while True:
            ctx.remaining(); attempt+=1; budget.tool_attempts+=1
            try:
                async def perform():
                    try:
                        async with self.semaphore:
                            scope={}
                            if name=='create_ticket':
                                scope['ticket_operation']=ctx.invocation_id
                            elif name=='begin_refund_workflow':
                                scope['refund_preview_scope']=(ctx.conversation_id,ctx.invocation_id)
                            raw=await self.gateway.call(ctx.principal,name,args,skill_name,single_attempt=True,
                                                        timeout=ctx.remaining(),operation_id=operation,attempt_no=attempt,
                                                        context=ctx,tool_call_id=proposal_id,required_max_price=required_max_price,
                                                        rewritten_query=rewritten_query,
                                                        access='direct_image' if direct_image else 'agent',**scope)
                            schema = RetrievalResponse if name in RETRIEVAL_TOOLS else ToolResponse
                            result=schema.model_validate(raw,strict=True).model_dump(mode='json',exclude={'diagnostics'})
                            if result['ok']:
                                validate_payload(name,result['data'],ctx,args)
                                if name in RETRIEVAL_TOOLS:
                                    diagnostics = raw['diagnostics']
                                    if diagnostics['retrieval_result_count'] != len(result['data']):
                                        raise ValueError('Retrieval result count mismatch')
                                    if name == 'search_knowledge' and any(
                                            chunk['snapshot'] != diagnostics['knowledge_snapshot'] for chunk in result['data']):
                                        raise ValueError('Knowledge evidence snapshot mismatch')
                            elif not write and result['error']['retryable'] and result['error']['code'] in TRANSIENT_READ_ERRORS:
                                raise DomainError(result['error']['code'],result['error']['message'])
                            telemetry.emit('tool.result_validated',tool_name=name,operation_id=operation,
                                           attempt_no=attempt,status='success' if result['ok'] else 'error',
                                           error_code=result['error']['code'] if result['error'] else None,**correlation)
                            return result
                    finally:
                        if write: budget.write_in_flight=False
                if write:
                    # Persist ownership before scheduling a shielded task (including the scheduling gap).
                    budget.write_started=True; budget.write_in_flight=True
                    self.controller.update(ctx,write_started=1)
                    task=self.controller.retain(asyncio.create_task(perform()))
                    raw=await asyncio.shield(task)
                else: raw=await perform()
                outcome='SUCCESS' if raw['ok'] else 'UNKNOWN_OUTCOME' if raw['error']['code']=='UNKNOWN_OUTCOME' else 'KNOWN_FAILURE'
                result=ToolExecutionResult(tool_name=name,**raw,operation_id=operation,attempts=attempt,latency_ms=(time.perf_counter()-start)*1000,outcome=outcome)
                telemetry.emit('tool.completed' if result.ok else 'tool.failed',tool_name=name,request_id=result.request_id,operation_id=operation,attempts=attempt,attempt_no=attempt,outcome=outcome,error_code=result.error['code'] if result.error else None,status='success' if result.ok else 'error',duration_ms=result.latency_ms,**correlation)
                return self.record_result(result,args,operation,**correlation)
            except asyncio.CancelledError:
                if write: telemetry.emit('recovery.started',error_class='WRITTEN_BUT_RESPONSE_UNKNOWN',action='explicit_status_query_required')
                raise
            except Exception as exc:
                code=exc.code if isinstance(exc,DomainError) else 'TOOL_BAD_RESULT' if isinstance(exc,(ValueError,TypeError,ValidationError)) else 'TOOL_TIMEOUT' if isinstance(exc,TimeoutError) else 'TOOL_UPSTREAM_ERROR'
                if write and code not in ('FORBIDDEN','BUSINESS_RULE_FAILED','INVALID_ARGUMENT','ALREADY_EXISTS','POLICY_DENIED'):
                    code='UNKNOWN_OUTCOME'
                telemetry.emit('tool.failed',tool_name=name,operation_id=operation,attempt_no=attempt,attempts=attempt,error_code=code,status='error',outcome='UNKNOWN_OUTCOME' if code=='UNKNOWN_OUTCOME' else 'KNOWN_FAILURE',**correlation)
                if not write and code in TRANSIENT_READ_ERRORS and attempt<=server.read_retries:
                    await backoff(ctx,attempt,telemetry,code); continue
                if ctx.deadline_monotonic<=time.monotonic() and not write: raise HarnessFailure('RUN_TIMEOUT')
                result=ToolExecutionResult(tool_name=name,ok=False,error={'code':code,'message':'申请状态暂未确认，请查询原操作状态' if code=='UNKNOWN_OUTCOME' else '工具查询暂不可用，请稍后重试或转人工','retryable':False},request_id=operation,operation_id=operation,attempts=attempt,latency_ms=(time.perf_counter()-start)*1000,outcome='UNKNOWN_OUTCOME' if code=='UNKNOWN_OUTCOME' else 'KNOWN_FAILURE')
                return self.record_result(result,args,operation,**correlation)
