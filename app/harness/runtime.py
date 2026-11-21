"""Facade around existing Agent graphs; owns requests, never business authorization."""
import asyncio,json,time
from dataclasses import dataclass
from sqlalchemy import select
from app.business import DomainError
from app.db import agent_invocations,now
from app.observability import TraceRecorder,active_trace,evaluation_context,clean
from app.harness.run_controller import RunController
from app.harness.execution_context import RunBudget
from app.harness.contracts import HarnessFailure
from app.harness.model_gateway import ModelGateway,FakeModelGateway
from app.harness.policy_guard import PolicyGuard
from app.harness.tool_executor import ToolExecutor
from app.harness.telemetry import Telemetry
from app.harness.recovery import error_class

@dataclass
class HarnessServices:
    model:object
    tools:ToolExecutor
    telemetry:Telemetry
    budget:RunBudget
    controller:RunController
    context:object=None
    image_ids:tuple[str, ...]=()
    allow_external_image_send:bool=False

class HarnessFacade:
    def __init__(self,agent,model_gateway=None,fake_gateway=None):
        self.agent=agent; self.config=agent.config; self.b=agent.b
        self.controller=RunController(self.b,self.config)
        from app.multimodal.vlm_image_adapter import VLMImageAdapter
        self.model=model_gateway or ModelGateway(self.config,client_factory=lambda:agent.llm(),
                                                 image_adapter=VLMImageAdapter(agent.images))
        self.fake_model=fake_gateway or FakeModelGateway(self.config)
        self.guard=PolicyGuard(self.b,agent.skills,self.config)
        self.tools=ToolExecutor(agent.gateway,self.guard,self.config,self.controller)
    async def close(self):
        await self.controller.close()
        await self.model.close()
    async def invoke_chat(self,*,principal,conversation_id,message,client_request_id=None,evaluation=None,mode='real',skill_mode='selected',legacy=False,image_ids=(),allow_external_image_send=False):
        body={'kind':'chat','message':message,'mode':mode,'skill_mode':skill_mode}
        if image_ids:
            if not self.config.multimodal_enabled:
                raise DomainError('MULTIMODAL_DISABLED')
            if len(image_ids) > self.config.multimodal_max_chat_images or len(set(image_ids)) != len(image_ids):
                raise DomainError('INVALID_ARGUMENT', '图片数量超过限制或包含重复附件')
            body['image_ids'] = list(image_ids)
        if allow_external_image_send:
            if not image_ids or mode != 'real':
                raise DomainError('INVALID_ARGUMENT', '外发授权须用于包含图片的真实模型请求')
            body['allow_external_image_send'] = True
        return await self._invoke(principal,conversation_id,'conv:'+conversation_id,body,client_request_id,None,mode,evaluation,legacy,lambda ctx,services:self.agent._chat(principal,conversation_id,message,mode,skill_mode,ctx,services))
    async def resume_workflow(self,*,principal,workflow_id,confirmation_payload,client_request_id=None,legacy=False):
        workflow=self.b.workflow(principal,workflow_id)
        nonce=confirmation_payload['confirm_nonce']
        body={'kind':'confirm','workflow_id':workflow_id,'confirmation_hash':__import__('hashlib').sha256(nonce.encode()).hexdigest()}
        async def confirm(ctx,services):
            result=await self.agent.refunds.confirm(principal,workflow,nonce,ctx,services)
            return self.agent.refund_response(result,workflow,trace_id=ctx.trace_id)
        return await self._invoke(principal,workflow['conversation_id'],workflow['graph_thread_id'],body,
                                  client_request_id,workflow_id,'real',None,legacy,confirm,
                                  workflow=workflow,confirmation_nonce=nonce)
    async def stream_chat(self,**kwargs):
        def event(kind,data): return 'event: '+kind+'\ndata: '+json.dumps(data,ensure_ascii=False)+'\n\n'
        yield event('status',{'status':'running'})
        try:
            result=await self.invoke_chat(**kwargs)
            if result.get('processing'):
                yield event('status',result); return
            if result.get('error_code'):
                yield event('error',{'code':result['error_code'],'message':result['answer'],'invocation_id':result['invocation_id'],'trace_id':result['trace_id']}); return
            for tool in result.get('tool_actions',[]): yield event('tool',{'name':tool['name'],'ok':tool['result']['ok']})
            if result.get('pending_confirmation'): yield event('interrupt',result['pending_confirmation'])
            yield event('token',{'text':result['answer']})
            yield event('done',result)
        except DomainError as error:
            yield event('error',{'code':error.code,'message':error.message,'invocation_id':getattr(error,'invocation_id',None),'trace_id':getattr(error,'trace_id',None)})
    async def _invoke(self,principal,conv,thread,body,key,wid,mode,evaluation,legacy,operation,*,workflow=None,confirmation_nonce=None):
        if not self.config.harness_enabled: raise HarnessFailure('CONFIG_ERROR','HARNESS_ENABLED=false 已暂停 Agent 执行；不会绕过运行时安全控制')
        ctx,replay=self.controller.begin(principal,conv,thread,body,key,wid,mode,evaluation or evaluation_context.get())
        if replay is not None:
            cached_refund=replay.get('refund_workflow_result') or {}
            if (workflow and not replay.get('processing') and cached_refund.get('reason')!='RECEIPT_MISMATCH'
                    and replay.get('error_code') in (None,'UNKNOWN_OUTCOME','INTERNAL_ERROR')):
                self.b.check_refund_nonce(principal,wid,confirmation_nonce)
                result=await self.agent.refunds.reconcile(principal,workflow)
                replay=self.agent.refund_response(result,workflow) | {
                    'invocation_id':replay['invocation_id'],'trace_id':replay['trace_id'],
                    'invocation_status':{'VERIFIED_SUCCESS':'SUCCEEDED','FAILED':'FAILED','UNKNOWN':'UNKNOWN_OUTCOME'}[result.status],
                }
            if body.get('image_ids'):
                await asyncio.to_thread(self.agent.images.bind, principal, conv, body['image_ids'])
            if 'cards' in replay:
                from app.memory.recommendations import ceiling
                replay['cards'] = await asyncio.to_thread(
                    self.agent.multimodal.revalidate_cards, principal, replay.get('tool_actions', []),
                    ceiling(body['message']))
            return replay
        budget=RunBudget(); telemetry=Telemetry(ctx,self.config.harness_trace_enabled)
        model_id = ('FakeModelGateway' if mode == 'deterministic' else
                    self.config.vlm_model if body.get('allow_external_image_send') else self.config.llm_model)
        recorder=TraceRecorder(self.b,principal,conv,self.config,trace_id=ctx.trace_id,invocation_id=ctx.invocation_id,parent_invocation_id=ctx.parent_invocation_id,graph_thread_id=ctx.graph_thread_id,workflow_id=wid,model_id=model_id)
        # Injected services are explicit; audit ContextVar is not an authorization source.
        services=HarnessServices(self.fake_model if mode=='deterministic' else self.model,self.tools,telemetry,budget,self.controller)
        from app.harness.context_manager import ContextManager
        services.context=ContextManager(self.b,self.config)
        services.image_ids=tuple(body.get('image_ids', ()))
        services.allow_external_image_send=body.get('allow_external_image_send', False)
        async def execute():
            marker=active_trace.set(recorder); result=None; status='FAILED'; code=None
            self.controller.update(ctx,status='RUNNING',started_at=now())
            telemetry.emit('invocation.started',status='running',operation=body['kind'])
            try:
                async with asyncio.timeout(ctx.remaining()):
                    async with self.controller.locks.acquire(thread):
                        ctx.remaining()
                        if services.image_ids:
                            await asyncio.to_thread(self.agent.images.bind, ctx.principal, conv, services.image_ids)
                            if not services.allow_external_image_send:
                                telemetry.emit('vlm.image.rejected', error_code='VLM_IMAGE_SEND_DISABLED',
                                               status='blocked', count=len(services.image_ids))
                        if legacy and mode=='real' and body['kind']=='chat':
                            try: self.model.get_client()
                            except HarnessFailure as e:
                                if e.code=='CONFIG_ERROR': raise DomainError('LLM_NOT_CONFIGURED',e.message) from None
                                raise
                        result=await operation(ctx,services)
                code=result.get('error_code')
                if workflow and code:
                    recorder.emit('error',status='error',error_code=code)
                status='UNKNOWN_OUTCOME' if code=='UNKNOWN_OUTCOME' else 'TIMED_OUT' if code=='RUN_TIMEOUT' else 'FAILED' if code else 'SUCCEEDED'
                pending=result.get('pending_confirmation')
                if pending:
                    # The graph helper returns only after checking the persisted interrupt.
                    wid_result=pending['workflow_id']
                    self.controller.update(ctx,workflow_id=wid_result)
                    recorder.payload['workflow_id']=wid_result
                    if pending['status']=='WAITING_CONFIRMATION': status='INTERRUPTED'
                result.update(invocation_id=ctx.invocation_id,trace_id=ctx.trace_id,invocation_status=status)
                return result
            except BaseException as exc:
                code=exc.code if isinstance(exc,DomainError) else 'RUN_TIMEOUT' if isinstance(exc,TimeoutError) else 'RUN_CANCELLED' if isinstance(exc,asyncio.CancelledError) else 'INTERNAL_ERROR'
                if workflow and budget.write_started and isinstance(exc,(Exception,asyncio.CancelledError)):
                    if code=='RUN_TIMEOUT':
                        telemetry.emit('run.timeout',error_code=code,status='error')
                    outcome=await self.agent.refunds.reconcile(principal,workflow,response_missing=True)
                    code=outcome.error_code
                    status={'VERIFIED_SUCCESS':'SUCCEEDED','FAILED':'FAILED','UNKNOWN':'UNKNOWN_OUTCOME'}[outcome.status]
                    result=self.agent.refund_response(outcome,workflow)
                    result.update(invocation_id=ctx.invocation_id,trace_id=ctx.trace_id,invocation_status=status)
                    return result
                if budget.write_started:
                    status='UNKNOWN_OUTCOME'; code='UNKNOWN_OUTCOME'
                else: status='UNKNOWN_OUTCOME' if code=='UNKNOWN_OUTCOME' else 'TIMED_OUT' if code=='RUN_TIMEOUT' else 'CANCELLED' if code=='RUN_CANCELLED' else 'FAILED'
                public_code={'INTERNAL_ERROR':'SYSTEM_ERROR','RUN_TIMEOUT':'TIMEOUT','RUN_CANCELLED':'CANCELLED'}.get(code,code) if legacy else code
                recorder.emit('error',status='error',error_code=public_code,error_type=type(exc).__name__)
                if code in ('RUN_TIMEOUT','RUN_CANCELLED','STEP_LIMIT_EXCEEDED'):
                    telemetry.emit({'RUN_TIMEOUT':'run.timeout','RUN_CANCELLED':'run.cancelled','STEP_LIMIT_EXCEEDED':'run.step_limit'}[code],error_code=code,status='error',step_count=budget.steps,remaining_seconds=max(0,ctx.deadline_monotonic-time.monotonic()))
                if isinstance(exc,asyncio.CancelledError) and not budget.write_started: raise
                message=exc.message if isinstance(exc,DomainError) else '本次执行暂不可用，请重试或转人工'
                result={'error_code':public_code,'answer':message,'invocation_id':ctx.invocation_id,'trace_id':ctx.trace_id,'invocation_status':status}
                if isinstance(exc,DomainError) and not isinstance(exc,HarnessFailure):
                    exc.invocation_id=ctx.invocation_id; exc.trace_id=ctx.trace_id
                    raise
                failure=DomainError(public_code,message)
                failure.invocation_id=ctx.invocation_id; failure.trace_id=ctx.trace_id
                raise failure from None
            finally:
                if any(event['type']=='loop.started' for event in recorder.payload['events']) and not any(event['type']=='loop.terminated' for event in recorder.payload['events']):
                    outstanding={event['tool_call_id'] for event in recorder.payload['events'] if event['type']=='loop.tool_proposed'}-{event['tool_call_id'] for event in recorder.payload['events'] if event['type']=='loop.tool_observed'}
                    telemetry.emit('loop.terminated',terminal_reason={'TIMED_OUT':'TIMEOUT','CANCELLED':'CANCELLED'}.get(status,'LIMIT' if code=='STEP_LIMIT_EXCEEDED' else 'ERROR'),iterations=budget.steps,tool_calls=budget.tool_calls,error_code=code,unobserved_tool_call_ids=sorted(outstanding))
                telemetry.emit('invocation.finished',status=status,error_code=code,step_count=budget.steps,model_calls=budget.model_calls,tool_calls=budget.tool_calls,model_attempts=budget.model_attempts,tool_attempts=budget.tool_attempts,duration_ms=(time.perf_counter()-recorder.started)*1000)
                safe=clean(result) if result is not None else {'error_code':code,'answer':'执行未完成，请查询原操作状态','invocation_id':ctx.invocation_id,'trace_id':ctx.trace_id,'invocation_status':status}
                if safe.get('pending_confirmation'): safe['pending_confirmation']['confirmation_credential_available']=False
                self.controller.update(ctx,status=status,error_code=code,finished_at=now(),response_message_id=(result or {}).get('message_id'),response_json=json.dumps(safe,ensure_ascii=False),model_calls=budget.model_calls,tool_calls=budget.tool_calls,prompt_tokens=budget.prompt_tokens if budget.provider_usage_complete else None,completion_tokens=budget.completion_tokens if budget.provider_usage_complete else None)
                recorder.payload.update(model_attempts=budget.model_attempts,tool_calls=budget.tool_calls,tool_attempts=budget.tool_attempts,step_count=budget.steps,error_class=error_class(code) if code else None,unknown_outcome=status=='UNKNOWN_OUTCOME',invocation_status=status)
                try:
                    refund=(result or {}).get('refund_workflow_result') or {}
                    business_outcome=refund.get('status','NONE')
                    if refund.get('verified'):
                        business_outcome='REQUEST_ACCEPTED'
                    recorder.finish(business_outcome=business_outcome)
                finally:
                    self.controller.active.discard(ctx.invocation_id)
                    active_trace.reset(marker)
        task=self.controller.retain(asyncio.create_task(execute()))
        try: return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not budget.write_started:
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
            # Writes remain owned by this controller task and durable original operation key.
            raise
