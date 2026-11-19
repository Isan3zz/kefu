import asyncio
import json
import re
import time
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.types import interrupt,Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import select
from app.config import settings
from app.business import DomainError
from app.db import *
from app.skills import SkillManager
from app.rag import RAG
from app.observability import active_trace
from app.memory.service import MemoryService
from app.memory.trigger import management_request, recommendation_request, save_request_only

ORDER_IDS = re.compile(r'\bO[0-9]+\b', re.ASCII)
ORDER_ORDINAL = re.compile(r'第([一二三四五六七八九十]|[0-9]{1,2})\s*(?:个|单|笔)')


class State(TypedDict,total=False):
    conversation_id:str
    tenant_id:str
    user_id:str
    messages:list[dict]
    active_workflow_id:str|None
    active_product_name:str|None
    message:str
    intent:str
    selected_skill:str|None
    active_order_id:str|None
    order_ids:list[str]
    final_response:str
    pending_confirmation:dict|None
    tool_events:list[dict]
    retrieved_chunks:list[dict]
    workflow_id:str
    result:dict
    trace_id:str

def fake_next_action(text):
    order_reference = ORDER_IDS.search(text) or ORDER_ORDINAL.search(text) or any(word in text for word in ('这单','那单','刚才'))
    if any(word in text for word in ('人工','投诉','解决不了')):
        return 'handoff'
    if any(word in text for word in ('退','取消订单')):
        if any(word in text for word in ('进度','状态','到账','受理')):
            return 'refund_status'
        return 'check_refund' if order_reference or any(word in text for word in ('申请','帮我')) else 'answer_policy'
    if any(word in text for word in ('物流','到哪','快递')):
        return 'track_shipment'
    if '订单' in text:
        return 'lookup_order' if order_reference else 'lookup_orders'
    if recommendation_request(text) or any(word in text for word in ('库存','价格','多少钱')):
        return 'search_products'
    if any(word in text for word in ('颜色','蓝色','耳机','保修','支持','政策')):
        return 'answer_product'
    return 'clarify'


def resolve_order(text, bound_order, order_ids):
    explicit = set(ORDER_IDS.findall(text))
    ordinals = set(ORDER_ORDINAL.findall(text))
    if len(explicit) > 1 or len(ordinals) > 1:
        return None
    ordinal = next(iter(ordinals), None)
    if ordinal:
        position = int(ordinal) if ordinal.isascii() else '一二三四五六七八九十'.index(ordinal) + 1
        if not 1 <= position <= len(order_ids) or (explicit and order_ids[position - 1] not in explicit):
            return None
        return order_ids[position - 1]
    return next(iter(explicit), bound_order)

class Agent:
    def __init__(self,business,gateway,config=settings,rag=None):
        self.b,self.gateway,self.config=business,gateway,config
        self.skills=SkillManager(); self.rag=rag or RAG(business.db,config=config)
        self._llm_client=None
        from app.harness.runtime import HarnessFacade
        self.harness=HarnessFacade(self)
        self.memory = MemoryService(business, self.rag, config,
                                    lambda mode: self.harness.fake_model if mode == 'deterministic' else self.harness.model)
    def llm(self):
        if not all((self.config.llm_api_key,self.config.llm_base_url,self.config.llm_model)):
            from app.harness.contracts import HarnessFailure
            raise HarnessFailure('CONFIG_ERROR','真实聊天需要配置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL；离线演示请显式选择 deterministic',source='model')
        from openai import AsyncOpenAI
        if self._llm_client is None:
            self._llm_client=AsyncOpenAI(api_key=self.config.llm_api_key,base_url=self.config.llm_base_url,timeout=self.config.llm_timeout_seconds,max_retries=0)
        return self._llm_client
    async def close(self):
        await self.harness.close()
        if self._llm_client is not None: await self._llm_client.close()
        self.rag.close()
    async def chat(self,a,conv,text,mode='real',skill_mode='selected',request_id=None):
        return await self.harness.invoke_chat(principal=a,conversation_id=conv,message=text,mode=mode,skill_mode=skill_mode,client_request_id=request_id,legacy=True)
    async def _chat(self,a,conv,text,mode,skill_mode,ctx,services):
        from app.agent_loop import AgentLoop
        from app.harness.tool_protocol import MAX_TOOL_BATCH
        recorder=active_trace.get()
        trace_id=recorder.trace_id
        try:
            consent_generation=self.memory.consent_snapshot(ctx.principal)
        except Exception as error:
            consent_generation=-1
            self.memory.event('skipped',reason_code='CONSENT_UNAVAILABLE',error_type=type(error).__name__)
        loop=AgentLoop(self,ctx,services,text,skill_mode)
        Path(self.config.checkpoint_db_path).parent.mkdir(parents=True,exist_ok=True)
        with self.b.db.tx() as connection:
            rows=connection.execute(select(messages.c.id,messages.c.role,messages.c.content).where(messages.c.conversation_id==conv).order_by(messages.c.created_at.desc()).limit(200)).mappings().all()
            history=[dict(row) for row in reversed(rows)]
            active=connection.execute(select(workflow_instances.c.id).where(workflow_instances.c.conversation_id==conv,workflow_instances.c.user_id==a.user_id,workflow_instances.c.tenant_id==a.tenant_id,workflow_instances.c.status.in_(['WAITING_CONFIRMATION','CONFIRMED'])).order_by(workflow_instances.c.updated_at.desc()).limit(1)).scalar()
        recorder.emit('state_restore',history_messages=len(history),active_workflow_id=active)
        if self.config.memory_enabled and (management_request(text) or save_request_only(text)):
            result = dict(intent='other', retrieved_chunks=[], tool_events=[], final_response='', terminal_reason='COMPLETED')
        else:
            async with AsyncSqliteSaver.from_conn_string(self.config.checkpoint_db_path) as saver:
                compiled=loop.graph().compile(checkpointer=saver)
                config = {'configurable':{'thread_id':ctx.graph_thread_id},
                          'recursion_limit':self.config.agent_max_steps*(1+MAX_TOOL_BATCH)+4}
                result=await compiled.ainvoke(dict(conversation_id=conv,message=text,tenant_id=a.tenant_id,user_id=a.user_id,messages=history,active_workflow_id=active),config,durability='sync')
        mid, source_id = uid('msg-'), uid('msg-')
        memory_dispatch = None
        with self.b.db.tx() as c:
            if not result['final_response']:
                if management_request(text):
                    result['final_response'] = '请在长期记忆设置中确认要删除的条目或关闭保存；这条消息不会新增记忆。'
                elif consent_generation < 0:
                    result['final_response'] = '长期记忆保存未启用。请在长期记忆设置中授权后重新发送。'
                else:
                    from app.memory.validator import prohibited
                    result['final_response'] = ('这条信息不符合长期记忆保存规则。' if prohibited(text) else
                                                '已收到保存请求，将在校验后处理；请在长期记忆列表查看是否保存。')
            for ident,role,content in [(source_id,'user',text),(mid,'assistant',result['final_response'])]: c.execute(messages.insert().values(id=ident,conversation_id=conv,role=role,content=content,created_at=self.b.clock().isoformat(),trace_id=trace_id))
            c.execute(conversations.update().where(conversations.c.id==conv).values(updated_at=now()))
            if self.config.memory_enabled:
                try:
                    with c.begin_nested():
                        memory_dispatch = self.memory.register_turn(c, ctx.principal, conv, source_id, mode, consent_generation)
                except Exception as error:
                    self.memory.event('skipped', reason_code='MEMORY_EXTRACTION_ERROR', error_type=type(error).__name__)
        try:
            if memory_dispatch is not None:
                await self.memory.finish_turn(source_id, *memory_dispatch)
        except Exception as error:
            self.memory.event('skipped', reason_code='MEMORY_EXTRACTION_ERROR', error_type=type(error).__name__)
        budget=services.budget
        usage={'source':'provider','input_tokens':budget.prompt_tokens,'output_tokens':budget.completion_tokens} if budget.model_calls and budget.provider_usage_complete else {'source':'unavailable','estimated_budget_consumed':budget.token_estimate_spent}
        pending = result.get('pending_confirmation')
        if pending and loop.confirmation_nonce:
            pending = {**pending, 'confirm_nonce':loop.confirmation_nonce}
        return dict(conversation_id=conv,message_id=mid,answer=result['final_response'],intent=result['intent'],skill={'name':result.get('selected_skill'),'version':recorder.payload.get('skill_version')},citations=result['retrieved_chunks'],tool_actions=result['tool_events'],pending_confirmation=pending,trace_id=trace_id,terminal_reason=result['terminal_reason'],iteration_count=budget.steps,tool_call_count=budget.tool_calls,model_usage=usage,error_code=next((e.get('error_code') for e in recorder.payload['events'] if e['type']=='error'),None))
    async def commit_refund(self,a,wid,ctx=None,services=None):
        if ctx is None or services is None: raise DomainError('FORBIDDEN','退款提交必须经过 Harness workflow 能力入口')
        previous=self.b.workflow(a,wid)['status']
        services.telemetry.emit('workflow.transition',workflow_id=wid,previous=previous,next='SUBMITTING')
        normalized=await services.tools.execute_refund_from_workflow(ctx=ctx,workflow_id=wid,budget=services.budget,telemetry=services.telemetry)
        if not normalized.ok: raise DomainError(normalized.error['code'],normalized.error['message'])
        actual=self.b.refund_outcome(a,wid)
        if actual is None: raise DomainError('UNKNOWN_OUTCOME','没有可核对的退款受理记录，请查询原操作状态')
        services.telemetry.emit('workflow.transition',workflow_id=wid,previous='SUBMITTING',next=self.b.workflow(a,wid)['status'])
        return normalized.data
    def refund_graph(self,a,saver,ctx=None,services=None):
        async def wait(s):
            interrupt({'kind':'refund_confirmation','workflow_id':s['workflow_id']})
            w=self.b.workflow(a,s['workflow_id'])
            if w['status'] not in ('CONFIRMED','REQUEST_ACCEPTED'): raise DomainError('FORBIDDEN')
            return {}
        async def commit(s):
            return {'result':await self.commit_refund(a,s['workflow_id'],ctx,services)}
        g=StateGraph(State); g.add_node('confirmation',wait); g.add_node('commit',commit); g.add_edge(START,'confirmation'); g.add_edge('confirmation','commit'); g.add_edge('commit',END)
        return g.compile(checkpointer=saver)
    async def start_workflow(self,a,wid,ctx=None,services=None):
        async def start():
            async with AsyncSqliteSaver.from_conn_string(self.config.checkpoint_db_path) as saver:
                graph=self.refund_graph(a,saver,ctx,services); config={'configurable':{'thread_id':'wf:'+wid}}
                state=await graph.aget_state(config)
                if not state.values or ('confirmation' in state.next and not state.interrupts):
                    await graph.ainvoke(None if state.values else {'workflow_id':wid},config,durability='sync')
                    state=await graph.aget_state(config)
                if not state.interrupts or state.interrupts[0].value['kind']!='refund_confirmation': raise DomainError('SYSTEM_ERROR','工作流未持久化确认中断')
                if services: services.telemetry.emit('workflow.interrupted',workflow_id=wid,graph_thread_id='wf:'+wid,interrupt_kind='refund_confirmation')
        if services:
            async with self.harness.controller.locks.acquire('wf:'+wid): await start()
        else: await start()
    async def confirm(self,a,wid,nonce,request_id=None):
        return await self.harness.resume_workflow(principal=a,workflow_id=wid,confirmation_payload={'confirm_nonce':nonce},client_request_id=request_id,legacy=True)
    async def _confirm_and_archive(self,a,wid,nonce,ctx,services):
        workflow=self.b.workflow(a,wid)
        result=await self._confirm(a,wid,nonce,ctx,services)
        result={**result,'answer':'退款申请已受理，申请编号：'+result['refund_request_id']+'。受理不代表退款已到账。'}
        with self.b.db.tx() as connection:
            exists=connection.execute(select(messages.c.id).where(messages.c.conversation_id==workflow['conversation_id'],messages.c.role=='assistant',messages.c.content==result['answer'])).first()
            if not exists:
                connection.execute(messages.insert().values(id=uid('msg-'),conversation_id=workflow['conversation_id'],role='assistant',content=result['answer'],created_at=now(),trace_id=ctx.trace_id))
                connection.execute(conversations.update().where(conversations.c.id==workflow['conversation_id']).values(updated_at=now()))
        active_trace.get().emit('workflow',to=self.b.workflow(a,wid)['status'],result=result)
        return result
    async def _confirm(self,a,wid,nonce,ctx,services):
        previous=self.b.workflow(a,wid)['status']
        w=self.b.confirm(a,wid,nonce)
        services.telemetry.emit('workflow.transition',workflow_id=wid,previous=previous,next=w['status'],version=w['version'])
        if w['status']=='REQUEST_ACCEPTED':
            result=self.b.refund_outcome(a,wid)
            if result is None: raise DomainError('UNKNOWN_OUTCOME')
            return result
        async with AsyncSqliteSaver.from_conn_string(self.config.checkpoint_db_path) as saver:
            graph=self.refund_graph(a,saver,ctx,services); config={'configurable':{'thread_id':w['graph_thread_id']}}
            state=await graph.aget_state(config)
            if not state.values or ('confirmation' in state.next and not state.interrupts):
                await graph.ainvoke(None if state.values else {'workflow_id':wid},config,durability='sync')
                state=await graph.aget_state(config)
            services.telemetry.emit('workflow.resumed',workflow_id=wid,graph_thread_id=w['graph_thread_id'],new_invocation_id=ctx.invocation_id)
            if state.next:
                result=await graph.ainvoke(Command(resume=True) if state.interrupts else None,config)
                return result['result']
            return await self.commit_refund(a,wid,ctx,services)

def pending_status(result):
    return result['pending_confirmation']['status'] if result.get('pending_confirmation') else 'NONE'
