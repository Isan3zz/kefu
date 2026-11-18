import asyncio,json,time
from dataclasses import replace
from types import SimpleNamespace
import httpx,pytest
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy import select,func
from app.db import refund_requests
from app.harness.model_gateway import ModelGateway
from app.harness.execution_context import RunBudget
from app.harness.run_controller import RunController
from app.harness.policy_guard import PolicyGuard
from app.harness.tool_executor import ToolExecutor
from app.harness.telemetry import Telemetry
from app.harness.contracts import HarnessFailure
from app.mcp import decode_tool_result

def services(env):
    c=RunController(env['b'],env['cfg']); ctx,_=c.begin(env['a'],env['conv'],'conv:'+env['conv'],{})
    return c,ctx,RunBudget(),Telemetry(ctx)

class Label(BaseModel): value:int

@pytest.mark.parametrize('statuses,expected,code', [([429,200],2,None),([401],1,'CONFIG_ERROR'),([503,503,503],3,'MODEL_UPSTREAM_ERROR')])
async def test_H03_H04_model_retry_single_layer(env,statuses,expected,code):
    controller,ctx,budget,tel=services(env); calls=[]
    async def provider(request):
        calls.append(request)
        status=statuses[min(len(calls)-1,len(statuses)-1)]
        if status!=200: return httpx.Response(status,json={'error':{'message':'secret provider detail','type':'rate_limit'}})
        return httpx.Response(200,json={'id':'test','choices':[{'index':0,'message':{'role':'assistant','content':'{"value":1}'},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}})
    client=AsyncOpenAI(api_key='not-real',base_url='http://mock/v1',max_retries=2,http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)))
    gateway=ModelGateway(env['cfg'],lambda:client)
    try:
        if code:
            with pytest.raises(HarnessFailure) as caught: await gateway.complete(ctx=ctx,messages=[{'role':'system','content':'JSON'}],purpose='next_action',response_schema=Label,budget=budget,telemetry=tel)
            assert caught.value.code==code
        else:
            result=await gateway.complete(ctx=ctx,messages=[{'role':'system','content':'JSON'}],purpose='next_action',response_schema=Label,budget=budget,telemetry=tel)
            assert result.structured=={'value':1} and result.attempts==expected
            assert result.usage.source=='unavailable' # the failed 429 attempt supplied no usage
        assert len(calls)==expected and budget.model_attempts==expected
        assert budget.provider_usage_complete is False
    finally: await client.close()

async def test_H05_bad_schema_repaired_once(env):
    _,ctx,budget,tel=services(env); calls=[]
    async def provider(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'test','choices':[{'index':0,'message':{'role':'assistant','content':'{"value":"not-int"}'},'finish_reason':'stop'}]})
    client=AsyncOpenAI(api_key='not-real',base_url='http://mock/v1',http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)))
    try:
        with pytest.raises(HarnessFailure) as error: await ModelGateway(env['cfg'],lambda:client).complete(ctx=ctx,messages=[{'role':'system','content':'JSON'}],purpose='plan',response_schema=Label,budget=budget,telemetry=tel)
        assert error.value.code=='MODEL_BAD_OUTPUT' and len(calls)==2
        assert 'Correct it once' in calls[1]['messages'][-1]['content']
    finally: await client.close()

async def test_H09_read_timeout_retry_and_H10_business_failure(env,monkeypatch):
    controller,ctx,budget,tel=services(env); attempts=0
    original=env['gateway']._call
    async def once(*args,**kwargs):
        nonlocal attempts
        attempts+=1
        if attempts==1: raise TimeoutError()
        return await original(*args,**kwargs)
    monkeypatch.setattr(env['gateway'],'_call',once)
    executor=ToolExecutor(env['gateway'],PolicyGuard(env['b'],env['agent'].skills,env['cfg']),env['cfg'],controller)
    result=await executor.invoke(ctx=ctx,skill_name='order-tracking',tool_name='get_order',arguments={'order_id':'O1001'},budget=budget,telemetry=tel)
    assert result.ok and result.attempts==2 and attempts==2
    result=await executor.invoke(ctx=ctx,skill_name='order-tracking',tool_name='get_order',arguments={'order_id':'O2001'},budget=budget,telemetry=tel)
    assert not result.ok and result.outcome=='KNOWN_FAILURE' and result.error['code']=='NOT_FOUND' and result.attempts==1

async def test_H11_transport_protocol_and_bad_results(env,monkeypatch):
    from app.business import DomainError
    with pytest.raises(DomainError) as error: decode_tool_result(SimpleNamespace(is_error=True))
    assert error.value.code=='TOOL_UPSTREAM_ERROR'
    controller,ctx,budget,tel=services(env)
    async def invalid(*a,**kw): return {'ok':True,'data':{},'error':{'code':'BAD','message':'bad'},'request_id':'x'}
    monkeypatch.setattr(env['gateway'],'_call',invalid)
    executor=ToolExecutor(env['gateway'],PolicyGuard(env['b'],env['agent'].skills,env['cfg']),env['cfg'],controller)
    result=await executor.invoke(ctx=ctx,skill_name='order-tracking',tool_name='get_order',arguments={'order_id':'O1001'},budget=budget,telemetry=tel)
    assert result.error['code']=='TOOL_BAD_RESULT' and result.attempts==1

async def test_H12_workflow_source_string_never_grants_write(env):
    controller,ctx,budget,tel=services(env)
    executor=ToolExecutor(env['gateway'],PolicyGuard(env['b'],env['agent'].skills,env['cfg']),env['cfg'],controller)
    with pytest.raises(HarnessFailure) as error:
        await executor.invoke(ctx=ctx,skill_name='refund-processing',tool_name='submit_refund_request',arguments={'workflow_id':'fake'},source='workflow',budget=budget,telemetry=tel)
    assert error.value.code=='POLICY_DENIED'
    with env['db'].tx() as c: assert c.scalar(select(func.count()).select_from(refund_requests))==0

async def test_H07_H08_steps_and_no_progress(env):
    _,ctx,budget,_=services(env)
    budget.step(ctx,1)
    with pytest.raises(HarnessFailure) as error: budget.step(ctx,1)
    assert error.value.code=='STEP_LIMIT_EXCEEDED'
    budget.action('get_order',{'order_id':'O1001'},1,2)
    with pytest.raises(HarnessFailure) as error: budget.action('get_order',{'order_id':'O1001'},1,2)
    assert error.value.code=='REPEATED_NO_PROGRESS'
    budget.action('get_order',{'order_id':'O1001'},2,2) # actual new state is progress

async def test_H06_no_new_model_or_tool_after_deadline(env):
    controller,ctx,budget,tel=services(env); ctx=replace(ctx,deadline_monotonic=0)
    def forbidden(): raise AssertionError('Model must not start')
    with pytest.raises(HarnessFailure) as error: await ModelGateway(env['cfg'],forbidden).complete(ctx=ctx,messages=[],purpose='route',budget=budget,telemetry=tel)
    assert error.value.code=='RUN_TIMEOUT' and budget.model_calls==0
    executor=ToolExecutor(env['gateway'],PolicyGuard(env['b'],env['agent'].skills,env['cfg']),env['cfg'],controller)
    with pytest.raises(HarnessFailure): await executor.invoke(ctx=ctx,skill_name='order-tracking',tool_name='get_orders',arguments={},budget=budget,telemetry=tel)
    assert budget.tool_calls==0
