import asyncio,json,os,subprocess,sys
from pathlib import Path
from sqlalchemy import select,func
import httpx,pytest
from fastmcp import Client
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.db import agent_invocations,refund_requests,tickets,traces
from app.business import DomainError
from app.agent import Agent

def records(env):
    with env['db'].tx() as c: return [json.loads(r) for r in c.scalars(select(traces.c.payload))]

async def test_H01_gateway_decides_once_and_answers_from_evidence(env):
    env['rag'].build('knowledge')
    result=await env['agent'].harness.invoke_chat(principal=env['a'],conversation_id=env['conv'],message='耳机有哪些颜色',mode='deterministic')
    assert result['citations']
    assert [c['purpose'] for c in env['agent'].harness.fake_model.calls]==['next_action','final_answer']
    assert {c['invocation_id'] for c in env['agent'].harness.fake_model.calls}=={result['invocation_id']}
    assert env['agent'].harness.controller.read(env['a'],result['invocation_id'])['model_calls']==2

async def test_H02_real_config_error_and_fake_mode(env):
    with pytest.raises(DomainError) as error: await env['agent'].harness.invoke_chat(principal=env['a'],conversation_id=env['conv'],message='你好')
    assert error.value.code=='CONFIG_ERROR'
    assert records(env)[0]['invocation_status']=='FAILED'
    result=await env['agent'].harness.invoke_chat(principal=env['a'],conversation_id=env['conv'],message='你好',mode='deterministic')
    assert result['invocation_status']=='SUCCEEDED'

async def test_H06_global_timeout_marks_invocation_and_stops_new_calls(env,monkeypatch):
    agent=env['agent']; monkeypatch.setattr(agent.config,'agent_timeout_seconds',.5)
    original=agent.harness.fake_model.complete; calls=0
    async def slow(**kwargs):
        nonlocal calls
        calls+=1; await asyncio.sleep(1); return await original(**kwargs)
    monkeypatch.setattr(agent.harness.fake_model,'complete',slow)
    with pytest.raises(DomainError) as error: await agent.harness.invoke_chat(principal=env['a'],conversation_id=env['conv'],message='查订单',mode='deterministic')
    assert error.value.code=='RUN_TIMEOUT' and calls==1
    record=records(env)[0]
    assert record['invocation_status']=='TIMED_OUT'
    assert any(e['type']=='run.timeout' for e in record['events'])
    assert not any(e['type']=='tool.started' for e in record['events'])

async def test_H07_step_limit_no_tools(env,monkeypatch):
    monkeypatch.setattr(env['cfg'],'agent_max_steps',0)
    with pytest.raises(DomainError) as error: await env['agent'].harness.invoke_chat(principal=env['a'],conversation_id=env['conv'],message='查订单',mode='deterministic')
    assert error.value.code=='STEP_LIMIT_EXCEEDED'
    assert not any(e['type']=='tool.started' for e in records(env)[0]['events'])

async def test_H13_timeout_after_refund_commit_reconciles_original_key(env,monkeypatch):
    original=Client.call_tool; writes=0
    async def lose(self,name,*args,**kwargs):
        nonlocal writes
        result=await original(self,name,*args,**kwargs)
        if name=='submit_refund_request': writes+=1; raise TimeoutError('lost response')
        return result
    monkeypatch.setattr(Client,'call_tool',lose)
    preview=await env['agent'].chat(env['a'],env['conv'],'退订单 O1001','deterministic'); p=preview['pending_confirmation']
    confirmed=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    replay=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert confirmed['refund_request_id']==replay['refund_request_id'] and writes==1
    with env['db'].tx() as c: assert c.scalar(select(func.count()).select_from(refund_requests))==1
    assert any(e['type']=='recovery.completed' and e.get('result')=='found' for r in records(env) for e in r['events'])

async def test_H14_H15_pending_checkpoint_survives_consultation_and_reentry(env):
    env['rag'].build('knowledge')
    agent=env['agent']; first=await agent.chat(env['a'],env['conv'],'退订单 O1001','deterministic'); p=first['pending_confirmation']
    async with AsyncSqliteSaver.from_conn_string(env['cfg'].checkpoint_db_path) as saver:
        graph=agent.refund_graph(env['a'],saver); cfg={'configurable':{'thread_id':'wf:'+p['workflow_id']}}
        initial=await graph.aget_state(cfg)
        await agent.chat(env['a'],env['conv'],'耳机有哪些颜色','deterministic')
        after=await graph.aget_state(cfg)
        assert initial.config==after.config and after.interrupts
    one=await agent.confirm(env['a'],p['workflow_id'],p['confirm_nonce']); two=await agent.confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert one['refund_request_id']==two['refund_request_id']
    with env['db'].tx() as c:
        assert c.scalar(select(func.count()).select_from(refund_requests))==1
        source=c.execute(select(agent_invocations).where(agent_invocations.c.invocation_id==first['invocation_id'])).mappings().one()
        resumed=c.execute(select(agent_invocations).where(agent_invocations.c.invocation_id==one['invocation_id'])).mappings().one()
    assert source['status']=='INTERRUPTED' and resumed['parent_invocation_id']==source['invocation_id']
    assert resumed['graph_thread_id']=='wf:'+p['workflow_id']

async def test_H16_real_interpreter_restart_uses_new_invocation(env):
    first=await env['agent'].chat(env['a'],env['conv'],'退订单 O1001','deterministic'); p=first['pending_confirmation']
    process_env=os.environ.copy()
    for field in ('sqlite_url','checkpoint_db_path','chroma_path','bm25_index_path'): process_env[field.upper()]=getattr(env['cfg'],field)
    process_env['TEST_WORKFLOW']=p['workflow_id']; process_env['TEST_NONCE']=p['confirm_nonce']
    code="""import asyncio,os,json
from app.config import settings
from app.db import DB
from app.business import Business,Actor
from app.mcp import Gateway,create_mcp
from app.agent import Agent
async def main():
 b=Business(DB(settings.sqlite_url)); g=Gateway(b,settings,create_mcp(b,settings.mcp_service_secret,testing=True)); agent=Agent(b,g,settings)
 result=await agent.confirm(Actor('user_a','a'),os.environ['TEST_WORKFLOW'],os.environ['TEST_NONCE'])
 print(json.dumps(result)); await agent.close(); b.db.engine.dispose()
asyncio.run(main())"""
    result=await asyncio.to_thread(subprocess.run,[sys.executable,'-c',code],env=process_env,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
    output=json.loads(result.stdout.strip().splitlines()[-1])
    assert output['invocation_id']!=first['invocation_id'] and output['status']=='ACCEPTED'
    row=env['agent'].harness.controller.read(env['a'],output['invocation_id'])
    assert row['parent_invocation_id']==first['invocation_id'] and row['graph_thread_id']=='wf:'+p['workflow_id']

async def test_H17_api_idempotency_replay_and_conflict(env):
    from app.main import create_app
    app=create_app(env['cfg'],env['db'],env['gateway'],env['rag']); token=env['b'].login('user_a')['token']
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',headers={'Authorization':'Bearer '+token}) as client:
        body={'conversation_id':env['conv'],'message':'退订单 O1001','mode':'deterministic','request_id':'stable-key'}
        first=await client.post('/api/v1/chat',json=body); again=await client.post('/api/v1/chat',json=body)
        a,b=first.json()['data'],again.json()['data']; assert a['invocation_id']==b['invocation_id']
        assert a['pending_confirmation']['workflow_id']==b['pending_confirmation']['workflow_id']
        assert 'confirm_nonce' not in b['pending_confirmation'] # original one-time credential is never persisted
        assert (await client.post('/api/v1/chat',json={**body,'message':'different'})).status_code==409
        p=a['pending_confirmation']; url='/api/v1/workflows/'+p['workflow_id']+'/confirm'; payload={'confirm_nonce':p['confirm_nonce'],'request_id':'confirm-key'}
        one=await client.post(url,json=payload); two=await client.post(url,json=payload)
        assert one.json()['data']['invocation_id']==two.json()['data']['invocation_id']
        assert (await client.get('/api/v1/invocations/'+a['invocation_id'])).status_code==200
    with env['db'].tx() as c: assert c.scalar(select(func.count()).select_from(refund_requests))==1

async def test_H18_actual_graph_serialization_and_cross_thread_parallelism(env,monkeypatch):
    agent=env['agent']; entered=asyncio.Event(); release=asyncio.Event(); calls=[]; original=agent._chat
    async def slow(a,conv,text,*args):
        calls.append((conv,text))
        if text=='first': entered.set(); await release.wait()
        return await original(a,conv,text,*args)
    monkeypatch.setattr(agent,'_chat',slow)
    one=asyncio.create_task(agent.chat(env['a'],env['conv'],'first','deterministic')); await entered.wait()
    two=asyncio.create_task(agent.chat(env['a'],env['conv'],'second','deterministic'))
    another=env['b'].conversation(env['a'])['id']
    await asyncio.wait_for(agent.chat(env['a'],another,'parallel','deterministic'),2)
    assert (env['conv'],'second') not in calls
    release.set(); await asyncio.gather(one,two)
    assert not agent.harness.controller.locks.entries

async def test_H19_sse_disconnect_after_committed_write_does_not_retry_or_cancel(env,monkeypatch):
    agent=env['agent']; committed=asyncio.Event(); release=asyncio.Event(); original=env['gateway']._call; writes=0
    async def delayed(*args,**kwargs):
        nonlocal writes
        result=await original(*args,**kwargs)
        if args[1]=='create_ticket': writes+=1; committed.set(); await release.wait()
        return result
    monkeypatch.setattr(env['gateway'],'_call',delayed)
    stream=agent.harness.stream_chat(principal=env['a'],conversation_id=env['conv'],message='转人工',mode='deterministic',client_request_id='sse-write')
    assert 'status' in await anext(stream)
    consumer=asyncio.create_task(anext(stream)); await committed.wait(); consumer.cancel()
    with pytest.raises(asyncio.CancelledError): await consumer
    release.set(); await agent.harness.controller.close()
    replay=await agent.harness.invoke_chat(principal=env['a'],conversation_id=env['conv'],message='转人工',mode='deterministic',client_request_id='sse-write')
    assert replay['invocation_status']=='SUCCEEDED' and writes==1
    with env['db'].tx() as c: assert c.scalar(select(func.count()).select_from(tickets))==1
    assert not any(r['invocation_status']=='CANCELLED' for r in records(env))

async def test_H23_trace_schema_and_secret_redaction(env):
    result=await env['agent'].chat(env['a'],env['conv'],'查订单 O1001；电话13900000000 Bearer raw-token sk-test-secret','deterministic')
    record=records(env)[0]; encoded=json.dumps(record)
    assert all(value not in encoded for value in ('13900000000','raw-token','sk-test-secret'))
    for e in record['events']:
        assert e['invocation_id']==result['invocation_id']
        assert all(key in e for key in ('schema_version','trace_id','event_id','span_id','ts','attributes','conversation_id','graph_thread_id'))
    kinds={e['type'] for e in record['events']}
    assert {'invocation.started','invocation.finished','model.completed','context.built','tool.completed'}<=kinds

async def test_H24_optional_langfuse_unavailable_does_not_block(env,monkeypatch):
    monkeypatch.setenv('LANGFUSE_ENABLED','true'); monkeypatch.setenv('LANGFUSE_BASE_URL','http://127.0.0.1:1')
    result=await env['agent'].chat(env['a'],env['conv'],'查订单 O1001','deterministic')
    assert result['invocation_status']=='SUCCEEDED' and records(env)
