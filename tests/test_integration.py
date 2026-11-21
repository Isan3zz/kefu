"""Real sockets and interpreter boundaries, not mocks of SDK interfaces."""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import pytest
import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from app.mcp import Gateway,sign_actor
from app.mcp import decode_tool_result
from app.db import orders,refund_requests,workflow_instances
from app.business import DomainError
from app.agent import Agent

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); return s.getsockname()[1]

@pytest.fixture
def http_mcp(env,tmp_path):
    port=free_port()
    variables=os.environ.copy(); variables.update(SQLITE_URL=env['cfg'].sqlite_url,MCP_SERVICE_SECRET=env['cfg'].mcp_service_secret)
    log=(tmp_path/'mcp.log').open('w',encoding='utf-8')
    code="from app.mcp import create_mcp; from app.db import DB; from app.business import Business; from app.config import settings; create_mcp(Business(DB(settings.sqlite_url)),settings.mcp_service_secret).run(transport='http',host='127.0.0.1',port="+str(port)+")"
    proc=subprocess.Popen([sys.executable,'-c',code],env=variables,stdout=log,stderr=log)
    try:
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            if proc.poll() is not None: raise AssertionError('MCP exited early')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.2): break
            except OSError: time.sleep(.1)
        else: raise AssertionError('MCP startup timed out')
        yield env['cfg'].model_copy(update={'mcp_base_url':f'http://127.0.0.1:{port}/mcp'})
    finally:
        proc.terminate(); proc.wait(timeout=10); log.close()

async def test_http_fastmcp_actor_boundary(env,http_mcp):
    gateway=Gateway(env['b'],http_mcp)
    r=await gateway.call(env['a'],'get_order',{'order_id':'O1001'},'order-tracking')
    assert r['ok'] and r['data']['id']=='O1001'
    async with Client(http_mcp.mcp_base_url) as client:
        tools=await client.list_tools()
        assert len(tools)==7
        by_name={tool.name:tool for tool in tools}
        assert 'submit_refund_request' not in by_name
        assert set(by_name['begin_refund_workflow'].inputSchema['properties'])=={'order_id'}
        assert all({'ok','data','error','request_id'}.issubset(t.outputSchema['properties']) for t in tools)
        assert all('user_id' not in t.inputSchema.get('properties',{}) and 'service_secret' not in t.inputSchema.get('properties',{}) for t in tools)
        r=await client.call_tool('get_orders',{})
        result=decode_tool_result(r)
        assert not result['ok'] and result['error']['code']=='FORBIDDEN'
    token=sign_actor(env['a'],http_mcp.mcp_service_secret)
    async with Client(StreamableHttpTransport(http_mcp.mcp_base_url,headers={'X-Internal-Auth':'Bearer '+token})) as client:
        result=decode_tool_result(await client.call_tool('begin_refund_workflow',{'order_id':'O1001'}))
        assert not result['ok'] and result['error']['code']=='FORBIDDEN'
        forbidden=await client.call_tool('submit_refund_request',{'workflow_id':'wf-unused'},raise_on_error=False)
        assert forbidden.is_error
    token=sign_actor(env['a'],http_mcp.mcp_service_secret,refund_preview_scope=(env['conv'],'invalid-invocation'))
    async with Client(StreamableHttpTransport(http_mcp.mcp_base_url,headers={'X-Internal-Auth':'Bearer '+token})) as client:
        result=decode_tool_result(await client.call_tool('begin_refund_workflow',{'order_id':'O1001'}))
        assert not result['ok'] and result['error']['code']=='NOT_FOUND'
    with env['db'].tx() as connection:
        assert connection.execute(workflow_instances.select()).first() is None
    for token in (sign_actor(env['a'],'wrong-secret'),sign_actor(env['a'],http_mcp.mcp_service_secret,expires=time.time()-5)):
        async with Client(StreamableHttpTransport(http_mcp.mcp_base_url,headers={'X-Internal-Auth':'Bearer '+token})) as client:
            r=await client.call_tool('get_orders',{}); assert not decode_tool_result(r)['ok']

async def test_T08_separate_process_resume(env,http_mcp,tmp_path):
    agent=Agent(env['b'],Gateway(env['b'],http_mcp),http_mcp,env['rag'])
    try:
        r=await agent.chat(env['a'],env['conv'],'退订单 O1001','deterministic')
        pending=r['pending_confirmation']
        assert [tool['name'] for tool in r['tool_actions']]==['begin_refund_workflow']
        assert pending['confirm_nonce'] and pending['status']=='WAITING_CONFIRMATION'
        assert pending['confirm_nonce'] not in json.dumps(r['tool_actions'])
    finally:
        await agent.close()
    # Credential travels via stdin, never argv or logs.
    code="""
import asyncio,json,sys
from app.config import settings
from app.db import DB
from app.business import Business,Actor
from app.mcp import Gateway
from app.agent import Agent
p=json.load(sys.stdin)
b=Business(DB(settings.sqlite_url))
r=asyncio.run(Agent(b,Gateway(b)).confirm(Actor('user_a','a'),p['workflow_id'],p['confirm_nonce']))
print(json.dumps(r))
"""
    variables=os.environ.copy(); variables.update(SQLITE_URL=env['cfg'].sqlite_url,CHECKPOINT_DB_PATH=env['cfg'].checkpoint_db_path,MCP_BASE_URL='http://127.0.0.1:1/mcp')
    p=await asyncio.to_thread(subprocess.run,[sys.executable,'-c',code],input=json.dumps(pending),env=variables,capture_output=True,text=True,timeout=40)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['status']=='ACCEPTED'
    assert env['b'].workflow(env['a'],pending['workflow_id'])['status']=='REQUEST_ACCEPTED'

async def test_confirm_changed_after_authorization(env):
    p=env['b'].preview(env['a'],env['conv'],'O1001'); env['b'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    with env['db'].tx() as c: c.execute(orders.update().where(orders.c.id=='O1001').values(paid_amount_cents=1,version=2))
    with pytest.raises(DomainError): env['b'].submit(env['a'],p['workflow_id'])

async def test_expired_nonce_cancel_and_cross_owner(env):
    from datetime import timedelta
    from app.db import workflow_instances
    p=env['b'].preview(env['a'],env['conv'],'O1001')
    with pytest.raises(DomainError): env['b'].confirm(env['other'],p['workflow_id'],p['confirm_nonce'])
    with pytest.raises(DomainError): env['b'].confirm(env['a'],p['workflow_id'],'wrong')
    with env['db'].tx() as c: c.execute(workflow_instances.update().values(nonce_expires_at='2020-01-01T00:00:00+00:00'))
    with pytest.raises(DomainError): env['b'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert env['b'].cancel(env['a'],p['workflow_id'])['status']=='CANCELLED'

async def test_api_complete_flow_and_sse(env):
    from app.main import create_app
    app=create_app(env['cfg'],env['db'],env['gateway'],env['rag'])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        login=await client.post('/api/v1/auth/demo-login',json={'demo_user_key':'user_a'})
        headers={'Authorization':'Bearer '+login.json()['data']['token']}
        conv=(await client.post('/api/v1/conversations',headers=headers)).json()['data']['id']
        resp=await client.post('/api/v1/chat',headers=headers,json={'conversation_id':conv,'message':'退订单 O1001','mode':'deterministic'})
        assert resp.status_code==200,resp.text
        p=resp.json()['data']['pending_confirmation']
        url='/api/v1/workflows/'+p['workflow_id']+'/confirm'
        first=await client.post(url,headers=headers,json={'confirm_nonce':p['confirm_nonce']})
        second=await client.post(url,headers=headers,json={'confirm_nonce':p['confirm_nonce']})
        assert first.json()['data']['refund_request_id']==second.json()['data']['refund_request_id']
        stream=await client.post('/api/v1/chat/stream',headers=headers,json={'conversation_id':conv,'message':'你好','mode':'deterministic'})
        assert 'event: done' in stream.text and 'event: token' in stream.text

async def test_judge_network_failure_not_score(env):
    from app.evaluation import judge
    cfg=env['cfg'].model_copy(update={'judge_enabled':True,'judge_api_key':'fake','judge_base_url':'http://127.0.0.1:1/v1','judge_model':'unavailable'})
    assert (await judge('hello',[],cfg))['status']=='NOT_EVALUATED'
