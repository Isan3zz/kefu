import asyncio
import json
import os
import subprocess
import sys

import httpx
import pytest
from sqlalchemy import select,func
from app.db import refund_requests,traces,workflow_instances
from app.db import shipments
from app.db import orders
from app.business import DomainError

@pytest.mark.parametrize('fields', [
    {'idempotency_key':'different-operation'},
    {'policy_version':'different-policy'},
])
def test_business_refund_outcome_checks_original_key_and_policy(env,fields):
    business, actor = env['b'], env['a']
    preview=business.preview(actor,env['conv'],'O1001')
    wid=preview['workflow_id']
    business.confirm(actor,wid,preview['confirm_nonce'])
    receipt=business.submit(actor,wid)
    assert business.refund_outcome(actor,wid)['refund_request_id']==receipt['refund_request_id']
    with env['db'].tx() as connection:
        connection.execute(refund_requests.update().where(
            refund_requests.c.id==receipt['refund_request_id']).values(**fields))
    assert business.refund_outcome(actor,wid) is None


async def test_submit_success_without_business_receipt_cannot_finish_workflow(env,monkeypatch):
    agent=env['agent']
    p=(await agent.chat(env['a'],env['conv'],'退订单 O1001','deterministic'))['pending_confirmation']
    attempts=0
    def forged(actor, wid):
        nonlocal attempts
        assert wid==p['workflow_id']
        attempts+=1
        return {'refund_request_id':'forged-id','status':'ACCEPTED','idempotent_replay':False}
    with monkeypatch.context() as patch:
        patch.setattr(env['b'],'submit',forged)
        with pytest.raises(DomainError) as error: await agent.confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert error.value.code=='UNKNOWN_OUTCOME' and attempts==1
    verdict=error.value.workflow_result
    assert verdict['status']=='UNKNOWN' and not verdict['verified'] and verdict['refund_request_id'] is None
    assert '暂未确认' in error.value.message and '已受理' not in error.value.message
    with env['db'].tx() as connection: assert connection.scalar(select(func.count()).select_from(refund_requests))==0
    assert env['b'].workflow(env['a'],p['workflow_id'])['status']=='SUBMITTING'
    with pytest.raises(DomainError,match='暂未确认'):
        await agent.confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    for conv in (env['conv'], env['b'].conversation(env['a'])['id']):
        repeated=await agent.chat(env['a'],conv,'退订单 O1001','deterministic')
        assert repeated['refund_workflow_result']['status']=='UNKNOWN'
        assert repeated['refund_workflow_result']['workflow_id']==p['workflow_id']
        assert not repeated['pending_confirmation'] and '暂未确认' in repeated['answer']
    progress=await agent.chat(env['a'],env['conv'],'订单 O1001 退款进度','deterministic')
    assert '暂未确认' in progress['answer'] and '已受理' not in progress['answer']
    with env['db'].tx() as connection:
        assert connection.scalar(select(func.count()).select_from(workflow_instances))==1
        assert connection.scalar(select(func.count()).select_from(refund_requests))==0


@pytest.mark.parametrize('fault', ['receipt_id','receipt_amount','receipt_status','malformed_receipt',
                                  'db_amount','db_tenant','db_user','db_order','db_status','workflow_status'])
async def test_refund_verification_detects_disagreement(env,monkeypatch,fault):
    p=env['b'].preview(env['a'],env['conv'],'O1001')
    original=env['b'].submit
    actual_id=None
    writes=0

    def disagree(*args,**kwargs):
        nonlocal actual_id,writes
        result=original(*args,**kwargs)
        writes+=1
        actual_id=result['refund_request_id']
        if fault=='malformed_receipt':
            result.pop('refund_request_id')
        elif fault.startswith('receipt_'):
            field,value={'receipt_id':('refund_request_id','forged-id'),
                         'receipt_status':('status','REJECTED'),
                         'receipt_amount':('amount_cents',1)}[fault]
            result[field]=value
        else:
            with env['db'].tx() as connection:
                if fault=='workflow_status':
                    connection.execute(workflow_instances.update().where(
                        workflow_instances.c.id==p['workflow_id']).values(status='SUBMITTING'))
                else:
                    fields={'db_amount':{'amount_cents':1},'db_tenant':{'tenant_id':'b'},
                            'db_user':{'user_id':'user_b'},'db_order':{'order_id':'O1003'},
                            'db_status':{'status':'REJECTED'}}[fault]
                    connection.execute(refund_requests.update().where(refund_requests.c.id==actual_id).values(**fields))
        return result

    monkeypatch.setattr(env['b'],'submit',disagree)
    result=await env['agent'].harness.resume_workflow(principal=env['a'],workflow_id=p['workflow_id'],
                                                       confirmation_payload=p)
    verdict=result['refund_workflow_result']
    assert verdict['status']=='UNKNOWN' and not verdict['verified']
    assert verdict['amount_cents'] is None and verdict['refund_request_id'] is None
    assert actual_id not in result['answer'] and '已受理' not in result['answer']
    with env['db'].tx() as connection:
        trace=json.loads(connection.scalar(select(traces.c.payload).where(traces.c.id==result['trace_id'])))
        assert connection.scalar(select(func.count()).select_from(refund_requests))==1
    assert trace['business_outcome']=='UNKNOWN'
    assert any(event['type']=='workflow.verified' and event['result']=='UNKNOWN' for event in trace['events'])
    if fault.startswith('db_') or fault=='workflow_status':
        with pytest.raises(DomainError,match='暂未确认'):
            await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert writes==1


async def test_refund_verification_recovers_after_read_outage_with_same_api_request(env,monkeypatch):
    from app.main import create_app

    p=env['b'].preview(env['a'],env['conv'],'O1001')
    app=create_app(env['cfg'],env['db'],env['gateway'],env['rag'])
    token=env['b'].login('user_a')['token']
    path='/api/v1/workflows/'+p['workflow_id']
    params={'confirm_nonce':p['confirm_nonce'],'request_id':'original-confirm'}
    original=app.state.agent.b.submit
    writes=0

    def dispatch(*args,**kwargs):
        nonlocal writes
        writes+=1
        return original(*args,**kwargs)

    def unavailable(*args,**kwargs):
        raise TimeoutError('temporary lookup outage')

    monkeypatch.setattr(app.state.agent.b,'submit',dispatch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test',
                                 headers={'Authorization':'Bearer '+token}) as client:
        with monkeypatch.context() as patch:
            patch.setattr(app.state.agent.b,'refund_outcome',unavailable)
            first=await client.post(path+'/confirm',json=params)
            assert first.status_code==409
            error=first.json()['error']
            assert error['refund_workflow_result']['status']=='UNKNOWN'
            assert error['refund_workflow_result']['reason']=='LOOKUP_UNAVAILABLE'
            assert (await client.get(path)).json()['data']['refund_workflow_result']['status']=='UNKNOWN'
        verified=(await client.get(path)).json()['data']['refund_workflow_result']
        assert verified['status']=='VERIFIED_SUCCESS' and verified['verified']
        replay=await client.post(path+'/confirm',json=params)
        data=replay.json()['data']
        assert replay.status_code==200 and data['invocation_id']==error['invocation_id']
        assert data['refund_request_id']==verified['refund_request_id']
        with env['db'].tx() as connection:
            connection.execute(refund_requests.update().where(
                refund_requests.c.id==data['refund_request_id']).values(amount_cents=1))
        inconsistent=await client.post(path+'/confirm',json=params)
        assert inconsistent.status_code==409
        assert inconsistent.json()['error']['refund_workflow_result']['status']=='UNKNOWN'
    assert writes==1


async def test_refund_submission_rejection_is_definitive_and_not_retried(env,monkeypatch):
    p=env['b'].preview(env['a'],env['conv'],'O1001')
    original=env['b'].submit
    attempts=0

    def change_order_after_confirmation(*args,**kwargs):
        nonlocal attempts
        attempts+=1
        with env['db'].tx() as connection:
            connection.execute(orders.update().where(orders.c.id=='O1001').values(status='CANCELLED',version=2))
        return original(*args,**kwargs)

    monkeypatch.setattr(env['b'],'submit',change_order_after_confirmation)
    for _ in range(2):
        with pytest.raises(DomainError,match='订单或资格已变化') as error:
            await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
        result=error.value.workflow_result
        assert result['status']=='FAILED' and result['business_status']=='REJECTED' and not result['verified']
    assert attempts==1
    with env['db'].tx() as connection:
        assert connection.scalar(select(func.count()).select_from(refund_requests))==0


async def test_refund_restart_after_commit_before_verification(env,monkeypatch):
    p=env['b'].preview(env['a'],env['conv'],'O1001')
    config=env['cfg']
    process_env=os.environ.copy()
    for field in ('sqlite_url','checkpoint_db_path','chroma_path','bm25_index_path'):
        process_env[field.upper()]=getattr(config,field)
    process_env.update(TEST_WORKFLOW=p['workflow_id'],TEST_NONCE=p['confirm_nonce'])
    code="""
import asyncio
import os
from app.agent import Agent
from app.business import Actor, Business
from app.config import settings
from app.db import DB
from app.mcp import Gateway, create_mcp

business = Business(DB(settings.sqlite_url))
gateway = Gateway(business, settings, create_mcp(business, settings.mcp_service_secret, testing=True))
agent = Agent(business, gateway, settings)
agent.refunds.verify = lambda *args, **kwargs: os._exit(27)
asyncio.run(agent.confirm(Actor('user_a', 'a'), os.environ['TEST_WORKFLOW'], os.environ['TEST_NONCE']))
"""
    exited=await asyncio.to_thread(subprocess.run,[sys.executable,'-c',code],env=process_env,
                                   capture_output=True,text=True,timeout=60)
    assert exited.returncode==27,exited.stderr
    with env['db'].tx() as connection:
        actual=connection.execute(select(refund_requests.c.id)).scalar_one()

    def unexpected_submit(*args,**kwargs):
        pytest.fail('Restart must enter verification without submitting again')

    monkeypatch.setattr(env['b'],'submit',unexpected_submit)
    recovered=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    verdict=recovered['refund_workflow_result']
    assert verdict['verified'] and verdict['refund_request_id']==actual

async def test_accepted_replay_reads_business_truth_without_mcp(env,monkeypatch):
    p=(await env['agent'].chat(env['a'],env['conv'],'退订单 O1001','deterministic'))['pending_confirmation']
    first=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert first['refund_workflow_result']['status']=='VERIFIED_SUCCESS'
    repeated=await env['agent'].chat(env['a'],env['conv'],'退订单 O1001','deterministic')
    assert not repeated['pending_confirmation'] and repeated['refund_workflow_result']['verified']
    assert repeated['refund_workflow_result']['workflow_id']==p['workflow_id']
    assert first['refund_request_id'] in repeated['answer']
    async def must_not_call(*args,**kwargs): raise AssertionError('Accepted replay must not depend on MCP availability')
    monkeypatch.setattr(env['gateway'],'call',must_not_call)
    replay=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert replay['refund_request_id']==first['refund_request_id'] and replay['idempotent_replay']


async def test_another_pending_preview_cannot_dispatch_while_original_outcome_is_unknown(env,monkeypatch):
    first=env['b'].preview(env['a'],env['conv'],'O1001')
    other=env['b'].preview(env['a'],env['b'].conversation(env['a'])['id'],'O1001')
    attempts=[]

    def unknown(actor, wid):
        attempts.append(wid)
        raise TimeoutError('submission result is unavailable')

    monkeypatch.setattr(env['b'],'submit',unknown)
    for preview in (first,other,first,other):
        with pytest.raises(DomainError) as error:
            await env['agent'].confirm(env['a'],preview['workflow_id'],preview['confirm_nonce'])
        assert error.value.workflow_result['status']=='UNKNOWN'
    assert attempts==[first['workflow_id']]

def test_paid_order_cannot_refund_if_shipment_changes_before_confirmation(env):
    p=env['b'].preview(env['a'],env['conv'],'O1003')
    with env['db'].tx() as connection:
        connection.execute(shipments.update().where(shipments.c.order_id=='O1003').values(status='运输中'))
    with pytest.raises(DomainError): env['b'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    with env['db'].tx() as connection: assert connection.scalar(select(func.count()).select_from(refund_requests))==0

async def test_refund_preview_does_not_round_integer_cents_through_float(env):
    amount=12345678901234567
    with env['db'].tx() as connection:
        connection.execute(orders.update().where(orders.c.id=='O1001').values(paid_amount_cents=amount))
    response=await env['agent'].chat(env['a'],env['conv'],'退订单 O1001','deterministic')
    assert response['pending_confirmation']['amount_cents']==amount
    assert '123456789012345.67 元' in response['answer']
