import pytest
from sqlalchemy import select,func
from app.db import refund_requests
from app.db import shipments
from app.db import orders
from app.business import DomainError

async def test_mcp_success_without_business_receipt_cannot_finish_workflow(env,monkeypatch):
    p=env['b'].preview(env['a'],env['conv'],'O1001')
    original=env['gateway']._call
    async def forged(*args,**kwargs):
        return {'ok':True,'data':{'refund_request_id':'forged-id','status':'ACCEPTED','idempotent_replay':False},'error':None,'request_id':'test-tool-call'}
    monkeypatch.setattr(env['gateway'],'_call',forged)
    with pytest.raises(DomainError) as error: await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert error.value.code=='UNKNOWN_OUTCOME'
    with env['db'].tx() as connection: assert connection.scalar(select(func.count()).select_from(refund_requests))==0
    assert env['b'].workflow(env['a'],p['workflow_id'])['status']=='CONFIRMED'
    monkeypatch.setattr(env['gateway'],'_call',original)
    result=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert result['status']=='ACCEPTED'

async def test_accepted_replay_reads_business_truth_without_mcp(env,monkeypatch):
    p=env['b'].preview(env['a'],env['conv'],'O1001')
    first=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    async def must_not_call(*args,**kwargs): raise AssertionError('Accepted replay must not depend on MCP availability')
    monkeypatch.setattr(env['gateway'],'call',must_not_call)
    replay=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert replay['refund_request_id']==first['refund_request_id'] and replay['idempotent_replay']

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
