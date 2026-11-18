from sqlalchemy import select,func
from app.db import refund_requests,messages

async def test_refund_status_uses_receipt_and_confirmation_is_archived_once(env):
    first=await env['agent'].chat(env['a'],env['conv'],'退订单 O1001','deterministic')
    pending=first['pending_confirmation']
    waiting=await env['agent'].chat(env['a'],env['conv'],'这单退款进度','deterministic')
    assert '没有已受理' in waiting['answer'] and not waiting['pending_confirmation']
    accepted=await env['agent'].confirm(env['a'],pending['workflow_id'],pending['confirm_nonce'])
    await env['agent'].confirm(env['a'],pending['workflow_id'],pending['confirm_nonce'])
    status=await env['agent'].chat(env['a'],env['conv'],'退款状态是什么','deterministic')
    assert accepted['refund_request_id'] in status['answer'] and '不代表退款已到账' in status['answer']
    assert status['tool_actions'][0]['name']=='get_order'
    with env['db'].tx() as connection:
        assert connection.scalar(select(func.count()).select_from(refund_requests))==1
        assert connection.scalar(select(func.count()).select_from(messages).where(messages.c.content==accepted['answer']))==2  # one confirmation archive + the later status reply
