import json
from sqlalchemy import select
from app.db import tickets,traces
from app.db import orders
from datetime import datetime,timezone,timedelta

async def test_recent_orders_use_paid_time_before_id_tie_break(env):
    with env['db'].tx() as connection:
        connection.execute(orders.update().where(orders.c.id=='O1001').values(paid_at=(datetime.now(timezone.utc)-timedelta(days=10)).isoformat()))
    result=await env['agent'].chat(env['a'],env['conv'],'最近的订单','deterministic')
    rows=result['tool_actions'][0]['result']['data']
    assert rows[0]['id']=='O1002' and rows[-1]['id']=='O1001'

async def test_handoff_includes_prior_problem_without_putting_summary_in_trace(env):
    await env['agent'].chat(env['a'],env['conv'],'订单 O1002 物流到哪了','deterministic')
    result=await env['agent'].chat(env['a'],env['conv'],'转人工','deterministic')
    with env['db'].tx() as connection:
        row=connection.execute(select(tickets)).mappings().one()
        payload=connection.execute(select(traces.c.payload).where(traces.c.id==result['trace_id'])).scalar()
    assert '订单 O1002 物流到哪了' in row['summary'] and '转人工' in row['summary']
    assert row['reason']=='customer_handoff'
    assert '历史问题' not in payload
    assert any(e['type']=='state_restore' and e['history_messages']==2 for e in json.loads(payload)['events'])

async def test_bound_product_pronoun_rewrites_only_retrieval_query(env,monkeypatch):
    env['rag'].build('knowledge')
    await env['agent'].chat(env['a'],env['conv'],'耳机库存','deterministic')
    original=env['rag'].search; queries=[]
    def search(query,*args,**kwargs): queries.append(query); return original(query,*args,**kwargs)
    monkeypatch.setattr(env['rag'],'search',search)
    result=await env['agent'].chat(env['a'],env['conv'],'它保修多久','deterministic')
    assert queries==['云听耳机：它保修多久']
    assert result['citations'] and not result['pending_confirmation']
