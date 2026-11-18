import asyncio
import json
from pathlib import Path
import pytest
from sqlalchemy import select
from app.db import traces
from app.business import DomainError
from app.observability import clean
from fastmcp import Client

def payloads(env):
    with env['db'].tx() as c:
        return [json.loads(row['payload']) for row in c.execute(select(traces)).mappings()]

async def test_missing_llm_config_is_audited_without_raw_input(env):
    with pytest.raises(DomainError):
        await env['agent'].chat(env['a'],env['conv'],'private input 13900000000')
    record=payloads(env)[0]
    assert record['errors'][0]['error_code']=='LLM_NOT_CONFIGURED'
    assert 'private input' not in json.dumps(record)
    assert all(key in record['events'][0] for key in ('event_id','span_id','parent_span_id','ts','duration_ms','status','error_code'))

async def test_decision_exception_and_cancellation_are_persisted(env,monkeypatch):
    import app.agent as module
    def fail(text): raise RuntimeError('private provider failure')
    monkeypatch.setattr(module,'fake_next_action',fail)
    with pytest.raises(DomainError) as error:
        await env['agent'].chat(env['a'],env['conv'],'你好','deterministic')
    assert error.value.code=='SYSTEM_ERROR'
    assert payloads(env)[-1]['errors'][0]['error_type']=='RuntimeError'
    assert 'private provider failure' not in json.dumps(payloads(env))
    started=asyncio.Event()
    async def slow(*args):
        started.set(); await asyncio.Event().wait()
    monkeypatch.setattr(env['agent'],'_chat',slow)
    task=asyncio.create_task(env['agent'].chat(env['a'],env['conv'],'你好','deterministic'))
    await started.wait(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert payloads(env)[-1]['errors'][0]['error_code']=='CANCELLED'

async def test_gateway_exception_has_actual_tool_evidence(env,monkeypatch):
    async def fail(*args,**kwargs): raise DomainError('UPSTREAM_UNAVAILABLE')
    monkeypatch.setattr(env['gateway'],'_call',fail)
    response=await env['agent'].chat(env['a'],env['conv'],'查订单 O1001','deterministic')
    assert response['error_code']=='UPSTREAM_UNAVAILABLE'
    record=payloads(env)[0]; tool=[e for e in record['events'] if e['type']=='tool'][0]
    assert tool['name']=='get_order' and tool['args']=={'order_id':'O1001'}
    assert tool['status']=='error' and tool['error_code']=='UPSTREAM_UNAVAILABLE'

async def test_invalid_confirm_does_not_fabricate_submit_call(env):
    p=env['b'].preview(env['a'],env['conv'],'O1001')
    with pytest.raises(DomainError): await env['agent'].confirm(env['a'],p['workflow_id'],'wrong nonce')
    record=payloads(env)[0]
    assert not any(e['type']=='tool' for e in record['events'])
    assert record['errors'][0]['error_code']=='FORBIDDEN'

async def test_jsonl_failure_preserves_sqlite_and_accepted_result(env,monkeypatch):
    p=env['b'].preview(env['a'],env['conv'],'O1001')
    original=Path.open
    def deny_trace_file(self,*args,**kwargs):
        if self.suffix=='.jsonl': raise PermissionError('injected file failure')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',deny_trace_file)
    response=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert response['status']=='ACCEPTED'
    record=payloads(env)[0]
    assert record['storage_errors'][0]['backend']=='jsonl'
    assert record['business_outcome']=='REQUEST_ACCEPTED'
    assert [e for e in record['events'] if e['type']=='tool'][0]['result']['data']['status']=='ACCEPTED'

def test_redaction_keeps_usage_metrics():
    value=clean({'Authorization':'Bearer raw','api_key':'secret','confirm_nonce':'nonce','summary':'private','usage':{'prompt_tokens':123,'completion_tokens':45},'estimated_tokens':20})
    assert set(value)=={'usage','estimated_tokens'}
    assert value['usage']['prompt_tokens']==123
    identifier='aa13900000000bbbbbbbbbbb'
    assert clean({'chunk_id':identifier})['chunk_id']==identifier
    assert clean({'graph_thread_id':'conv:conv-'+identifier})['graph_thread_id']=='conv:conv-'+identifier
    assert clean({'operation_id':'model-op-'+identifier})['operation_id']=='model-op-'+identifier
    assert clean({'note':'13900000000'})['note']=='[PHONE]'

async def test_unknown_ticket_write_never_silently_returns_none_or_retries(env,monkeypatch):
    from app.db import tickets
    original=Client.call_tool; calls=0
    async def after_commit(self,*args,**kwargs):
        nonlocal calls
        calls+=1
        await original(self,*args,**kwargs)
        raise TimeoutError('lost ticket response after database commit')
    monkeypatch.setattr(Client,'call_tool',after_commit)
    response=await env['agent'].chat(env['a'],env['conv'],'转人工','deterministic')
    assert response['error_code']=='UNKNOWN_OUTCOME' and calls==1
    with env['db'].tx() as c: assert len(c.execute(select(tickets)).all())==1
    assert any(e.get('error_code')=='UNKNOWN_OUTCOME' for e in payloads(env)[0]['events'])
