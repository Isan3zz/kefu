import asyncio
import json
import time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import pytest
from sqlalchemy import select,func
from app.business import DomainError,Actor
from app.db import *
from app.agent import Agent
from app.mcp import create_mcp,sign_actor,verify_actor
from app.rag import RAG,LocalModels,require_model
from app.skills import SkillManager
from app.evaluation import assertions,run,compare,judge

def count(e):
    with e['db'].tx() as c: return c.scalar(select(func.count()).select_from(refund_requests))
def preview(e): return e['b'].preview(e['a'],e['conv'],'O1001')

async def test_T01_tenant_order(env):
    r=await env['gateway'].call(env['a'],'get_order',{'order_id':'O2001'},'order-tracking')
    assert not r['ok'] and r['error']['code']=='NOT_FOUND' and r['data'] is None

async def test_T02_unconfirmed(env):
    p=preview(env)
    with pytest.raises(DomainError,match='尚未确认'): env['b'].submit(env['a'],p['workflow_id'])
    assert env['b'].workflow(env['a'],p['workflow_id'])['status']=='WAITING_CONFIRMATION' and count(env)==0

async def test_T03_confirm(env):
    r=await env['agent'].chat(env['a'],env['conv'],'退订单 O1001','deterministic'); p=r['pending_confirmation']
    result=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert result['status']=='ACCEPTED' and count(env)==1
    assert env['b'].workflow(env['a'],p['workflow_id'])['status']=='REQUEST_ACCEPTED'

async def test_T04_duplicate_and_concurrent(env):
    p=preview(env); env['b'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    results=await asyncio.gather(*[asyncio.to_thread(env['b'].submit,env['a'],p['workflow_id']) for _ in range(5)])
    assert len({r['refund_request_id'] for r in results})==1 and count(env)==1

async def test_T05_changed_order(env):
    p=preview(env)
    with env['db'].tx() as c: c.execute(orders.update().where(orders.c.id=='O1001').values(status='CANCELLED',version=2))
    with pytest.raises(DomainError): env['b'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert count(env)==0

async def test_T06_timeout_committed(env,monkeypatch):
    p=preview(env)
    original=env['b'].submit
    writes=0

    def lose_result(actor,wid):
        nonlocal writes
        original(actor,wid)
        writes+=1
        raise TimeoutError('submission result lost after commit')

    monkeypatch.setattr(env['b'],'submit',lose_result)
    first=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    replay=await env['agent'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert first['refund_workflow_result']['status']=='VERIFIED_SUCCESS'
    assert first['refund_request_id']==replay['refund_request_id']
    assert writes==1 and count(env)==1

async def test_T07_unrelated_question(env):
    r=await env['agent'].chat(env['a'],env['conv'],'退订单 O1001','deterministic'); p=r['pending_confirmation']
    await env['agent'].chat(env['a'],env['conv'],'你们有蓝色耳机吗','deterministic')
    assert env['b'].workflow(env['a'],p['workflow_id'])['status']=='WAITING_CONFIRMATION' and count(env)==0

async def test_T08_persistent_restart(env):
    r=await env['agent'].chat(env['a'],env['conv'],'退订单 O1001','deterministic'); p=r['pending_confirmation']
    # A fresh agent opens the same on-disk saver. A separate interpreter is covered in integration.
    fresh=Agent(env['b'],env['gateway'],env['cfg'],env['rag'])
    r=await fresh.confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    assert r['status']=='ACCEPTED' and count(env)==1
    # Neither DB nor checkpoint persists the plaintext confirmation credential.
    assert p['confirm_nonce'].encode() not in Path(env['cfg'].checkpoint_db_path).read_bytes()

def test_T09_stale_knowledge(env):
    rag=env['rag']; rag.build('knowledge')
    with env['db'].tx() as c: c.execute(documents.update().values(status='withdrawn'))
    assert rag.search('退款','a')==[]

def test_T10_missing_models_offline(env,monkeypatch):
    monkeypatch.setenv('HF_HUB_OFFLINE','1'); monkeypatch.setenv('TRANSFORMERS_OFFLINE','1')
    with pytest.raises(FileNotFoundError,match='显式运行'): LocalModels(env['cfg'].model_copy(update={'embedding_model_path':'missing-model'}))

def test_T11_chroma_local_embeddings(env,monkeypatch):
    rag=env['rag']; rag.build('knowledge')
    import chromadb.api.models.Collection
    from chromadb.api.models.Collection import Collection
    original=Collection.query; observed=[]
    def query(self,*args,**kw):
        observed.append(kw); return original(self,*args,**kw)
    monkeypatch.setattr(Collection,'query',query)
    assert rag.search('耳机颜色','a',mode='dense')
    assert len(observed[0]['query_embeddings'][0])==1024 and 'query_texts' not in observed[0]

def test_T12_hybrid(env):
    env['rag'].build('knowledge'); rows=env['rag'].search('耳机颜色','a')
    assert rows and len({r['chunk_id'] for r in rows})==len(rows)
    assert all(r['rrf_score']>0 and r['rerank_score'] is not None for r in rows)
    assert rows[0]['rerank_score']>=rows[-1]['rerank_score']

def test_T13_skill_lazy_hash():
    sm=SkillManager(); assert len(sm.catalog)==4
    assert all('instructions' not in m for m in sm.catalog.values())
    s=sm.load('refund-processing'); assert s['instructions'] and len(s['hash'])==64
    with pytest.raises(ValueError): sm.load('../refund-processing')

async def test_T14_skill_not_authority(env):
    with pytest.raises(DomainError): await env['gateway'].call(env['a'],'submit_refund_request',{'workflow_id':'bogus'},'refund-processing')
    token=sign_actor(Actor('user_a','b'),env['cfg'].mcp_service_secret)
    with pytest.raises(DomainError): verify_actor(token,env['cfg'].mcp_service_secret,env['b'])
    for bad in ('',sign_actor(env['a'],env['cfg'].mcp_service_secret,expires=time.time()-10)):
        with pytest.raises(DomainError): verify_actor(bad,env['cfg'].mcp_service_secret,env['b'])

async def test_T15_no_llm(env):
    with pytest.raises(DomainError,match='LLM_BASE_URL'): await env['agent'].chat(env['a'],env['conv'],'你好')
    assert (await env['agent'].chat(env['a'],env['conv'],'你好','deterministic'))['intent']=='other'

async def test_T16_evaluator_report(env):
    r=await run('datasets/workflow.jsonl',config=env['cfg'])
    assert r['total']==10 and r['passed']==10, [c for c in r['cases'] if not c['passed']]
    assert r['metrics']['workflow_status']['denominator']>0
    assert (Path('data/eval_runs')/r['run_id']/'report.html').is_file()

def test_T17_version_compare():
    def report(passed,category=None):
        return dict(dataset_sha256='same',cases=[dict(case_id='c',passed=passed,results=[dict(evaluator='refund',passed=passed,failure_category=category)])],metrics={},release={},latency={},failure_categories={})
    a,b=report(True),report(False,'IDEMPOTENCY_VIOLATION')
    assert compare(a,b)['gate']=='BLOCKED' and compare(a,b)['changes'][0]['change']=='pass→fail'
    assert compare(b,a)['changes'][0]['change']=='fail→pass'

def test_T18_rebuild_failure(env,monkeypatch):
    rag=env['rag']; rag.build('knowledge'); old=(rag.path/'active.json').read_bytes()
    def fail(texts): raise RuntimeError('injected embedding failure')
    monkeypatch.setattr(rag.models,'encode',fail)
    with pytest.raises(RuntimeError): rag.build('knowledge')
    assert (rag.path/'active.json').read_bytes()==old

def test_T18_manifest_failure_rolls_back_documents(env,monkeypatch,tmp_path):
    import shutil
    import app.rag as module
    rag=env['rag']; rag.build('knowledge'); old=(rag.path/'active.json').read_bytes()
    copy=tmp_path/'new-knowledge'; shutil.copytree('knowledge',copy)
    path=copy/'policies/refund_policy_v1.md'
    path.write_text(path.read_text(encoding='utf-8').replace('version: "1"','version: "2"'),encoding='utf-8')
    original=module.atomic_json
    def fail_pointer(path,value):
        if Path(path).name=='active.json': raise OSError('injected manifest write failure')
        return original(path,value)
    monkeypatch.setattr(module,'atomic_json',fail_pointer)
    with pytest.raises(OSError): rag.build(copy)
    assert (rag.path/'active.json').read_bytes()==old
    with env['db'].tx() as c:
        assert c.scalar(select(documents.c.version).where(documents.c.id=='refund_policy_v1'))=='1'
    assert rag.search('退款','a')

def test_T19_cross_conversation_trace(env):
    from app.main import create_app
    from fastapi.testclient import TestClient
    app=create_app(env['cfg'],env['db'],env['gateway'],env['rag'])
    client=TestClient(app)
    tok=env['b'].login('user_b')['token']; headers={'Authorization':'Bearer '+tok}
    assert client.get('/api/v1/conversations/'+env['conv']+'/messages',headers=headers).status_code==404
    with env['db'].tx() as c: c.execute(traces.insert().values(id='trace-a',tenant_id='a',user_id='user_a',conversation_id=env['conv'],payload='{}'))
    assert client.get('/api/v1/traces/trace-a',headers=headers).status_code==404
    assert client.get('/api/v1/conversations').status_code==401

async def test_T20_judge_unavailable(env):
    result=await judge('answer',[],env['cfg']); assert result['status']=='NOT_EVALUATED' and result['score'] is None

@pytest.mark.models
def test_real_local_models():
    from app.config import settings
    try:
        require_model(settings.embedding_model_path)
        require_model(settings.reranker_model_path)
    except FileNotFoundError as error: pytest.skip(str(error)+'; no real-model result')
    m=LocalModels(); assert m.encode(['退款']).shape==(1,1024); assert len(m.score([('退款','七天退款政策')]))==1
