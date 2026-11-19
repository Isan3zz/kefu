import json
from dataclasses import replace
import pytest
from sqlalchemy import select,func
from app.db import messages,uid,now,context_summaries
from app.harness.context_manager import ContextManager,SECURITY_RULES
from app.harness.run_controller import RunController
from app.harness.telemetry import Telemetry
from app.harness.contracts import ContextBudget,HarnessFailure
from app.business import DomainError

async def test_H20_three_layers_preserve_facts_refs_and_original_history(env):
    p=env['b'].preview(env['a'],env['conv'],'O1001')
    ctx,_=RunController(env['b'],env['cfg']).begin(env['a'],env['conv'],'conv:'+env['conv'],{})
    history=[{'id':'history-'+str(i),'role':'user','content':'过往产品咨询；'+('重复历史内容'*200)} for i in range(30)]
    with env['db'].tx() as c:
        for m in history: c.execute(messages.insert().values(**m,conversation_id=env['conv'],created_at=now()))
    evidence=[{'chunk_id':'chunk-ref','version':'v1','document_title':'政策','content':'申请需确认。'}]
    tools=[{'name':'get_order','result':{'ok':True,'request_id':'req1','data':{'id':'O1001','status':'DELIVERED','verbose':'X'*20000,'phone':'13912345678'}}}]*4
    cfg=env['cfg'].model_copy(update={'context_recent_message_limit':1})
    manager=ContextManager(env['b'],cfg)
    output=await manager.build_context(ctx=ctx,task_state={'active_workflow_id':p['workflow_id'],'workflow_state':'REQUEST_ACCEPTED','message':'查一下进度'},recent_messages=history,selected_skill={'name':'refund-processing','instructions':'先验证本人订单并等待明确确认','version':'1'},evidence=evidence,tool_results=tools,budget=ContextBudget(model_context_window=16000,reserved_output_tokens=2000,safety_margin_tokens=1000,max_input_tokens=11000))
    assert output.compression_level=='L0+L1+L2' and output.summary_id
    assert output.token_estimate<output.before_tokens and output.preserved_refs==['chunk-ref@v1']
    assert SECURITY_RULES in output.messages[0]['content'] and '先验证本人订单' in output.messages[0]['content']
    payload=json.loads(output.messages[1]['content'])
    assert payload['current_task']['verified_workflow']['status']=='WAITING_CONFIRMATION'
    assert payload['history_summary']['semantic_summary_generated'] is False
    assert payload['evidence'][0]['chunk_id']=='chunk-ref'
    assert '13912345678' not in json.dumps(output.model_dump())
    with env['db'].tx() as c: assert c.scalar(select(func.count()).select_from(messages))==30
    # Exact source is reused; updated authoritative business state invalidates the summary.
    assert manager.read_summary(ctx,output.summary_id)['summary_version']=='verified-facts-v1'
    env['b'].confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
    newer=await manager.build_context(ctx=ctx,task_state={'active_workflow_id':p['workflow_id']},recent_messages=history,evidence=evidence,budget=manager.budget())
    assert newer.summary_id!=output.summary_id

async def test_H21_summary_owner_and_thread_scope(env):
    controller=RunController(env['b'],env['cfg']); ctx,_=controller.begin(env['a'],env['conv'],'conv:'+env['conv'],{})
    manager=ContextManager(env['b'],env['cfg'])
    row=manager._summary(ctx,[{'id':'m1'}],{'status':'NONE'},[],[])
    for other in (replace(ctx,principal=type(ctx.principal)(tenant_id='b',user_id='user_b')),replace(ctx,graph_thread_id='conv:another')):
        with pytest.raises(DomainError): manager.read_summary(other,row['id'])

async def test_H22_context_overflow_is_explicit_without_semantic_invention(env):
    ctx,_=RunController(env['b'],env['cfg']).begin(env['a'],env['conv'],'conv:'+env['conv'],{})
    manager=ContextManager(env['b'],env['cfg'])
    with pytest.raises(HarnessFailure) as error:
        budget=ContextBudget(model_context_window=16384,reserved_output_tokens=2048,safety_margin_tokens=1024,max_input_tokens=12000)
        await manager.build_context(ctx=ctx,task_state={'message':'必要的当前问题'*5000},recent_messages=[],budget=budget)
    assert error.value.code=='CONTEXT_BUDGET_EXCEEDED'
    with env['db'].tx() as c: assert c.scalar(select(func.count()).select_from(context_summaries))==0
