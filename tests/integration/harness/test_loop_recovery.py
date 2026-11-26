import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import pytest
from sqlalchemy import func, select

from app.agent import Agent
from app.business import Actor, Business, DomainError
from app.config import settings
from app.db import DB, agent_invocations, refund_requests, tickets, workflow_instances
from app.fakes import FakeModels
from app.mcp import Gateway, create_mcp
from app.rag import RAG
from app.retrieval_mcp import RetrievalService, create_retrieval_mcp


async def crash_chat(scenario, conversation_id, text):
    business = Business(DB(settings.sqlite_url))
    rag = RAG(business.db, FakeModels(), settings)
    retrieval = create_retrieval_mcp(RetrievalService(business, settings, rag=rag), settings)
    gateway = Gateway(business, settings, create_mcp(business, settings.business_secret, testing=True),
                      retrieval_server=retrieval)
    agent = Agent(business, gateway, settings, rag)
    dispatch = gateway._call
    if scenario == 'batch':
        def call(name, arguments):
            return SimpleNamespace(type='function', id='call-'+name,
                                   function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))

        async def decide(**kwargs):
            kwargs['fake_message'] = lambda: SimpleNamespace(content='', tool_calls=[
                call('load_skill', {'name':'refund-processing'}),
                call('search_knowledge', {'query':'耳机退款政策'}),
                call('get_orders', {}),
                call('create_ticket', {}),
            ])
            return await complete(**kwargs)

        complete = agent.harness.fake_model.complete_with_tools
        agent.harness.fake_model.complete_with_tools = decide

    async def dispatch_then_crash(actor, name, *args, **kwargs):
        if name == 'create_ticket' and scenario == 'batch':
            os._exit(27)
        result = await dispatch(actor, name, *args, **kwargs)
        if name == 'create_ticket' and scenario == 'ticket' and result['ok']:
            os._exit(27)
        return result

    async def crash_before_workflow(*args):
        os._exit(27)

    gateway._call = dispatch_then_crash
    if scenario == 'preview':
        agent.refunds.start = crash_before_workflow
    if scenario == 'interrupt':
        import app.refund_workflow as workflow_module

        def crash_before_interrupt(*args):
            os._exit(27)

        workflow_module.interrupt = crash_before_interrupt
    await agent.chat(Actor('user_a', 'a'), conversation_id, text, 'deterministic', request_id='crash-key')


async def crash_process(env, scenario, text):
    process_env = os.environ.copy()
    for field in ('sqlite_url','checkpoint_db_path','chroma_path','bm25_index_path'):
        process_env[field.upper()] = getattr(env['cfg'],field)
    code = ('import asyncio,runpy,sys; '
            'asyncio.run(runpy.run_path(sys.argv[1])["crash_chat"](*sys.argv[2:]))')
    process = await asyncio.to_thread(
        subprocess.run, [sys.executable, '-c', code, str(Path(__file__).resolve()), scenario, env['conv'], text],
        env=process_env, capture_output=True, text=True, timeout=45,
    )
    assert process.returncode == 27, (process.stdout, process.stderr)
    with env['db'].tx() as connection:
        return connection.execute(select(agent_invocations)).mappings().one()


async def checkpoint(env, thread_id=None):
    async with AsyncSqliteSaver.from_conn_string(env['cfg'].checkpoint_db_path) as saver:
        checkpoint = await saver.aget({'configurable':{'thread_id':thread_id or 'conv:'+env['conv']}})
    return checkpoint['channel_values']


async def test_loop_saved_observations_before_dispatching_next_tool(env):
    env['rag'].build('knowledge')
    env['rag'].close()
    text = '帮我查售后政策和订单，再转人工'
    invocation = await crash_process(env, 'batch', text)
    state = await checkpoint(env)
    assert state['loop_invocation_id'] == invocation['invocation_id']
    assert (state['iteration_count'], state['tool_call_count'], state['batch_cursor']) == (1, 3, 3)
    assert [item['name'] for item in state['loop_observations']] == ['load_skill','search_knowledge','get_orders']
    assert state['loop_observations'][0]['result']['data']['hash']
    assert 'guidance' not in state['loop_observations'][0]['result']['data']
    assert {entry.split(':')[0] for entry in state['loop_seen']} == {'load_skill','search_knowledge','get_orders'}
    assert state['retrieved_chunks'][0]['chunk_id']
    assert [item.tool_call_id for item in state['loop_messages'][1:]] == [
        'call-load_skill','call-search_knowledge','call-get_orders',
    ]
    agent = env['agent']
    replay = await agent.chat(env['a'], env['conv'], text, 'deterministic', request_id='crash-key')
    assert replay['invocation_status'] == 'UNKNOWN_OUTCOME' and replay['invocation_id'] == invocation['invocation_id']
    fresh = await agent.chat(env['a'], env['conv'], '你好', 'deterministic')
    state = await checkpoint(env)
    assert state['loop_invocation_id'] == fresh['invocation_id'] != invocation['invocation_id']
    assert not state['loop_seen'] and not state['loop_observations'] and not state['retrieved_chunks']
    with env['db'].tx() as connection:
        assert connection.scalar(select(func.count()).select_from(tickets)) == 0
    # The original downstream operation may commit after the first lookup.
    receipt = await env['gateway'].call(env['a'], 'create_ticket',
        {'category':'handoff','summary':'late receipt','conversation_id':env['conv']}, 'complaint-handling',
        single_attempt=True, ticket_operation=invocation['invocation_id'])
    settled = await agent.chat(env['a'], env['conv'], text, 'deterministic', request_id='crash-key')
    assert receipt['ok'] and settled['ticket_id'] == receipt['data']['ticket_id']
    assert settled['invocation_id'] == invocation['invocation_id'] and settled['invocation_status'] == 'SUCCEEDED'
    assert (await checkpoint(env))['loop_invocation_id'] == fresh['invocation_id']


@pytest.mark.parametrize(('scenario','text'), [('ticket','请转人工'),('preview','退订单 O1001')])
async def test_loop_same_key_recovers_committed_operation_without_dispatch(env, monkeypatch, scenario, text):
    invocation = await crash_process(env, scenario, text)
    assert invocation['status'] == 'RUNNING' and invocation['write_started']
    agent = env['agent']

    async def forbidden(*args, **kwargs):
        pytest.fail('same-key recovery must not invoke the graph or another tool')

    monkeypatch.setattr(agent, '_chat', forbidden)
    replay = await agent.chat(env['a'], env['conv'], text, 'deterministic', request_id='crash-key')
    assert replay['invocation_id'] == invocation['invocation_id'] and replay['recovered']
    assert replay == await agent.chat(env['a'], env['conv'], text, 'deterministic', request_id='crash-key')
    stored = agent.harness.controller.read(env['a'], invocation['invocation_id'])
    assert stored['response'] == replay
    with pytest.raises(DomainError, match='NOT_FOUND'):
        agent.harness.controller.read(env['other'], invocation['invocation_id'])
    with pytest.raises(DomainError, match='同一 request_id'):
        await agent.chat(env['a'], env['conv'], '新的内容', 'deterministic', request_id='crash-key')
    with env['db'].tx() as connection:
        assert connection.scalar(select(func.count()).select_from(refund_requests)) == 0
        if scenario == 'ticket':
            assert replay['invocation_status'] == 'SUCCEEDED' and replay['terminal_reason'] == 'HANDOFF'
            assert connection.scalars(select(tickets.c.id)).all() == [replay['ticket_id']]
        else:
            assert replay['invocation_status'] == 'INTERRUPTED'
            pending = replay['pending_confirmation']
            assert not pending['confirmation_credential_available'] and 'confirm_nonce' not in pending
            assert connection.scalars(select(workflow_instances.c.id)).all() == [pending['workflow_id']]
            assert pending['workflow_id'] == stored['workflow_id']


async def test_explicit_confirmation_recovers_workflow_killed_before_interrupt(env):
    customer, conversation = env['a'], env['conv']
    original = env['b'].preview(customer, conversation, 'O1001')
    wid, nonce = original['workflow_id'], original['confirm_nonce']
    text = '退订单 O1001'
    invocation = await crash_process(env, 'interrupt', text)
    assert (await checkpoint(env, 'wf:'+wid))['workflow_id'] == wid
    agent = env['agent']
    replay = await agent.chat(customer, conversation, text, 'deterministic', request_id='crash-key')
    assert replay['invocation_id'] == invocation['invocation_id']
    assert replay['pending_confirmation']['workflow_id'] == wid
    assert 'confirm_nonce' not in replay['pending_confirmation']
    with pytest.raises(DomainError, match='FORBIDDEN'):
        await agent.confirm(customer, wid, 'invalid')
    first = await agent.confirm(customer, wid, nonce)
    repeated = await agent.confirm(customer, wid, nonce)
    assert first['refund_request_id'] == repeated['refund_request_id']
    with env['db'].tx() as connection:
        assert connection.scalar(select(func.count()).select_from(refund_requests)) == 1
