import asyncio
import json

import httpx
import openai
import pytest
from sqlalchemy import select

from app.db import traces
from app.memory.contracts import MemoryInput
from app.memory.service import MemoryService


@pytest.mark.parametrize('failure', ['429', '503', 'timeout'])
async def test_transient_model_failure_recovers_once(env, monkeypatch, failure):
    attempts = []
    original = openai.AsyncOpenAI

    async def provider(request):
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            if failure == 'timeout':
                await asyncio.sleep(1)
            else:
                return httpx.Response(int(failure), json={'error': {'message': 'test transient'}})
        return httpx.Response(200, json={'id': 'recovery', 'choices': [{'index': 0, 'message': {
            'role': 'assistant', 'content': '您好。'}, 'finish_reason': 'stop'}]})

    monkeypatch.setattr(openai, 'AsyncOpenAI', lambda **kwargs: original(**kwargs,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider))))
    config = env['cfg'].model_copy(update={'llm_api_key': 'fake', 'llm_base_url': 'http://mock/v1',
        'llm_model': 'model-fault-test', 'llm_timeout_seconds': .3, 'model_max_retries': 1})
    from app.agent import Agent
    agent = Agent(env['b'], env['gateway'], config, env['rag'])
    try:
        response = await agent.chat(env['a'], env['conv'], '你好', 'real')
        assert response['answer'] == '您好。' and len(attempts) == 2 and not response['error_code']
    finally:
        await agent.close()


async def test_missing_rag_index_degrades_while_orders_work(env):
    response = await env['agent'].chat(env['a'], env['conv'], '耳机保修多久', 'deterministic')
    assert not response['citations']
    assert '知识服务暂不可用' in response['answer'] and response['error_code'] == 'KNOWLEDGE_NOT_READY'
    with env['db'].tx() as connection:
        payload = json.loads(connection.scalar(select(traces.c.payload).where(traces.c.id==response['trace_id'])))
    assert any(event['type']=='tool' and event.get('name')=='search_knowledge' and not event['ok'] for event in payload['events'])
    order = await env['agent'].chat(env['a'], env['conv'], '查订单 O1001', 'deterministic')
    assert any(tool['name'] == 'get_order' and tool['result']['ok'] for tool in order['tool_actions'])


async def test_stale_active_memory_projection_is_ignored(env, monkeypatch):
    service = MemoryService(env['b'], env['rag'], env['cfg'])
    actor = env['a']
    item = await service.create_memory(principal=actor, data=MemoryInput(memory_type='preference',
        subject_key='headphones.style', content='我喜欢头戴式耳机'))
    assert await service.search(principal=actor, query='头戴式耳机')

    def unavailable(*args, **kwargs):
        raise RuntimeError('test index write outage')

    with monkeypatch.context() as fault:
        fault.setattr(service.index, 'sync', unavailable)
        updated = await service.update_memory(principal=actor, memory_id=item.memory_id,
            content='我现在更喜欢入耳式耳机', expected_version=1)
        assert updated.version == 2
        assert service.index.collection().get(ids=[item.memory_id])['metadatas'][0]['version'] == 1
        assert not await service.search(principal=actor, query='头戴式耳机')
    await service.sync_index()
    assert service.index.collection().get(ids=[item.memory_id])['metadatas'][0]['version'] == 2
