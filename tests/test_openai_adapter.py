"""Mock HTTP OpenAI-compatible protocol; no live LLM or billable request."""
import json

import httpx
import pytest
from openai import AsyncOpenAI
from sqlalchemy import func, select

from app.db import products, refund_requests, traces


@pytest.fixture
async def mock_chat(env, monkeypatch):
    async def chat(text, respond, skill_mode='selected'):
        requests = []

        async def provider(request):
            body = json.loads(request.content)
            requests.append(body)
            content = respond(body)
            return httpx.Response(200, json={
                'id': 'mock-completion', 'model': 'mock-model',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
            })

        transport = httpx.MockTransport(provider)
        async with AsyncOpenAI(api_key='offline-test-key', base_url='http://mock.local/v1',
                               http_client=httpx.AsyncClient(transport=transport)) as client:
            monkeypatch.setattr(env['agent'], 'llm', lambda: client)
            result = await env['agent'].chat(env['a'], env['conv'], text, 'real', skill_mode)
        with env['db'].tx() as connection:
            payload = connection.scalar(select(traces.c.payload).where(traces.c.id == result['trace_id']))
        return result, requests, json.loads(payload)

    return chat


async def test_one_decision_for_refund_preview_and_no_submission(env, mock_chat):
    result, requests, record = await mock_chat('退订单 O1001', lambda _: '{"action":"check_refund"}')
    assert len(requests) == 1
    assert result['intent'] == 'refund' and result['skill']['name'] == 'refund-processing'
    assert result['pending_confirmation']['status'] == 'WAITING_CONFIRMATION'
    with env['db'].tx() as connection:
        assert connection.scalar(select(func.count()).select_from(refund_requests)) == 0
    calls = [event for event in record['events'] if event['type'] == 'llm']
    assert [event['stage'] for event in calls] == ['next_action']
    assert calls[0]['usage']['prompt_tokens'] == 10
    assert 'offline-test-key' not in json.dumps(record)


@pytest.mark.parametrize('skill_mode', ['none', 'all', 'selected'])
async def test_knowledge_generation_loads_skill_only_after_decision(env, mock_chat, skill_mode):
    env['rag'].build('knowledge')

    def respond(request):
        payload = json.loads(request['messages'][1]['content'])
        if not payload['evidence']:
            return '{"action":"answer_policy"}'
        return '请查看政策 [' + payload['evidence'][0]['chunk_id'] + ']'

    result, requests, _ = await mock_chat('七天可以退吗', respond, skill_mode)
    assert len(requests) == 2 and result['intent'] == 'faq' and result['citations']
    decision, answer = [request['messages'][0]['content'] for request in requests]
    for name in env['agent'].skills.catalog:
        instructions = env['agent'].skills.load(name)['instructions']
        assert instructions not in decision
        assert (instructions in answer) == (skill_mode == 'all' or skill_mode == 'selected' and name == 'product-consultation')


@pytest.mark.parametrize('proposal', ['{"action":"submit_refund_request"}', '{"action":"check_refund","order_id":"O1001"}'])
async def test_invalid_model_action_is_repaired_at_most_once_and_never_executes(env, mock_chat, proposal):
    result, requests, record = await mock_chat('退订单 O1001', lambda _: proposal)
    assert len(requests) == 2 and 'Correct it once' in requests[1]['messages'][-1]['content']
    assert not result['pending_confirmation'] and not result['tool_actions']
    with env['db'].tx() as connection:
        assert connection.scalar(select(func.count()).select_from(refund_requests)) == 0
    assert any(event['type'] == 'agent_proposal_rejected' for event in record['events'])


async def test_shipping_resolves_order_ordinal_in_code_and_calls_provider_once(env, mock_chat):
    listing = await env['agent'].chat(env['a'], env['conv'], '查一下我的订单', 'deterministic')
    expected = listing['tool_actions'][0]['result']['data'][1]['id']
    result, requests, _ = await mock_chat('第二单到哪了', lambda _: '{"action":"track_shipment"}')
    assert len(requests) == 1 and result['intent'] == 'shipping_query'
    assert result['tool_actions'][0]['args'] == {'order_id': expected}
    result, requests, _ = await mock_chat('第99单到哪了', lambda _: '{"action":"track_shipment"}')
    assert len(requests) == 1 and not result['tool_actions'] and '订单号' in result['answer']


async def test_refund_without_order_clarifies_without_another_model_call(mock_chat):
    result, requests, _ = await mock_chat('帮我退款', lambda _: '{"action":"check_refund"}')
    assert len(requests) == 1 and result['intent'] == 'refund'
    assert not result['tool_actions'] and not result['pending_confirmation'] and '订单号' in result['answer']


async def test_recommendation_without_memory_uses_only_the_action_decision(mock_chat):
    result, requests, _ = await mock_chat('推荐一款耳机，预算不超过100元', lambda _: '{"action":"search_products"}')
    assert len(requests) == 1 and result['tool_actions'][0]['name'] == 'search_products'
    assert '没有找到预算内' in result['answer']


async def test_stock_uses_live_product_tool(env):
    with env['db'].tx() as connection:
        connection.execute(products.update().where(products.c.id == 'headphones-a').values(stock=3))
    result = await env['agent'].chat(env['a'], env['conv'], '耳机库存有多少', 'deterministic')
    assert result['tool_actions'][0]['name'] == 'search_products'
    assert result['tool_actions'][0]['result']['data'][0]['stock'] == 3
