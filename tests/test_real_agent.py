import pytest
from sqlalchemy import select,func
from app.config import settings
from app.agent import Agent
from app.db import tickets

@pytest.mark.real
async def test_real_llm_business_e2e(env):
    if not all((settings.llm_base_url,settings.llm_api_key,settings.llm_model)):
        pytest.skip('Explicit real LLM configuration required; no real-agent result')
    cfg=env['cfg'].model_copy(update={k:getattr(settings,k) for k in ('llm_base_url','llm_api_key','llm_model')})
    agent=Agent(env['b'],env['gateway'],cfg,env['rag'])
    try:
        r=await agent.chat(env['a'],env['conv'],'帮我退订单 O1001','real')
        assert r['intent']=='refund' and r['pending_confirmation']
        p=r['pending_confirmation']
        assert env['b'].workflow(env['a'],p['workflow_id'])['status']=='WAITING_CONFIRMATION'
        result=await agent.confirm(env['a'],p['workflow_id'],p['confirm_nonce'])
        assert result['status']=='ACCEPTED'
    finally: await agent.close()

@pytest.mark.real
async def test_real_llm_explicit_handoff_creates_ticket(env):
    if not all((settings.llm_base_url,settings.llm_api_key,settings.llm_model)):
        pytest.skip('Explicit real LLM configuration required; no real-agent result')
    cfg=env['cfg'].model_copy(update={k:getattr(settings,k) for k in ('llm_base_url','llm_api_key','llm_model')})
    agent=Agent(env['b'],env['gateway'],cfg,env['rag'])
    try:
        response=await agent.chat(env['a'],env['conv'],'转人工','real')
        assert response['intent']=='complaint'
        assert any(tool['name']=='create_ticket' and tool['result']['ok'] for tool in response['tool_actions'])
        with env['db'].tx() as connection: assert connection.scalar(select(func.count()).select_from(tickets))==1
    finally: await agent.close()

@pytest.mark.real
async def test_real_llm_retrieves_knowledge_before_grounded_answer(env):
    if not all((settings.llm_base_url,settings.llm_api_key,settings.llm_model)):
        pytest.skip('Explicit real LLM configuration required; no real-agent result')
    env['rag'].build('knowledge')
    cfg=env['cfg'].model_copy(update={k:getattr(settings,k) for k in ('llm_base_url','llm_api_key','llm_model')})
    agent=Agent(env['b'],env['gateway'],cfg,env['rag'])
    try:
        response=await agent.chat(env['a'],env['conv'],'耳机支持七天无理由退款吗','real')
        assert response['citations'] and not response['error_code']
        assert '生成回答缺少有效引用' not in response['answer']
        assert any('['+chunk['chunk_id']+']' in response['answer'] for chunk in response['citations'])
    finally: await agent.close()
