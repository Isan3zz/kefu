import asyncio
from datetime import datetime, timedelta, timezone
import json

import pytest
from sqlalchemy import select

from app.business import Actor
from app.db import memories, memory_jobs, memory_operations, memory_pending, memory_revisions, messages, traces, uid
from app.fakes import FixtureMemoryGateway
from app.memory.contracts import TriggerDecision
from app.memory.service import MemoryService


@pytest.fixture
def batch(env):
    current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def advance(seconds):
        nonlocal current
        current += timedelta(seconds=seconds)

    env['b'].clock = lambda: current
    cfg = env['cfg'].model_copy(update={'memory_min_similarity': .15})
    gateway = FixtureMemoryGateway(cfg, {})
    service = MemoryService(env['b'], env['rag'], cfg, lambda _: gateway)

    async def turn(text, *, principal=env['a'], conv=env['conv'], role='user', subject=None, excerpt=None,
                   memory_type='preference', expected_generation=None, **override):
        message_id = uid('msg-')
        advance(.001)
        with env['db'].tx() as connection:
            connection.execute(messages.insert().values(id=message_id, conversation_id=conv, content=text,
                               role=role, created_at=current.isoformat()))
        if subject:
            gateway.fixtures[text] = [dict(type=memory_type, subject_key=subject, content=excerpt or text,
                                           evidence_excerpt=excerpt or text, **override)]
        return await service.consider_extraction(principal=principal, conversation_id=conv, last_user_message_id=message_id,
                                                 mode='deterministic', expected_generation=expected_generation)

    def table_rows(table):
        with env['db'].tx() as connection:
            return list(connection.execute(select(table)).mappings())

    return service, gateway, turn, advance, table_rows


@pytest.mark.asyncio
async def test_M01_M02_M03_M25_default_consent_and_urgent_evidence(env, batch):
    service, gateway, turn, _, rows = batch
    principal = env['a']
    text = '请记住我喜欢头戴式耳机'
    assert (await turn(text, subject='headphones.form_factor')).reason_code == 'OPT_IN_REQUIRED'
    assert rows(memory_pending) == rows(memory_jobs) == rows(memories) == []
    assert service.update_settings(principal, auto_write_enabled=True)['auto_write_enabled']
    job = await turn(text, subject='headphones.form_factor')
    assert job.decision == 'URGENT' and job.batch_job_id
    assert await service.worker.process(job.batch_job_id)
    memory, = rows(memories)
    source_id, = json.loads(rows(memory_jobs)[0]['source_message_ids_json'])
    assert memory['source_message_id'] == source_id and memory['evidence_excerpt'] == text
    assert memory['status'] == 'active' and memory['source_conversation_id'] == env['conv']
    assert len(gateway.calls) == 1


@pytest.mark.parametrize('role', ['assistant', 'tool', 'retrieval'], ids=['M04', 'M05_tool', 'M05_rag'])
@pytest.mark.asyncio
async def test_only_user_sources_are_eligible(env, batch, role):
    service, gateway, turn, _, rows = batch
    service.update_settings(env['a'], auto_write_enabled=True)
    result = await turn('请记住我喜欢头戴式耳机', role=role, subject='headphones.form_factor')
    assert result.reason_code == 'SOURCE_NOT_OWNED_USER'
    assert rows(memories) == rows(memory_pending) == rows(memory_jobs) == [] and len(gateway.calls) == 0


@pytest.mark.asyncio
async def test_M06_fabricated_source_excerpt_and_paraphrase_rejected(env, batch):
    service, gateway, turn, _, rows = batch
    service.update_settings(env['a'], auto_write_enabled=True)
    bad_id = await turn('请记住我喜欢头戴式耳机', subject='headphones.form_factor', source_message_id='invented')
    await service.worker.process(bad_id.batch_job_id)
    bad_excerpt = await turn('请记住我不喜欢头戴式耳机', subject='headphones.form_factor', excerpt='喜欢头戴式耳机')
    await service.worker.process(bad_excerpt.batch_job_id)
    unsupported = await turn('请记住我倾向头戴式耳机', subject='headphones.form_factor')
    gateway.fixtures['请记住我倾向头戴式耳机'][0]['content'] = '我一直不买耳机'
    await service.worker.process(unsupported.batch_job_id)
    assert rows(memories) == []
    assert sum('MEMORY_EVIDENCE_UNSUPPORTED' in trace['payload'] for trace in rows(traces)) == 3


@pytest.mark.asyncio
async def test_M07_M08_dedupe_and_explicit_conflict_with_revision(env, batch):
    service, _, turn, _, rows = batch
    service.update_settings(env['a'], auto_write_enabled=True)
    for text in ('请记住我喜欢头戴式耳机', '请记住我更喜欢头戴式耳机', '我现在更喜欢入耳式耳机'):
        dispatch = await turn(text, subject='headphones.form_factor')
        assert await service.worker.process(dispatch.batch_job_id)
    item, = rows(memories)
    assert item['version'] == 2 and '入耳式' in item['content']
    assert [row['operation'] for row in rows(memory_operations)] == ['CREATE', 'REINFORCE', 'UPDATE']
    assert {row['new_version'] for row in rows(memory_revisions)} == {1, 2}
    hits = await service.search(principal=env['a'], query='推荐耳机')
    assert len(hits) == 1 and '头戴式' not in hits[0].content


@pytest.mark.asyncio
async def test_M16_M23_secrets_and_small_talk_make_no_jobs(env, batch):
    service, gateway, turn, _, rows = batch
    service.update_settings(env['a'], auto_write_enabled=True)
    for text in ('请记住我的验证码是 946731', '记住 API Key: sk-fictional-1234', '记住收货地址虚构路 2 号'):
        assert (await turn(text)).reason_code == 'SENSITIVE_CONTENT'
    for text in ('你好', '谢谢', '看一下订单 O1001', '物流到哪了'):
        assert (await turn(text)).decision == TriggerDecision.NOOP
    assert rows(memory_pending) == rows(memory_jobs) == rows(memories) == [] and len(gateway.calls) == 0


@pytest.mark.asyncio
async def test_M26_three_messages_one_batch_three_sources(env, batch):
    service, gateway, turn, _, rows = batch
    service.update_settings(env['a'], auto_write_enabled=True)
    fixtures = [('我通常选择头戴式耳机', 'form_factor', 'preference'),
                ('我买耳机的预算一般不超过一千', 'budget', 'stable_constraint'),
                ('我平时在办公室用耳机开会', 'use_case', 'stable_fact')]
    for position, (text, key, kind) in enumerate(fixtures):
        dispatch = await turn(text, subject='headphones.' + key, memory_type=kind)
        assert dispatch.decision == 'CANDIDATE' and bool(dispatch.batch_job_id) == (position == 2)
    await service.worker.process(dispatch.batch_job_id)
    assert len(gateway.calls) == 1
    job, = rows(memory_jobs)
    assert job['flush_reason'] == 'COUNT' and job['status'] == 'COMPLETED'
    assert [entry['source_message_id'] for entry in rows(memories)] == json.loads(job['source_message_ids_json'])
    assert {entry['memory_type'] for entry in rows(memories)} == {fixture[2] for fixture in fixtures}


@pytest.mark.parametrize('idle, reason', [(120, 'IDLE'), (350, 'MAX_WAIT')])
@pytest.mark.asyncio
async def test_M27_fake_clock_flush_thresholds(env, batch, idle, reason):
    from dataclasses import replace
    service, _, turn, advance, rows = batch
    service.scheduler.policy = replace(service.policy, idle_seconds=idle)
    service.update_settings(env['a'], auto_write_enabled=True)
    await turn('我平时在电脑开会时用耳机')
    delay = min(idle, service.policy.max_wait_seconds)
    advance(delay - 1)
    assert service.scheduler.schedule_due() == []
    advance(1)
    job_id, = service.scheduler.schedule_due()
    assert rows(memory_jobs)[0]['flush_reason'] == reason
    assert await service.worker.process(job_id)


@pytest.mark.parametrize('operation', ['M15_delete_all', 'M28_opt_out', 'M31_forget'])
@pytest.mark.asyncio
async def test_cancelled_in_flight_batches_never_write_after_reopening(env, batch, monkeypatch, operation):
    service, _, turn, _, rows = batch
    principal = env['a']
    service.update_settings(principal, auto_write_enabled=True)
    dispatch = await turn('请记住我优先选择降噪耳机', subject='headphones.noise')
    started, proceed = asyncio.Event(), asyncio.Event()
    original = service.worker.extractor.extract

    async def blocked(*args):
        started.set()
        await proceed.wait()
        return await original(*args)

    monkeypatch.setattr(service.worker.extractor, 'extract', blocked)
    task = asyncio.create_task(service.worker.process(dispatch.batch_job_id))
    await asyncio.wait_for(started.wait(), 5)
    pending = await turn('我平时在办公室用耳机')
    assert pending.pending_message_id and not pending.batch_job_id
    if operation.startswith('M15'):
        await service.delete_all(principal=principal)
    elif operation.startswith('M28'):
        service.update_settings(principal, auto_write_enabled=False)
    else:
        assert (await turn('忘记我的耳机偏好')).reason_code == 'MANAGEMENT_REQUIRED'
    assert service.update_settings(principal, auto_write_enabled=True)['consent_generation'] == 1
    proceed.set()
    await task
    assert rows(memories) == []
    assert {row['status'] for row in rows(memory_pending)} == {'CANCELLED'}
    assert rows(memory_jobs)[0]['status'] == 'CANCELLED'


@pytest.mark.asyncio
async def test_M18_M29_restart_lease_recovery_and_idempotent_processing(env, batch):
    service, gateway, turn, advance, rows = batch
    service.update_settings(env['a'], auto_write_enabled=True)
    first = await turn('请记住我平时用头戴式耳机', subject='headphones.style')
    assert service.worker.claim(first.batch_job_id)['attempts'] == 1
    assert service.worker.claim(first.batch_job_id) is None
    advance(service.config.agent_timeout_seconds + 15)
    restored = MemoryService(env['b'], env['rag'], service.config, lambda _: gateway)
    assert await restored.worker.process(first.batch_job_id)
    second = await turn('我现在更喜欢入耳式耳机', subject='headphones.style')
    assert await restored.worker.process(second.batch_job_id)
    assert not await restored.worker.process(second.batch_job_id)
    assert len(rows(memories)) == 1 and len(rows(memory_revisions)) == 2 and len(rows(memory_operations)) == 2
    assert max(row['attempts'] for row in rows(memory_jobs)) == 2
    assert restored.rebuild_index() == 1


@pytest.mark.asyncio
async def test_M30_scopes_never_cross_sessions_users_or_tenants(env, batch):
    service, _, turn, _, rows = batch
    actor = env['a']
    scopes = [(actor, env['conv']), (actor, env['b'].conversation(actor)['id'])]
    scopes += [(other, env['b'].conversation(other)['id']) for other in (env['other'], Actor('admin', 'a'))]
    for principal, conversation in scopes:
        service.update_settings(principal, auto_write_enabled=True)
        result = await turn('我日常使用耳机开会', principal=principal, conv=conversation)
        assert result.pending_message_id and not result.batch_job_id
    assert rows(memory_jobs) == []
    for principal, conversation in scopes:
        await service.flush_pending(principal=principal, conversation_id=conversation, reason='CONVERSATION_ENDED')
    assert len(rows(memory_jobs)) == len(scopes)
    assert all(len(json.loads(row['source_message_ids_json'])) == 1 for row in rows(memory_jobs))
