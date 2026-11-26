import pytest

from app.business import Actor, DomainError
from app.db import memories, memory_revisions
from app.memory.contracts import MemoryInput
from app.memory.service import MemoryService
from app.rag import RAG


@pytest.mark.asyncio
async def test_memory_storage_owner_version_delete_and_rebuild(env, monkeypatch):
    service = MemoryService(env['b'], env['rag'], env['cfg'])
    principal = env['a']
    assert service.settings(principal)['auto_write_enabled'] is False
    stored = await service.create_memory(principal=principal, data=MemoryInput(
        memory_type='preference', subject_key='headphones.style', content='我喜欢头戴式耳机'))
    query = '头戴式耳机'
    assert [item.memory_id for item in await service.search(principal=principal, query=query)] == [stored.memory_id]
    for other in (env['other'], Actor('user_a', 'b'), Actor('admin', 'a')):
        assert await service.search(principal=other, query=query) == []
        with pytest.raises(DomainError, match='NOT_FOUND'):
            await service.delete_memory(principal=other, memory_id=stored.memory_id)
    assert service.rebuild_index(dry_run=True) == service.rebuild_index() == 1
    service.update_settings(principal, retrieval_enabled=False)
    assert await service.search(principal=principal, query=query) == []
    assert len(await service.list_memories(principal=principal)) == 1
    service.update_settings(principal, retrieval_enabled=True)
    updated = await service.update_memory(principal=principal, memory_id=stored.memory_id,
                                          content='我喜欢入耳式耳机', expected_version=1)
    assert updated.version == 2
    with pytest.raises(DomainError, match='MEMORY_VERSION_CONFLICT'):
        await service.update_memory(principal=principal, memory_id=stored.memory_id, content=query, expected_version=1)
    backup = service.index.sync

    def offline(*_):
        raise RuntimeError('test index outage')

    monkeypatch.setattr(service.index, 'sync', offline)
    await service.delete_memory(principal=principal, memory_id=stored.memory_id)
    assert service.index.collection().get(ids=[stored.memory_id])['ids'] == [stored.memory_id]
    assert await service.search(principal=principal, query=query) == []
    with service.db.tx() as connection:
        row = connection.execute(memories.select()).mappings().one()
        assert row['status'] == 'deleted' and row['content'] == '' and row['evidence_excerpt'] is None
        assert connection.execute(memory_revisions.select()).all() == []
    monkeypatch.setattr(service.index, 'sync', backup)
    await service.sync_index()
    assert service.index.collection().get(ids=[stored.memory_id])['ids'] == []


async def test_memory_projection_rehydrates_existing_rows_in_separate_store(env, tmp_path):
    principal = env['a']
    original = MemoryService(env['b'], env['rag'], env['cfg'])
    stored = await original.create_memory(principal=principal, data=MemoryInput(
        memory_type='preference', subject_key='headphones.style', content='我喜欢头戴式耳机'))
    config = env['cfg'].model_copy(update={'chroma_path': str(tmp_path/'separate-memory')})
    projection = RAG(env['db'], env['rag'].models, config)
    service = MemoryService(env['b'], projection, config)
    try:
        service.index.queue_missing_projection(service.repository)
        await service.sync_index()
        hits = await service.search(principal=principal, query='头戴式耳机')
        assert [item.memory_id for item in hits] == [stored.memory_id]
        assert not await service.search(principal=env['other'], query='头戴式耳机')
    finally:
        projection.close()
