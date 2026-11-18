from contextlib import contextmanager
import logging
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from app.db import uid
from app.harness.execution_context import Principal
from app.memory.contracts import MemoryInput
from app.observability import TraceRecorder, active_trace


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    auto_write_enabled: StrictBool | None = None
    retrieval_enabled: StrictBool | None = None


class MemoryPatch(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    content: str = Field(min_length=2, max_length=240)
    expected_version: int = Field(ge=1, strict=True)


class DeleteAll(BaseModel):
    model_config = ConfigDict(extra='forbid')
    confirm: Literal['DELETE_ALL_MEMORIES']


def memory_router(service, actor_auth, envelope):
    router = APIRouter(prefix='/api/v1/memories', tags=['长期记忆'])

    def principal(actor=Depends(actor_auth)):
        return Principal(actor.tenant_id, actor.user_id, actor.role)

    @contextmanager
    def audit(owner, conversation_id=None):
        recorder = TraceRecorder(service.repository.business, owner, conversation_id, service.config,
                                 invocation_id=uid('mem-inv-'))
        marker = active_trace.set(recorder)
        try:
            yield
        finally:
            active_trace.reset(marker)
            try:
                if recorder.payload['events']:
                    recorder.finish()
            except Exception as error:
                logging.getLogger(__name__).error('Memory management trace unavailable: %s', type(error).__name__)

    @router.get('/settings')
    def settings(owner=Depends(principal)):
        return envelope(service.settings(owner))

    @router.patch('/settings')
    def edit_settings(body: SettingsPatch, owner=Depends(principal)):
        with audit(owner):
            return envelope(service.update_settings(owner, **body.model_dump()))

    @router.get('')
    async def list_memories(owner=Depends(principal), offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
        items = await service.list_memories(principal=owner, offset=offset, limit=limit)
        return envelope([item.model_dump() for item in items])

    @router.post('', status_code=201)
    async def create_memory(body: MemoryInput, owner=Depends(principal)):
        with audit(owner):
            item = await service.create_memory(principal=owner, data=body)
        return envelope(item.model_dump())

    @router.patch('/{memory_id}')
    async def edit_memory(memory_id: str, body: MemoryPatch, owner=Depends(principal)):
        with audit(owner):
            item = await service.update_memory(principal=owner, memory_id=memory_id, **body.model_dump())
        return envelope(item.model_dump())

    @router.get('/{memory_id}/revisions')
    def revisions(memory_id: str, owner=Depends(principal), offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100)):
        return envelope(service.revisions(owner, memory_id, offset, limit))

    @router.delete('/{memory_id}', status_code=204)
    async def delete_memory(memory_id: str, owner=Depends(principal)):
        with audit(owner):
            await service.delete_memory(principal=owner, memory_id=memory_id)
        return Response(status_code=204)

    @router.delete('', status_code=204)
    async def delete_all(body: DeleteAll, owner=Depends(principal)):
        with audit(owner):
            await service.delete_all(principal=owner)
        return Response(status_code=204)

    @router.post('/conversations/{conversation_id}/finish')
    async def finish_conversation(conversation_id: str, owner=Depends(principal)):
        with audit(owner, conversation_id):
            job_id = await service.flush_pending(principal=owner, conversation_id=conversation_id,
                                                 reason='CONVERSATION_ENDED', wait=False)
        return envelope({'batch_job_id': job_id})

    return router
