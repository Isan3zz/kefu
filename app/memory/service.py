import asyncio
from dataclasses import replace
import time

from sqlalchemy import select

from app.business import DomainError
from app.db import conversations, memories, memory_revisions
from app.harness.model_gateway import estimate_tokens
from app.memory.batch_scheduler import BatchExtractionScheduler
from app.memory.contracts import PIPELINE_VERSION, MemoryInput, MemoryPolicy, TriggerDecision, TriggerDispatch
from app.memory.extractor import BatchExtractor
from app.memory.jobs import MemoryJobWorker
from app.memory.repository import SqliteMemoryRepository, owner, retrieved, validate_manual
from app.memory.trigger import FORGET, MemoryTrigger
from app.memory.validator import SENSITIVE, normalized, prohibited
from app.memory.vector_index import ChromaMemoryIndex
from app.observability import active_trace, emit


class MemoryService:
    def __init__(self, business, rag, config, resolve_gateway=None):
        self.repository = SqliteMemoryRepository(business)
        self.db, self.config = business.db, config
        self.index = ChromaMemoryIndex(rag, config)
        self.policy = MemoryPolicy(
            eligible_count=config.memory_batch_max_eligible_messages, max_messages=config.memory_batch_max_messages,
            idle_seconds=config.memory_batch_idle_seconds, max_wait_seconds=config.memory_batch_max_wait_seconds,
            candidates_per_message=config.memory_extraction_max_candidates_per_message,
            candidates_per_batch=config.memory_extraction_max_candidates_per_batch,
        )
        self.trigger = MemoryTrigger()
        self.scheduler = BatchExtractionScheduler(self.repository, self.policy)
        self.worker = MemoryJobWorker(self, BatchExtractor(resolve_gateway, config)) if resolve_gateway else None

    def event(self, kind, **attributes):
        recorder = active_trace.get()
        if recorder is not None and self.config.harness_trace_enabled:
            attrs = {'embedding_version': self.index.version, **attributes}
            emit('memory.' + kind, memory_pipeline_version=PIPELINE_VERSION,
                 tenant_id=recorder.actor.tenant_id, **attrs)

    def consent_snapshot(self, principal):
        if not self.config.memory_enabled:
            return -1
        current = self.settings(principal)
        return current['consent_generation'] if current['auto_write_enabled'] else -1

    def revalidate(self, principal, items):
        if not items or not self.config.memory_enabled:
            return []
        try:
            with self.db.tx() as connection:
                if not self.repository.settings(connection, principal)['retrieval_enabled']:
                    return []
                query = self.repository.active_rows(connection, principal, [item.memory_id for item in items])
                versions = dict(connection.execute(query.with_only_columns(memories.c.id, memories.c.version)).all())
            return [item for item in items if versions.get(item.memory_id) == item.version]
        except Exception as error:
            self.event('search_failed', status='error', error_type=type(error).__name__)
            return []

    async def consider_extraction(self, *, principal, conversation_id, last_user_message_id,
                                  mode='real', expected_generation=None):
        if not self.config.memory_enabled:
            return TriggerDispatch(TriggerDecision.REJECT, 'MEMORY_DISABLED')
        with self.db.tx() as connection:
            dispatch, opted_in = self.register_turn(connection, principal, conversation_id, last_user_message_id, mode, expected_generation)
        await self.finish_turn(last_user_message_id, dispatch, opted_in)
        return dispatch

    def register_turn(self, connection, principal, conversation_id, last_user_message_id, mode, expected_generation):
        recorder = active_trace.get()
        settings = self.repository.settings(connection, principal)
        opted_in = settings['auto_write_enabled'] and expected_generation in (None, settings['consent_generation'])
        source = self.repository.sources(connection, principal, conversation_id, [last_user_message_id]).get(last_user_message_id)
        dispatch = self.trigger.check(source, opted_in=opted_in, max_bytes=self.policy.max_message_bytes)
        if source and FORGET.search(normalized(source['content'] or '')):
            self.repository.set_settings(connection, principal, {}, invalidate=True)
            dispatch = TriggerDispatch(TriggerDecision.REJECT, 'MANAGEMENT_REQUIRED')
        if dispatch.decision in (TriggerDecision.CANDIDATE, TriggerDecision.URGENT):
            pending_id, job_id = self.scheduler.enqueue(connection, principal, source, settings['consent_generation'], dispatch,
                                                     mode, recorder.payload.get('invocation_id') if recorder else None)
            dispatch = replace(dispatch, pending_message_id=pending_id, batch_job_id=job_id)
        return dispatch, opted_in

    async def finish_turn(self, last_user_message_id, dispatch, opted_in):
        self.event('trigger_checked', message_id=last_user_message_id, decision=dispatch.decision.value,
                   reason_code=dispatch.reason_code, opt_in=opted_in)
        if dispatch.reason_code == 'MANAGEMENT_REQUIRED':
            self.event('batch_cancelled', reason_code='MANAGEMENT_REQUIRED')
        if dispatch.decision == TriggerDecision.REJECT:
            self.event('trigger_rejected', message_id=last_user_message_id, reason_code=dispatch.reason_code)
        if dispatch.pending_message_id:
            self.event('pending_buffered', message_id=last_user_message_id, pending_id=dispatch.pending_message_id)
        if dispatch.batch_job_id:
            self.event('batch_created', job_id=dispatch.batch_job_id, pending_id=dispatch.pending_message_id,
                       message_id=last_user_message_id)
            await self.dispatch_jobs([dispatch.batch_job_id])

    async def dispatch_jobs(self, jobs, wait=False):
        if jobs and self.worker:
            if wait or self.config.memory_job_mode == 'sync':
                for job_id in jobs:
                    await self.worker.process(job_id)
            else:
                self.worker.wake.set()

    async def flush_pending(self, *, principal, conversation_id, reason='manual_test', wait=True):
        if not self.config.memory_enabled:
            return None
        if reason not in ('manual_test', 'CONVERSATION_ENDED'):
            raise ValueError('Unknown memory flush reason')
        with self.db.tx() as connection:
            self.repository.business.owned(connection, conversations, principal, conversation_id)
            jobs = self.scheduler.flush(connection, principal, conversation_id, reason=reason)
        for job_id in jobs:
            self.event('batch_created', job_id=job_id, flush_reason=reason)
        await self.dispatch_jobs(jobs, wait=wait)
        return jobs[0] if jobs else None

    def settings(self, principal):
        with self.db.tx() as connection:
            return self.repository.settings(connection, principal)

    def update_settings(self, principal, *, auto_write_enabled=None, retrieval_enabled=None):
        flags = {key: value for key, value in dict(auto_write_enabled=auto_write_enabled,
                                                  retrieval_enabled=retrieval_enabled).items() if value is not None}
        with self.db.tx() as connection:
            settings = self.repository.set_settings(connection, principal, flags)
        if auto_write_enabled is False:
            self.event('batch_cancelled', reason_code='OPT_OUT', consent_generation=settings['consent_generation'])
        return settings

    async def search(self, *, principal, query, limit=5, token_budget=600):
        started = time.perf_counter()
        self.event('search_started')
        try:
            result = await asyncio.to_thread(self._search, principal, query, limit, token_budget)
            self.event('search_completed', matched_count=len(result), memory_ids=[row.memory_id for row in result],
                       embedding_version=self.index.version, duration_ms=(time.perf_counter() - started) * 1000)
            return result
        except Exception as error:
            self.event('search_failed', status='error', error_type=type(error).__name__,
                       error_code='MEMORY_INDEX_SYNC_ERROR', duration_ms=(time.perf_counter() - started) * 1000)
            return []

    def _search(self, principal, query, limit, token_budget):
        limit = min(limit, self.config.memory_top_k)
        remaining = min(token_budget, self.config.memory_token_budget)
        if not self.config.memory_enabled or limit <= 0 or remaining <= 0 or not query.strip() or SENSITIVE.search(normalized(query)):
            return []
        with self.db.tx() as connection:
            if not self.repository.settings(connection, principal)['retrieval_enabled'] or not connection.execute(
                    self.repository.active_rows(connection, principal).limit(1)).first():
                return []
        hits = self.index.query(principal, query[:500], self.config.memory_candidate_k)
        with self.db.tx() as connection:
            rows = connection.execute(self.repository.active_rows(connection, principal, [hit[0] for hit in hits])).mappings()
            by_id = {row['id']: row for row in rows}
        version = self.index.version
        valid = [(by_id[ident], score) for ident, score, metadata in hits if ident in by_id
                 and metadata.get('version') == by_id[ident]['version'] and not by_id[ident]['index_pending']
                 and metadata.get('embedding_version') == by_id[ident]['embedding_version'] == version
                 and score >= self.config.memory_min_similarity and not prohibited(by_id[ident]['content'])]
        if self.config.memory_rerank_enabled and valid:
            scores = self.index.rag.model().score([(query[:500], row['content']) for row, _ in valid])
            valid = sorted(zip((row for row, _ in valid), scores, strict=True), key=lambda item: -item[1])
        with self.db.tx() as connection:
            if not self.repository.settings(connection, principal)['retrieval_enabled']:
                return []
            latest = {row['id']: row['version'] for row in connection.execute(
                self.repository.active_rows(connection, principal, [row['id'] for row, _ in valid])).mappings()}
            result = []
            for row, score in valid:
                cost = estimate_tokens([{'content': row['content']}])
                if latest.get(row['id']) == row['version'] and cost <= remaining:
                    result.append(retrieved(row, float(score)))
                    remaining -= cost
                    if len(result) == limit:
                        break
            return result

    async def list_memories(self, *, principal, include_deleted=False, offset=0, limit=50):
        with self.db.tx() as connection:
            query = self.repository.active_rows(connection, principal)
            if include_deleted:
                query = select(memories).where(*owner(memories, principal))
            return [retrieved(row) for row in connection.execute(query.order_by(memories.c.updated_at.desc(), memories.c.id)
                                                                .offset(offset).limit(limit)).mappings()]

    def revisions(self, principal, memory_id, offset, limit):
        with self.db.tx() as connection:
            self.repository.owned(connection, principal, memory_id)
            query = select(memory_revisions).where(memory_revisions.c.memory_id == memory_id)
            return [dict(row) for row in connection.execute(query.order_by(memory_revisions.c.new_version.desc())
                                                            .offset(offset).limit(limit)).mappings()]

    async def create_memory(self, *, principal, data):
        if not self.config.memory_enabled:
            raise DomainError('FORBIDDEN', 'MEMORY_DISABLED')
        validate_manual(data)
        with self.db.tx() as connection:
            self.repository.settings(connection, principal)
            if connection.execute(self.repository.active_rows(connection, principal).where(memories.c.subject_key == data.subject_key)).first():
                raise DomainError('MEMORY_SUBJECT_EXISTS')
            item = self.repository.save(connection, principal, data)
        self.event('upserted', memory_id=item.memory_id, operation='CREATE')
        await self.sync_index([item.memory_id])
        return item

    async def update_memory(self, *, principal, memory_id, content, expected_version):
        if not self.config.memory_enabled:
            raise DomainError('FORBIDDEN', 'MEMORY_DISABLED')
        with self.db.tx() as connection:
            row = self.repository.owned(connection, principal, memory_id)
            if row['version'] != expected_version:
                raise DomainError('MEMORY_VERSION_CONFLICT')
            data = MemoryInput(memory_type=row['memory_type'], subject_key=row['subject_key'], content=content)
            validate_manual(data)
            item = self.repository.save(connection, principal, data, previous=row)
        self.event('updated', memory_id=item.memory_id, operation='UPDATE', version=item.version)
        await self.sync_index([item.memory_id])
        return item

    async def delete_memory(self, *, principal, memory_id):
        await self.delete_all(principal=principal, memory_id=memory_id)

    async def delete_all(self, *, principal, memory_id=None):
        with self.db.tx() as connection:
            ids = self.repository.delete(connection, principal, memory_id)
        self.event('deleted', memory_ids=ids, deleted_count=len(ids))
        await self.sync_index(ids)

    async def sync_index(self, ids=None):
        if ids is None:
            with self.db.tx() as connection:
                query = select(memories.c.id).where(memories.c.index_pending == 1)
                if not self.config.memory_enabled:
                    query = query.where(memories.c.status == 'deleted')
                ids = list(connection.execute(query.order_by(memories.c.status != 'deleted', memories.c.updated_at).limit(25)).scalars())
        for memory_id in ids:
            try:
                await asyncio.to_thread(self.index.sync, self.repository, memory_id)
            except Exception as error:
                self.event('index_sync_failed', memory_id=memory_id, status='error', error_type=type(error).__name__,
                           error_code='MEMORY_INDEX_SYNC_ERROR')

    def rebuild_index(self, dry_run=False):
        count = self.index.rebuild(self.repository, dry_run)
        if not dry_run:
            self.event('index_rebuilt', indexed_count=count, embedding_version=self.index.version)
        return count
