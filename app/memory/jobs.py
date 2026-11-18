import asyncio
from collections import Counter
from datetime import datetime, timedelta
import json
import logging
import time

from sqlalchemy import and_, case, or_, select

from app.db import memories, memory_jobs, memory_pending, uid
from app.harness.execution_context import ExecutionContext, Principal, RunBudget
from app.harness.telemetry import Telemetry
from app.memory.consolidator import Consolidator, validated
from app.memory.contracts import PIPELINE_VERSION
from app.memory.trigger import MemoryTrigger
from app.observability import TraceRecorder, active_trace, evaluation_context


class MemoryJobWorker:
    def __init__(self, service, extractor):
        self.service, self.extractor = service, extractor
        self.repository, self.policy, self.config = service.repository, service.policy, service.config
        self.consolidator = Consolidator(self.repository)
        self.wake = asyncio.Event()
        self.stopping = False

    def authorized(self, connection, principal, job):
        settings = self.repository.settings(connection, principal)
        return settings['auto_write_enabled'] and settings['consent_generation'] == job['consent_generation']

    def claim(self, job_id=None):
        stamp = self.repository.stamp()
        lease = (datetime.fromisoformat(stamp) + timedelta(seconds=self.config.agent_timeout_seconds + 15)).isoformat()
        table = memory_jobs
        due = or_(and_(table.c.status == 'PENDING', or_(table.c.next_retry_at.is_(None), table.c.next_retry_at <= stamp)),
                  and_(table.c.status == 'PROCESSING', table.c.lease_expires_at <= stamp))
        with self.repository.db.tx() as connection:
            filters = [due]
            if job_id is not None:
                filters.append(table.c.id == job_id)
            priority = case((table.c.flush_reason == 'URGENT', 0), else_=1)
            job = connection.execute(select(table).where(*filters).order_by(priority, table.c.created_at, table.c.id).limit(1)).mappings().first()
            if not job:
                return None
            principal = Principal(job['tenant_id'], job['user_id'])
            if not self.authorized(connection, principal, job) or job['attempts'] >= self.policy.max_attempts:
                failed = job['attempts'] >= self.policy.max_attempts
                self.terminal(connection, job['id'], 'FAILED' if failed else 'CANCELLED',
                              'MEMORY_BATCH_FLUSH_TIMEOUT' if failed else 'MEMORY_BATCH_CANCELLED')
                return None
            updates = dict(status='PROCESSING', attempts=job['attempts'] + 1, lease_expires_at=lease, updated_at=stamp)
            if connection.execute(table.update().where(table.c.id == job['id'], due).values(**updates)).rowcount != 1:
                return None
            return {**job, **updates}

    def current(self, connection, job, principal):
        row = connection.execute(select(memory_jobs.c.status, memory_jobs.c.attempts, memory_jobs.c.lease_expires_at)
                                 .where(memory_jobs.c.id == job['id'])).one()
        return (row.status == 'PROCESSING' and row.attempts == job['attempts']
                and row.lease_expires_at > self.repository.stamp() and self.authorized(connection, principal, job))

    def terminal(self, connection, job_id, status, error_code=None):
        stamp = self.repository.stamp()
        connection.execute(memory_jobs.update().where(memory_jobs.c.id == job_id).values(
            status=status, updated_at=stamp, error_code=error_code, lease_expires_at=None, next_retry_at=None))
        connection.execute(memory_pending.update().where(memory_pending.c.batch_id == job_id, memory_pending.c.status == 'CLAIMED')
                           .values(status='CANCELLED' if status == 'CANCELLED' else 'DONE', updated_at=stamp))

    def eligible_sources(self, connection, principal, job, source_ids):
        sources = self.repository.sources(connection, principal, job['conversation_id'], source_ids)
        trigger = MemoryTrigger()
        return {ident: sources[ident] for ident in source_ids if ident in sources and
                trigger.check(sources[ident], opted_in=True, max_bytes=self.policy.max_message_bytes).reason_code not in (
                    'SENSITIVE_CONTENT', 'INSTRUCTION_CONTENT', 'MANAGEMENT_REQUIRED', 'SOURCE_TOO_LONG',
                    'DYNAMIC_BUSINESS_FACT', 'NOT_USER_ASSERTED')}

    async def process(self, job_id=None):
        job = self.claim(job_id)
        if job is None:
            return False
        principal = Principal(job['tenant_id'], job['user_id'])
        recorder = TraceRecorder(self.repository.business, principal, job['conversation_id'], self.config,
                                 origin_trace_id=job['origin_trace_id'], origin_invocation_id=job['origin_invocation_id'],
                                 invocation_id=uid('mem-inv-'), graph_thread_id='conv:' + job['conversation_id'],
                                 memory_job_id=job['id'])
        evaluation = evaluation_context.get()
        ctx = ExecutionContext(recorder.payload['invocation_id'], recorder.trace_id, job['conversation_id'],
                               recorder.payload['graph_thread_id'], principal, self.config.agent_release_id,
                               time.monotonic() + self.config.agent_timeout_seconds, evaluation.get('case_id'),
                               evaluation.get('run_id'), mode=job['mode'])
        telemetry = Telemetry(ctx, self.config.harness_trace_enabled)
        marker = active_trace.set(recorder)
        budget = RunBudget()
        try:
            source_ids = json.loads(job['source_message_ids_json'])
            self.service.event('batch_flushed', job_id=job['id'], source_message_ids=source_ids,
                               flush_reason=job['flush_reason'], batch_size=len(source_ids), attempt=job['attempts'])
            with self.repository.db.tx() as connection:
                if not self.current(connection, job, principal):
                    self.service.event('batch_cancelled', job_id=job['id'], reason_code='NO_CURRENT_LEASE_OR_CONSENT')
                    return True
                sources = self.eligible_sources(connection, principal, job, source_ids)
                keys = list(connection.execute(self.repository.active_rows(connection, principal)
                            .with_only_columns(memories.c.subject_key).order_by(memories.c.subject_key).limit(12)).scalars())
            if sources:
                candidates, usage = await self.extractor.extract(ctx, list(sources.values()), keys, budget, telemetry)
                self.service.event('extracted', job_id=job['id'], candidate_count=len(candidates),
                                   extractor_call_count=budget.model_attempts, **usage)
            else:
                candidates = []
            changes = []
            with self.repository.db.tx() as connection:
                if not self.current(connection, job, principal):
                    self.service.event('batch_cancelled', job_id=job['id'], reason_code='NO_CURRENT_LEASE_OR_CONSENT')
                    return True
                latest = self.eligible_sources(connection, principal, job, source_ids)
                verified = {ident: row for ident, row in latest.items() if sources.get(ident) == row}
                counts = Counter()
                order = {ident: position for position, ident in enumerate(source_ids)}
                for candidate in sorted(candidates, key=lambda item: order.get(item.source_message_id, len(order)))[:self.policy.candidates_per_batch]:
                    ident = candidate.source_message_id
                    counts[ident] += 1
                    reason = 'CANDIDATE_LIMIT' if counts[ident] > self.policy.candidates_per_message else validated(candidate, verified)
                    if reason:
                        changes.append(('SKIP', None, reason))
                        continue
                    action, memory_id = self.consolidator.apply(connection, principal, job, candidate, verified[ident])
                    changes.append((action, memory_id, 'MEMORY_CONFLICT_UNRESOLVED' if action == 'SKIP' else None))
                pending_times = list(connection.execute(select(memory_pending.c.created_at).where(memory_pending.c.batch_id == job['id'])).scalars())
                self.terminal(connection, job['id'], 'COMPLETED')
            changed_ids = set()
            for action, memory_id, reason in changes:
                self.service.event({'CREATE': 'upserted', 'UPDATE': 'updated', 'REINFORCE': 'upserted'}.get(action, 'skipped'),
                                   operation=action, memory_id=memory_id, job_id=job['id'], reason_code=reason)
                if action in ('CREATE', 'UPDATE'):
                    changed_ids.add(memory_id)
            clock = self.repository.business.clock()
            self.service.event('batch_completed', job_id=job['id'], batch_size=len(source_ids), extractor_call_count=budget.model_attempts,
                               flush_delays_ms=[(clock - datetime.fromisoformat(stamp)).total_seconds() * 1000 for stamp in pending_times])
            await self.service.sync_index(sorted(changed_ids))
        except Exception as error:
            code = getattr(error, 'code', 'MEMORY_EXTRACTION_ERROR')
            with self.repository.db.tx() as connection:
                if self.current(connection, job, principal):
                    if job['attempts'] >= self.policy.max_attempts or code in ('CONFIG_ERROR', 'CONTEXT_BUDGET_EXCEEDED'):
                        self.terminal(connection, job['id'], 'FAILED', code)
                    else:
                        retry = self.repository.business.clock() + timedelta(seconds=2 ** job['attempts'])
                        connection.execute(memory_jobs.update().where(memory_jobs.c.id == job['id']).values(
                            status='PENDING', next_retry_at=retry.isoformat(), error_code=code, lease_expires_at=None,
                            updated_at=self.repository.stamp()))
            self.service.event('batch_failed', job_id=job['id'], status='error', error_code='MEMORY_EXTRACTION_ERROR',
                               reason_code=code, extractor_call_count=budget.model_attempts)
        finally:
            try:
                if self.config.harness_trace_enabled:
                    recorder.finish()
            except Exception as error:
                logging.getLogger(__name__).error('Memory trace unavailable: %s', type(error).__name__)
            finally:
                active_trace.reset(marker)
        return True

    async def serve(self):
        while not self.stopping:
            self.wake.clear()
            try:
                if self.config.memory_enabled:
                    self.service.scheduler.schedule_due()
                    for attempt in range(16):
                        if self.stopping or not await self.process():
                            break
                await self.service.sync_index()
            except Exception as error:
                logging.getLogger(__name__).error('Memory worker iteration failed: %s', type(error).__name__)
            if self.stopping:
                break
            try:
                await asyncio.wait_for(self.wake.wait(), self.config.memory_batch_worker_poll_seconds)
            except TimeoutError:
                pass

    def stop(self):
        self.stopping = True
        self.wake.set()
