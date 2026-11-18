import json
from datetime import datetime

from sqlalchemy import select

from app.db import digest, memory_jobs, memory_pending, messages, uid
from app.harness.execution_context import Principal
from app.memory.contracts import EXTRACTOR_VERSION, TriggerDecision
from app.memory.repository import owner


class BatchExtractionScheduler:
    def __init__(self, repository, policy):
        self.repository, self.policy = repository, policy

    def enqueue(self, connection, principal, source, generation, dispatch, mode, invocation_id=None):
        filters = (*owner(memory_pending, principal), memory_pending.c.message_id == source['id'])
        existing = connection.execute(select(memory_pending).where(*filters)).mappings().first()
        if existing:
            return existing['id'], None
        stamp = self.repository.stamp()
        ident = uid('mem-pending-')
        connection.execute(memory_pending.insert().values(id=ident, tenant_id=principal.tenant_id, user_id=principal.user_id,
                           conversation_id=source['conversation_id'], message_id=source['id'], consent_generation=generation,
                           trigger_class=dispatch.decision.value, trigger_reason_code=dispatch.reason_code, status='PENDING',
                           created_at=stamp, updated_at=stamp, mode=mode, origin_trace_id=source['trace_id'],
                           origin_invocation_id=invocation_id))
        reason = 'URGENT' if dispatch.decision == TriggerDecision.URGENT else None
        batches = self.flush(connection, principal, source['conversation_id'], reason=reason, mode=mode)
        return ident, batches[-1] if batches else None

    def flush(self, connection, principal, conversation_id, reason=None, mode=None):
        settings = self.repository.settings(connection, principal)
        if not settings['auto_write_enabled']:
            return []
        generation = settings['consent_generation']
        query = select(memory_pending).join(messages, messages.c.id == memory_pending.c.message_id).where(
            *owner(memory_pending, principal), memory_pending.c.conversation_id == conversation_id,
            memory_pending.c.consent_generation == generation, memory_pending.c.status == 'PENDING')
        if mode is not None:
            query = query.where(memory_pending.c.mode == mode)
        rows = list(connection.execute(query.order_by(messages.c.created_at, messages.c.id)).mappings())
        stamp = self.repository.stamp()
        batch_ids = []
        for current_mode in sorted({row['mode'] for row in rows}):
            group = [row for row in rows if row['mode'] == current_mode]
            flush_reason = reason or self.due_reason(group, datetime.fromisoformat(stamp))
            if not flush_reason:
                continue
            for start in range(0, len(group), self.policy.max_messages):
                selected = group[start:start + self.policy.max_messages]
                source_ids = [row['message_id'] for row in selected]
                ident = uid('mem-job-')
                fingerprint = digest(json.dumps([principal.tenant_id, principal.user_id, conversation_id, generation,
                                                sorted(set(source_ids)), EXTRACTOR_VERSION], separators=(',', ':')))
                connection.execute(memory_jobs.insert().values(
                    id=ident, tenant_id=principal.tenant_id, user_id=principal.user_id, conversation_id=conversation_id,
                    consent_generation=generation, batch_fingerprint=fingerprint, source_message_ids_json=json.dumps(source_ids),
                    extractor_version=EXTRACTOR_VERSION, flush_reason=flush_reason, status='PENDING', mode=current_mode,
                    origin_trace_id=selected[-1]['origin_trace_id'], origin_invocation_id=selected[-1]['origin_invocation_id'],
                    created_at=stamp, updated_at=stamp))
                connection.execute(memory_pending.update().where(memory_pending.c.id.in_([row['id'] for row in selected]),
                                   memory_pending.c.status == 'PENDING').values(status='CLAIMED', batch_id=ident, updated_at=stamp))
                batch_ids.append(ident)
        return batch_ids

    def due_reason(self, group, clock):
        if not group:
            return None
        if any(row['trigger_class'] == 'URGENT' for row in group):
            return 'URGENT'
        if len(group) >= self.policy.eligible_count:
            return 'COUNT'
        times = [datetime.fromisoformat(row['created_at']) for row in group]
        if (clock - min(times)).total_seconds() >= self.policy.max_wait_seconds:
            return 'MAX_WAIT'
        if (clock - max(times)).total_seconds() >= self.policy.idle_seconds:
            return 'IDLE'
        return None

    def schedule_due(self):
        with self.repository.db.tx() as connection:
            scopes = connection.execute(select(memory_pending.c.tenant_id, memory_pending.c.user_id,
                memory_pending.c.conversation_id, memory_pending.c.mode).where(memory_pending.c.status == 'PENDING').distinct()).all()
            return [ident for tenant, user, conversation, mode in scopes
                    for ident in self.flush(connection, Principal(tenant, user), conversation, mode=mode)]
