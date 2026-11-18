from sqlalchemy import or_, select

from app.business import DomainError
from app.db import conversations, memories, memory_jobs, memory_pending, memory_revisions, memory_settings, messages, uid, users
from app.memory.contracts import RetrievedMemory
from app.memory.validator import prohibited


def owner(table, principal):
    return table.c.tenant_id == principal.tenant_id, table.c.user_id == principal.user_id


def active(stamp):
    return (memories.c.status == 'active', or_(memories.c.expires_at.is_(None), memories.c.expires_at > stamp))


def retrieved(row, score=None):
    return RetrievedMemory(memory_id=row['id'], memory_type=row['memory_type'], content=row['content'],
                           subject_key=row['subject_key'], version=row['version'],
                           source_message_id=row['source_message_id'], score=score, status=row['status'])


class SqliteMemoryRepository:
    def __init__(self, business):
        self.business, self.db = business, business.db

    def stamp(self):
        return self.business.clock().isoformat()

    def settings(self, connection, principal):
        if not connection.scalar(select(users.c.id).where(users.c.id == principal.user_id, users.c.tenant_id == principal.tenant_id)):
            raise DomainError('NOT_FOUND')
        row = connection.execute(select(memory_settings).where(*owner(memory_settings, principal))).mappings().first()
        return dict(auto_write_enabled=bool(row and row['auto_write_enabled']),
                    retrieval_enabled=not row or bool(row['retrieval_enabled']),
                    consent_generation=row['consent_generation'] if row else 0)

    def set_settings(self, connection, principal, values, *, invalidate=False):
        current = self.settings(connection, principal)
        stamp = self.stamp()
        if invalidate or (current['auto_write_enabled'] and values.get('auto_write_enabled') is False):
            current['consent_generation'] += 1
            for table, statuses in ((memory_jobs, ('PENDING', 'PROCESSING')), (memory_pending, ('PENDING', 'CLAIMED'))):
                connection.execute(table.update().where(*owner(table, principal), table.c.status.in_(statuses))
                                   .values(status='CANCELLED', updated_at=stamp))
        current.update(values)
        from sqlalchemy.dialects.sqlite import insert
        statement = insert(memory_settings).values(tenant_id=principal.tenant_id, user_id=principal.user_id,
                                                   created_at=stamp, updated_at=stamp, **current)
        connection.execute(statement.on_conflict_do_update(index_elements=['tenant_id', 'user_id'],
                                                            set_={**current, 'updated_at': stamp}))
        return current

    def sources(self, connection, principal, conversation_id, source_ids):
        self.business.owned(connection, conversations, principal, conversation_id)
        rows = connection.execute(select(messages).where(messages.c.conversation_id == conversation_id,
                                  messages.c.id.in_(source_ids), messages.c.role == 'user')).mappings()
        return {row['id']: dict(row) for row in rows}

    def owned(self, connection, principal, memory_id):
        row = self.business.owned(connection, memories, principal, memory_id)
        if row['status'] != 'active' or (row['expires_at'] and row['expires_at'] <= self.stamp()):
            raise DomainError('NOT_FOUND')
        return row

    def active_rows(self, connection, principal, ids=None):
        query = select(memories).where(*owner(memories, principal), *active(self.stamp()))
        return query if ids is None else query.where(memories.c.id.in_(ids))

    def save(self, connection, principal, data, previous=None, source=None):
        stamp = self.stamp()
        if previous is None:
            connection.execute(memories.update().where(*owner(memories, principal), memories.c.subject_key == data.subject_key,
                               memories.c.status == 'active', memories.c.expires_at <= stamp)
                               .values(status='superseded', index_pending=1, updated_at=stamp))
        memory_id = previous['id'] if previous else uid('mem-')
        version = previous['version'] + 1 if previous else 1
        values = {**data.model_dump(), 'status': 'active', 'version': version, 'updated_at': stamp, 'index_pending': 1,
                  'source_conversation_id': source['conversation_id'] if source else None,
                  'source_message_id': source['id'] if source else None,
                  'source_created_at': source['created_at'] if source else stamp,
                  'source_trace_id': source.get('trace_id') if source else None,
                  'evidence_excerpt': data.content if source else None,
                  'extractor_version': source['extractor_version'] if source else 'manual'}
        if previous:
            updated = connection.execute(memories.update().where(*owner(memories, principal), memories.c.id == memory_id,
                        memories.c.version == previous['version']).values(**values))
            if updated.rowcount != 1:
                raise DomainError('MEMORY_VERSION_CONFLICT')
        else:
            connection.execute(memories.insert().values(id=memory_id, tenant_id=principal.tenant_id,
                               user_id=principal.user_id, created_at=stamp, **values))
        connection.execute(memory_revisions.insert().values(id=uid('memrev-'), memory_id=memory_id,
                           operation='UPDATE' if previous else 'CREATE', old_version=previous['version'] if previous else None,
                           new_version=version, prior_content=previous['content'] if previous else None,
                           new_content=data.content, source_message_id=values['source_message_id'], created_at=stamp))
        return retrieved({'id': memory_id, **values})

    def delete(self, connection, principal, memory_id=None):
        if memory_id is not None:
            self.owned(connection, principal, memory_id)
        self.set_settings(connection, principal, {}, invalidate=True)
        filters = [*owner(memories, principal), memories.c.status != 'deleted']
        if memory_id is not None:
            filters.append(memories.c.id == memory_id)
        ids = list(connection.execute(select(memories.c.id).where(*filters)).scalars())
        connection.execute(memory_revisions.delete().where(memory_revisions.c.memory_id.in_(ids)))
        connection.execute(memories.update().where(*filters).values(status='deleted', content='', evidence_excerpt=None,
                           source_message_id=None, source_conversation_id=None, source_created_at=None, source_trace_id=None,
                           version=memories.c.version + 1, updated_at=self.stamp(), index_pending=1))
        return ids


def validate_manual(data):
    reason = prohibited(data.content) or prohibited(data.subject_key)
    if reason:
        raise DomainError('INVALID_ARGUMENT', reason)
