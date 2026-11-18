import json
import re

from sqlalchemy import select

from app.db import digest, memories, memory_operations
from app.memory.contracts import MemoryInput
from app.memory.trigger import CHANGE, FORGET, STABLE, TRANSIENT
from app.memory.validator import REPORTED, normalized, prohibited


def canonical(content):
    text = re.sub(r'[\W_]+', '', normalized(content).casefold())
    text = re.sub(r'^(?:请记住|请记下)?(?:我)?(?:平时|通常|一般)?', '', text)
    return re.sub(r'^(?:比较喜欢|更喜欢|偏爱|偏好|钟意|倾向于)', '喜欢', text)


def validated(candidate, sources):
    source = sources.get(candidate.source_message_id)
    if not source or not candidate.is_user_asserted:
        return 'MEMORY_EVIDENCE_UNSUPPORTED'
    raw = source['content'] or ''
    if prohibited(raw) or REPORTED.search(raw) or FORGET.search(raw) or prohibited(candidate.subject_key):
        return 'UNSAFE_SOURCE'
    clauses = [part.strip() for part in re.split(r'[，。！？；,!?;\n]+', raw) if part.strip()]
    excerpt = candidate.evidence_excerpt
    if candidate.content != excerpt or (excerpt != raw.strip() and excerpt not in clauses):
        return 'MEMORY_EVIDENCE_UNSUPPORTED'
    if not (STABLE.search(excerpt) or CHANGE.search(excerpt)):
        return 'NOT_STABLE_ASSERTION'
    if re.search(r'[?？]|(?:吗|么|是不是|能否|是否)[。！!]?$', raw.strip()) or TRANSIENT.search(raw) or re.search(r'如果|假如|假设|举例|[“”「」]', excerpt):
        return 'NOT_STABLE_ASSERTION'
    return None


class Consolidator:
    def __init__(self, repository):
        self.repository = repository

    def apply(self, connection, principal, job, candidate, source):
        fingerprint = digest(json.dumps([canonical(candidate.content), candidate.subject_key,
                                         candidate.source_message_id], ensure_ascii=False))
        if connection.scalar(select(memory_operations.c.operation).where(memory_operations.c.job_id == job['id'],
                             memory_operations.c.candidate_fingerprint == fingerprint)):
            return 'REPLAY', None
        rows = list(connection.execute(self.repository.active_rows(connection, principal)).mappings())
        previous = next((row for row in rows if row['subject_key'] == candidate.subject_key), None)
        equivalent = next((row for row in rows if row['memory_type'] == candidate.type
                           and canonical(row['content']) == canonical(candidate.content)), None)
        newer = not previous or (source['created_at'], source['id']) > (previous['source_created_at'], previous['source_message_id'] or '\uffff')
        change = CHANGE.search(candidate.content) or (candidate.type == 'stable_constraint' and '预算' in candidate.content)
        if equivalent:
            action, memory_id = 'REINFORCE', equivalent['id']
        elif previous and (not newer or not change or previous['memory_type'] != candidate.type):
            action, memory_id = 'SKIP', previous['id']
        else:
            data = MemoryInput(memory_type=candidate.type, subject_key=candidate.subject_key, content=candidate.content)
            item = self.repository.save(connection, principal, data, previous=previous,
                                        source={**source, 'extractor_version': job['extractor_version']})
            action, memory_id = ('UPDATE' if previous else 'CREATE'), item.memory_id
        connection.execute(memory_operations.insert().values(job_id=job['id'], candidate_fingerprint=fingerprint,
                           operation=action, memory_id=memory_id, created_at=self.repository.stamp()))
        return action, memory_id
