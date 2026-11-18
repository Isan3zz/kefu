"""Local trace authority shared by Agent, gateway and evaluation.

ContextVars carry audit context through LangGraph tasks, never actor authorization.
"""
import json
import logging
import re
import time
from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from app.db import traces, uid, now

active_trace = ContextVar("active_trace", default=None)
evaluation_context = ContextVar('evaluation_context',default={})

@contextmanager
def evaluation_scope(run_id,case_id):
    marker=evaluation_context.set({'run_id':run_id,'case_id':case_id})
    try: yield
    finally: evaluation_context.reset(marker)

def clean(value):
    if isinstance(value, dict):
        result={}
        for key,item in value.items():
            if key.lower() in ('address','shipping_address','billing_address','full_address','phone','mobile','email'): continue
            if key not in ('summary_id','summary_version','semantic_summary_generated','summary_method') and re.search(r"(?i)(nonce|(^|_)token($|_)|secret|authorization|api[_-]?key|instructions|summary|operation_key)",key): continue
            identifier=key.endswith('_id') or key in ('hash','content_hash','skill_hash','revision')
            trusted_hex=isinstance(item,str) and re.fullmatch(r'(?:(?:conv|wf):)?(?:[a-z][a-z0-9-]*-)?[a-fA-F0-9]{24,64}',item)
            if key in ('memory_ids', 'source_message_ids') and isinstance(item, list):
                prefix = 'mem-' if key == 'memory_ids' else 'msg-'
                result[key] = [ident if isinstance(ident, str) and re.fullmatch(prefix + r'[a-f0-9]{24}', ident)
                               else clean(ident) for ident in item]
                continue
            result[key]=item if identifier and trusted_hex else clean(item)
        return result
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", value)
        value = re.sub(r"\bsk-[\w-]+", "[REDACTED]", value)
        return re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[PHONE]", value)
    return value

class TraceRecorder:
    def __init__(self, business, actor, conversation_id, config, **metadata):
        self.business, self.actor, self.config = business, actor, config
        self.started = time.perf_counter()
        self.payload = dict(trace_id=metadata.pop('trace_id',None) or uid("trace-"), conversation_id=conversation_id,
                            tenant_id=actor.tenant_id, user_id=actor.user_id,
                            agent_release_id=config.agent_release_id,
                            prompt_version="agent-loop-v1", started_at=now(), events=[], errors=[], **metadata)
        context=evaluation_context.get()
        self.payload.update(run_id=context.get('run_id'),case_id=context.get('case_id'))
        from app.rag import model_manifest
        manifest=model_manifest()
        for key in ('embedding','reranker'):
            entry=manifest.get(key,'NOT_DOWNLOADED')
            self.payload[key+'_revision']=entry.get('revision') if isinstance(entry,dict) else entry
    @property
    def trace_id(self): return self.payload["trace_id"]
    def emit(self, kind, **data):
        entry = dict(event_id=uid(), span_id=uid(), parent_span_id=self.trace_id,
                     ts=now(), type=kind, duration_ms=0, status="success", error_code=None)
        entry.update(clean(data))
        entry.update(schema_version='1.1',trace_id=self.trace_id,
                     invocation_id=self.payload.get('invocation_id'),
                     evaluation_run_id=self.payload.get('run_id'),evaluation_case_id=self.payload.get('case_id'),
                     conversation_id=self.payload['conversation_id'],graph_thread_id=data.get('graph_thread_id',self.payload.get('graph_thread_id')))
        entry.setdefault('attributes',clean(data))
        self.payload["events"].append(entry)
        if kind=='retrieval' and data.get('chunks'):
            self.payload['kb_snapshot_id']=data['chunks'][0]['snapshot']
            for role in ('embedding','reranker'):
                if role+'_revision' in data['chunks'][0]: self.payload[role+'_revision']=data['chunks'][0][role+'_revision']
        if entry["status"] in ("error", "blocked") or kind == "error":
            self.payload["errors"].append(entry)
        return entry
    def finish(self, **metadata):
        self.payload.update(clean(metadata))
        self.payload["latency_ms"] = (time.perf_counter() - self.started) * 1000
        self.payload['completed_at']=now()
        from app.db import workflow_instances
        from sqlalchemy import select
        with self.business.db.tx() as connection:
            rows=connection.execute(select(workflow_instances.c.id,workflow_instances.c.order_id,workflow_instances.c.status,workflow_instances.c.amount_cents).where(
                workflow_instances.c.tenant_id==self.actor.tenant_id,
                workflow_instances.c.user_id==self.actor.user_id,
                workflow_instances.c.conversation_id==self.payload['conversation_id']
            )).mappings()
            self.payload['workflow_outcomes']=[dict(row) for row in rows]
        directory = Path(self.config.checkpoint_db_path).parent / "traces"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (datetime.now(timezone.utc).date().isoformat() + ".jsonl")
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(self.payload, ensure_ascii=False) + "\n")
        except OSError as error:
            # A file-system issue must not disguise an accepted refund as a failure.
            self.payload["storage_errors"] = [{"backend": "jsonl", "type": type(error).__name__}]
            logging.getLogger(__name__).warning("JSONL trace unavailable; retaining SQLite copy")
        with self.business.db.tx() as c:
            c.execute(traces.insert().values(id=self.trace_id, tenant_id=self.actor.tenant_id,
                       user_id=self.actor.user_id, conversation_id=self.payload["conversation_id"],
                       payload=json.dumps(self.payload, ensure_ascii=False)))
        return self.payload

def emit(kind, **data):
    recorder = active_trace.get()
    if recorder is not None: return recorder.emit(kind, **data)
