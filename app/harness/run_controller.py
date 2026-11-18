"""Single-worker graph serialization and durable request identity, not a graph scheduler."""
import asyncio,json,hashlib,time
from contextlib import asynccontextmanager
from datetime import datetime,timezone,timedelta
from sqlalchemy import select
from app.db import agent_invocations,conversations,uid,now
from app.business import DomainError
from app.harness.execution_context import ExecutionContext,Principal

class ThreadLocks:
    def __init__(self): self.entries={}
    @asynccontextmanager
    async def acquire(self,key):
        entry=self.entries.setdefault(key,[asyncio.Lock(),0]); entry[1]+=1
        try:
            async with entry[0]: yield
        finally:
            entry[1]-=1
            if entry[1]==0: self.entries.pop(key,None)

class RunController:
    def __init__(self,business,config):
        self.b,self.config=business,config
        self.locks=ThreadLocks()
        self.tasks=set()
        self.active=set()
    def begin(self,principal,conversation_id,graph_thread_id,body,client_request_id=None,workflow_id=None,mode='real',evaluation=None):
        if client_request_id is not None and (not client_request_id.strip() or len(client_request_id)>200): raise DomainError('INVALID_ARGUMENT')
        body_hash=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        p=Principal(tenant_id=principal.tenant_id,user_id=principal.user_id,role=principal.role)
        parent=None
        with self.b.db.tx() as c:
            self.b.owned(c,conversations,p,conversation_id)
            if client_request_id:
                old=c.execute(select(agent_invocations).where(agent_invocations.c.tenant_id==p.tenant_id,agent_invocations.c.user_id==p.user_id,agent_invocations.c.conversation_id==conversation_id,agent_invocations.c.client_request_id==client_request_id)).mappings().first()
                if old:
                    if old['request_body_hash']!=body_hash: raise DomainError('REQUEST_CONFLICT','同一 request_id 不能用于不同请求')
                    if old['status'] in ('CREATED','RUNNING') and old['invocation_id'] not in self.active:
                        # Explicit retry after restart: never redispatch an abandoned write.
                        status='UNKNOWN_OUTCOME' if old['write_started'] else 'FAILED'
                        response={'invocation_id':old['invocation_id'],'trace_id':old['trace_id'],'invocation_status':status,'error_code':'UNKNOWN_OUTCOME' if old['write_started'] else 'INTERNAL_ERROR','answer':'原执行进程状态不可确认，请查询原工作流或使用新的请求显式恢复'}
                        c.execute(agent_invocations.update().where(agent_invocations.c.invocation_id==old['invocation_id']).values(status=status,error_code=response['error_code'],response_json=json.dumps(response,ensure_ascii=False),finished_at=now(),updated_at=now()))
                        return None,response
                    response=json.loads(old['response_json']) if old['response_json'] else {'invocation_id':old['invocation_id'],'trace_id':old['trace_id'],'status':old['status'],'processing':True}
                    return None,response
            if workflow_id:
                parent=c.execute(select(agent_invocations.c.invocation_id).where(agent_invocations.c.workflow_id==workflow_id,agent_invocations.c.tenant_id==p.tenant_id,agent_invocations.c.user_id==p.user_id).order_by(agent_invocations.c.created_at.desc()).limit(1)).scalar()
            ident,trace=uid('inv-'),uid('trace-'); stamp=now()
            c.execute(agent_invocations.insert().values(invocation_id=ident,trace_id=trace,conversation_id=conversation_id,graph_thread_id=graph_thread_id,workflow_id=workflow_id,tenant_id=p.tenant_id,user_id=p.user_id,parent_invocation_id=parent,client_request_id=client_request_id,request_body_hash=body_hash,status='CREATED',agent_release_id=self.config.agent_release_id,created_at=stamp,updated_at=stamp,deadline_at_utc=(datetime.now(timezone.utc)+timedelta(seconds=self.config.agent_timeout_seconds)).isoformat()))
        self.active.add(ident)
        evaluation=evaluation or {}
        return ExecutionContext(invocation_id=ident,trace_id=trace,conversation_id=conversation_id,graph_thread_id=graph_thread_id,principal=p,agent_release_id=self.config.agent_release_id,deadline_monotonic=time.monotonic()+self.config.agent_timeout_seconds,workflow_id=workflow_id,parent_invocation_id=parent,mode=mode,evaluation_case_id=evaluation.get('case_id'),evaluation_run_id=evaluation.get('run_id')),None
    def update(self,ctx,**values):
        if 'status' in values:
            from app.harness.contracts import InvocationStatus
            values['status']=InvocationStatus(values['status']).value
        with self.b.db.tx() as c:
            c.execute(agent_invocations.update().where(agent_invocations.c.invocation_id==ctx.invocation_id,agent_invocations.c.tenant_id==ctx.principal.tenant_id,agent_invocations.c.user_id==ctx.principal.user_id).values(updated_at=now(),**values))
    def read(self,principal,ident):
        with self.b.db.tx() as c:
            row=c.execute(select(agent_invocations).where(agent_invocations.c.invocation_id==ident,agent_invocations.c.tenant_id==principal.tenant_id,agent_invocations.c.user_id==principal.user_id)).mappings().first()
            if not row: raise DomainError('NOT_FOUND')
            result=dict(row)
            result.pop('request_body_hash',None)
            result['response']=json.loads(result.pop('response_json')) if result['response_json'] else None
            return result
    def retain(self,task):
        self.tasks.add(task)
        def completed(done):
            self.tasks.discard(done)
            if not done.cancelled(): done.exception()
        task.add_done_callback(completed)
        return task
    async def close(self):
        if self.tasks: await asyncio.gather(*list(self.tasks),return_exceptions=True)
