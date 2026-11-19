import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,Depends,Header,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse,JSONResponse,StreamingResponse,RedirectResponse
from pydantic import BaseModel,Field,ConfigDict
from sqlalchemy import select
from app.config import settings,ROOT
from app.db import *
from app.business import Business,DomainError
from app.mcp import Gateway
from app.agent import Agent
from app.memory.api import memory_router
from app.multimodal.api import ERROR_STATUS, image_router, retrieval_router

def create_app(config=settings,db=None,gateway=None,rag=None):
    db=db or DB(config.sqlite_url); db.initialize(); b=Business(db)
    agent=Agent(b,gateway or Gateway(b,config),config,rag)
    @asynccontextmanager
    async def lifespan(app):
        worker = agent.memory.worker
        worker.stopping = False
        task = asyncio.create_task(worker.serve()) if config.memory_job_mode == 'async' else None
        try: yield
        finally:
            if task is not None:
                worker.stop()
                await task
            await agent.close()
    app=FastAPI(title='Skill 客服 Agent',version='0.1.0',lifespan=lifespan); app.state.agent=agent; app.state.harness=agent.harness
    def envelope(data): return dict(code='OK',data=data,error=None,request_id=uid('req-'))
    @app.exception_handler(DomainError)
    async def domain_error(request,e):
        status={'UNAUTHORIZED':401,'FORBIDDEN':403,'POLICY_DENIED':403,'NOT_FOUND':404,'INVALID_ARGUMENT':422,'LLM_NOT_CONFIGURED':503,'CONFIG_ERROR':503,'KNOWLEDGE_NOT_READY':503,'UPSTREAM_UNAVAILABLE':503,'TOOL_UPSTREAM_ERROR':503,'MODEL_UPSTREAM_ERROR':503,'MODEL_RATE_LIMIT':503,'MODEL_BAD_OUTPUT':502,'TOOL_BAD_RESULT':502,'SYSTEM_ERROR':500,'INTERNAL_ERROR':500,'TIMEOUT':504,'RUN_TIMEOUT':504,'MODEL_TIMEOUT':504,'TOOL_TIMEOUT':504}.get(e.code,409)
        status = ERROR_STATUS.get(e.code, status)
        if e.code=='MODEL_TOOL_CALLING_UNSUPPORTED': status=503
        detail={'code':e.code,'message':e.message}
        if getattr(e,'invocation_id',None): detail.update(invocation_id=e.invocation_id,trace_id=e.trace_id)
        return JSONResponse(status_code=status,content=dict(code=e.code,data=None,error=detail,request_id=uid('req-')))
    @app.exception_handler(RequestValidationError)
    async def invalid_argument(request,e):
        return JSONResponse(status_code=422,content=dict(code='INVALID_ARGUMENT',data=None,error={'code':'INVALID_ARGUMENT','message':'请求参数不符合接口定义'},request_id=uid('req-')))
    @app.exception_handler(Exception)
    async def unexpected_error(request,error):
        request_id=uid('req-')
        logging.getLogger(__name__).error('Request failed: type=%s request_id=%s',type(error).__name__,request_id)
        return JSONResponse(status_code=500,content=dict(code='SYSTEM_ERROR',data=None,error={'code':'SYSTEM_ERROR','message':'服务暂不可用，请根据请求编号核查'},request_id=request_id))
    def actor(authorization:str=Header(default='')):
        if not authorization.startswith('Bearer '): raise DomainError('UNAUTHORIZED')
        return b.authenticate(authorization[7:])
    app.include_router(memory_router(agent.memory, actor, envelope))
    if config.multimodal_enabled:
        app.include_router(image_router(agent.images, agent.multimodal_index, actor, envelope))
        app.include_router(retrieval_router(agent.multimodal, actor, envelope))

        @app.get('/multimodal', include_in_schema=False)
        def multimodal_console():
            return FileResponse(Path(__file__).with_name('multimodal')/'console.html', media_type='text/html',
                                headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
                                         'Content-Security-Policy': "default-src 'self'; script-src 'unsafe-inline'; "
                                                                    "style-src 'unsafe-inline'; img-src 'self' blob:; "
                                                                    "frame-ancestors 'none'; base-uri 'none'"})
    def admin(a=Depends(actor)):
        if a.role!='admin': raise DomainError('FORBIDDEN')
        return a
    @app.get('/healthz')
    def health():
        with db.tx() as c: c.exec_driver_sql('SELECT 1')
        from app.rag import local_models
        return envelope({'db':'ok','models_loaded':local_models.cache_info().currsize>0})
    class Input(BaseModel): model_config=ConfigDict(extra='forbid')
    class Login(Input): demo_user_key:str
    @app.post('/api/v1/auth/demo-login')
    def login(body:Login):
        if not config.demo_mode: raise DomainError('FORBIDDEN')
        return envelope(b.login(body.demo_user_key))
    @app.post('/api/v1/conversations')
    def conversation(a=Depends(actor)): return envelope(b.conversation(a))
    @app.get('/api/v1/conversations')
    def list_conversations(a=Depends(actor)):
        with db.tx() as c: return envelope([dict(r) for r in c.execute(select(conversations).where(conversations.c.user_id==a.user_id,conversations.c.tenant_id==a.tenant_id)).mappings()])
    @app.get('/api/v1/conversations/{cid}/messages')
    def history(cid:str,a=Depends(actor)):
        with db.tx() as c:
            b.owned(c,conversations,a,cid)
            return envelope([dict(r) for r in c.execute(select(messages).where(messages.c.conversation_id==cid).order_by(messages.c.created_at)).mappings()])
    class Chat(Input):
        conversation_id:str
        message:str=Field(min_length=1,max_length=8000)
        mode:Literal['real','deterministic']='real'
        request_id:str|None=Field(default=None,min_length=1,max_length=200)
        image_ids:list[str]=Field(default_factory=list,max_length=config.multimodal_max_chat_images)
        allow_external_image_send:bool=Field(default=False,strict=True,
            description='Explicitly allow this request’s uploaded images to be sent to the configured external VLM.')
    @app.post('/api/v1/chat')
    async def chat(body:Chat,a=Depends(actor)):
        result=await agent.harness.invoke_chat(principal=a,conversation_id=body.conversation_id,message=body.message,mode=body.mode,client_request_id=body.request_id,
                                               image_ids=body.image_ids,allow_external_image_send=body.allow_external_image_send)
        if result.get('error_code'):
            error = DomainError(result['error_code'],result['answer'])
            error.trace_id, error.invocation_id = result['trace_id'], result['invocation_id']
            raise error
        return envelope(result)
    @app.post('/api/v1/chat/stream')
    async def stream(body:Chat,a=Depends(actor)):
        with db.tx() as c: b.owned(c,conversations,a,body.conversation_id)
        return StreamingResponse(agent.harness.stream_chat(principal=a,conversation_id=body.conversation_id,message=body.message,mode=body.mode,client_request_id=body.request_id,
                                                                         image_ids=body.image_ids,allow_external_image_send=body.allow_external_image_send),
                                 media_type='text/event-stream')
    @app.get('/api/v1/workflows/{wid}')
    def workflow(wid:str,a=Depends(actor)): return envelope(b.workflow(a,wid))
    class Confirm(Input):
        confirm_nonce:str
        request_id:str|None=Field(default=None,min_length=1,max_length=200)
    @app.post('/api/v1/workflows/{wid}/confirm')
    async def confirm(wid:str,body:Confirm,a=Depends(actor)):
        result=await agent.harness.resume_workflow(principal=a,workflow_id=wid,confirmation_payload={'confirm_nonce':body.confirm_nonce},client_request_id=body.request_id)
        if result.get('error_code'): raise DomainError(result['error_code'],result['answer'])
        return envelope(result)
    @app.get('/api/v1/invocations/{ident}')
    def invocation(ident:str,a=Depends(actor)): return envelope(agent.harness.controller.read(a,ident))
    @app.post('/api/v1/workflows/{wid}/cancel')
    def cancel(wid:str,a=Depends(actor)): return envelope(b.cancel(a,wid))
    @app.get('/api/v1/skills')
    def skills(a=Depends(actor)): return envelope(agent.skills.discover())
    @app.get('/api/v1/traces/{tid}')
    def trace(tid:str,a=Depends(actor)):
        with db.tx() as c:
            r=b.owned(c,traces,a,tid)
            return envelope(json.loads(r['payload']))
    @app.post('/api/v1/knowledge/rebuild')
    async def rebuild(a=Depends(admin)):
        return envelope(await asyncio.to_thread(agent.rag.build,ROOT/'knowledge'))
    class Eval(Input): dataset:str; mode:Literal['deterministic','rag-real','agent-real']='deterministic'; agent_release:str='local'
    @app.post('/api/v1/evaluations/runs')
    async def evaluate(body:Eval,a=Depends(admin)):
        from app.evaluation import run
        path=(ROOT/body.dataset).resolve()
        if not path.is_relative_to(ROOT/'datasets') or path.suffix!='.jsonl': raise DomainError('INVALID_ARGUMENT')
        if not path.is_file(): raise DomainError('NOT_FOUND')
        from app.dataset import load_dataset
        try: load_dataset(path)
        except ValueError as error: raise DomainError('INVALID_ARGUMENT','数据集不符合评测契约') from error
        report=await run(path,body.mode,body.agent_release,config=config)
        with db.tx() as c: c.execute(evaluation_runs.insert().values(id=report['run_id'],tenant_id=a.tenant_id,user_id=a.user_id,payload=json.dumps(report,ensure_ascii=False)))
        return envelope(report)
    def read_run(rid,a):
        with db.tx() as c: return json.loads(b.owned(c,evaluation_runs,a,rid)['payload'])
    @app.get('/api/v1/evaluations/runs/{rid}')
    def get_run(rid:str,a=Depends(admin)): return envelope(read_run(rid,a))
    @app.get('/api/v1/evaluations/runs/{rid}/failures')
    def failures(rid:str,a=Depends(admin)): return envelope([c for c in read_run(rid,a)['cases'] if c['passed'] is False])
    @app.get('/api/v1/evaluations/runs/{rid}/traces/{tid}')
    def evaluation_trace(rid:str,tid:str,a=Depends(admin)):
        read_run(rid,a)
        from app.evaluation import read_eval_trace
        return envelope(read_eval_trace(rid,tid))
    class Comparison(Input): baseline:str; candidate:str
    @app.post('/api/v1/evaluations/compare')
    def compare(body:Comparison,a=Depends(admin)):
        from app.evaluation import compare
        try: return envelope(compare(read_run(body.baseline,a),read_run(body.candidate,a)))
        except ValueError as error: raise DomainError('INVALID_ARGUMENT',str(error)) from error
    @app.get('/')
    def ui(): return RedirectResponse('/docs')
    return app

app=create_app()
