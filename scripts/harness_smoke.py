"""Offline runtime walkthrough: explicit fake models, real in-process FastMCP and isolated DB."""
import asyncio,json
from pathlib import Path
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy import select,func
from app.config import settings
from app.db import DB,refund_requests,traces
from app.business import Business,Actor
from app.agent import Agent
from app.mcp import Gateway,create_mcp
from app.rag import RAG,atomic_json
from app.fakes import FakeModels
from app.evaluation import run,case_workspace
from app.harness.model_gateway import ModelGateway

async def smoke():
    output=Path('data/harness_smoke'); output.mkdir(parents=True,exist_ok=True)
    with case_workspace() as folder:
        root=Path(folder)
        config=settings.model_copy(update={'sqlite_url':'sqlite:///'+str(root/'app.db'),'checkpoint_db_path':str(root/'cp.db'),'chroma_path':str(root/'chroma'),'bm25_index_path':str(root/'bm25'),'llm_api_key':'','llm_base_url':'','llm_model':''})
        db=DB(config.sqlite_url); db.seed(); b=Business(db); actor=Actor('user_a','a'); conv=b.conversation(actor)['id']
        from app.retrieval_mcp import RetrievalService, create_retrieval_mcp
        rag=RAG(db,FakeModels(),config); rag.build('knowledge')
        retrieval=create_retrieval_mcp(RetrievalService(b,config,rag=rag),config)
        gateway=Gateway(b,config,create_mcp(b,config.business_secret,testing=True),retrieval_server=retrieval)
        agent=Agent(b,gateway,config,rag)
        try:
            ordinary=await agent.chat(actor,conv,'耳机保修多久','deterministic')
            assert ordinary['citations']
            preview=await agent.chat(actor,conv,'退订单 O1001','deterministic'); pending=preview['pending_confirmation']
            assert preview['invocation_status']=='INTERRUPTED'
            await agent.chat(actor,conv,'耳机还有蓝色吗','deterministic')
            assert b.workflow(actor,pending['workflow_id'])['status']=='WAITING_CONFIRMATION'
            accepted=await agent.confirm(actor,pending['workflow_id'],pending['confirm_nonce']); replay=await agent.confirm(actor,pending['workflow_id'],pending['confirm_nonce'])
            assert accepted['refund_request_id']==replay['refund_request_id']
            original=gateway._call; tool_attempts=0
            async def timeout_once(*args,**kwargs):
                nonlocal tool_attempts
                tool_attempts+=1
                if tool_attempts==1: raise TimeoutError('injected read timeout')
                return await original(*args,**kwargs)
            gateway._call=timeout_once
            queried=await agent.chat(actor,conv,'查订单 O1001','deterministic')
            assert not queried['error_code'] and tool_attempts==2
            gateway._call=original
            model_attempts=0
            async def provider(request):
                nonlocal model_attempts
                model_attempts+=1
                if model_attempts==1: raise httpx.ReadTimeout('injected model timeout',request=request)
                return httpx.Response(200,json={'id':'mock','choices':[{'index':0,'message':{'role':'assistant','content':'{"ok":true}'},'finish_reason':'stop'}]})
            class Probe(BaseModel): ok:bool
            client=AsyncOpenAI(api_key='offline-mock',base_url='http://mock.local/v1',http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)))
            model=ModelGateway(config.model_copy(update={'llm_model':'mock-protocol'}),lambda:client)
            async def probe(ctx,services):
                result=await model.complete(ctx=ctx,messages=[{'role':'system','content':'Return JSON'}],purpose='smoke_timeout_probe',response_schema=Probe,budget=services.budget,telemetry=services.telemetry)
                return {'answer':'mock protocol probe','attempts':result.attempts}
            try: result=await agent.harness._invoke(actor,conv,'conv:'+conv,{'kind':'model_timeout_probe'},None,None,'deterministic',None,False,probe)
            finally: await client.close()
            assert result['attempts']==model_attempts==2
            with db.tx() as c:
                count=c.scalar(select(func.count()).select_from(refund_requests)); assert count==1
                records=[json.loads(value) for value in c.scalars(select(traces.c.payload))]
            (output/'traces.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
            evaluation=await run('datasets/workflow.jsonl',config=config,agent_release='harness-smoke')
            assert evaluation['passed']==evaluation['total']==10
            artifact={'status':'PASS','model_mode':'EXPLICIT_FAKE_AND_MOCK_HTTP_NOT_REAL_LLM','mcp_transport':'real FastMCP inprocess test server','refund_rows':count,'refund_request_id':accepted['refund_request_id'],'invocation_ids':[r['invocation_id'] for r in records],'tool_timeout_attempts':tool_attempts,'model_timeout_attempts':model_attempts,'evaluation_run_id':evaluation['run_id'],'harness_metrics':evaluation['harness']}
            atomic_json(output/'result.json',artifact)
            print(json.dumps(artifact,ensure_ascii=False,indent=2))
        finally:
            await agent.close(); db.engine.dispose()
    return artifact

if __name__=='__main__': asyncio.run(smoke())
