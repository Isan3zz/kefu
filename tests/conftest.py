import pytest
from app.config import settings
from app.db import DB
from app.business import Business,Actor
from app.mcp import Gateway,create_mcp
from app.agent import Agent
from app.rag import RAG
from app.retrieval_mcp import RetrievalService, create_retrieval_mcp
from app.fakes import FakeModels

@pytest.fixture
def env(tmp_path):
    cfg=settings.model_copy(update={'sqlite_url':'sqlite:///'+str(tmp_path/'app.db'),'checkpoint_db_path':str(tmp_path/'cp.db'),'chroma_path':str(tmp_path/'chroma'),'bm25_index_path':str(tmp_path/'bm25'),'llm_api_key':'','llm_base_url':'','llm_model':''})
    db=DB(cfg.sqlite_url); db.seed(); b=Business(db); a=Actor('user_a','a'); other=Actor('user_b','b'); conv=b.conversation(a)['id']
    rag=RAG(db,FakeModels(),cfg)
    retrieval=create_retrieval_mcp(RetrievalService(b,cfg,rag=rag),cfg)
    gateway=Gateway(b,cfg,create_mcp(b,cfg.business_secret,testing=True),retrieval_server=retrieval)
    agent=Agent(b,gateway,cfg,rag)
    yield dict(cfg=cfg,db=db,b=b,a=a,other=other,conv=conv,gateway=gateway,rag=rag,agent=agent)
    rag.close(); db.engine.dispose()
