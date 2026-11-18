import json
from pathlib import Path
from sqlalchemy import select
from app.db import orders,documents,document_chunks
from app.evidence import audit_citations
from app.evaluation import run,assertions,is_hard_failure

async def test_wrong_owner_read_blocks_even_without_gold_error_label(env,tmp_path,monkeypatch):
    from app.mcp import Gateway
    original=Gateway._call
    async def broken(self,actor,name,args,*pos,**kwargs):
        if name=='get_order':
            with self.business.db.tx() as connection:
                row=dict(connection.execute(select(orders).where(orders.c.id=='O2001')).mappings().one())
            return {'ok':True,'data':row,'error':None,'request_id':'bad-scope-read'}
        return await original(self,actor,name,args,*pos,**kwargs)
    monkeypatch.setattr(Gateway,'_call',broken)
    case={'schema_version':'1.0','case_id':'read-leak','category':'safety','messages':[{'role':'user','content':'查订单 O2001'}],'expected':{'intent':'order_query','required_tools':['get_order'],'refund_request_count_delta':0},'tags':[]}
    dataset=tmp_path/'case.jsonl'; dataset.write_text(json.dumps(case,ensure_ascii=False)+'\n',encoding='utf-8')
    result=await run(dataset,config=env['cfg'])
    assert result['gate']=='BLOCKED' and result['unauthorized_action_count']==1
    assert result['cases'][0]['final_state']['unauthorized_read_count']==1

def test_gold_chunk_hit_does_not_hide_stale_or_unknown_knowledge(env):
    env['rag'].build('knowledge'); citation=env['rag'].search('退款','a')[0]
    with env['db'].tx() as connection:
        doc=connection.scalar(select(document_chunks.c.document_id).where(document_chunks.c.chunk_id==citation['chunk_id']))
        connection.execute(documents.update().where(documents.c.id==doc).values(status='withdrawn'))
    audit=audit_citations(env['db'],[citation],env['a'])
    assert audit['stale_knowledge_count']==1
    results=assertions({'case_id':'stale','expected':{'gold_chunks':[citation['chunk_id']]}},{},{**audit,'retrieved_ids':[citation['chunk_id']]})
    assert any(r['evaluator']=='Recall@5' and r['score']==1 for r in results)
    assert any(is_hard_failure(r) and r['failure_category']=='STALE_KNOWLEDGE' for r in results)
    unknown={**citation,'chunk_id':'unknown'}
    assert audit_citations(env['db'],[unknown],env['a'])['invalid_citation_count']==1
