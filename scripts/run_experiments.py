"""Explicit fake-model control-flow experiments. No claimed real-model benefit."""
import argparse,asyncio,json
from pathlib import Path
from app.evaluation import run,compare
from app.rag import atomic_json
from app.config import settings
from app.db import DB
from app.rag import RAG

async def real_rag_only():
    # No main LLM or Judge is involved. Warm both local models once for comparable timings.
    models=RAG(DB(settings.sqlite_url)).model()
    models.encode(['本地模型预热'])
    models.score([('退款政策','符合条件可申请退款。')])
    reports={}
    for mode in ('bm25','dense','hybrid','hybrid_rerank'):
        result=await run('datasets/retrieval.jsonl',mode='rag-real',retrieval_mode=mode,agent_release='local-bge-v1')
        reports[mode]={key:result[key] for key in ('run_id','passed','total','metrics','gate','retrieval_latency','knowledge_snapshot','dataset_sha256')}
    import torch,platform
    artifact={'mode':'rag-real','warmup':True,'main_llm_called':False,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'device':str(models.embedder.device),'cpu_threads':torch.get_num_threads()},'reports':reports}
    atomic_json('data/real_rag_ablation.json',artifact)
    print(json.dumps(artifact,ensure_ascii=False,indent=2))
async def experiments(real=False):
    reports={}
    for dataset in ('intent','skill','workflow','safety'):
        r=await run(f'datasets/{dataset}.jsonl',agent_release='agent-v1')
        reports[dataset]={k:r[k] for k in ('run_id','passed','total','metrics','gate')}
    for mode in ('bm25','dense','hybrid','hybrid_rerank'):
        r=await run('datasets/retrieval.jsonl',mode='rag-real' if real else 'deterministic',retrieval_mode=mode)
        reports['rag-'+mode]={k:r[k] for k in ('run_id','passed','total','metrics','gate')}
    for mode in ('none','all','selected'):
        r=await run('datasets/skill.jsonl',mode='agent-real' if real else 'deterministic',skill_mode=mode)
        reports['skills-'+mode]={k:r[k] for k in ('run_id','passed','total','skill_load_estimated_tokens','latency')}
    baseline=await run('datasets/fault_baseline.jsonl',agent_release='fault-injected',inject_unsafe_action=True)
    candidate=await run('datasets/fault_baseline.jsonl',agent_release='guarded',inject_unsafe_action=False)
    reports['fault']={'baseline':baseline['run_id'],'candidate':candidate['run_id'],'attempts':[baseline['illegal_action_attempts'],candidate['illegal_action_attempts']],'unauthorized_side_effects':[baseline['unauthorized_side_effects'],candidate['unauthorized_side_effects']],'comparison':compare(baseline,candidate),'review_status':'PENDING_HUMAN_REVIEW'}
    atomic_json('data/experiment_summary.json',{'model_mode':'real' if real else 'FAKE_MODELS_NOT_REAL_QUALITY','reports':reports})
    print(json.dumps(reports,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); group=p.add_mutually_exclusive_group(); group.add_argument('--real',action='store_true'); group.add_argument('--rag-real-only',action='store_true'); a=p.parse_args()
    asyncio.run(real_rag_only() if a.rag_real_only else experiments(a.real))
