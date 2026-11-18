"""Same fixed multi-turn fixture, explicit FakeModelGateway, three context policies."""
import asyncio,json
from collections import Counter
from app.config import settings
from app.evaluation import run
from app.rag import atomic_json

async def experiment():
    outputs={}
    for strategy in ('none','L0+L1','L0+L1+L2'):
        report=await run('datasets/harness_context.jsonl',config=settings.model_copy(update={'context_strategy':strategy}),agent_release='harness-context-v1')
        context=[e for c in report['cases'] for t in c['traces'] for e in t['events'] if e['type']=='context.built']
        repeats=0; lost=0; checked=0; not_reached=0
        for case in report['cases']:
            actions=Counter((e['name'],json.dumps(e.get('args',{}),sort_keys=True)) for e in case['tools'])
            repeats+=sum(max(0,n-1) for n in actions.values())
            if 'workflow_status' in case['final_state']:
                checked+=1; lost+=int(case['final_state']['workflow_status']!='WAITING_CONFIRMATION')
            else: not_reached+=1
        outputs[strategy]={'run_id':report['run_id'],'passed':report['passed'],'total':report['total'],'estimated_input_tokens':sum(e['after_tokens'] for e in context),'estimation':'conservative UTF-8 byte estimate, not provider usage','context_compression_ratio':report['harness']['context_compression_ratio'],'key_entity_loss':{'count':lost if checked else None,'checked_cases':checked,'status':'MEASURED' if checked else 'NOT_REACHED','definition':'Expected O1001 binding missing from pending workflow after completing all steps; budget-aborted cases excluded'},'entity_check_not_reached':not_reached,'repeated_tool_calls':repeats,'latency_ms':report['latency'],'latency_scope':report['latency_scope'],'fixture':report['fixture'],'dataset_sha256':report['dataset_sha256'],'harness':report['harness']}
    result={'model':'EXPLICIT_FAKE_NOT_REAL_SEMANTIC_QUALITY','summary_method':'verified_structured_facts_fallback','strategies':outputs}
    atomic_json('data/harness-context-experiment.json',result); print(json.dumps(result,ensure_ascii=False,indent=2)); return result

if __name__=='__main__': asyncio.run(experiment())
