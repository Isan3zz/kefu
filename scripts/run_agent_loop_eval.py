import argparse
import asyncio
import json
from pathlib import Path

from app.config import settings
from app.evaluation import compare, run
from app.rag import atomic_json


async def experiment(dataset, mode, baseline_report=None):
    if mode == 'agent-real' and not all((settings.llm_base_url,settings.llm_api_key,settings.llm_model)):
        return {'status':'SKIPPED','reason':'Real tool calling requires configured main model.'}
    results={}
    for label,skill_mode in (('loop_without_dynamic_skill','none'),('loop_on_demand_skill','selected')):
        results[label]=await run(dataset,mode=mode,agent_release='agent-loop-v1',skill_mode=skill_mode)
    baseline=results['loop_without_dynamic_skill']; candidate=results['loop_on_demand_skill']
    comparison={'no_skill_vs_on_demand':compare(baseline,candidate)}
    if baseline_report:
        old=json.loads(Path(baseline_report).read_text(encoding='utf-8'))
        if old['mode']!=candidate['mode'] or old['release']['llm']!=candidate['release']['llm'] or old.get('knowledge_corpus_sha256')!=candidate.get('knowledge_corpus_sha256'):
            raise ValueError('Legacy comparison requires the same model mode, LLM, and knowledge corpus')
        comparison['legacy_vs_on_demand']=compare(old,candidate)
    comparison.update({label:{key:result[key] for key in ('run_id','passed','total','gate','loop','llm_usage')}
                       for label,result in results.items()})
    output=Path('data/eval_runs')/candidate['run_id']/'loop_comparison.json'
    atomic_json(output,comparison)
    return {'status':'MEASURED','report':str(output),'groups':{label:f"{result['passed']}/{result['total']}"
                                                            for label,result in results.items()}}


def main():
    parser=argparse.ArgumentParser(description='Compare bounded Agent Loop orchestration on isolated cases')
    parser.add_argument('--dataset',default='datasets/agent_loop.jsonl')
    parser.add_argument('--mode',choices=('deterministic','agent-real'),default='deterministic')
    parser.add_argument('--baseline-report')
    args=parser.parse_args()
    print(json.dumps(asyncio.run(experiment(args.dataset,args.mode,args.baseline_report)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
