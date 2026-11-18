import argparse,asyncio,json
from app.evaluation import run,promote,review,judge_existing
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset'); p.add_argument('--mode',default='deterministic',choices=['deterministic','rag-real','agent-real','judge']); p.add_argument('--run-id'); p.add_argument('--agent-release',default='local'); p.add_argument('--retrieval-mode',default='hybrid_rerank'); p.add_argument('--skill-mode',default='selected',choices=['none','all','selected']); p.add_argument('--promote-failures'); p.add_argument('--confirmed-only',action='store_true'); p.add_argument('--review-run'); p.add_argument('--case-id'); p.add_argument('--reviewer'); p.add_argument('--reason'); a=p.parse_args()
    if a.mode=='judge':
        if not a.run_id: p.error('judge mode requires --run-id')
        print(json.dumps(asyncio.run(judge_existing(a.run_id)),ensure_ascii=False,indent=2))
    elif a.promote_failures:
        if not a.confirmed_only: p.error('Promotion requires --confirmed-only')
        print(promote(a.promote_failures))
    elif a.review_run:
        if not all((a.case_id,a.reviewer,a.reason)): p.error('Review requires case-id, reviewer, reason')
        review(a.review_run,a.case_id,a.reviewer,a.reason)
    elif a.dataset:
        r=asyncio.run(run(a.dataset,a.mode,a.agent_release,retrieval_mode=a.retrieval_mode,skill_mode=a.skill_mode)); print(json.dumps({k:v for k,v in r.items() if k!='cases'},ensure_ascii=False,indent=2))
    else: p.error('Choose --dataset, --review-run or --promote-failures')
if __name__=='__main__': main()
