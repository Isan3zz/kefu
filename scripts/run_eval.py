import argparse
import asyncio
import json

from app.evaluation import judge_existing, promote, rescore_existing, review, run


def main():
    parser = argparse.ArgumentParser(description='Run, review or replay isolated evaluation cases.')
    parser.add_argument('--dataset')
    parser.add_argument('--mode', default='deterministic', choices=['deterministic', 'rag-real', 'agent-real', 'judge'])
    parser.add_argument('--run-id')
    parser.add_argument('--agent-release', default='local')
    parser.add_argument('--retrieval-mode', default='hybrid_rerank', choices=['bm25', 'dense', 'hybrid', 'hybrid_rerank'])
    parser.add_argument('--skill-mode', default='selected', choices=['none', 'all', 'selected'])
    parser.add_argument('--enable-judge', action='store_true', help='Allow the configured external Judge for an agent-real run')
    parser.add_argument('--inject-unsafe-action', action='store_true', help='Activate model_proposal in the isolated fault dataset')
    parser.add_argument('--rescore-run', help='Reapply deterministic assertions to recorded evidence without invoking the Agent')
    parser.add_argument('--promote-failures')
    parser.add_argument('--confirmed-only', action='store_true')
    parser.add_argument('--review-run')
    parser.add_argument('--case-id')
    parser.add_argument('--reviewer')
    parser.add_argument('--reason')
    args = parser.parse_args()
    if args.mode == 'judge':
        if not args.run_id:
            parser.error('Judge mode requires --run-id')
        result = asyncio.run(judge_existing(args.run_id))
    elif args.rescore_run:
        result = rescore_existing(args.rescore_run)
    elif args.promote_failures:
        if not args.confirmed_only:
            parser.error('Promotion requires --confirmed-only')
        result = {'regression_cases': promote(args.promote_failures)}
    elif args.review_run:
        if not all((args.case_id, args.reviewer, args.reason)):
            parser.error('Review requires --case-id, --reviewer and --reason')
        review(args.review_run, args.case_id, args.reviewer, args.reason)
        result = {'reviewed': args.case_id}
    elif args.dataset:
        result = asyncio.run(run(args.dataset, args.mode, args.agent_release, retrieval_mode=args.retrieval_mode,
                                 skill_mode=args.skill_mode, inject_unsafe_action=args.inject_unsafe_action,
                                 enable_judge=args.enable_judge))
    else:
        parser.error('Choose --dataset, --rescore-run, --review-run or --promote-failures')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
