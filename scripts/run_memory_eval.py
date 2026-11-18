import argparse
import asyncio
from pathlib import Path

from app.memory.evaluation import run_memory


def main():
    parser = argparse.ArgumentParser(description='Isolated long-term memory evaluation and ablations')
    parser.add_argument('--dataset', type=Path, default=Path('datasets/memory.jsonl'))
    parser.add_argument('--mode', choices=('deterministic', 'agent-real'), default='deterministic')
    parser.add_argument('--embedding', choices=('fake', 'local'), default='fake')
    parser.add_argument('--ablation', default='trigger-batch')
    parser.add_argument('--case-id')
    args = parser.parse_args()
    report = asyncio.run(run_memory(args.dataset, tuple(args.ablation.split(',')), args.mode,
                                    args.embedding, case_id=args.case_id))
    print('data/eval_runs/' + report['run_id'] + '/report.md')
    for name, result in report['runs'].items():
        print(f'{name}: {result["passed"]}/{result["total"]}, gate={result["gate"]}')
    return int(any(run['gate'] == 'BLOCKED' for run in report['runs'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
