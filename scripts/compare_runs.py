import argparse
import json
from pathlib import Path

from app.eval_gate import GatePolicy
from app.evaluation import compare
from app.rag import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--gate-policy', type=Path)
    parser.add_argument('--vary', action='append', default=[], choices=['agent_model_config', 'judge_config', 'models', 'retrieval', 'skill_mode', 'harness_config', 'fault_injection', 'knowledge'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--enforce', action='store_true', help='Exit nonzero for FAIL or BLOCKED')
    args = parser.parse_args()
    root = Path('data/eval_runs').resolve()

    def load(run_id):
        directory = (root / run_id).resolve()
        if directory.parent != root:
            parser.error('Expected a local run ID')
        return json.loads((directory / 'report.json').read_text(encoding='utf-8'))

    policy = GatePolicy.model_validate_json(args.gate_policy.read_bytes()) if args.gate_policy else GatePolicy()
    try:
        result = compare(load(args.baseline), load(args.candidate), policy, vary=args.vary)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'gate': 'BLOCKED', 'gate_reasons': [str(error)], 'case_comparison': []}
    result.update(baseline_run_id=args.baseline, candidate_run_id=args.candidate, gate_policy=policy.model_dump())
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.enforce and result['gate'] != 'PASS':
        raise SystemExit(2 if result['gate'] == 'BLOCKED' else 1)


if __name__ == '__main__':
    main()
