"""Score frozen Judge anchors and compare optional independent human ratings."""
import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict

from app.config import settings
from app.db import now, uid
from app.eval_judge import DIMENSIONS, Rating, judge, judge_config
from app.harness.metrics import rate
from app.rag import atomic_json


class HumanRating(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    case_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    at: datetime
    reason: str = Field(min_length=1)
    scores: dict[str, Rating]


def agreement(verdicts, ratings):
    matches = defaultdict(list)
    for rating in ratings:
        for dimension, human in rating.scores.items():
            measured = verdicts.get(rating.case_id, {}).get('dimensions', {}).get(dimension)
            if measured is not None:
                matches[dimension].append(measured == human)
    return {dimension: rate(sum(matches[dimension]), len(matches[dimension])) for dimension in DIMENSIONS}


async def calibrate(dataset, ratings=(), config=settings):
    raw = Path(dataset).read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = {case['case_id'] for case in cases}
    rating_keys = {(row.case_id, row.reviewer) for row in ratings}
    if not cases or len(ids) != len(cases) or len(rating_keys) != len(ratings):
        raise ValueError('Calibration case IDs and reviewer/case pairs must be unique')
    if any(row.case_id not in ids or set(row.scores) != set(DIMENSIONS) for row in ratings):
        raise ValueError('Human ratings must refer to a calibration case and cover all dimensions')
    verdicts = {case['case_id']: await judge(case['answer'], case['evidence'], config) for case in cases}
    assessed = [case for case in cases if verdicts[case['case_id']].get('passed') is not None]
    return {'run_id': uid('judge-calibration-'), 'created_at': now(), 'dataset_sha256': hashlib.sha256(raw).hexdigest(),
            'judge_config': judge_config(config), 'verdicts': verdicts,
            'anchor_agreement': rate(sum(verdicts[case['case_id']]['passed'] == case['expected_pass'] for case in assessed), len(assessed)),
            'human_rating_count': len(ratings), 'human_ratings': [row.model_dump(mode='json') for row in ratings],
            'human_agreement': agreement(verdicts, ratings), 'total': len(cases)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='datasets/judge/calibration_v1.jsonl')
    parser.add_argument('--human-ratings', type=Path)
    args = parser.parse_args()
    ratings = [HumanRating.model_validate_json(line) for line in args.human_ratings.read_bytes().splitlines() if line.strip()] if args.human_ratings else []
    output = asyncio.run(calibrate(args.dataset, ratings))
    path = Path('data/eval_runs') / output['run_id'] / 'report.json'
    path.parent.mkdir(parents=True)
    atomic_json(path, output)
    print(path)


if __name__ == '__main__':
    main()
