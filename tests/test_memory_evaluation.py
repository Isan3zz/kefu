from pathlib import Path

import pytest

from app.memory.contracts import TriggerDecision
from app.memory.evaluation import Turn, load_memory_dataset, run_memory
from app.memory.trigger import MemoryTrigger

DATASET = Path('datasets/memory.jsonl')


@pytest.mark.asyncio
async def test_M22_memory_on_off_distinct_sessions_and_reusable_reports(tmp_path, env):
    report = await run_memory(DATASET, ('memory-off', 'memory-on'), config=env['cfg'],
                              output_root=tmp_path, case_id='cross-session-noise')
    off, on = (report['runs'][variant] for variant in ('memory-off', 'memory-on'))
    for run in (off, on):
        facts = run['cases'][0]['final_state']
        assert facts['write_conversation_id'] != facts['query_id']
        assert run['release']['batch_policy_version'] and run['fixture']['version']
        assert (tmp_path / report['run_id'] / run['variant'] / 'index.sqlite').exists()
    assert off['cases'][0]['final_state']['query_id'] != on['cases'][0]['final_state']['query_id']
    assert not off['cases'][0]['final_state']['task_success'] and on['cases'][0]['final_state']['task_success']
    assert report['comparisons']['memory-off__memory-on']['changes'][0]['change'] == 'fail→pass'
    assert (tmp_path / report['run_id'] / 'report.md').exists()


def test_M24_labeled_trigger_variations():
    _, cases = load_memory_dataset(DATASET)
    expected, selected = set(), set()
    for case in cases:
        for position, turn in enumerate(case.turns):
            key = case.case_id, position
            truth = turn.expected_trigger in (TriggerDecision.URGENT, TriggerDecision.CANDIDATE)
            decision = MemoryTrigger().check({'role': 'user', 'content': turn.text}, opted_in=case.opt_in, max_bytes=950)
            if truth:
                expected.add(key)
            if decision.decision in (TriggerDecision.URGENT, TriggerDecision.CANDIDATE):
                selected.add(key)
    false_negatives = expected - selected
    assert len(expected) > 20 and not false_negatives
    assert len(expected & selected) / len(selected) >= .9


@pytest.mark.asyncio
async def test_M32_extraction_strategy_metrics_use_actual_calls(tmp_path, env):
    _, cases = load_memory_dataset(DATASET)
    case = next(case for case in cases if case.case_id == 'batch-three')
    case.turns.append(Turn(text='谢谢解答', expected_trigger='NOOP'))
    dataset = tmp_path / 'dataset.jsonl'
    dataset.write_text(case.model_dump_json(), encoding='utf-8')
    variants = ('extraction-per-turn', 'trigger-only', 'trigger-batch')
    report = await run_memory(dataset, variants, config=env['cfg'], output_root=tmp_path)
    for variant, count in zip(variants, (4, 3, 1), strict=True):
        run = report['runs'][variant]
        assert run['memory']['extractor_calls'] == count
        assert run['metrics']['extractor_calls_per_100_turns']['value'] == count * 25
        assert run['metrics']['extraction_recall']['value'] == 1
        assert run['memory']['estimated_extractor_input_tokens'] > 0
        assert run['memory']['provider_tokens'] is None
        assert run['memory']['flush_delay_ms']['samples'] >= 3
