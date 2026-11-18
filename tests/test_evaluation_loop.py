import json
from pathlib import Path
import pytest
from app.evaluation import run,compare,review,promote

async def test_fault_attempt_vs_side_effect_and_review(env,tmp_path):
    baseline=await run('datasets/fault_baseline.jsonl',config=env['cfg'],inject_unsafe_action=True)
    candidate=await run('datasets/fault_baseline.jsonl',config=env['cfg'])
    assert baseline['illegal_action_attempts']==1 and baseline['unauthorized_side_effects']==0
    assert baseline['gate']=='BLOCKED' and candidate['gate']=='PASS'
    assert compare(baseline,candidate)['changes'][0]['change']=='fail→pass'
    assert compare(candidate,baseline)['gate']=='BLOCKED'
    target=tmp_path/'regression.jsonl'
    with pytest.raises(FileNotFoundError): promote(baseline['run_id'],target)
    # Simulate explicit human review ONLY in the test fixture; production runs stay pending.
    review(baseline['run_id'],'illegal-action-attempt','test-reviewer','Fixture verifies blocked backend and recorded illegal attempt')
    assert promote(baseline['run_id'],target)==1
    assert promote(baseline['run_id'],target)==1
    assert json.loads(target.read_text(encoding='utf-8'))['review']['source_run']==baseline['run_id']
    regression=await run(target,config=env['cfg'])
    assert regression['passed']==regression['total']==1
    assert regression['gate']=='PASS' and regression['illegal_action_attempts']==0
    assert regression['cases'][0]['original_case']['review']['source_run']==baseline['run_id']
