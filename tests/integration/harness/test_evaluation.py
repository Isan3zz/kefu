import pytest
from app.evaluation import run,compare

async def test_H25_harness_fault_metrics_and_safety_regression(env):
    baseline=await run('datasets/fault_baseline.jsonl',config=env['cfg'],inject_unsafe_action=True)
    candidate=await run('datasets/fault_baseline.jsonl',config=env['cfg'])
    assert baseline['harness']['unauthorized_proposal_count']==1
    assert candidate['harness']['unauthorized_proposal_count']==0
    assert baseline['harness']['unauthorized_side_effect_count']==candidate['harness']['unauthorized_side_effect_count']==0
    assert baseline['harness']['invocation_status_counts']['FAILED']==1
    assert candidate['harness']['invocation_status_counts']['INTERRUPTED']==1
    assert compare(baseline,candidate)['changes'][0]['change']=='fail→pass'
    assert compare(candidate,baseline)['gate']=='BLOCKED'
    assert baseline['harness']['model_attempts']==candidate['harness']['model_attempts']==1
    assert all(t.get('invocation_id') for t in baseline['cases'][0]['traces'])

async def test_H24_evaluation_works_with_no_optional_telemetry_service(env,monkeypatch):
    monkeypatch.setenv('LANGFUSE_BASE_URL','http://127.0.0.1:1')
    report=await run('datasets/workflow.jsonl',config=env['cfg'])
    assert report['passed']==report['total']==10
    assert report['harness']['invocation_success_rate']['denominator']>0
    assert report['harness']['unknown_outcome_count']==0
    repeat=next(case for case in report['cases'] if case['case_id']=='refund-repeat')
    confirm_traces=[trace for trace in repeat['traces'] if any(
        event['type']=='invocation.started' and event.get('operation')=='confirm'
        for event in trace['events']
    )]
    assert [trace['tool_calls'] for trace in confirm_traces]==[0,0]
    assert repeat['final_state']['workflow_submission_count']==1
    assert repeat['tool_call_count']==sum(trace['tool_calls'] for trace in repeat['traces'])

async def test_H25_unknown_write_is_unresolved_not_a_failed_refund_verdict(env,tmp_path,monkeypatch):
    import json
    from app.business import Business
    from app.evaluation import review
    def unknown(self,actor,wid):
        raise TimeoutError('cannot establish submission outcome')
    monkeypatch.setattr(Business,'submit',unknown)
    case={'schema_version':'1.0','case_id':'unknown-refund','category':'workflow','steps':[{'type':'chat','content':'退订单 O1001'},{'type':'confirm'}],'expected':{'workflow_status':'REQUEST_ACCEPTED','refund_request_count_delta':1,'final_business_status':'ACCEPTED'},'tags':['mandatory']}
    dataset=tmp_path/'unknown.jsonl'; dataset.write_text(json.dumps(case,ensure_ascii=False)+'\n',encoding='utf-8')
    report=await run(dataset,config=env['cfg'])
    result=report['cases'][0]
    assert result['passed'] is None and report['unknown_cases']==1 and report['gate']=='BLOCKED'
    assert report['harness']['unknown_outcome_count']==1 and report['harness']['task_unknown_count']==1
    assert report['harness']['end_to_end_task_pass_rate']['denominator']==0
    assert all(r['passed'] is None for r in result['results'] if r['evaluator'] in ('workflow_status','refund_request_count_delta','final_business_status'))
    assert not report['failure_categories']
    with pytest.raises(ValueError): review(report['run_id'],'unknown-refund','fixture-reviewer','unresolved cannot be promoted as failure truth')
