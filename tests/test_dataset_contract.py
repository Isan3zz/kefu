import json
import pytest
from app.dataset import load_dataset
from app.evaluation import assertions,compare
from app.evaluation import FieldEvaluator

def test_independent_evaluator_has_explicit_not_applicable_state():
    evaluator=FieldEvaluator('intent','INTENT_ERROR')
    assert evaluator.evaluate({'case_id':'c','expected':{}},{},{}).passed is None
    result=evaluator.evaluate({'case_id':'c','expected':{'intent':'refund'}},{},{'intent':'order_query'})
    assert result.passed is False and result.failure_category=='INTENT_ERROR'

def base_case():
    return dict(schema_version='1.0',case_id='case-001',category='workflow',fixture='user_a_delivered_refundable',messages=[dict(role='user',content='你好')],expected={'intent':'other'})

@pytest.mark.parametrize('mutation',[
    lambda c:c['expected'].update(typo_expectation=1),
    lambda c:c.update(fixture='unknown_fixture'),
    lambda c:c['expected'].update(refund_request_count_delta=True),
    lambda c:c['expected'].update(required_tools='get_orders'),
    lambda c:c['expected'].update(tool_arguments=[{'name':'get_order'}]),
    lambda c:c.update(steps=[{'type':'confirm','use_pending_nonce':False}]),
])
def test_invalid_dataset_cannot_silently_reduce_checks(tmp_path,mutation):
    case=base_case(); mutation(case); path=tmp_path/'case.jsonl'; path.write_text(json.dumps(case),encoding='utf-8')
    with pytest.raises(ValueError): load_dataset(path)

def test_tool_order_consumes_each_required_occurrence():
    case=base_case(); case['expected']={'tool_order':['get_order','get_order']}
    result=assertions(case,{'tool_events':[{'name':'get_order'}]},{} )
    assert result[0]['passed'] is False

def test_tool_argument_mismatch_is_not_a_tool_selection_success():
    case=base_case(); case['expected']={'required_tools':['get_order'],'tool_arguments':[{'name':'get_order','args':{'order_id':'O1001'}}]}
    result=assertions(case,{'tool_events':[{'name':'get_order','args':{'order_id':'O2001'}}]},{} )
    assert result[0]['passed'] is True
    assert result[1]['failure_category']=='TOOL_ARGUMENT_ERROR'

def test_new_safety_category_blocks_even_when_case_was_already_failing():
    def report(category):
        return dict(dataset_sha256='same',cases=[dict(case_id='case',passed=False,results=[dict(evaluator='execution',passed=False,failure_category=category)])],metrics={},release={},latency={},failure_categories={})
    assert compare(report('GENERATION_ERROR'),report('AUTHORIZATION_VIOLATION'))['gate']=='BLOCKED'
    a=report('GENERATION_ERROR'); b=report('GENERATION_ERROR'); b['fixture']={'clock':'different'}
    with pytest.raises(ValueError,match='fixture'): compare(a,b)
