import json
from pathlib import Path
from app.evaluation import run
from app.evaluation import summarize_llm_usage

def test_missing_usage_is_not_reported_as_zero_cost(env):
    cfg=env['cfg'].model_copy(update={'llm_input_cost_per_million':'2','llm_output_cost_per_million':'6'})
    usage,cost=summarize_llm_usage([{'usage':{}}],cfg)
    assert usage['status']=='NOT_PROVIDED' and usage['input_tokens'] is None
    assert cost['status']=='NOT_EVALUATED' and cost['value'] is None
    usage,cost=summarize_llm_usage([{'usage':{'prompt_tokens':1000,'completion_tokens':100}}],cfg)
    assert usage['status']=='MEASURED' and cost['value']=='0.0026'

async def test_skill_negative_and_tool_case_denominators_are_distinct(env,tmp_path):
    cases=[
        {'schema_version':'1.0','case_id':'negative','category':'skill','messages':[{'role':'user','content':'你好'}],'expected':{'intent':'other','skill':None}},
        {'schema_version':'1.0','case_id':'positive','category':'skill','messages':[{'role':'user','content':'最近的订单'}],'expected':{'intent':'order_query','skill':'order-tracking','required_tools':['get_orders']}},
    ]
    path=tmp_path/'cases.jsonl'; path.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases),encoding='utf-8')
    report=await run(path,config=env['cfg'])
    assert report['metrics']['skill']['numerator']==report['metrics']['skill']['denominator']==1
    assert report['metrics']['skill_false_activation_rate']['numerator']==0
    assert report['metrics']['skill_false_activation_rate']['denominator']==1
    assert report['metrics']['tool_selection_accuracy']['denominator']==1
    assert [case['tool_call_count'] for case in report['cases']]==[0,1]
    assert report['metrics']['avg_tool_calls_per_task']['value']==0.5
    assert report['metrics']['avg_tool_calls_per_task']['denominator']==2
    assert report['agent_latency']['samples']==2
    assert report['duplicate_refund_count']==0
    text=(Path('data/eval_runs')/report['run_id']/'report.md').read_text(encoding='utf-8')
    assert 'Duplicate Refund Count: 0' in text
