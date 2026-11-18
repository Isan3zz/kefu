from collections import Counter
from .metrics import rate


def distribution(values):
    ordered=sorted(values)
    return {'samples':len(ordered),'avg':sum(ordered)/len(ordered),
            'p95':ordered[(95*len(ordered)+99)//100-1]} if ordered else {'samples':0,'avg':None,'p95':None}


def loop_metrics(cases):
    runs=[(trace,event) for case in cases for trace in case['traces'] for event in trace['events'] if event['type']=='loop.terminated']
    applicable=[case for case in cases if case['passed'] is not None and any(
        event['type']=='loop.terminated' for trace in case['traces'] for event in trace['events'])]
    required_yes=required_total=activated_total=0
    for case in cases:
        labels=case['original_case']['expected']
        if 'activated_skills' in labels:
            required=set(labels['activated_skills']); got=set(case['final_state'].get('activated_skills',()))
            required_yes+=len(required & got); required_total+=len(required); activated_total+=len(got)
    proposed=sum(event['type']=='loop.tool_proposed' for trace,_ in runs for event in trace['events'])
    repeats=sum(event['type']=='loop.no_progress' for trace,_ in runs for event in trace['events'])
    return {
        'task_success':rate(sum(case['passed'] for case in applicable),len(applicable)),
        'no_progress_rate':rate(sum(any(item['type']=='loop.no_progress' for item in trace['events']) for trace,_ in runs),len(runs)),
        'redundant_proposal_rate':rate(repeats,proposed,scope='identical normalized proposals rejected within an invocation'),
        'skill_activation_precision':rate(required_yes,activated_total),'skill_activation_recall':rate(required_yes,required_total),
        'iterations':distribution([event['iterations'] for _,event in runs]),
        'tool_calls':distribution([event['tool_calls'] for _,event in runs]),
        'latency_ms':distribution([trace['latency_ms'] for trace,_ in runs]),
        'provider_tokens':distribution([event['input_tokens']+event['output_tokens'] for _,event in runs
                                       if event.get('input_tokens') is not None and event.get('output_tokens') is not None]),
        'terminal_reasons':dict(Counter(event['terminal_reason'] for _,event in runs)),
    }
