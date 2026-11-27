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
    plan_runs = {trace['trace_id'] for trace, _ in runs
                 if any(e['type'] == 'task.plan.created' for e in trace['events'])}
    control_runs = [trace['trace_id'] for case in cases if 'task-plan-control' in case.get('tags', ())
                    for trace in case['traces'] if any(e['type'] == 'loop.terminated' for e in trace['events'])]
    target_runs = [trace['trace_id'] for case in cases if 'task-plan-target' in case.get('tags', ())
                   for trace in case['traces'] if any(e['type'] == 'loop.terminated' for e in trace['events'])]
    planning_calls = [sum(e['type'] == 'tool' and e.get('layer') == 'harness' and e.get('name') == 'create_task_plan'
                          for e in trace['events']) for trace, _ in runs]
    task_events = Counter(e['type'] for trace, _ in runs for e in trace['events'] if e['type'].startswith('task.'))
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
        'task_plan_usage_rate':rate(len(plan_runs),len(runs)),
        'task_plan_target_usage_rate':rate(sum(ident in plan_runs for ident in target_runs),len(target_runs)),
        'task_plan_false_activation_rate':rate(sum(ident in plan_runs for ident in control_runs),len(control_runs)),
        'task_plan_tool_calls':distribution(planning_calls),
        'other_tool_calls':distribution([e['tool_calls'] - n for (_, e), n in zip(runs, planning_calls)]),
        'task_plan_events':dict(task_events),
        'task_gate_continuations':sum(e.get('continued') is True for trace, _ in runs for e in trace['events']
                                      if e['type'] == 'task.gate.checked'),
    }
