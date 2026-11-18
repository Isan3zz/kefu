"""Deterministic aggregation of actual Harness events in the existing eval report."""
def rate(n,d,**extra): return {'numerator':n,'denominator':d,'value':n/d if d else None,'status':'MEASURED' if d else 'NOT_AVAILABLE',**extra}

def harness_metrics(report):
    traces=[t for c in report['cases'] for t in c['traces']]
    events=[e for t in traces for e in t['events']]
    inv=[e for e in events if e['type']=='invocation.finished']
    model_operations={e['operation_id']:e for e in events if e['type'] in ('model.started','model.completed','model.failed') and e.get('operation_id')}
    models=list(model_operations.values())
    # Failed attempts are not failed business calls: group by operation_id.
    operations={e['operation_id']:e for e in events if e['type'] in ('tool.completed','tool.failed') and e.get('operation_id')}
    context=[e for e in events if e['type']=='context.built' and e.get('before_tokens',0)>0]
    return {
        'invocation_success_rate':rate(sum(e['status']=='SUCCEEDED' for e in inv),len(inv)),
        'invocation_status_counts':{status:sum(e['status']==status for e in inv) for status in ('SUCCEEDED','INTERRUPTED','FAILED','TIMED_OUT','CANCELLED','UNKNOWN_OUTCOME')},
        'model_retry_rate':rate(sum(e.get('attempts',1)>1 for e in models),len(models)),
        'model_attempts':sum(t.get('model_attempts',0) for t in traces),
        'tool_error_rate':rate(sum(e['type']=='tool.failed' for e in operations.values()),len(operations)),
        'tool_attempts':sum(t.get('tool_attempts',0) for t in traces),
        'step_limit_hit_rate':rate(sum(e.get('error_code')=='STEP_LIMIT_EXCEEDED' for e in inv),len(inv)),
        'context_compression_ratio':rate(sum(e['before_tokens']-e['after_tokens'] for e in context),sum(e['before_tokens'] for e in context),estimate_source='estimated'),
        'unknown_outcome_count':sum(e['status']=='UNKNOWN_OUTCOME' for e in inv),
        'unauthorized_proposal_count':sum(e['type']=='policy.denied' for e in events),
        'unauthorized_side_effect_count':report['unauthorized_side_effects'],
        'end_to_end_task_pass_rate':rate(report['passed'],sum(c['passed'] is not None for c in report['cases'])),
        'task_unknown_count':sum(c['passed'] is None for c in report['cases']),
    }
