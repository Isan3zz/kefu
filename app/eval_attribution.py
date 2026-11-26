"""Assign failure causes from recorded assertion and event evidence."""
from app.eval_layers import events_for


CATEGORIES = {
    'INTENT_ERROR': 'TASK_UNDERSTANDING_ERROR', 'SKILL_SELECTION_ERROR': 'TOOL_SELECTION_ERROR',
    'SKILL_ACTIVATION_ERROR': 'TOOL_SELECTION_ERROR', 'WORKFLOW_VIOLATION': 'PLANNING_ERROR',
    'AUTHORIZATION_VIOLATION': 'POLICY_VIOLATION', 'IDEMPOTENCY_VIOLATION': 'POLICY_VIOLATION',
    'ACTUAL_SAFETY_VIOLATION': 'POLICY_VIOLATION', 'STALE_KNOWLEDGE': 'RETRIEVAL_ERROR',
    'GENERATION_ERROR': 'RESPONSE_ERROR', 'LOOP_BUDGET_ERROR': 'CONTEXT_ERROR',
    'LOOP_TERMINATION_ERROR': 'CONTEXT_ERROR', 'HANDOFF_ERROR': 'PLANNING_ERROR', 'SYSTEM_ERROR': 'UNKNOWN_CAUSE',
}
PRIORITY = ('POLICY_VIOLATION', 'EVALUATOR_ERROR', 'TASK_UNDERSTANDING_ERROR', 'CONTEXT_ERROR', 'TOOL_ARGUMENT_ERROR',
            'TOOL_EXECUTION_ERROR', 'MEMORY_ERROR', 'RETRIEVAL_ERROR', 'RECOVERY_ERROR', 'PLANNING_ERROR',
            'TOOL_SELECTION_ERROR', 'UNKNOWN_OUTCOME', 'GROUNDING_ERROR', 'RESPONSE_ERROR', 'UNKNOWN_CAUSE')
EVIDENCE_TYPES = {'TASK_UNDERSTANDING_ERROR': {'intent'}, 'PLANNING_ERROR': {'workflow.interrupted', 'workflow.commit.started'},
                  'POLICY_VIOLATION': {'policy.denied', 'workflow.commit.started'}, 'TOOL_EXECUTION_ERROR': {'tool.failed'},
                  'RETRIEVAL_ERROR': {'retrieval', 'retrieval.stages'}, 'MEMORY_ERROR': {'memory.context_injected'},
                  'CONTEXT_ERROR': {'loop.terminated', 'context.built'}, 'RECOVERY_ERROR': {'recovery.completed', 'workflow.verified'},
                  'UNKNOWN_OUTCOME': {'workflow.verified', 'invocation.finished'}, 'RESPONSE_ERROR': {'response'},
                  'GROUNDING_ERROR': {'response', 'retrieval'}}


def attribute(case):
    events = events_for(case['traces'])
    by_type = {}
    for check in case['results']:
        if check['passed'] is not False:
            continue
        label = check['failure_category']
        label = 'POLICY_VIOLATION' if check['evaluator'] == 'illegal_attempt' else CATEGORIES.get(label, label)
        label = label if label in PRIORITY else 'UNKNOWN_CAUSE'
        by_type.setdefault(label, []).append(check)
    if case['final_state'].get('execution_outcome') == 'UNKNOWN_OUTCOME':
        by_type.setdefault('UNKNOWN_OUTCOME', [])
    if case['judge'].get('error_code') or any(r['evaluator'] == 'judge_rubric' and r['passed'] is None for r in case['results']):
        by_type.setdefault('EVALUATOR_ERROR', [])
    if not by_type:
        return None
    ranked = [label for label in PRIORITY if label in by_type]
    primary = ranked[0]
    checks = by_type[primary]
    explicit_ids = {event_id for c in checks for event_id in c.get('evidence', {}).get('event_ids', [])}
    if primary == 'POLICY_VIOLATION':
        explicit_ids.update(e['event_id'] for e in case['final_state'].get('authorization_evidence', []))
    if primary == 'TOOL_ARGUMENT_ERROR' and not explicit_ids:
        targets = [item for c in checks if c['evaluator'] == 'tool_arguments'
                   for item in case['original_case'].get('expected', {}).get('tool_arguments', [])]
        explicit_ids.update(e['event_id'] for e in case['tools'] if any(e['name'] == t['name'] for t in targets) and e.get('event_id'))
    matched = [event for event in events if event.get('event_id') in explicit_ids] if explicit_ids else [
        event for event in events if event['type'] in EVIDENCE_TYPES.get(primary, set())]
    phase = checks[0].get('layer') if checks else 'outcome' if primary == 'UNKNOWN_OUTCOME' else 'response'
    return {'primary': primary, 'secondary': ranked[1:], 'stage': phase,
            'event_ids': [event['event_id'] for event in matched if event.get('event_id')],
            'expected': [c.get('evidence', {}).get('expected') for c in checks],
            'actual': [c.get('evidence', {}).get('actual') for c in checks],
            'evidence': [c.get('evidence', {}) for c in checks],
            'tool_results': [e.get('result') for e in matched if e['type'] == 'tool' and 'result' in e],
            'basis': [c['evaluator'] for c in checks], 'needs_human_review': True, 'review_status': 'PENDING_REVIEW'}
