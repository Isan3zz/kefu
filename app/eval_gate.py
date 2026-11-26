"""Compatibility checks and release gates over immutable run artifacts."""
from pydantic import BaseModel, ConfigDict, Field


class GatePolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    core_scenarios: list[str] = Field(default_factory=list)
    max_pass_rate_drop: float | None = Field(default=None, ge=0, le=1)
    max_avg_tool_calls_increase: float | None = Field(default=None, ge=0)
    max_avg_model_calls_increase: float | None = Field(default=None, ge=0)
    max_input_tokens_increase: int | None = Field(default=None, ge=0)
    max_p95_ms_increase: float | None = Field(default=None, ge=0)
    require_judge: bool = False


def compatibility(baseline, candidate):
    def sources(report):
        snapshot = report.get('knowledge_snapshot') or {}
        return {'evaluator': report.get('release', {}).get('evaluator'), 'mode': report.get('mode'),
                'evaluator_sha256':report.get('release', {}).get('evaluator_sha256'),
                'agent_model_config': report.get('agent_model_config'), 'judge_config': report.get('judge_config'),
                'models': {key: value.get('revision') if isinstance(value, dict) else value for key, value in
                           report.get('release', {}).get('models', {}).items()} if report.get('mode') != 'deterministic' else {},
                'knowledge': snapshot.get('snapshot') or snapshot.get('snapshot_id') or report.get('knowledge_corpus_sha256'),
                'retrieval': (report.get('retrieval_mode'), report.get('retrieval_config')),
                'skill_mode': report.get('skill_mode'), 'harness_config': report.get('harness_config'),
                'fault_injection': report.get('inject_unsafe_action')}
    old, new = sources(baseline), sources(candidate)
    return {key: {'baseline': old[key], 'candidate': new[key]} for key in old if old[key] != new[key]}


def gate_reasons(candidate, baseline=None, policy=None):
    policy = policy or GatePolicy()
    blockers, failures = [], []
    ready = candidate.get('release', {}).get('evaluator') == 'v3'
    old = {c['case_id']: c for c in (baseline or {}).get('cases', [])}
    for case in candidate['cases']:
        case_id = case['case_id']
        tags = set(case.get('tags', []))
        prior = old.get(case_id, {})
        scenario = case.get('original_case', {}).get('scenario') or case.get('original_case', {}).get('category')
        state = case.get('final_state', {})
        if 'mandatory' in tags and case['passed'] is not True:
            blockers.append(f'Mandatory case is not passing: {case_id}')
        if ('core' in tags or scenario in policy.core_scenarios) and prior.get('passed') is True and case['passed'] is not True:
            blockers.append(f'Core scenario regression: {case_id}')
        if policy.require_judge and case.get('answer'):
            verdict = case.get('judge', {})
            if verdict.get('passed') is None:
                blockers.append(f'Required Judge verdict unavailable: {case_id}')
            elif not verdict['passed']:
                failures.append(f'Required Judge rubric failed: {case_id}')
        if any(r['passed'] is False and r['failure_category'] in ('ACTUAL_SAFETY_VIOLATION', 'AUTHORIZATION_VIOLATION',
                                                                 'IDEMPOTENCY_VIOLATION', 'STALE_KNOWLEDGE') for r in case['results']):
            blockers.append(f'Hard safety assertion failed: {case_id}')
        if ready and (any(k not in case or case[k] is None for k in ('traces', 'original_case', 'facts', 'layers'))
                      or not case['traces'] or any(not t.get('trace_id') for t in case['traces'])):
            blockers.append(f'Missing evaluation evidence: {case_id}')
        if state.get('execution_outcome') == 'UNKNOWN_OUTCOME' and 'core' in tags:
            blockers.append(f'Core write remains unresolved: {case_id}')
        if any(r['passed'] is False and r['failure_category']=='EVALUATOR_ERROR' for r in case['results']):
            blockers.append(f'Inconsistent evaluation oracle: {case_id}')
    if ready and (not candidate.get('dataset_sha256') or not candidate.get('fixture') or not candidate.get('dataset')
                  or not candidate['cases'] or len(candidate['cases']) != candidate.get('total')
                  or not (candidate.get('knowledge_corpus_sha256') or candidate.get('knowledge_snapshot'))):
        blockers.append('Incomplete run manifest, knowledge snapshot or case coverage')
    if baseline:
        def metric(report, key):
            return report.get('metrics', {}).get(key, {}).get('value')
        limits = [
            ('pass rate decrease', policy.max_pass_rate_drop, metric(candidate, 'end_to_end_task_pass_rate'), metric(baseline, 'end_to_end_task_pass_rate')),
            ('average tool call increase', policy.max_avg_tool_calls_increase, metric(baseline, 'avg_tool_calls_per_task'), metric(candidate, 'avg_tool_calls_per_task')),
            ('average model call increase', policy.max_avg_model_calls_increase, metric(baseline, 'avg_model_calls_per_task'), metric(candidate, 'avg_model_calls_per_task')),
            ('agent input token increase', policy.max_input_tokens_increase, baseline.get('llm_usage', {}).get('input_tokens'), candidate.get('llm_usage', {}).get('input_tokens')),
            ('agent p95 latency increase', policy.max_p95_ms_increase, baseline.get('agent_latency', {}).get('p95'), candidate.get('agent_latency', {}).get('p95')),
        ]
        for label, threshold, before, after in limits:
            if threshold is None:
                continue
            if before is None or after is None:
                blockers.append(f'Required comparison metric unavailable: {label}')
            elif after - before > threshold:
                failures.append(f'{label} {after - before:.4g} exceeds {threshold}')
    return blockers, failures
