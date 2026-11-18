from collections import Counter
from datetime import datetime, timedelta
import hashlib
import html
import json
from pathlib import Path
import statistics
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from app.agent import Agent
from app.business import Actor, Business
from app.config import settings
from app.dataset import FIXTURE_TIME
from app.db import DB, memory_jobs, messages, now, refund_requests, traces, uid, users
from app.evaluation import FieldEvaluator, case_workspace, compare, persist_run_evidence, release
from app.fakes import FakeModels, FixtureMemoryGateway
from app.harness.metrics import rate
from app.mcp import Gateway, create_mcp
from app.memory.contracts import BATCH_POLICY_VERSION, EXTRACTOR_VERSION, PIPELINE_VERSION, TRIGGER_VERSION, MemoryType, TriggerDecision, TriggerDispatch
from app.observability import TraceRecorder, active_trace, evaluation_scope
from app.rag import RAG, atomic_json

VARIANTS = ('memory-off', 'memory-on', 'extraction-per-turn', 'trigger-only', 'trigger-batch')
POSITIVE = {TriggerDecision.CANDIDATE, TriggerDecision.URGENT}
FIXTURE_VERSION = 'memory-fixture-v1'


class Fact(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    key: str
    content: str
    type: MemoryType = 'preference'


class Turn(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1, max_length=950)
    expected_trigger: TriggerDecision
    expected_memories: list[Fact] = Field(default_factory=list)
    fixture_memories: list[Fact] = Field(default_factory=list)
    advance_seconds: int = Field(default=1, ge=0, le=600)
    action: Literal['opt_out_reopen', 'delete_all', 'pause_retrieval', 'conversation_end'] | None = None


class MemoryCase(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal['1.0'] = '1.0'
    case_id: str
    scenario: str
    turns: list[Turn] = Field(min_length=1, max_length=40)
    query: str = '推荐一款耳机'
    query_as: Literal['owner', 'same_tenant', 'other_tenant'] = 'owner'
    opt_in: bool = True
    expected_stored: list[Fact] = Field(default_factory=list)
    expected_retrieved_keys: list[str] = Field(default_factory=list)
    expected_answer_substrings: list[str] = Field(default_factory=list)
    expected_workflow_status: str | None = None
    expected_batch_reason: str | None = None
    idle_seconds: int | None = Field(default=None, ge=1, le=600)
    mandatory: bool = False

    @model_validator(mode='after')
    def labeled(self):
        if self.query_as != 'owner' and self.expected_retrieved_keys:
            raise ValueError('A foreign principal must not recall the fixture owner’s memories')
        if not set(self.expected_retrieved_keys) <= {fact.key for fact in self.expected_stored}:
            raise ValueError('Recall labels must refer to expected stored facts')
        return self


def load_memory_dataset(path):
    raw = path.read_bytes()
    cases = [MemoryCase.model_validate_json(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError('Memory dataset must have unique case IDs and at least one case')
    return raw, cases


def percentiles(values):
    ordered = sorted(values)
    return {'p50': statistics.median(ordered), 'p95': ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
            'samples': len(ordered)} if ordered else {'p50': None, 'p95': None, 'samples': 0}


async def drain(service):
    if service.config.memory_enabled:
        service.scheduler.schedule_due()
        while await service.worker.process():
            pass


async def run_case(case, variant, config, mode, embedding, run_id):
    with evaluation_scope(run_id, case.case_id), case_workspace() as folder:
        root = Path(folder)
        enabled = variant != 'memory-off'
        cfg = config.model_copy(update={'sqlite_url': 'sqlite:///' + str(root / 'app.db'),
            'checkpoint_db_path': str(root / 'checkpoints.db'), 'chroma_path': str(root / 'chroma'),
            'bm25_index_path': str(root / 'bm25'), 'memory_enabled': enabled, 'memory_job_mode': 'async',
            'harness_trace_enabled': True, 'trace_content_enabled': False,
            'memory_batch_idle_seconds': case.idle_seconds or config.memory_batch_idle_seconds,
            'memory_min_similarity': .15 if embedding == 'fake' else config.memory_min_similarity,
            'memory_batch_max_eligible_messages': 1 if variant in ('extraction-per-turn', 'trigger-only') else config.memory_batch_max_eligible_messages})
        fixture_clock = datetime.fromisoformat(FIXTURE_TIME)
        db = DB(cfg.sqlite_url)
        db.seed(lambda: fixture_clock)
        business = Business(db, lambda: fixture_clock)
        owner = Actor('user_a', 'a')
        with db.tx() as connection:
            connection.execute(users.insert().values(id='memory_same_tenant', tenant_id='a', role='customer'))
        history_id = business.conversation(owner)['id']
        gateway = Gateway(business, cfg, create_mcp(business, cfg.mcp_service_secret, testing=True))
        rag = RAG(db, FakeModels() if embedding == 'fake' else None, cfg)
        agent = Agent(business, gateway, cfg, rag)
        model_mode = 'deterministic' if mode == 'deterministic' else 'agent-real'
        fixtures = {turn.text: [dict(type=fact.type, subject_key=fact.key, content=fact.content, evidence_excerpt=fact.content)
                                for fact in turn.fixture_memories] for turn in case.turns}
        agent.harness.fake_model = FixtureMemoryGateway(cfg, fixtures)
        service = agent.memory
        observations = []
        try:
            if case.opt_in and enabled:
                service.update_settings(owner, auto_write_enabled=True)
            for offset, turn in enumerate(case.turns):
                fixture_clock += timedelta(seconds=turn.advance_seconds, microseconds=1)
                await drain(service)
                message_id = uid('msg-')
                recorder = TraceRecorder(business, owner, history_id, cfg, invocation_id=uid('eval-mem-inv-'),
                                         graph_thread_id='conv:' + history_id, fixture_archived_turn=True)
                source = dict(id=message_id, conversation_id=history_id, role='user', content=turn.text,
                              trace_id=recorder.trace_id, created_at=fixture_clock.isoformat())
                with db.tx() as connection:
                    connection.execute(messages.insert().values(**source))
                observed = service.trigger.check(source, opted_in=case.opt_in, max_bytes=service.policy.max_message_bytes)
                observations.append({'turn': offset, 'expected_trigger': turn.expected_trigger.value,
                                     'expected_memories': [fact.key for fact in turn.expected_memories],
                                     'actual_trigger': observed.decision.value, 'reason_code': observed.reason_code})
                marker = active_trace.set(recorder)
                try:
                    if variant == 'extraction-per-turn' and case.opt_in and observed.reason_code in ('NO_STABLE_SIGNAL', 'NOT_STABLE_ASSERTION'):
                        with db.tx() as connection:
                            consent = service.repository.settings(connection, owner)
                            pending_id, job_id = service.scheduler.enqueue(connection, owner, source, consent['consent_generation'],
                                TriggerDispatch(TriggerDecision.CANDIDATE, 'OFFLINE_PER_TURN'), model_mode, recorder.payload['invocation_id'])
                        service.event('offline_baseline_dispatch', message_id=message_id, pending_id=pending_id, job_id=job_id)
                    else:
                        await service.consider_extraction(principal=owner, conversation_id=history_id,
                                                          last_user_message_id=message_id, mode=model_mode)
                    if turn.action == 'opt_out_reopen':
                        service.update_settings(owner, auto_write_enabled=False)
                        service.update_settings(owner, auto_write_enabled=True)
                    elif turn.action == 'delete_all':
                        await service.delete_all(principal=owner)
                    elif turn.action == 'pause_retrieval':
                        service.update_settings(owner, retrieval_enabled=False)
                    elif turn.action == 'conversation_end':
                        await service.flush_pending(principal=owner, conversation_id=history_id, reason='CONVERSATION_ENDED', wait=False)
                    recorder.finish()
                finally:
                    active_trace.reset(marker)
                await drain(service)
            await service.flush_pending(principal=owner, conversation_id=history_id, reason='CONVERSATION_ENDED')
            principal = {'owner': owner, 'same_tenant': Actor('memory_same_tenant', 'a'),
                         'other_tenant': Actor('user_b', 'b')}[case.query_as]
            query_id = business.conversation(principal)['id']
            started = time.perf_counter()
            response = await agent.harness.invoke_chat(principal=principal, conversation_id=query_id,
                                                        message=case.query, mode=model_mode)
            latency = (time.perf_counter() - started) * 1000
            stored = await service.list_memories(principal=owner, limit=100)
            by_id = {row.memory_id: row.subject_key for row in stored}
            with db.tx() as connection:
                case_traces = [json.loads(row.payload) for row in connection.execute(select(traces.c.payload)).all()]
                refunds = len(connection.execute(select(refund_requests.c.id)).all())
                jobs = list(connection.execute(select(memory_jobs)).mappings())
            raw_trace = next(row for row in case_traces if row['trace_id'] == response['trace_id'])
            injection_ids = {ident for event in raw_trace['events'] if event['type'] == 'memory.context_injected'
                             for ident in event.get('memory_ids', [])}
            retrieved_ids = {ident for event in raw_trace['events'] if event['type'] == 'memory.search_completed'
                             for ident in event.get('memory_ids', [])}
            actual = {'stored_facts': sorted((row.subject_key, row.content, row.memory_type) for row in stored),
                      'retrieved_keys': sorted(by_id.get(ident, 'unexpected-id') for ident in retrieved_ids),
                      'injected_keys': sorted(by_id.get(ident, 'unexpected-id') for ident in injection_ids),
                      'foreign_recall_count': len(injection_ids | retrieved_ids) if case.query_as != 'owner' else 0,
                      'unauthorized_write_count': refunds, 'workflow_status': (response.get('pending_confirmation') or {}).get('status'),
                      'task_success': (not response.get('error_code') and all(fragment in response['answer'] for fragment in case.expected_answer_substrings)
                                       and set(case.expected_retrieved_keys) <= {by_id.get(ident) for ident in injection_ids}),
                      'batch_reasons': sorted({row['flush_reason'] for row in jobs}),
                      'failed_jobs': sum(row['status'] == 'FAILED' for row in jobs),
                      'cancelled_jobs': sum(row['status'] == 'CANCELLED' for row in jobs),
                      'duplicate_active_facts': len(stored) - len({row.subject_key for row in stored}),
                      'query_id': query_id, 'write_conversation_id': history_id}
            gold = {'stored_facts': sorted((fact.key, fact.content, fact.type) for fact in case.expected_stored) if enabled else [],
                    'retrieved_keys': sorted(case.expected_retrieved_keys) if enabled else [],
                    'task_success': True, 'foreign_recall_count': 0, 'unauthorized_write_count': 0,
                    'workflow_status': case.expected_workflow_status, 'failed_jobs': 0, 'duplicate_active_facts': 0}
            evaluators = {'stored_facts': 'MEMORY_CONFLICT_UNRESOLVED' if case.scenario == 'conflict' else 'MEMORY_EXTRACTION_ERROR',
                          'retrieved_keys': 'MEMORY_STALE', 'task_success': 'MEMORY_CONTEXT_INJECTION',
                          'foreign_recall_count': 'AUTHORIZATION_VIOLATION', 'unauthorized_write_count': 'WORKFLOW_VIOLATION',
                          'workflow_status': 'WORKFLOW_VIOLATION', 'failed_jobs': 'MEMORY_EXTRACTION_ERROR',
                          'duplicate_active_facts': 'MEMORY_DUPLICATE'}
            if case.expected_batch_reason and variant in ('memory-on', 'trigger-batch'):
                gold['expected_flush'] = True
                actual['expected_flush'] = case.expected_batch_reason in actual['batch_reasons']
                evaluators['expected_flush'] = 'MEMORY_BATCH_FLUSH_TIMEOUT'
            label = {'case_id': case.case_id, 'expected': gold}
            results = [FieldEvaluator(field, category).evaluate(label, raw_trace, actual).model_dump()
                       for field, category in evaluators.items()]
            return {'case_id': case.case_id, 'scenario': case.scenario, 'passed': all(item['passed'] is True for item in results),
                    'tags': ['mandatory'] if case.mandatory else [], 'results': results, 'traces': case_traces,
                    'trace_ids': [row['trace_id'] for row in case_traces], 'final_state': actual, 'trigger': observations,
                    'gold_facts': sorted((fact.key, fact.content, fact.type) for fact in case.expected_stored),
                    'gold_retrieval': case.expected_retrieved_keys, 'latency_ms': latency,
                    'embedding_provenance': rag.provenance(), 'similarity_threshold': cfg.memory_min_similarity,
                    'batch_fixture_policy': {'eligible_count': service.policy.eligible_count, 'idle_seconds': service.policy.idle_seconds,
                                             'max_wait_seconds': service.policy.max_wait_seconds}}
        finally:
            await agent.close()
            db.engine.dispose()


def summarize(cases):
    labels = [(case['case_id'], turn) for case in cases for turn in case['trigger']]
    triggered = [(case_id, turn) for case_id, turn in labels if turn['actual_trigger'] in POSITIVE]
    wanted = [(case_id, turn) for case_id, turn in labels if turn['expected_trigger'] in POSITIVE]
    hits = sum(turn['expected_trigger'] in POSITIVE for _, turn in triggered)
    all_events = [event for case in cases for trace in case['traces'] for event in trace['events']]
    extraction = [event for event in all_events if event['type'] in ('memory.extracted', 'memory.batch_failed')]
    operations = {event['operation_id']: event for event in all_events if event['type'] in ('model.completed', 'model.failed')
                  and event.get('purpose') == 'memory_extract' and event.get('operation_id')}
    attempts = sum(event['attempts'] for event in operations.values())
    batches = [event for event in all_events if event['type'] == 'memory.batch_completed']
    token_events = [event for event in all_events if event['type'] == 'memory.context_injected']
    total_expected = sum(len(case['gold_facts']) for case in cases)
    total_stored = sum(len(case['final_state']['stored_facts']) for case in cases)
    facts_found = sum(len(set(case['gold_facts']) & set(case['final_state']['stored_facts'])) for case in cases)
    retrieval_found = sum(len(set(case['gold_retrieval']) & set(case['final_state']['retrieved_keys'])) for case in cases)
    metrics = {'trigger_recall': rate(hits, len(wanted)), 'trigger_precision': rate(hits, len(triggered)),
               'trigger_accuracy': rate(sum(turn['expected_trigger'] == turn['actual_trigger'] for _, turn in labels), len(labels)),
               'extraction_recall': rate(facts_found, total_expected), 'extraction_precision': rate(facts_found, total_stored),
               'recall_at_k': rate(retrieval_found, sum(len(case['gold_retrieval']) for case in cases)),
               'precision_at_k': rate(retrieval_found, sum(len(case['final_state']['retrieved_keys']) for case in cases)),
               'duplicate_rate': rate(sum(case['final_state']['duplicate_active_facts'] for case in cases), total_stored),
               'forgetting_success': rate(sum(case['passed'] for case in cases if case['scenario'] == 'delete'),
                                          sum(case['scenario'] == 'delete' for case in cases)),
               'grounded_task_success': rate(sum(case['final_state']['task_success'] for case in cases), len(cases)),
               'personalization_task_success': rate(sum(case['final_state']['task_success'] for case in cases if case['gold_retrieval']),
                                                    sum(bool(case['gold_retrieval']) for case in cases)),
               'conflict_update_accuracy': rate(sum(case['passed'] for case in cases if case['scenario'] == 'conflict'),
                                                sum(case['scenario'] == 'conflict' for case in cases)),
               'extractor_calls_per_100_turns': rate(attempts * 100, len(labels), extractor_attempts=attempts,
                                                    scope='write_conversation_user_turns'),
               'average_batch_size': rate(sum(event['batch_size'] for event in batches), len(batches))}
    return metrics, {'total_user_turns': len(labels), 'extractor_calls': attempts, 'successful_calls': sum(
                event['type'] == 'model.completed' for event in operations.values()),
             'candidate_count': sum(event.get('candidate_count', 0) for event in extraction),
             'failed_attempts': attempts - sum(event['type'] == 'model.completed' for event in operations.values()),
             'failed_jobs': sum(case['final_state']['failed_jobs'] for case in cases),
             'cancelled_jobs': sum(case['final_state']['cancelled_jobs'] for case in cases),
             'isolation_violations': sum(case['final_state']['foreign_recall_count'] for case in cases),
             'unauthorized_business_writes': sum(case['final_state']['unauthorized_write_count'] for case in cases),
             'estimated_extractor_input_tokens': sum(event.get('estimated_input_tokens', 0) for event in extraction),
             'estimated_injected_tokens': sum(event.get('estimated_tokens', 0) for event in token_events),
             'provider_tokens': ({'input': sum(event['input_tokens'] for event in operations.values()),
                                  'output': sum(event['output_tokens'] for event in operations.values())}
                                 if operations and all(event.get('usage_source') == 'provider' for event in operations.values()) else None),
             'flush_delay_ms': percentiles([delay for event in batches for delay in event['flush_delays_ms']]),
             'search_latency_ms': percentiles([event['duration_ms'] for event in all_events if event['type'] == 'memory.search_completed']),
             'false_negatives': [{'case_id': case_id, **turn} for case_id, turn in wanted if turn['actual_trigger'] not in POSITIVE],
             'false_positives': [{'case_id': case_id, **turn} for case_id, turn in triggered if turn['expected_trigger'] not in POSITIVE]}


async def run_memory(dataset, variants=('trigger-batch',), mode='deterministic', embedding='fake', config=settings,
                     output_root=Path('data/eval_runs'), case_id=None):
    raw, cases = load_memory_dataset(Path(dataset))
    if case_id:
        cases = [case for case in cases if case.case_id == case_id]
    if not cases or not variants or not set(variants) <= set(VARIANTS):
        raise ValueError('Select valid memory variants and at least one matching case')
    if mode not in ('deterministic', 'agent-real') or embedding not in ('fake', 'local'):
        raise ValueError('Invalid model or embedding mode')
    if mode == 'agent-real' and not all((config.llm_base_url, config.llm_api_key, config.llm_model)):
        raise ValueError('Real mode requires explicitly configured LLM_BASE_URL, LLM_API_KEY and LLM_MODEL')
    run_id = uid('mem-eval-')
    directory = Path(output_root) / run_id
    directory.mkdir(parents=True)
    snapshot = release(config)
    report = {'run_id': run_id, 'mode': mode, 'embedding': embedding, 'created_at': now(), 'runs': {}, 'comparisons': {}}
    for variant in variants:
        results = [await run_case(case, variant, config, mode, embedding, run_id + '-' + variant) for case in cases]
        metrics, detail = summarize(results)
        failures = Counter(item['failure_category'] for case in results for item in case['results'] if item['failure_category'])
        if detail['false_negatives']:
            failures['MEMORY_TRIGGER_FALSE_NEGATIVE'] = len(detail['false_negatives'])
        result = {'run_id': run_id + '-' + variant, 'dataset_sha256': hashlib.sha256(raw).hexdigest(),
                  'variant': variant, 'fixture': {'version': FIXTURE_VERSION, 'clock': FIXTURE_TIME},
                  'release': {**snapshot, 'memory_pipeline_version': PIPELINE_VERSION,
                              'trigger_rule_version': TRIGGER_VERSION, 'batch_policy_version': BATCH_POLICY_VERSION,
                              'extractor_version': EXTRACTOR_VERSION,
                              'extraction_model': 'FixtureMemoryGateway' if mode == 'deterministic' else config.memory_extraction_model or config.llm_model,
                              'embedding': results[0]['embedding_provenance']['embedding'],
                              'effective_similarity_threshold': results[0]['similarity_threshold'],
                              'memory_config': config.model_dump(include={key for key in type(config).model_fields if key.startswith('memory_')})},
                  'cases': results, 'passed': sum(case['passed'] for case in results), 'total': len(results),
                  'metrics': metrics, 'failure_categories': dict(failures), 'memory': detail,
                  'latency': percentiles([case['latency_ms'] for case in results])}
        result['gate'] = 'BLOCKED' if detail['isolation_violations'] or detail['unauthorized_business_writes'] or any(
            not case['passed'] for case in results if 'mandatory' in case['tags']) else 'PASS'
        output = directory / variant
        output.mkdir()
        persist_run_evidence(output, result)
        atomic_json(output / 'report.json', result)
        report['runs'][variant] = result
    pairs = [('memory-off', 'memory-on'), ('extraction-per-turn', 'trigger-only'),
             ('extraction-per-turn', 'trigger-batch'), ('trigger-only', 'trigger-batch')]
    for baseline, target in pairs:
        if baseline in report['runs'] and target in report['runs']:
            report['comparisons'][baseline + '__' + target] = compare(report['runs'][baseline], report['runs'][target])
    summary = [f'# {run_id}', f'Model mode: `{mode}`; embedding: `{embedding}`. Fake results are fixture checks, not production quality or cost evidence.',
               'Flush delays use the fixture clock; query latency uses wall-clock time. Provider tokens are unavailable for fakes.']
    for name, run in report['runs'].items():
        summary.append(f'## {name}: {run["passed"]}/{run["total"]} passed, gate {run["gate"]}')
        summary.extend(f'- {key}: {metric["value"]} ({metric["numerator"]}/{metric["denominator"]})' for key, metric in run['metrics'].items())
        summary.extend(f'- {key}: {value}' for key, value in run['memory'].items())
        summary.append(f'- Failure categories: {run["failure_categories"]}')
    for name, result in report['comparisons'].items():
        summary.append(f'## {name}: {result["gate"]}\n- Metric deltas: {result["metric_deltas"]}\n- Case changes: {result["changes"]}')
    markdown = '\n\n'.join(summary) + '\n'
    (directory / 'report.md').write_text(markdown, encoding='utf-8')
    (directory / 'report.html').write_text('<meta charset="utf-8"><pre>' + html.escape(markdown) + '</pre>', encoding='utf-8')
    atomic_json(directory / 'report.json', report)
    return report
