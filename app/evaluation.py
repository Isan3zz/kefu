import hashlib
import html
import json
import subprocess
import tempfile
import time
import sqlite3
import platform
from pathlib import Path
from collections import Counter
from filelock import FileLock
from pydantic import BaseModel
from sqlalchemy import select,func
from app.config import settings,ROOT
from app.db import DB,refund_requests,traces,uid,now
from app.business import Business,Actor,DomainError
from app.mcp import Gateway,create_mcp,TOOL_SCHEMA_VERSION
from app.agent import Agent
from app.fakes import FakeModels
from app.rag import RAG,atomic_json,model_manifest,CHUNKER_VERSION
from app.observability import TraceRecorder, clean, evaluation_scope
from app.dataset import load_dataset, FIXTURE_TIME, FIXTURE_VERSION
from app.eval_attribution import attribute
from app.eval_evidence import HOLDOUT_FIXTURE, collect_facts, displayed_turn, readonly_snapshot, seed_holdout
from app.eval_gate import GatePolicy, compatibility, gate_reasons
from app.eval_judge import RUBRIC, judge, judge_config, judge_evidence
from app.eval_layers import case_resolution, evaluate_layers, events_for, layer_for, logical_tools, normalized, summarize_layers
from app.evidence import audit_tool_reads,audit_citations,audit_retrieval_scope
from app.harness.metrics import harness_metrics, rate
from datetime import datetime
from contextlib import AsyncExitStack, contextmanager
from typing import Protocol
from dataclasses import dataclass
from decimal import Decimal

@contextmanager
def case_workspace():
    folder=tempfile.TemporaryDirectory(prefix='kefu-eval-')
    try: yield folder.name
    finally:
        # Chroma's native workers can release their final Windows handle just after close().
        # Retry only cleanup of this exact TemporaryDirectory; never ignore cleanup failure.
        for attempt in range(5):
            try:
                folder.cleanup()
                break
            except PermissionError:
                if attempt==4: raise
                time.sleep(.05*2**attempt)

class EvalResult(BaseModel):
    case_id:str
    evaluator:str
    passed:bool|None
    score:float|None=None
    failure_category:str|None=None
    reason:str
    evidence:dict={}

class Evaluator(Protocol):
    def evaluate(self,case:dict,trace:dict,final_state:dict)->EvalResult: ...

@dataclass(frozen=True)
class FieldEvaluator:
    field:str
    failure_category:str
    def evaluate(self,case,trace,final_state):
        expected=case.get('expected',{})
        if self.field not in expected:
            return EvalResult(case_id=case['case_id'],evaluator=self.field,passed=None,reason='NOT_APPLICABLE')
        actual=final_state.get(self.field); wanted=expected[self.field]; ok=actual==wanted
        return EvalResult(case_id=case['case_id'],evaluator=self.field,passed=ok,score=float(ok),
                          failure_category=None if ok else self.failure_category,
                          reason='matched' if ok else f'expected {wanted!r}, got {actual!r}',
                          evidence={'actual':actual,'expected':wanted})

def summarize_llm_usage(events,config=settings):
    measured=[e for e in events if all(type(e.get('usage',{}).get(key)) is int and e['usage'][key]>=0 for key in ('prompt_tokens','completion_tokens'))]
    complete=bool(events) and len(measured)==len(events)
    observed_in=sum(e['usage']['prompt_tokens'] for e in measured)
    observed_out=sum(e['usage']['completion_tokens'] for e in measured)
    usage={'calls':len(events),'measured_calls':len(measured),'input_tokens':observed_in if complete else None,'output_tokens':observed_out if complete else None,
           'status':'MEASURED' if complete else 'PARTIAL' if measured else 'NOT_PROVIDED' if events else 'NOT_EVALUATED'}
    cost={'status':'NOT_EVALUATED','value':None,'currency':config.llm_cost_currency,'scope':'main_llm_only'}
    if complete and config.llm_input_cost_per_million is not None and config.llm_output_cost_per_million is not None:
        amount=(Decimal(observed_in)*Decimal(config.llm_input_cost_per_million)+Decimal(observed_out)*Decimal(config.llm_output_cost_per_million))/Decimal(1_000_000)
        cost.update(status='CALCULATED',value=str(amount),input_price_per_million=str(config.llm_input_cost_per_million),output_price_per_million=str(config.llm_output_cost_per_million))
    return usage,cost

def persist_run_evidence(directory,report):
    """Keep trace JSONL plus a durable SQLite index outside disposable business fixtures."""
    all_traces=[trace for case in report['cases'] for trace in case['traces']]
    trace_file=directory/'traces.jsonl'
    trace_file.write_text(''.join(json.dumps(trace,ensure_ascii=False)+'\n' for trace in all_traces),encoding='utf-8')
    connection=sqlite3.connect(directory/'index.sqlite')
    connection.execute('PRAGMA foreign_keys=ON')
    try:
        with connection:
            connection.executescript('''
                CREATE TABLE cases (case_id TEXT PRIMARY KEY, passed INTEGER, results_json TEXT NOT NULL, final_state_json TEXT NOT NULL);
                CREATE TABLE traces (trace_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(case_id), payload_json TEXT NOT NULL);
            ''')
            for case in report['cases']:
                connection.execute('INSERT INTO cases VALUES (?,?,?,?)',(case['case_id'],int(case['passed']) if case['passed'] is not None else None,json.dumps(case['results'],ensure_ascii=False),json.dumps(case['final_state'],ensure_ascii=False)))
                for trace in case['traces']:
                    connection.execute('INSERT INTO traces VALUES (?,?,?)',(trace['trace_id'],case['case_id'],json.dumps(trace,ensure_ascii=False)))
    finally: connection.close()

def read_eval_trace(run_id,trace_id):
    directory=(Path('data/eval_runs')/run_id).resolve()
    if not directory.is_relative_to(Path('data/eval_runs').resolve()): raise ValueError('Run path escapes evaluation storage')
    index=directory/'index.sqlite'
    if not index.is_file(): raise DomainError('NOT_FOUND','Evaluation trace index not found')
    connection=sqlite3.connect(index)
    try: row=connection.execute('SELECT payload_json FROM traces WHERE trace_id=?',(trace_id,)).fetchone()
    finally: connection.close()
    if row is None: raise DomainError('NOT_FOUND')
    return json.loads(row[0])

def assertions(case,trace,final):
    expected=case.get('expected',{}).copy(); results=[]
    for field, legacy in [('required_actions', 'required_tools'), ('forbidden_actions', 'forbidden_tools'),
                          ('dependency_constraints', 'tool_order'), ('argument_constraints', 'tool_arguments')]:
        if field in case: expected.pop(legacy, None)
    def check(name,actual,wanted,category):
        ok=actual==wanted
        results.append(EvalResult(case_id=case['case_id'],evaluator=name,passed=ok,score=float(ok),failure_category=None if ok else category,reason='matched' if ok else f'expected {wanted!r}, got {actual!r}',evidence={'actual':actual,'expected':wanted}).model_dump())
    for name,category in [('intent','INTENT_ERROR'),('skill','SKILL_SELECTION_ERROR'),('workflow_status','WORKFLOW_VIOLATION'),('refund_request_count_delta','WORKFLOW_VIOLATION'),('final_business_status','WORKFLOW_VIOLATION'),('error_code','TOOL_EXECUTION_ERROR'),('eligibility_reason','WORKFLOW_VIOLATION'),('shipping_order_id','TOOL_ARGUMENT_ERROR'),('ticket_count_delta','HANDOFF_ERROR'),('duplicate_refund_count','IDEMPOTENCY_VIOLATION'),('unauthorized_action_count','AUTHORIZATION_VIOLATION')]:
        if name in expected:
            if final.get('execution_outcome')=='UNKNOWN_OUTCOME' and name in ('workflow_status','refund_request_count_delta','final_business_status','ticket_count_delta'):
                results.append(EvalResult(case_id=case['case_id'],evaluator=name,passed=None,reason='UNKNOWN_OUTCOME: business completion awaits reconciliation',evidence={'observed_snapshot':final.get(name)}).model_dump())
            else: results.append(FieldEvaluator(name,category).evaluate(case,trace,final).model_dump())
    tools=[t['name'] for t in trace.get('tool_events',[])]
    unexpected=[e for e in trace.get('tool_events',[]) if e.get('status')=='error' and e.get('error_code') not in ('UNKNOWN_OUTCOME',expected.get('error_code'))]
    if case.get('schema_version')=='2.0':
        # Required successful actions and business results evaluate recovery independently.
        unexpected=[]
    if unexpected: check('tool_execution',False,True,'TOOL_EXECUTION_ERROR')
    submit_count=final.get('workflow_submission_count',tools.count('submit_refund_request'))
    for key in ('required_tools','forbidden_tools'):
        if key in expected:
            if key=='required_tools':
                ok=set(expected[key]).issubset(tools)
            else:
                ok=not set(expected[key])&set(tools) and not ('submit_refund_request' in expected[key] and submit_count)
            check(key,ok,True,'TOOL_SELECTION_ERROR' if key=='required_tools' else 'WORKFLOW_VIOLATION')
    if 'tool_order' in expected:
        cursor=0; ordered=True
        for name in expected['tool_order']:
            try: cursor=tools.index(name,cursor)+1
            except ValueError: ordered=False; break
        check('tool_order',ordered,True,'TOOL_SELECTION_ERROR')
    if 'tool_arguments' in expected:
        ok=all(any(e['name']==wanted['name'] and all(e.get('args',{}).get(k)==v for k,v in wanted['args'].items()) for e in trace.get('tool_events',[])) for wanted in expected['tool_arguments'])
        check('tool_arguments',ok,True,'TOOL_ARGUMENT_ERROR')
    if 'submit_tool_count_max' in expected: check('submit_tool_count_max',submit_count<=expected['submit_tool_count_max'],True,'IDEMPOTENCY_VIOLATION')
    if 'terminal_reason' in expected: check('terminal_reason',final.get('terminal_reason'),expected['terminal_reason'],'LOOP_TERMINATION_ERROR')
    if 'max_iterations' in expected: check('max_iterations',final.get('iterations',0)<=expected['max_iterations'],True,'LOOP_BUDGET_ERROR')
    if 'activated_skills' in expected: check('activated_skills',sorted(final.get('activated_skills',[])),sorted(expected['activated_skills']),'SKILL_ACTIVATION_ERROR')
    if final.get('illegal_attempt'):
        check('illegal_attempt',True,False,'WORKFLOW_VIOLATION')
    for invariant,category in [('unauthorized_action_count','AUTHORIZATION_VIOLATION'),('duplicate_refund_count','IDEMPOTENCY_VIOLATION')]:
        if invariant in final and invariant not in expected:
            check(invariant,final[invariant],0,category)
        elif final.get(invariant) and final[invariant] == expected.get(invariant):
            check(invariant+'_invariant',final[invariant],0,category)
    if final.get('stale_knowledge_count'): check('evidence_currentness',final['stale_knowledge_count'],0,'STALE_KNOWLEDGE')
    if final.get('invalid_citation_count'): check('evidence_integrity',final['invalid_citation_count'],0,'RETRIEVAL_ERROR')
    gold=set(expected.get('gold_chunks',[]))
    if gold:
        ids=final.get('retrieved_ids',[]); recall=len(set(ids[:5])&gold)/len(gold); mrr=next((1/i for i,c in enumerate(ids[:10],1) if c in gold),0)
        for metric,value in [('Recall@5',recall),('MRR@10',mrr)]:
            results.append(EvalResult(case_id=case['case_id'],evaluator=metric,passed=value>0,score=value,failure_category=None if value>0 else 'RETRIEVAL_ERROR',reason='gold chunk comparison',evidence={'retrieved':ids,'gold':sorted(gold)}).model_dump())
    return results

def response_assertions(case_id, step_number, step, response):
    answer = response.get('answer', '')
    plain_answer = normalized(answer)
    required = step.get('answer_contains', [])
    alternatives = step.get('answer_contains_any', [])
    if required or alternatives:
        missing = [fragment for fragment in required if normalized(fragment) not in plain_answer]
        unmatched_alternatives = alternatives if not any(normalized(fragment) in plain_answer for fragment in alternatives) else []
        passed = not (missing or unmatched_alternatives)
        yield EvalResult(case_id=case_id, evaluator='step_answer_contains', passed=passed,
                         score=float(passed), failure_category=None if passed else 'GENERATION_ERROR',
                         reason='matched' if passed else 'Required answer text missing',
                         evidence={'step':step_number, 'missing':missing, 'unmatched_alternatives':unmatched_alternatives,
                                   'answer':answer}).model_dump()
    if step.get('citation_required'):
        cited = [row['chunk_id'] for row in response.get('citations', []) if f"[{row['chunk_id']}]" in answer]
        yield EvalResult(case_id=case_id, evaluator='step_citation_required', passed=bool(cited),
                         score=float(bool(cited)), failure_category=None if cited else 'RETRIEVAL_ERROR',
                         reason='matched' if cited else 'No retrieved citation in the answer',
                         evidence={'step':step_number, 'cited_chunk_ids':cited}).model_dump()


EVALUATOR_VERSION = 'v3'


def evaluator_sha256():
    digest=hashlib.sha256()
    for name in ('dataset.py','evaluation.py','eval_attribution.py','eval_evidence.py','eval_gate.py','eval_judge.py','eval_layers.py','evidence.py'):
        digest.update(name.encode())
        digest.update((ROOT/'app'/name).read_bytes())
    return digest.hexdigest()
def release(config):
    from app.skills import SkillManager
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: commit='UNVERSIONED'
    sm=SkillManager()
    source=hashlib.sha256()
    for path in sorted((ROOT/'app').rglob('*.py')):
        source.update(str(path.relative_to(ROOT)).encode()); source.update(path.read_bytes())
    lock=ROOT/'uv.lock'
    from app.agent_loop import PROMPT_VERSION
    return dict(git_commit=commit,source_sha256=source.hexdigest(),uv_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,llm=config.llm_model,temperature=0,prompt=PROMPT_VERSION,prompts={'loop':PROMPT_VERSION},skills={n:sm.load(n)['hash'] for n in sm.catalog},tool_schema=TOOL_SCHEMA_VERSION,models=model_manifest(),policy='P-REFUND-v1',evaluator=EVALUATOR_VERSION,evaluator_sha256=evaluator_sha256(),chunker=CHUNKER_VERSION)

async def run(dataset,mode='deterministic',agent_release='local',config=settings,retrieval_mode='hybrid_rerank',skill_mode='selected',inject_unsafe_action=False,*,enable_judge=False):
    raw,cases=load_dataset(dataset)
    if mode not in ('deterministic','rag-real','agent-real'): raise ValueError('Unknown evaluation mode')
    if mode=='rag-real' and any(c['category']!='retrieval' for c in cases): raise ValueError('rag-real requires retrieval cases')
    if mode=='agent-real': Agent(Business(DB(config.sqlite_url)),None,config).llm()
    if enable_judge and mode != 'agent-real': raise ValueError('Inline Judge requires agent-real; score existing runs with judge mode')
    report=dict(run_id=uid('run-'),created_at=now(),mode=mode,agent_release=agent_release,dataset_sha256=hashlib.sha256(raw).hexdigest(),release=release(config),retrieval_mode=retrieval_mode,skill_mode=skill_mode,cases=[])
    split=cases[0].get('dataset_split', 'regression' if Path(dataset).name=='regression.jsonl' else 'fault_injection' if Path(dataset).name=='fault_baseline.jsonl' else 'development')
    report['dataset']={'name':Path(dataset).name,'split':split,'schema_versions':sorted({case['schema_version'] for case in cases}),
                       'sha256':report['dataset_sha256'],'case_count':len(cases)}
    report['provenance']={'simulated_commerce':True,'real_agent':mode=='agent-real','real_local_models_requested':mode!='deterministic',
                          'fake_or_mock':mode=='deterministic','fault_injection':inject_unsafe_action or split=='fault_injection',
                          'independent_holdout':split=='holdout','human_reviewed_cases':sum(bool(case.get('review',{}).get('confirmed')) for case in cases)}
    report['environment']={'python':platform.python_version(),'platform':platform.platform(),'uv_lock_sha256':report['release']['uv_lock_sha256']}
    report['agent_model_config']={'adapter':'FakeModelGateway'} if mode=='deterministic' else {
        name:getattr(config,name) for name in ('llm_model','llm_timeout_seconds','model_json_mode','model_max_output_tokens')} if mode=='agent-real' else None
    report['judge_config']={**judge_config(config),'requested':enable_judge}
    report['retrieval_config']={key:getattr(config,key) for key in ('retrieval_dense_k','retrieval_bm25_k','retrieval_rrf_k','retrieval_rerank_k','retrieval_context_k','embed_batch_size','rerank_batch_size','embed_max_seq_length','rerank_max_length','model_device')}
    report['retrieval_config']['evaluation_top_k']=10
    fixture_report={'version':FIXTURE_VERSION,'clock':FIXTURE_TIME}
    boundary_selectors=sorted({case['fixture'] for case in cases if case['fixture'].startswith('orders_boundary:')})
    if boundary_selectors:
        from app.order_fixtures import ORDER_BUNDLE_PATH, load_order_fixture_data
        fixture_report['order_boundary_version']=load_order_fixture_data()['fixture_version']
        fixture_report['order_boundary_sha256']=hashlib.sha256(ORDER_BUNDLE_PATH.read_bytes()).hexdigest()
        fixture_report['selectors']=boundary_selectors
    report['fixture']=fixture_report
    report['harness_config']={name:getattr(config,name) for name in ('harness_enabled','context_strategy','model_context_window','context_max_input_tokens','context_reserved_output_tokens','context_safety_margin_tokens','context_compact_threshold','context_recent_message_limit','model_max_retries','model_max_token_budget','tool_read_max_retries','tool_timeout_seconds','tool_max_concurrency','agent_max_steps','agent_timeout_seconds')}
    for case in cases:
        start=time.perf_counter(); final={}; error=None; evidence=[]
        step_results=[]
        review_turns=[]
        citation_counts={'stale_knowledge_count':0,'invalid_citation_count':0,'unauthorized_citation_count':0}; forbidden_evidence_actions=0
        async with AsyncExitStack() as resources:
            resources.enter_context(evaluation_scope(report['run_id'],case['case_id']))
            folder=resources.enter_context(case_workspace())
            root=Path(folder)
            cfg=config.model_copy(update={'sqlite_url':'sqlite:///'+str(root/'app.db'),'checkpoint_db_path':str(root/'checkpoints.db'),'chroma_path':str(root/'chroma'),'bm25_index_path':str(root/'bm25'),'agent_release_id':agent_release})
            clock=lambda:datetime.fromisoformat(FIXTURE_TIME)
            db=DB(cfg.sqlite_url); resources.callback(db.engine.dispose)
            if case['fixture'].startswith('orders_boundary:'):
                from app.order_fixtures import seed_order_scenario
                seed_order_scenario(db,case['fixture'],clock)
            else:
                if case['fixture']=='holdout_commerce_v1':
                    seed_holdout(db,clock)
                    fixture_report['holdout_fixture_sha256']=hashlib.sha256(HOLDOUT_FIXTURE.read_bytes()).hexdigest()
                else:
                    db.seed(clock)
                # The 100-case E2E suite needs the full simulated catalog.
                # Do not change the fixtures used by existing evaluation cases.
                if set(case.get('tags', [])) & {'e2e-100', 'product-fixture-required'}:
                    from app.product_fixtures import import_product_fixtures, merge_missing_static_attributes
                    import_product_fixtures(db)
                    merge_missing_static_attributes(db)
                    from app.product_fixtures import PRODUCT_FIXTURE_PATH
                    fixture_report['product_sha256']=hashlib.sha256(PRODUCT_FIXTURE_PATH.read_bytes()).hexdigest()
            from app.db import users
            principal_id=case.get('actor_user_id') or 'user_a'
            with db.tx() as connection:
                principal=connection.execute(select(users).where(users.c.id==principal_id)).mappings().one()
            b=Business(db,clock); a=Actor(principal['id'],principal['tenant_id'],principal['role']); conv=b.conversation(a)['id']
            from app.db import tickets,orders
            with db.tx() as connection:
                owned_order_ids=set(connection.scalars(select(orders.c.id).where(orders.c.user_id==a.user_id,orders.c.tenant_id==a.tenant_id)))
                refund_ids_before=set(connection.scalars(select(refund_requests.c.id)))
                refund_count_before=len(refund_ids_before)
                readonly_before=readonly_snapshot(connection)
            if mode=='deterministic':
                rag=RAG(db,FakeModels(),cfg)
                resources.callback(rag.close)
                fixture_manifest=rag.build(ROOT/'knowledge')
                report.setdefault('knowledge_corpus_sha256',fixture_manifest['corpus_hash'])
            else:
                rag=RAG(DB(config.sqlite_url),config=config)
                resources.callback(rag.db.engine.dispose)
                resources.callback(rag.close)
                try:
                    report.setdefault('knowledge_snapshot',rag.active_manifest())
                except FileNotFoundError:
                    report.setdefault('knowledge_snapshot',None)
            from app.retrieval_mcp import RetrievalService, create_retrieval_mcp
            retrieval = create_retrieval_mcp(RetrievalService(b,cfg,rag=rag),cfg)
            gateway=Gateway(b,cfg,create_mcp(b,cfg.business_secret,testing=True),retrieval_server=retrieval)
            agent=Agent(b,gateway,cfg,rag); pending={}; response={}; workflow_id=None
            resources.push_async_callback(agent.close)
            steps=case.get('steps') or [{'type':'chat','content':m['content']} for m in case.get('messages',[])]
            try:
                for step_number, step in enumerate(steps, 1):
                    typ=step['type']
                    if typ=='chat':
                        response=await agent.chat(a,conv,step['content'],'real' if mode=='agent-real' else 'deterministic',skill_mode)
                        audited=audit_citations(rag.db,response.get('citations',[]),a)
                        for key,value in audited.items(): citation_counts[key]+=value
                        forbidden_evidence_actions+=int(audited['unauthorized_citation_count']>0)
                        if response.get('error_code'): final['error_code']=response['error_code']
                        if response.get('pending_confirmation'): pending=response['pending_confirmation']; workflow_id=pending['workflow_id']
                    elif typ=='confirm':
                        try:
                            result=await agent.confirm(a,pending['workflow_id'],pending['confirm_nonce'])
                        except DomainError as exc:
                            if hasattr(exc, 'workflow_result'):
                                review_turns.append(displayed_turn(step_number, typ, {'answer':exc.message,'trace_id':getattr(exc,'trace_id',None),'refund_workflow_result':exc.workflow_result}, b,a,workflow_id))
                            raise
                        response['answer']=result['answer']; response['citations']=[]
                    elif typ=='expect_interrupt':
                        if not pending: raise AssertionError('No pending confirmation')
                        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
                        async with AsyncSqliteSaver.from_conn_string(cfg.checkpoint_db_path) as saver:
                            state=await agent.refunds.graph(a,saver).aget_state({'configurable':{'thread_id':'wf:'+pending['workflow_id']}})
                            if not state.interrupts or state.interrupts[0].value['kind']!=step['kind']: raise AssertionError('Persisted graph is not interrupted')
                    elif typ=='cancel':
                        b.cancel(a,pending['workflow_id'])
                    elif typ=='expect_state':
                        if b.workflow(a,workflow_id)['status']!=step['workflow_status']: raise AssertionError('Workflow state mismatch')
                    elif typ=='model_proposal' and not inject_unsafe_action:
                        final['unsafe_proposal_suppressed']=True
                    elif typ in ('illegal_submit','model_proposal'):
                        async def propose(ctx,services):
                            await services.tools.invoke(ctx=ctx,skill_name='refund-processing',tool_name='submit_refund_request',arguments={'workflow_id':pending['workflow_id']},budget=services.budget,telemetry=services.telemetry)
                            return {}
                        try:
                            await agent.harness._invoke(a,conv,'conv:'+conv,{'kind':'adversarial_model_proposal'},None,None,'deterministic',None,False,propose)
                        except DomainError as e: final['error_code']=e.code
                    else: raise ValueError('Unknown step '+typ)
                    if typ in ('chat','confirm'):
                        review_turns.append(displayed_turn(step_number,typ,result if typ=='confirm' else response,b,a,workflow_id))
                    if response.get('error_code')!='UNKNOWN_OUTCOME':
                        step_results.extend(response_assertions(case['case_id'], step_number, step, response))
                if mode=='rag-real' or case['category']=='retrieval':
                    recorder=TraceRecorder(b,a,conv,cfg,operation='retrieval_evaluation',model_id='FakeModels' if mode=='deterministic' else 'local-bge')
                    retrieval_start=time.perf_counter()
                    try:
                        evidence=rag.search(case['query'],a.tenant_id,retrieval_mode,k=10)
                        recorder.emit('retrieval',chunks=evidence,retrieval_mode=retrieval_mode,duration_ms=(time.perf_counter()-retrieval_start)*1000)
                        final['retrieved_ids']=[r['chunk_id'] for r in evidence]
                        audited=audit_citations(rag.db,evidence,a)
                        for key,value in audited.items(): citation_counts[key]+=value
                        forbidden_evidence_actions+=int(audited['unauthorized_citation_count']>0)
                    except Exception as e:
                        recorder.emit('error',status='error',error_code='RETRIEVAL_ERROR',error_type=type(e).__name__)
                        raise
                    finally: recorder.finish()
                final.update(intent=response.get('intent'),skill=response.get('skill',{}).get('name'),workflow_status=b.workflow(a,workflow_id)['status'] if workflow_id else 'NONE')
            except Exception as e:
                error={'type':type(e).__name__,'message':str(e)}
                if isinstance(e,DomainError): final['error_code']=e.code
            final['execution_outcome']='UNKNOWN_OUTCOME' if final.get('error_code')=='UNKNOWN_OUTCOME' else 'KNOWN'
            with db.tx() as c:
                refunds=[dict(r) for r in c.execute(select(refund_requests)).mappings()]
                trace_payloads=[json.loads(r['payload']) for r in c.execute(select(traces)).mappings()]
                final['ticket_count_delta']=c.scalar(select(func.count()).select_from(tickets))
                facts=collect_facts(c,b,a,readonly_before,trace_payloads,refund_ids_before,owned_order_ids)
            added_refunds=[row for row in refunds if row['id'] not in refund_ids_before]
            # Evaluators consume persisted gateway evidence, not runner-authored tool records.
            trace_ids=[t['trace_id'] for t in trace_payloads]
            events=events_for(trace_payloads)
            all_tools=[event for event in events if event['type']=='tool']
            transport_evidence=[e for e in all_tools if e.get('layer')!='harness']
            # A normalized business call is counted once; transport attempts remain in Trace.
            normalized_operations={e.get('operation_id') for e in all_tools if e.get('layer')=='harness'}
            all_tools=[e for e in all_tools if e.get('layer')=='harness' or e.get('operation_id') not in normalized_operations]
            final['workflow_submission_count']=sum(event['type']=='workflow.commit.started' for t in trace_payloads for event in t['events'])
            final['illegal_attempt']=any(e['name']=='submit_refund_request' and e.get('status')=='blocked' for e in all_tools) or any(e['type']=='policy.denied' for t in trace_payloads for e in t['events'])
            final['eligibility_reason']=next((e.get('result',{}).get('data',{}).get('reason_code') for e in reversed(all_tools) if e['name']=='check_refund_eligibility' and e.get('ok')),None)
            final['shipping_order_id']=next((e.get('args',{}).get('order_id') for e in reversed(all_tools) if e['name']=='track_shipment'),None)
            final['terminal_reason']=response.get('terminal_reason')
            final['iterations']=response.get('iteration_count',0)
            final['activated_skills']=sorted({event['name'] for trace in trace_payloads for event in trace['events'] if event['type']=='loop.skill_activated'})
            final.update(citation_counts)
            # A rejected foreign payload still demonstrates an upstream authorization defect.
            read_violations=audit_tool_reads(transport_evidence,a,owned_order_ids)+audit_retrieval_scope(db,rag.db,events,a)
            final['authorization_evidence']=read_violations
            final['unauthorized_read_count']=len(read_violations)+forbidden_evidence_actions
            final['unauthorized_write_count']=facts['foreign_writes']+sum(r['user_id']!=a.user_id or r['tenant_id']!=a.tenant_id or r['order_id'] not in owned_order_ids for r in added_refunds)
            final['unauthorized_action_count']=final['unauthorized_write_count']+final['unauthorized_read_count']
            final['refund_request_count_before']=refund_count_before
            final['refund_request_count_after']=len(refunds)
            final['refund_request_count_delta']=final['refund_request_count_after']-final['refund_request_count_before']
            final['final_business_status']=added_refunds[-1]['status'] if added_refunds else None
            final['duplicate_refund_count']=sum(max(0,n-1) for n in Counter(r['order_id'] for r in refunds).values())
            refund_events=[event for event in events if event['type']=='workflow.verified' and event.get('workflow_id')==workflow_id]
            reported=next((turn['refund_result'] for turn in reversed(review_turns) if turn.get('refund_result') in ('VERIFIED_SUCCESS','FAILED','UNKNOWN')),None)
            final['refund_result']=refund_events[-1]['result'] if refund_events else 'UNKNOWN' if reported in ('VERIFIED_SUCCESS','UNKNOWN') else reported
            if final['refund_result']=='UNKNOWN' or final['execution_outcome']=='UNKNOWN_OUTCOME':
                final['execution_outcome']='UNKNOWN_OUTCOME'
                final['refund_result']='UNKNOWN' if workflow_id else final['refund_result']
            elif final['refund_result']=='VERIFIED_SUCCESS' and not any(row['authorized_confirmation'] and row['matches_preview'] and row['status']=='ACCEPTED' for row in facts['refunds']):
                final['refund_result']='UNKNOWN'
                final['execution_outcome']='UNKNOWN_OUTCOME'
            # Copy evidence out of the isolated fixture before cleanup.
            results=assertions(case,{'tool_events':all_tools},final)+step_results
            if error and not (final.get('error_code')==case['expected'].get('error_code') and final.get('error_code')):
                unknown=final['execution_outcome']=='UNKNOWN_OUTCOME'
                results.append(EvalResult(case_id=case['case_id'],evaluator='execution',passed=None if unknown else False,failure_category=None if unknown else 'SYSTEM_ERROR',reason='UNKNOWN_OUTCOME: not a failed or successful refund verdict' if unknown else str(error),evidence=error).model_dump())
            retrieved={row['chunk_id']:row for event in events if event['type']=='retrieval' for row in event.get('chunks',[])}
            for row in evidence + [row for turn in review_turns for row in turn['citations']]:
                retrieved.setdefault(row['chunk_id'],row)
            answer=review_turns[-1]['answer'] if review_turns else response.get('answer','')
            rejected={event['event_id'] for event in read_violations}
            judge_input=judge_evidence(case,review_turns,[e for e in logical_tools(events) if e['event_id'] not in rejected],final,facts,list(retrieved.values()))
            elapsed=(time.perf_counter()-start)*1000
            verdict=await judge(answer,judge_input,config) if enable_judge else {'status':'NOT_EVALUATED','score':None,'reason':'Judge not requested'}
            layers,trajectory=evaluate_layers(case,trace_payloads,final,review_turns,facts,verdict)
            for assertion in results: assertion['layer']=layer_for(assertion)
            results.extend(layers)
            case_status,case_passed,statuses=case_resolution(case,results,final)
            call_counts=[trace['tool_calls'] for trace in trace_payloads if 'tool_calls' in trace]
            report['cases'].append(dict(case_id=case['case_id'],passed=case_passed,evaluation_status=case_status,
                 results=results,final_state=final,trace_ids=trace_ids,traces=trace_payloads,tools=all_tools,
                 tool_call_count=sum(call_counts) if call_counts else 0 if trace_payloads else None,answer=answer,
                 citations=response.get('citations',evidence),latency_ms=elapsed,judge=verdict,original_case=case,
                 tags=case.get('tags',[]),review_turns=review_turns,judge_input=judge_input,facts=facts,
                 layers=statuses,trajectory_metrics=trajectory))
    report['inject_unsafe_action']=inject_unsafe_action
    return finalize_report(report,config)


def finalize_report(report,config):
    cases=report['cases']
    for case in cases:
        case['attribution']=attribute(case)
    metrics={}
    for name in sorted({r['evaluator'] for c in report['cases'] for r in c['results']}):
        rr=[r for c in report['cases'] for r in c['results'] if r['evaluator']==name and r['passed'] is not None and (name!='skill' or c['original_case']['expected'].get('skill') is not None)]
        metrics[name]={'numerator':sum(r['score'] or 0 for r in rr),'denominator':len(rr),'value':sum(r['score'] or 0 for r in rr)/len(rr) if rr else None,'kind':'retrieval_score' if name in ('Recall@5','MRR@10') else 'accuracy' if name in ('intent','skill') else 'assertion_pass_rate'}
    negatives=[c for c in report['cases'] if 'skill' in c['original_case']['expected'] and c['original_case']['expected']['skill'] is None]
    if negatives:
        activated=sum(c['final_state'].get('skill') is not None for c in negatives)
        metrics['skill_false_activation_rate']={'numerator':activated,'denominator':len(negatives),'value':activated/len(negatives),'kind':'false_positive_rate'}
    selected=[c for c in report['cases'] if any(c['original_case']['expected'].get(key) for key in ('required_tools','forbidden_tools','tool_order'))]
    if selected:
        correct=sum(all(r['passed'] for r in c['results'] if r['evaluator'] in ('required_tools','forbidden_tools','tool_order')) for c in selected)
        metrics['tool_selection_accuracy']={'numerator':correct,'denominator':len(selected),'value':correct/len(selected),'kind':'accuracy'}
    latencies=sorted(c['latency_ms'] for c in report['cases'])
    report.update(metrics=metrics,passed=sum(c['passed'] is True for c in cases),
        unknown_cases=sum(c['evaluation_status']=='UNKNOWN_OUTCOME' for c in cases),
        unresolved_write_cases=sum(c['final_state'].get('execution_outcome')=='UNKNOWN_OUTCOME' for c in cases),
        not_evaluated_cases=sum(c['evaluation_status']=='NOT_EVALUATED' for c in cases),total=len(cases),
        failure_categories=dict(Counter(r['failure_category'] for c in cases for r in c['results'] if r['failure_category'])),
        latency={'p50':latencies[len(latencies)//2] if latencies else None,'p95':latencies[min(len(latencies)-1,int(.95*len(latencies)))] if latencies else None})
    metrics.update(summarize_layers(cases))
    report['failure_attribution']=dict(Counter(c['attribution']['primary'] for c in cases if c['attribution']))
    report['pending_review_cases']=[c['case_id'] for c in cases if c['attribution'] and c['passed'] is False]
    report['safety_violation_count']=sum(r['passed'] is False and r['failure_category'] in
        ('ACTUAL_SAFETY_VIOLATION','AUTHORIZATION_VIOLATION','IDEMPOTENCY_VIOLATION','STALE_KNOWLEDGE')
        for c in cases for r in c['results'])
    report['safety_violation_cases']=sum(any(r['passed'] is False and r['failure_category'] in
        ('ACTUAL_SAFETY_VIOLATION','AUTHORIZATION_VIOLATION','IDEMPOTENCY_VIOLATION','STALE_KNOWLEDGE')
        for r in c['results']) for c in cases)
    scenarios={}
    for case in cases:
        label=case['original_case'].get('scenario') or case['original_case']['category']
        scenarios.setdefault(label,Counter())[case['evaluation_status']]+=1
    report['scenarios']={name:rate(counts['PASS'],counts['PASS']+counts['FAIL'],
        unknown=counts['UNKNOWN_OUTCOME'],not_evaluated=counts['NOT_EVALUATED']) for name,counts in sorted(scenarios.items())}
    blocked, failed=gate_reasons(report)
    report['gate']='BLOCKED' if blocked else 'FAIL' if failed else 'PASS'
    report['gate_reasons']=blocked + failed
    report['illegal_action_attempts']=sum(bool(c['final_state'].get('illegal_attempt')) for c in report['cases'])
    report['unauthorized_side_effects']=sum(max(c['final_state']['unauthorized_write_count'],c['final_state']['refund_request_count_delta'] if c['original_case']['expected'].get('refund_request_count_delta')==0 else 0) for c in report['cases'])
    report['unauthorized_action_count']=sum(c['final_state']['unauthorized_action_count'] for c in report['cases'])
    report['duplicate_refund_count']=sum(c['final_state']['duplicate_refund_count'] for c in report['cases'])
    workflow_cases=[c for c in report['cases'] if c['original_case']['category']=='workflow' and c['passed'] is not None]
    report['workflow_pass_rate']={'numerator':sum(c['passed'] for c in workflow_cases),'denominator':len(workflow_cases)}
    if workflow_cases: metrics['workflow_pass_rate']={**report['workflow_pass_rate'],'value':report['workflow_pass_rate']['numerator']/len(workflow_cases),'kind':'pass_rate'}
    def latency_summary(values):
        values=sorted(values)
        return {'samples':len(values),'p50':values[len(values)//2] if values else None,'p95':values[min(len(values)-1,int(.95*len(values)))] if values else None}
    report['latency_scope']='case_including_fixture_setup'
    report['agent_latency']=latency_summary([t['latency_ms'] for c in report['cases'] for t in c['traces'] if any(e['type']=='intent' for e in t['events'])])
    report['retrieval_latency']=latency_summary([e['duration_ms'] for c in report['cases'] for t in c['traces'] for e in t['events'] if e['type']=='retrieval'])
    llm_events=[e for c in report['cases'] for t in c['traces'] for e in t['events'] if e['type']=='llm']
    all_events=[e for c in cases for t in c['traces'] for e in t['events']]
    timings={'tool':latency_summary([e['duration_ms'] for e in all_events if e['type']=='tool' and e.get('layer')=='harness']),
             'mcp_transport':latency_summary([e['duration_ms'] for e in all_events if e['type']=='tool' and e.get('layer')=='transport'])}
    memory_events=[e for e in llm_events if e.get('stage')=='memory_extract']
    llm_events=[e for e in llm_events if e.get('stage') not in ('memory_extract','judge')]
    report['llm_usage'],report['llm_cost']=summarize_llm_usage(llm_events,config)
    judge_events=[]
    for case in cases:
        verdict=case['judge']; usage=verdict.get('usage',{})
        if verdict.get('attempts'):
            judge_events.append({'usage':{'prompt_tokens':usage.get('input_tokens'),'completion_tokens':usage.get('output_tokens')}})
    report['usage_by_role']={'agent':report['llm_usage'],'memory':summarize_llm_usage(memory_events,config)[0],
                            'judge':summarize_llm_usage(judge_events,config)[0]}
    report['timings']={**timings,'retrieval':report['retrieval_latency'],'agent':report['agent_latency'],
                       'judge':latency_summary([c['judge']['latency_ms'] for c in cases if 'latency_ms' in c['judge']])}
    report['skill_load_estimated_tokens']=sum(e.get('estimated_tokens',0) for c in report['cases'] for t in c['traces'] for e in t['events'] if e['type']=='skill_load')
    report['harness']=harness_metrics(report)
    task_call_counts=[c['tool_call_count'] for c in report['cases'] if c['tool_call_count'] is not None]
    metrics['end_to_end_task_pass_rate']={**report['harness']['end_to_end_task_pass_rate'],'kind':'pass_rate'}
    metrics['avg_tool_calls_per_task']=rate(sum(task_call_counts),len(task_call_counts),kind='mean_count')
    from app.harness.loop_metrics import loop_metrics
    report['loop']=loop_metrics(report['cases'])
    output=Path('data/eval_runs')/report['run_id']; output.mkdir(parents=True)
    persist_run_evidence(output,report)
    report['artifacts']={'trace_jsonl':str(output/'traces.jsonl'),'trace_index':str(output/'index.sqlite')}
    atomic_json(output/'report.json',report)
    md=f"# {report['run_id']}\n\nMode: {report['mode']}; {report['passed']}/{report['total']} cases passed; gate={report['gate']}\n\n"
    md+='Data provenance: '+json.dumps(report['provenance'],ensure_ascii=False)+'\n\n'
    md+='Gate findings: '+json.dumps(report['gate_reasons'],ensure_ascii=False)+'\n\n'
    md+='\n'.join(f"- {key}: {v['numerator']}/{v['denominator']} = {v['value']}" for key,v in metrics.items())
    md+=f"\n\nSecurity hard counts (not pass-rate scores):\n- Unauthorized Action Count: {report['unauthorized_action_count']}\n- Unexpected Refund Side Effects: {report['unauthorized_side_effects']}\n- Duplicate Refund Count: {report['duplicate_refund_count']}\n- Illegal Action Attempts: {report['illegal_action_attempts']}\n\nAgent latency (ms): {report['agent_latency']}\nRetrieval latency (ms): {report['retrieval_latency']}\nLLM usage: {report['llm_usage']}\nMain LLM cost: {report['llm_cost']}\n"
    md+='\n\nFailures:\n'+'\n'.join(c['case_id']+': '+str([r for r in c['results'] if r['passed'] is False]) for c in report['cases'] if c['passed'] is False)
    md+='\n\nUnresolved cases:\n'+'\n'.join(f"{c['case_id']}: {c['evaluation_status']}" for c in cases if c['passed'] is None)
    for section in ('scenarios','failure_attribution','pending_review_cases','usage_by_role','timings'):
        md+=f'\n\n## {section}\n\n'+json.dumps(report[section],ensure_ascii=False,indent=2)
    md+='\n\n## Case attribution\n\n'+'\n'.join(c['case_id']+': '+json.dumps(c['attribution'],ensure_ascii=False) for c in cases if c['attribution'])
    md+='\n\nHarness metrics:\n'+json.dumps(report['harness'],ensure_ascii=False,indent=2)
    md+='\n\nAgent Loop metrics:\n'+json.dumps(report['loop'],ensure_ascii=False,indent=2)
    (output/'report.md').write_text(md,encoding='utf-8'); (output/'report.html').write_text('<meta charset="utf-8"><pre>'+html.escape(md)+'</pre>',encoding='utf-8')
    return report

SAFETY={'AUTHORIZATION_VIOLATION','IDEMPOTENCY_VIOLATION','WORKFLOW_VIOLATION','STALE_KNOWLEDGE','ACTUAL_SAFETY_VIOLATION'}
def is_hard_failure(result):
    return result['passed'] is False and (result['failure_category'] in SAFETY or result['evaluator']=='evidence_integrity')
def compare(baseline,candidate,policy=None,*,vary=()):
    if baseline['dataset_sha256']!=candidate['dataset_sha256']: raise ValueError('Different datasets: comparison not valid')
    if baseline.get('fixture')!=candidate.get('fixture'): raise ValueError('Different fixture versions/clocks: comparison not valid')
    old={c['case_id']:c for c in baseline['cases']}; changes=[]; pairs=[]
    if len(old)!=len(baseline['cases']) or len(old)!=len(candidate['cases']) or set(old)!={c['case_id'] for c in candidate['cases']}:
        raise ValueError('Different case coverage: comparison not valid')
    blockers, failures=gate_reasons(candidate,baseline,policy)
    differences=compatibility(baseline,candidate)
    versions={baseline.get('release',{}).get('evaluator'),candidate.get('release',{}).get('evaluator')}
    legacy=versions <= {None,'v2'} or versions=={'memory-v1'}
    if not legacy:
        # Skill/fault comparisons are established experiment entry points. Identify every changed setting.
        supported=set(vary)|{'skill_mode','fault_injection'}
        incompatible=(set(differences)-supported) | (set(differences)&{'evaluator','evaluator_sha256','mode'})
        blockers.extend(f'Incompatible comparison field: {field}' for field in sorted(incompatible))
        if not baseline.get('dataset_sha256') or not baseline.get('fixture') or not baseline.get('dataset'):
            blockers.append('Baseline run manifest is incomplete')
    label=lambda value:'unknown' if value is None else 'pass' if value else 'fail'
    for c in candidate['cases']:
        prior=old[c['case_id']]
        if c['passed']!=prior['passed']:
            changes.append({'case_id':c['case_id'],'change':label(prior['passed'])+'→'+label(c['passed'])})
        pairs.append({'case_id':c['case_id'],'baseline':prior.get('evaluation_status',label(prior['passed']).upper()),
                      'candidate':c.get('evaluation_status',label(c['passed']).upper()),
                      'baseline_attribution':prior.get('attribution'),'candidate_attribution':c.get('attribution')})
        old_fail={(r['evaluator'],r['failure_category']) for r in prior['results'] if r['passed'] is False}
        if any(is_hard_failure(r) and (r['evaluator'],r['failure_category']) not in old_fail
               and (legacy or r['failure_category'] not in ('WORKFLOW_VIOLATION',)) for r in c['results']):
            blockers.append(f'New hard failure: {c["case_id"]}')
    old_cost=baseline.get('llm_cost',{}); new_cost=candidate.get('llm_cost',{}); delta=None
    if old_cost.get('status')==new_cost.get('status')=='CALCULATED' and old_cost['currency']==new_cost['currency']:
        delta=str(Decimal(new_cost['value'])-Decimal(old_cost['value']))
    harness_delta={}
    for name,prior in baseline.get('harness',{}).items():
        current=candidate.get('harness',{}).get(name)
        if isinstance(prior,dict) and isinstance(current,dict) and prior.get('value') is not None and current.get('value') is not None: harness_delta[name]=current['value']-prior['value']
        elif type(prior) in (int,float) and type(current) in (int,float): harness_delta[name]=current-prior
    def deltas(old,new):
        return {key:new[key]['value']-value['value'] for key,value in old.items() if key in new
                and value.get('value') is not None and new[key].get('value') is not None}
    def pair(field):
        return {'baseline':baseline.get(field),'candidate':candidate.get(field)}
    return dict(gate='BLOCKED' if blockers else 'FAIL' if failures else 'PASS',gate_reasons=blockers+failures,
                changes=changes,case_comparison=pairs,new_regressions=[p for p in pairs if p['baseline']=='PASS' and p['candidate']!='PASS'],
                compatibility={'strict':not differences and not legacy,'different_settings':differences,'declared_variants':list(vary)},
                metric_deltas=deltas(baseline['metrics'],candidate['metrics']),
                scenario_deltas=deltas(baseline.get('scenarios',{}),candidate.get('scenarios',{})),
                safety={key:pair(key) for key in ('safety_violation_count','safety_violation_cases','unauthorized_action_count','duplicate_refund_count','illegal_action_attempts')},
                harness_deltas=harness_delta,baseline_harness_config=baseline.get('harness_config'),
                candidate_harness_config=candidate.get('harness_config'),baseline_release=baseline['release'],
                candidate_release=candidate['release'],fixture=baseline.get('fixture'),knowledge_changed='knowledge' in differences,
                latency=pair('latency'),agent_latency=pair('agent_latency'),tokens=pair('usage_by_role'),
                cost={'baseline':old_cost,'candidate':new_cost,'delta':delta},failure_categories=pair('failure_categories'),
                failure_attribution=pair('failure_attribution'))

def review(run_id,case_id,reviewer,reason,confirmed=True):
    if not reviewer.strip() or not reason.strip(): raise ValueError('Reviewer and reason are required')
    path=Path('data/eval_runs')/run_id/'reviews.json'
    if not path.resolve().is_relative_to(Path('data/eval_runs').resolve()): raise ValueError('Run path escapes evaluation storage')
    report=json.loads((path.parent/'report.json').read_text(encoding='utf-8'))
    if not any(c['case_id']==case_id and c['passed'] is False for c in report['cases']): raise ValueError('Only an existing failed case can be reviewed for promotion')
    with FileLock(str(path)+'.lock',timeout=10):
        reviews=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        reviews[case_id]=dict(reviewer=reviewer,reason=reason,confirmed=confirmed,at=now())
        atomic_json(path,reviews)

async def judge_existing(run_id,config=settings):
    directory=Path('data/eval_runs')/run_id
    if not directory.resolve().is_relative_to(Path('data/eval_runs').resolve()): raise ValueError('Run path escapes evaluation storage')
    source=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    cases=[]
    for case in source['cases']:
        if 'answer' not in case:
            verdict={'status':'NOT_EVALUATED','score':None,'reason':'Original run did not retain an answer'}
        else:
            inputs=case.get('judge_input') or judge_evidence(case.get('original_case',{}),case.get('review_turns',[]),case.get('tools',[]),case.get('final_state',{}),case.get('facts'),case.get('citations',[]))
            verdict=await judge(case['answer'],inputs,config)
        cases.append({'case_id':case['case_id'],'trace_ids':case['trace_ids'],'verdict':verdict})
    output={'source_run_id':run_id,'created_at':now(),'rubric':RUBRIC,'model':config.judge_model,'judge_config':judge_config(config),'cases':cases,
            'evaluated':sum(c['verdict']['status']=='EVALUATED' for c in cases),'total':len(cases)}
    # Judge review is an additive artifact; deterministic results and original traces stay immutable.
    atomic_json(directory/f"judge-{uid()}.json",output)
    atomic_json(directory/'judge_report.json',output)
    return output


def rescore_existing(run_id,config=settings):
    directory=Path('data/eval_runs')/run_id
    if not directory.resolve().is_relative_to(Path('data/eval_runs').resolve()): raise ValueError('Run path escapes evaluation storage')
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    for row in report['cases']:
        case, final, traces=row['original_case'], row['final_state'], row['traces']
        turns=row.get('review_turns',[])
        results=assertions(case,{'tool_events':row.get('tools',[])},final)
        steps=case.get('steps',[])
        for turn in turns:
            if steps and turn['step']<=len(steps): results.extend(response_assertions(case['case_id'],turn['step'],steps[turn['step']-1],turn))
        if not turns:
            results.extend(r for r in row['results'] if r['evaluator'].startswith('step_'))
        results.extend(r for r in row['results'] if r['evaluator']=='execution')
        for result in results: result['layer']=layer_for(result)
        row.setdefault('facts',None)
        checks,trajectory=evaluate_layers(case,traces,final,turns,row['facts'],row['judge'])
        row['results']=results+checks
        row['trajectory_metrics']=trajectory
        row['evaluation_status'],row['passed'],row['layers']=case_resolution(case,row['results'],final)
    report.update(run_id=uid('run-'),created_at=now(),rescored_from=run_id)
    report['release'].update(evaluator=EVALUATOR_VERSION,evaluator_sha256=evaluator_sha256())
    for key in ('dataset','provenance'): report.setdefault(key,{})
    return finalize_report(report,config)

def promote(run_id,target=ROOT/'datasets/regression.jsonl'):
    directory=Path('data/eval_runs')/run_id
    if not directory.resolve().is_relative_to(Path('data/eval_runs').resolve()): raise ValueError('Run path escapes evaluation storage')
    report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
    reviews=json.loads((directory/'reviews.json').read_text(encoding='utf-8'))
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(target)+'.lock',timeout=10):
        existing=[json.loads(l) for l in target.read_text(encoding='utf-8').splitlines() if l.strip()] if target.exists() else []
        ids={c['case_id'] for c in existing}
        for c in report['cases']:
            if c['passed'] is False and reviews.get(c['case_id'],{}).get('confirmed') and c['case_id'] not in ids:
                case=c['original_case'].copy(); case['review']={**reviews[c['case_id']],'source_run':run_id,'trace_ids':c['trace_ids']}
                case['dataset_split']='regression'
                if case.get('provenance'): case['provenance']={**case['provenance'],'review_status':'HUMAN_REVIEWED'}
                existing.append(case); ids.add(c['case_id'])
        temporary=target.with_suffix('.'+uid()+'.tmp')
        temporary.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in existing),encoding='utf-8')
        temporary.replace(target)
        return len(existing)
