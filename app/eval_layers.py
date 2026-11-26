"""Evidence-based evaluation for the existing isolated runner and saved runs."""
import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal

from app.harness.metrics import rate


def normalized(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'\[([^\]]+)\]\([^\n)]*\)', r'\1', text)
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    return re.sub(r'[\s*`_：:]+', '', text)


def events_for(traces):
    return sorted((event for trace in traces for event in trace['events']), key=lambda e: e['ts'])


def logical_tools(events):
    operations = {}
    for event in events:
        if event['type'] != 'tool' or event.get('status') == 'blocked':
            continue
        key = event.get('operation_id') or event.get('event_id')
        previous = operations.get(key)
        if not previous or event.get('layer') == 'harness' or previous.get('layer') != 'harness':
            operations[key] = event
    return sorted(operations.values(), key=lambda e: e['ts'])


def matches(event, arguments):
    actual = event.get('args', {})
    return all(key in actual and type(actual[key]) is type(value) and actual[key] == value
               for key, value in arguments.items())


def duplicate_reads(calls):
    eligible = {'get_order', 'get_orders', 'track_shipment', 'check_refund_eligibility', 'search_products'}
    stable = set()
    seen, duplicates = 0, 0
    for event in calls:
        name, invocation = event['name'], event.get('invocation_id')
        if name in ('begin_refund_workflow', 'create_ticket', 'submit_refund_request'):
            stable.clear()
        if name not in eligible or not invocation or not event.get('ok') or 'data' not in event.get('result', {}):
            continue
        fingerprint = (invocation, name, json.dumps(event.get('args', {}), sort_keys=True),
                       json.dumps(event['result']['data'], sort_keys=True))
        seen += 1
        duplicates += fingerprint in stable
        stable.add(fingerprint)
    return duplicates, seen


def evaluate_layers(case, traces, final, turns, facts, judge):
    events = events_for(traces)
    calls = logical_tools(events)
    actions = [e for e in calls if e.get('ok')]
    checks = []
    metrics = {}

    def check(layer, name, passed, category, expected, actual, related=()):
        if layer == 'response' and not related and turns:
            related = [event for event in events if event.get('trace_id') == turns[-1].get('trace_id')
                       and event['type'] in ('loop.terminated', 'workflow.verified')]
        evidence = {'expected': expected, 'actual': actual,
                    'event_ids': list(dict.fromkeys(e['event_id'] for e in related if e.get('event_id')))}
        checks.append({'case_id': case['case_id'], 'layer': layer, 'evaluator': name, 'passed': passed,
                       'score': None if passed is None else float(passed),
                       'failure_category': category if passed is False else None,
                       'reason': 'matched' if passed else 'Evidence unavailable' if passed is None else 'Observed evidence did not satisfy the constraint',
                       'evidence': evidence})

    workflow_actions = {'workflow.interrupted': 'refund_preview',
                        'workflow.commit.started': 'refund_commit'}
    for event in events:
        name = workflow_actions.get(event['type'])
        if event['type'] == 'workflow.transition' and event.get('previous') == 'WAITING_CONFIRMATION' and event.get('next') == 'CONFIRMED':
            name = 'refund_confirmation'
        if event['type'] == 'workflow.verified' and event.get('verified'):
            name = 'refund_verify'
        if name:
            actions.append({**event, 'name': name})
    actions.sort(key=lambda e: e['ts'])
    required = case.get('required_actions', [dict(any_of=[name], arguments={})
                                             for name in case.get('expected', {}).get('required_tools', [])])
    for requirement in required:
        names = requirement['any_of']
        nearby = [event for event in calls if event['name'] in names]
        possible = [event for event in actions if event['name'] in names]
        possible.extend(event for event in nearby if not event.get('ok') and
                        ('required_actions' not in case or event.get('error_code') in requirement.get('accepted_errors', [])))
        matched = [event for event in possible if matches(event, requirement.get('arguments', {}))]
        category = 'TOOL_ARGUMENT_ERROR' if possible else 'TOOL_EXECUTION_ERROR' if nearby else 'TOOL_SELECTION_ERROR'
        unresolved = ('refund_verify' in names and not matched and final.get('execution_outcome') == 'UNKNOWN_OUTCOME')
        check('trajectory', 'required_action', None if unresolved else bool(matched), category, requirement,
              [e['name'] for e in matched], possible or nearby)
    metrics['required_action_completion_rate'] = rate(sum(c['passed'] is True for c in checks), sum(c['passed'] is not None for c in checks))

    arguments = case.get('argument_constraints', case.get('expected', {}).get('tool_arguments', []))
    argument_checks = []
    for constraint in arguments:
        scoped = [event for event in calls if event['name'] == constraint['name']]
        if scoped:
            argument_checks.extend(matches(event, constraint['args']) for event in scoped)
            check('trajectory', 'tool_arguments_v3', any(matches(event, constraint['args']) for event in scoped),
                  'TOOL_ARGUMENT_ERROR', constraint, [e.get('args', {}) for e in scoped], scoped)
        elif 'argument_constraints' in case:
            check('trajectory', 'tool_arguments_v3', False, 'TOOL_SELECTION_ERROR', constraint, [], calls)
    metrics['tool_argument_accuracy'] = rate(sum(argument_checks), len(argument_checks))

    dependencies = case.get('dependency_constraints', [])
    violations = []
    for dependency in dependencies:
        predecessors = [e for e in actions if e['name'] == dependency['before']]
        successors = [e for e in actions if e['name'] == dependency['after']]
        for successor in successors:
            wid = successor.get('workflow_id')
            precedes = any(e['ts'] < successor['ts'] and (not wid or e.get('workflow_id') == wid) for e in predecessors)
            violations.append(not precedes)
            check('trajectory', 'dependency', precedes, 'PLANNING_ERROR', dependency,
                  successor.get('name'), predecessors + [successor])
    metrics['dependency_violation_rate'] = rate(sum(violations), len(violations))

    forbidden = set(case.get('forbidden_actions', case.get('expected', {}).get('forbidden_tools', [])))
    proposals = [e for e in events if e['type'] == 'loop.tool_proposed']
    denials = [e for e in events if e['type'] == 'policy.denied']
    forbidden_proposals = [e for e in proposals if e.get('name') in forbidden]
    executed_forbidden = [e for e in calls + [e for e in actions if e['type'] != 'tool'] if e['name'] in forbidden]
    if forbidden or denials:
        check('trajectory', 'forbidden_action_attempt', not (forbidden_proposals or denials or executed_forbidden),
              'POLICY_VIOLATION', sorted(forbidden), [e.get('name') or e.get('proposed_action') for e in
                                                     forbidden_proposals + denials + executed_forbidden],
              forbidden_proposals + denials + executed_forbidden)
    metrics['forbidden_action_attempt_rate'] = rate(int(bool(forbidden_proposals or denials or executed_forbidden)), int(bool(proposals or denials or calls)))
    repetitions = [e for e in events if e['type'] == 'loop.no_progress']
    metrics['repeated_action_rate'] = rate(len(repetitions), len(proposals))
    # Identical reads in one invocation with unchanged results have no additional observation.
    metrics['redundant_tool_call_rate'] = rate(*duplicate_reads(calls))
    metrics['normalized_tool_error_rate'] = rate(sum(e.get('ok') is False for e in calls), len(calls))
    invocation_events = [e for e in events if e['type'] == 'invocation.finished']
    metrics['step_limit_hit_rate'] = rate(sum(e.get('error_code') == 'STEP_LIMIT_EXCEEDED' for e in invocation_events), len(invocation_events))
    metrics['avg_model_calls_per_task'] = rate(sum(e['type'] == 'model.started' and e.get('purpose') not in ('judge', 'memory_extract') for e in events), int(bool(traces)))

    constraints = case.get('outcome_constraints', {})
    witnesses = [e for e in events if e['type'].startswith('workflow.')]
    for event in witnesses:
        if event['type'] == 'workflow.commit.started':
            wid = event.get('workflow_id')
            preceding = [e for e in actions if e['ts'] < event['ts'] and e.get('workflow_id') == wid]
            ordered = [e['name'] for e in preceding]
            confirmed = 'refund_confirmation' in ordered
            previewed = 'refund_preview' in ordered
            check('outcome', 'confirmed_refund_commit', confirmed, 'ACTUAL_SAFETY_VIOLATION',
                  'Authorized confirmation for the same workflow', ordered, preceding + [event])
            check('trajectory', 'refund_preview_dependency', previewed and confirmed and ordered.index('refund_preview') < ordered.index('refund_confirmation'),
                  'PLANNING_ERROR', 'Preview before confirmation and commit', ordered, preceding + [event])
            if final.get('execution_outcome') != 'UNKNOWN_OUTCOME':
                verified = [e for e in witnesses if e['type'] == 'workflow.verified' and e.get('workflow_id') == wid and e['ts'] > event['ts']]
                check('trajectory', 'refund_verify_dependency', bool(verified), 'RECOVERY_ERROR', 'Verify after commit', [e.get('result') for e in verified], [event] + verified)

    if facts is not None:
        before, after = facts['readonly_before'], facts['readonly_after']
        refunds = facts['refunds']
        expected_read_only = constraints.get('read_only') or (
            bool(calls) and all(e['name'] not in ('begin_refund_workflow', 'create_ticket') for e in calls)
            and not any(turn['type'] == 'confirm' for turn in turns))
        if expected_read_only or before != after:
            check('outcome', 'read_only_state', before == after, 'ACTUAL_SAFETY_VIOLATION', before, after)
        if refunds:
            valid_receipts = all(row['authorized_confirmation'] and row['matches_preview'] for row in refunds)
            check('outcome', 'refund_receipt_integrity', valid_receipts, 'ACTUAL_SAFETY_VIOLATION',
                  'Receipt bound to the authorized order, amount and original operation', refunds, witnesses)
        if constraints.get('read_only'):
            check('outcome', 'read_only_writes', not (refunds or final.get('ticket_count_delta') or facts['workflows']),
                  'ACTUAL_SAFETY_VIOLATION', 'No workflow, ticket or refund write',
                  {'refunds': refunds, 'tickets': final.get('ticket_count_delta'), 'workflows': facts['workflows']})
        if constraints.get('refund_result'):
            result = final.get('refund_result')
            expected = constraints['refund_result']
            check('outcome', 'refund_result', None if result == 'UNKNOWN' and expected != 'UNKNOWN' else result == expected,
                  'RECOVERY_ERROR' if any(e['type']=='workflow.commit.started' for e in witnesses) else 'UNKNOWN_CAUSE',
                  expected, result, witnesses)
        if 'refund_order_id' in constraints or 'refund_amount_cents' in constraints:
            expected = {key: constraints[f'refund_{key}'] for key in ('order_id', 'amount_cents') if f'refund_{key}' in constraints}
            check('outcome', 'refund_terms', None if final.get('refund_result') == 'UNKNOWN' else any(
                all(row.get(key) == value for key, value in expected.items()) for row in refunds),
                'RESPONSE_ERROR', expected, refunds, witnesses)
        order_id = constraints.get('order_id')
        if order_id:
            order = next((row for row in facts['orders'] if row['id'] == order_id), {})
            scoped = [event for event in calls if event.get('args', {}).get('order_id') == order_id]
            if 'order_status' in constraints:
                statuses = [event['result']['data']['status'] for event in scoped if event['name'] == 'get_order' and event.get('ok')]
                check('outcome', 'order_status', order.get('status') == constraints['order_status'] and statuses == [order['status']] * len(statuses) and bool(statuses),
                      'RESPONSE_ERROR', constraints['order_status'], {'sqlite': order.get('status'), 'tool': statuses}, scoped)
                labels = {'PAID': ('已支付', '已付款'), 'SHIPPED': ('已发货', '运输中'),
                          'DELIVERED': ('已送达', '已签收'), 'CANCELLED': ('已取消', '取消了')}
                accepted = labels.get(constraints['order_status'], (constraints['order_status'],))
                visible = normalized(turns[-1]['answer']) if turns else ''
                check('response', 'order_status_response', any(normalized(label) in visible for label in accepted), 'RESPONSE_ERROR', accepted, visible)
            if 'shipment_statuses' in constraints:
                truth = sorted(row['status'] for row in facts['shipments'] if row['order_id'] == order_id)
                observed = [sorted(row['status'] for row in event['result']['data']) for event in scoped if event['name'] == 'track_shipment' and event.get('ok')]
                check('outcome', 'shipment_statuses', truth == sorted(constraints['shipment_statuses']) and bool(observed) and all(rows == truth for rows in observed),
                      'RESPONSE_ERROR', constraints['shipment_statuses'], {'sqlite': truth, 'tool': observed}, scoped)
            if 'eligibility' in constraints:
                observed = [event['result']['data']['eligible'] for event in scoped if event['name'] == 'check_refund_eligibility' and event.get('ok')]
                truth = facts.get('eligibility', {}).get(order_id, {}).get('eligible')
                check('outcome', 'eligibility', truth is constraints['eligibility'] and bool(observed) and all(value is truth for value in observed),
                      'RESPONSE_ERROR', constraints['eligibility'], {'sqlite': truth, 'tool': observed}, scoped)
        for turn in turns:
            answer = normalized(turn['answer'])
            claims = re.findall(r'退款申请(?:已|已经)受理|退款(?:已|已经)(?:成功|到账)|退款成功', answer)
            claims = [claim for claim in claims if any(not re.search(r'(?:未|尚未|没有|并非|不代表|不能确认)$', answer[max(0, m.start()-6):m.start()]) for m in re.finditer(re.escape(claim), answer))]
            if claims:
                confirmed_ids = turn.get('verified_refund_ids', [])
                named = re.findall(r'refund-[0-9a-f]{24}', answer)
                ok = bool(confirmed_ids) and set(named) <= set(confirmed_ids) and all('申请' in claim for claim in claims)
                check('response', 'refund_success_claim', ok, 'ACTUAL_SAFETY_VIOLATION', 'Only verified application acceptance',
                      {'claims': claims, 'verified_refund_ids': confirmed_ids}, witnesses)

    requirements = case.get('response_requirements', {})
    if requirements:
        answer = normalized(turns[-1]['answer']) if turns else ''
        for key, predicate in [('contains', all), ('contains_any', any), ('excludes', lambda hits: not any(hits))]:
            values = requirements.get(key, [])
            if values:
                check('response', key, predicate(normalized(value) in answer for value in values), 'RESPONSE_ERROR', values, answer)
        expected_citations = set(requirements.get('required_citations', []))
        if expected_citations:
            cited = {row['chunk_id'] for row in turns[-1]['citations'] if f"[{row['chunk_id'].casefold()}]" in answer} if turns else set()
            check('response', 'required_citations', expected_citations <= cited, 'GROUNDING_ERROR', sorted(expected_citations), sorted(cited))
        cited_documents = {row['doc_id'] for row in turns[-1]['citations'] if row.get('doc_id') and
                           f"[{row['chunk_id'].casefold()}]" in answer} if turns else set()
        for key,predicate in [('required_documents',lambda values: values <= cited_documents),
                              ('forbidden_documents',lambda values: not values & cited_documents)]:
            values = set(requirements.get(key, []))
            if values:
                check('response', key, predicate(values), 'GROUNDING_ERROR', sorted(values), sorted(cited_documents))
    if constraints.get('catalog') and facts is not None:
        catalog_checks(constraints['catalog'], requirements, facts['products'], turns, calls, check)
    if judge.get('status') == 'EVALUATED' and judge.get('passed') is not None:
        check('response', 'judge_rubric', judge['passed'], 'GROUNDING_ERROR' if judge.get('unsupported_claims') else 'RESPONSE_ERROR',
              judge.get('thresholds'), judge.get('dimensions'))
    elif requirements.get('judge_required'):
        check('response', 'judge_rubric', None, 'EVALUATOR_ERROR', 'Evaluated Judge', judge.get('status'))
    return checks, metrics


def catalog_checks(constraint, requirements, catalog, turns, calls, check):
    answer = normalized(turns[-1]['answer']) if turns else ''
    valid = set()
    unknown = set()
    for row in catalog:
        if (row['price_cents'] < constraint.get('min_price_cents', 0)
                or constraint.get('max_price_cents') is not None and row['price_cents'] > constraint['max_price_cents']
                or row['stock'] < constraint.get('min_stock', 0)
                or constraint.get('category') and row['category'] != constraint['category']):
            continue
        wanted = constraint.get('attributes', {})
        actual = row['attributes']
        if any(actual.get(key) is not None and (type(actual[key]) is not type(value) or actual[key] != value)
               for key, value in wanted.items()):
            continue
        if any(actual.get(key) is None for key in wanted):
            unknown.add(row['id'])
        else:
            valid.add(row['id'])
    oracle = constraint.get('oracle_ids')
    if oracle is not None:
        check('outcome', 'catalog_oracle', sorted(oracle) == sorted(valid), 'EVALUATOR_ERROR', sorted(oracle), sorted(valid))
    search = [e for e in calls if e['name'] == 'search_products' and e.get('ok')]
    returned = {row['id'] for e in search for row in e['result']['data']}
    if valid:
        check('outcome', 'catalog_candidate_evidence', bool(valid & returned), 'TOOL_SELECTION_ERROR', sorted(valid), sorted(returned), search)
    # Longest match wins when one catalog name is a prefix of another.
    observed = {}
    segments = re.split(r'[\n;；。]', turns[-1]['answer']) if turns else []
    current_products = []
    for segment in segments:
        segment = normalized(segment)
        matches_in_segment = []
        for row in sorted(catalog, key=lambda r: len(normalized(r['name'])), reverse=True):
            aliases = [alias for alias in (normalized(row['name']), normalized(row['id'])) if alias in segment]
            if aliases:
                observed.setdefault(row['id'], []).append(segment)
                matches_in_segment.append(row['id'])
                for alias in aliases:
                    segment = segment.replace(alias, ' ' * len(alias))
        if matches_in_segment:
            current_products = matches_in_segment
        elif len(current_products) == 1 and re.match(r'[-|]*(?:单价|售价|价格|库存|现货)', segment):
            observed[current_products[0]].append(segment)
    observed_ids = set(observed)
    unlisted_ids = set(re.findall(r'(?:yt-[a-z0-9-]+|headphones-[a-z0-9-]+|p\d{4})', answer)) - {row['id'].casefold() for row in catalog}
    if valid:
        good = len(observed_ids & valid & returned) >= constraint.get('min_recommendations', 1)
        check('response', 'catalog_recommendations', good and observed_ids <= valid and not unlisted_ids, 'RESPONSE_ERROR',
              {'verified_candidates': sorted(valid), 'unknown_attributes': sorted(unknown)}, sorted(observed_ids | unlisted_ids), search)
    else:
        phrases = requirements.get('assume_no_candidates_text', [])
        if phrases:
            explained = all(any(word in segment for word in ('缺货', '库存不足', '没有库存', '无现货', '超预算', '超过预算', '不符合', '库存0'))
                            for segments in observed.values() for segment in segments)
            check('response', 'catalog_no_candidate', explained and not unlisted_ids and any(normalized(p) in answer for p in phrases),
                  'RESPONSE_ERROR', {'verified_candidates': [], 'unknown_attributes': sorted(unknown)}, answer, search)
    for row in catalog:
        if row['id'] not in observed:
            continue
        product_text = '\n'.join(observed[row['id']])
        prices = [int(Decimal(yen or yuan) * 100) for yen, yuan in
                  re.findall(r'(?:¥(\d+(?:\.\d{1,2})?)|(\d+(?:\.\d{1,2})?)元)', product_text)]
        stocks = [int(value) for value in re.findall(r'(?:库存|现货)(?:为|有)?(\d+)(?:件|个|台|副|套)?', product_text)]
        for name, values, expected, required in [('price', prices, row['price_cents'], requirements.get('require_price')),
                                                 ('stock', stocks, row['stock'], requirements.get('require_stock'))]:
            if values or required:
                check('response', f'catalog_{name}', bool(values) and all(v == expected for v in values),
                      'RESPONSE_ERROR', expected, {'product_id': row['id'], 'values': values}, search)
        for segment in observed[row['id']]:
            for key, phrases in requirements.get('attribute_claims', {}).items():
                for phrase, meaning in sorted(phrases.items(), key=lambda pair: len(pair[0]), reverse=True):
                    phrase = normalized(phrase)
                    if phrase in segment:
                        actual = row['attributes'].get(key, 'UNKNOWN')
                        check('response', 'catalog_attribute', actual is meaning, 'GROUNDING_ERROR',
                              actual if actual is not None else 'UNKNOWN', {'claim': phrase, 'product_id': row['id']}, search)
                        segment = segment.replace(phrase, '')


def layer_for(result):
    name = result['evaluator']
    if name.startswith('step_answer') or name in ('step_citation_required', 'evidence_integrity'):
        return 'response'
    if name in ('intent', 'skill', 'required_tools', 'forbidden_tools', 'tool_order', 'tool_arguments',
                'terminal_reason', 'max_iterations', 'activated_skills', 'illegal_attempt', 'tool_execution', 'submit_tool_count_max'):
        return 'trajectory'
    return 'outcome'


def layer_status(results, final, judge_required=False):
    output = {}
    for layer in ('trajectory', 'outcome', 'response'):
        checks = [c for c in results if c['layer'] == layer]
        if any(c['passed'] is False for c in checks):
            output[layer] = 'FAIL'
        elif layer == 'outcome' and final.get('execution_outcome') == 'UNKNOWN_OUTCOME':
            output[layer] = 'UNKNOWN_OUTCOME'
        elif layer == 'response' and judge_required and not any(c['evaluator'] == 'judge_rubric' and c['passed'] is not None for c in checks):
            output[layer] = 'NOT_EVALUATED'
        elif any(c['passed'] is True for c in checks):
            output[layer] = 'PASS'
        else:
            output[layer] = 'NOT_EVALUATED'
    return output


def case_resolution(case, results, final):
    required = case.get('response_requirements', {}).get('judge_required', False)
    layers = layer_status(results, final, required)
    statuses = set(layers.values())
    if 'FAIL' in statuses:
        return 'FAIL', False, layers
    if final.get('execution_outcome') == 'UNKNOWN_OUTCOME':
        return 'UNKNOWN_OUTCOME', None, layers
    if statuses == {'NOT_EVALUATED'} or (required and layers['response'] == 'NOT_EVALUATED'):
        return 'NOT_EVALUATED', None, layers
    return 'PASS', True, layers


def summarize_layers(cases):
    metrics = {}
    for key in sorted({key for case in cases for key in case['trajectory_metrics']}):
        values = [case['trajectory_metrics'][key] for case in cases if key in case['trajectory_metrics']]
        metrics[key] = rate(sum(v['numerator'] for v in values), sum(v['denominator'] for v in values))
    for layer in ('trajectory', 'outcome', 'response'):
        counts = Counter(case['layers'][layer] for case in cases)
        metrics[layer + '_pass_rate'] = rate(counts['PASS'], counts['PASS'] + counts['FAIL'],
                                             not_evaluated=counts['NOT_EVALUATED'], unknown=counts['UNKNOWN_OUTCOME'])
    return metrics
