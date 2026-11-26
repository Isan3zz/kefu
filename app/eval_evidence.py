"""Read authority snapshots before disposing of an evaluation case database."""
import json
from datetime import timedelta

from sqlalchemy import select

from app.db import order_items, orders, products, refund_requests, shipments, tickets, workflow_instances
from app.config import ROOT
from app.eval_layers import events_for
from app.observability import clean

HOLDOUT_FIXTURE = ROOT / 'datasets/holdout/business_v1.json'


def seed_holdout(db, clock):
    db.seed(clock)
    source = json.loads(HOLDOUT_FIXTURE.read_text(encoding='utf-8'))
    stamp = clock()
    with db.tx() as connection:
        for row in source['orders']:
            oid = row['id']
            tenant = 'b' if row['user_id'] == 'user_b' else 'a'
            delivered = (stamp - timedelta(seconds=row['delivery_age_seconds'])).isoformat() if row.get('delivery_age_seconds') is not None else None
            connection.execute(orders.insert().values(id=oid, tenant_id=tenant, user_id=row['user_id'],
                status=row['status'], version=1, paid_amount_cents=row['paid_amount_cents'],
                paid_at=(stamp - timedelta(hours=row['paid_age_hours'])).isoformat(), delivered_at=delivered))
            connection.execute(order_items.insert().values(id='holdout-item-'+oid, order_id=oid, product_id='headphones-'+tenant,
                quantity=1, unit_price_cents=row['paid_amount_cents']))
            for index, status in enumerate(row.get('shipments', []), 1):
                connection.execute(shipments.insert().values(id=f'holdout-ship-{oid}-{index}', order_id=oid, carrier='模拟快递',
                    tracking_no=f'NEW{oid}{index}' if status != '待发货' else None, status=status, updated_at=stamp.isoformat()))


def rows(connection, table, order_column=None):
    statement = select(table).order_by(order_column if order_column is not None else table.c.id)
    return [dict(row) for row in connection.execute(statement).mappings()]


def readonly_snapshot(connection):
    return {table.name: clean(rows(connection, table)) for table in (orders, order_items, shipments, products)}


def collect_facts(connection, business, actor, before, traces, refund_ids_before, order_ids):
    after = readonly_snapshot(connection)
    workflows = rows(connection, workflow_instances)
    approved = {e['workflow_id'] for e in events_for(traces) if e['type'] == 'workflow.transition'
                and e.get('previous') == 'WAITING_CONFIRMATION' and e.get('next') == 'CONFIRMED'}
    by_operation = {row['operation_key']: row for row in workflows}
    refunds = []
    for row in rows(connection, refund_requests):
        if row['id'] in refund_ids_before:
            continue
        workflow = by_operation.get(row['idempotency_key'], {})
        scope = ('tenant_id', 'user_id', 'order_id', 'amount_cents', 'policy_version')
        refunds.append({key: row[key] for key in ('id', *scope, 'status')} | {
            'workflow_id': workflow.get('id'), 'authorized_confirmation': workflow.get('id') in approved,
            'matches_preview': (all(row[key] == workflow.get(key) for key in scope)
                                and row['idempotency_key'] == f"{workflow.get('id')}:refund"
                                and row['order_id'] in order_ids
                                and (row['status'], workflow.get('status')) in (('ACCEPTED', 'REQUEST_ACCEPTED'), ('REJECTED', 'FAILED'))),
        })
    selected_products = [{key: row[key] for key in ('id', 'name', 'category', 'price_cents', 'stock')}
                         | {'attributes': json.loads(row['attributes_json'])}
                         for row in after['products'] if row['tenant_id'] == actor.tenant_id]
    scoped_orders = [row for row in after['orders'] if row['id'] in order_ids]
    scoped_shipments = [row for row in after['shipments'] if row['order_id'] in order_ids]
    eligibility = {row['id']: business.eligibility(connection, actor, row['id']) for row in scoped_orders}
    return {'products': selected_products, 'orders': scoped_orders, 'shipments': scoped_shipments,
            'eligibility': eligibility,
            'refunds': refunds, 'readonly_before': before, 'readonly_after': after,
            'workflows': [{key: row[key] for key in ('id', 'status', 'order_id', 'amount_cents')} for row in workflows],
            'foreign_writes': sum(row['tenant_id'] != actor.tenant_id or row['user_id'] != actor.user_id
                                  for row in rows(connection, tickets) + workflows)}


def displayed_turn(step, kind, response, business, actor, workflow_id):
    outcome = business.refund_outcome(actor, workflow_id) if workflow_id else None
    verifier_ids = [outcome['refund_request_id']] if outcome and outcome.get('status') == 'ACCEPTED' else []
    pending = response.get('pending_confirmation') or {}
    return clean({'step': step, 'type': kind, 'trace_id': response.get('trace_id'), 'answer': response.get('answer', ''),
                  'citations': response.get('citations', []), 'pending_confirmation': {key: pending[key] for key in
                      ('order_id', 'amount_cents', 'status') if key in pending}, 'verified_refund_ids': verifier_ids,
                  'refund_result': (response.get('refund_workflow_result') or {}).get('status')})
