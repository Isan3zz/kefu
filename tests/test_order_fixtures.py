import json
from datetime import datetime

import pytest
from sqlalchemy import func, select

from app.config import settings
from app.dataset import FIXTURE_TIME, load_dataset
from app.db import DB, order_items, orders, refund_requests, shipments
from app.evaluation import run
from app.order_fixtures import (
    ORDER_FIXTURE_ALL,
    evaluate_order_boundary_oracle,
    import_order_fixture,
    load_order_fixture_data,
    load_order_oracle,
    seed_order_scenario,
)


def test_order_fixture_isolated_idempotent_and_preserves_deliberate_anomalies(tmp_path):
    fixture_time = datetime.fromisoformat(FIXTURE_TIME)
    db = DB("sqlite:///" + str(tmp_path / "orders.db"))
    db.seed(lambda: fixture_time)
    with db.tx() as connection:
        baseline_orders = [dict(row) for row in connection.execute(select(orders).order_by(orders.c.id)).mappings()]
        baseline_items = [dict(row) for row in connection.execute(select(order_items).order_by(order_items.c.id)).mappings()]

    first = seed_order_scenario(db, ORDER_FIXTURE_ALL, lambda: fixture_time)
    data = load_order_fixture_data()
    assert first["products"]["inserted"] == 19
    assert first["orders"]["inserted"] == {"orders": 50, "order_items": 61, "shipments": 58, "refund_requests": 2}

    # The source rows, including intentionally invalid business values, are inserted verbatim.
    with db.tx() as connection:
        for table, source_rows in ((orders, data["orders"]), (order_items, data["order_items"]),
                                   (shipments, data["shipments"])):
            actual = [
                dict(row)
                for row in connection.execute(
                    select(table)
                    .where(table.c.id.in_([source["id"] for source in source_rows]))
                    .order_by(table.c.id)
                ).mappings()
            ]
            expected = sorted(source_rows, key=lambda row: row["id"])
            assert actual == expected, f"{table.name} source/actual mismatch: expected={expected!r}, actual={actual!r}"
        actual_refunds = [
            dict(row)
            for row in connection.execute(
                select(refund_requests)
                .where(refund_requests.c.id.in_([row["id"] for row in data["refund_requests"]]))
                .order_by(refund_requests.c.id)
            ).mappings()
        ]
        assert actual_refunds == sorted(data["refund_requests"], key=lambda row: row["id"])
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        for table in (orders, order_items, shipments, refund_requests):
            assert set(table.c.keys()) == set(data[table.name][0])

    anomalies = {row["id"]: row for row in data["orders"]}
    assert anomalies["O3009"]["paid_amount_cents"] == 0
    assert anomalies["O3010"]["paid_amount_cents"] == -100
    assert anomalies["O3040"]["status"] == "RETURNED"
    assert anomalies["O3005"]["status"] == "PAID"

    with db.tx() as connection:
        # A runtime status change must survive the next fixture initialization.
        connection.execute(orders.update().where(orders.c.id == "O3001").values(status="CANCELLED"))
    second = seed_order_scenario(db, ORDER_FIXTURE_ALL, lambda: fixture_time)
    assert second["products"]["inserted"] == 0
    assert second["orders"]["inserted"] == {"orders": 0, "order_items": 0, "shipments": 0, "refund_requests": 0}
    with db.tx() as connection:
        final_baseline_orders = [
            dict(row)
            for row in connection.execute(
                select(orders).where(orders.c.id.in_([row["id"] for row in baseline_orders])).order_by(orders.c.id)
            ).mappings()
        ]
        final_baseline_items = [
            dict(row)
            for row in connection.execute(
                select(order_items)
                .where(order_items.c.id.in_([row["id"] for row in baseline_items]))
                .order_by(order_items.c.id)
            ).mappings()
        ]
        changed_fixture_order = connection.execute(select(orders.c.status).where(orders.c.id == "O3001")).scalar_one()
        assert connection.scalar(select(func.count()).select_from(orders)) == 55
        assert connection.scalar(select(func.count()).select_from(order_items)) == 66
    assert final_baseline_orders == baseline_orders
    assert final_baseline_items == baseline_items
    assert changed_fixture_order == "CANCELLED"
    db.engine.dispose()


def test_fixture_selector_loads_only_requested_order_and_scenario(tmp_path):
    fixture_time = datetime.fromisoformat(FIXTURE_TIME)
    db = DB("sqlite:///" + str(tmp_path / "selected.db"))
    single = seed_order_scenario(db, "orders_boundary:O3041", lambda: fixture_time)
    assert single["orders"]["order_ids"] == ["O3041"]
    with db.tx() as connection:
        assert connection.scalar(select(func.count()).select_from(orders)) == 6  # five defaults plus the selected order
        assert connection.scalar(select(func.count()).select_from(refund_requests)) == 1
        assert connection.scalar(select(orders.c.id).where(orders.c.id == "O1001")) == "O1001"
    # Separate database proves a named scenario selector loads only that scenario's order IDs.
    scenario_db = DB("sqlite:///" + str(tmp_path / "scenario.db"))
    selected = seed_order_scenario(scenario_db, "orders_boundary:delivered_exactly_7d", lambda: fixture_time)
    assert selected["orders"]["order_ids"] == ["O3028", "O3029"]
    scenario_db.engine.dispose()
    db.engine.dispose()


def test_order_oracle_records_actual_and_expected_for_all_boundary_cases(tmp_path):
    fixture_time = datetime.fromisoformat(FIXTURE_TIME)
    db = DB("sqlite:///" + str(tmp_path / "oracle.db"))
    seed_order_scenario(db, ORDER_FIXTURE_ALL, lambda: fixture_time)
    report = evaluate_order_boundary_oracle(db)

    assert len(load_order_oracle()["cases"]) == 50
    assert len(report["cases"]) == 50
    assert report["fixture_clock_utc"] == FIXTURE_TIME
    assert report["refund_request_count_before"] == report["refund_request_count_after"] == 2
    assert report["refund_request_count_delta"] == 0
    assert report["failures"] == [], json.dumps(report["failures"], ensure_ascii=False, indent=2)
    db.engine.dispose()


async def test_order_chat_shows_items_and_latest_status_per_tracking_number(env):
    clock = lambda: datetime.fromisoformat(FIXTURE_TIME)
    for order_id in ('O3007', 'O3011'):
        seed_order_scenario(env['db'], 'orders_boundary:' + order_id, clock)
    with env['db'].tx() as connection:
        connection.execute(shipments.insert().values(id='second-parcel', order_id='O3011', carrier='模拟快递',
            tracking_no='PARCEL2', status='已签收', updated_at=FIXTURE_TIME))

    agent, actor, conversation = env['agent'], env['a'], env['conv']
    empty = await agent.chat(actor, conversation, '查询订单 O3007', 'deterministic')
    assert [action['name'] for action in empty['tool_actions']] == ['get_order']
    assert empty['tool_actions'][0]['result']['data']['shipments'] == []
    assert '云听 Travel 长续航降噪耳机 × 2' in empty['answer'] and '实付 1098.00 元' in empty['answer']
    assert '暂无物流记录' in empty['answer']

    split = await agent.chat(actor, conversation, '查询订单 O3011', 'deterministic')
    data = split['tool_actions'][0]['result']['data']
    assert [row['status'] for row in data['shipments']] == ['已签收', '运输中']
    assert '已签收' in split['answer'] and '运输中' in split['answer']
    assert '揽收' not in split['answer'] and 'PARCEL2' not in split['answer']

    detail = await agent.chat(actor, conversation, '查 O3011 的物流单号', 'deterministic')
    assert [action['name'] for action in detail['tool_actions']] == ['track_shipment']
    assert 'PARCEL2' in detail['answer'] and 'DEMOO3011' in detail['answer']
    with env['db'].tx() as connection:
        assert connection.scalar(select(func.count()).select_from(refund_requests)) == 0


@pytest.mark.asyncio
async def test_eval_case_selects_boundary_fixture_and_uses_refund_delta(tmp_path):
    case_with_seed_refund = {
        "schema_version": "1.0",
        "case_id": "boundary-existing-refund-count",
        "category": "workflow",
        "fixture": "orders_boundary:O3041",
        "messages": [{"role": "user", "content": "查订单 O3041"}],
        "expected": {"required_tools": ["get_order"], "refund_request_count_delta": 0},
        "tags": [],
    }
    case_same_tenant_admin = {
        "schema_version": "1.0",
        "case_id": "boundary-same-tenant-admin-owner",
        "category": "workflow",
        "fixture": "orders_boundary:O3044",
        "actor_user_id": "admin",
        "messages": [{"role": "user", "content": "查订单 O3044"}],
        "expected": {"required_tools": ["get_order"], "refund_request_count_delta": 0},
        "tags": [],
    }
    dataset = tmp_path / "boundary_existing_refund.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in (case_with_seed_refund, case_same_tenant_admin)) + "\n",
        encoding="utf-8",
    )
    _raw, cases = load_dataset(dataset)
    assert cases[0]["fixture"] == "orders_boundary:O3041"
    assert cases[1]["actor_user_id"] == "admin"

    report = await run(dataset, mode="deterministic", config=settings.model_copy(update={"memory_enabled": False}))
    evaluations = {item["case_id"]: item for item in report["cases"]}
    assert set(evaluations) == {case_with_seed_refund["case_id"], case_same_tenant_admin["case_id"]}
    for evaluated in evaluations.values():
        assert evaluated["passed"] is True, json.dumps(evaluated["results"], ensure_ascii=False, indent=2)
        assert evaluated["final_state"]["refund_request_count_after"] - evaluated["final_state"]["refund_request_count_before"] == 0
        assert evaluated["final_state"]["refund_request_count_delta"] == 0
    assert evaluations[case_with_seed_refund["case_id"]]["final_state"]["refund_request_count_before"] == 1
