"""Isolated order scenario fixtures and their business-boundary Oracle."""

import json
import zipfile
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.business import Actor, Business, DomainError
from app.dataset import FIXTURE_TIME
from app.db import (
    DB,
    order_items,
    orders,
    products,
    refund_requests,
    shipments,
    tenants,
    users,
)
from app.product_fixtures import import_product_fixtures


ORDER_BUNDLE_PATH = Path(__file__).resolve().parents[1] / "kefu_orders_boundary_50_bundle.zip"
ORDER_DATA_MEMBER = "kefu_orders_boundary_50.json"
ORDER_ORACLE_MEMBER = "kefu_orders_boundary_50_oracle.json"
ORDER_FIXTURE_PREFIX = "orders_boundary:"
ORDER_FIXTURE_ALL = ORDER_FIXTURE_PREFIX + "all"
ORDER_TABLES = {
    "orders": orders,
    "order_items": order_items,
    "shipments": shipments,
    "refund_requests": refund_requests,
}


class OrderFixtureError(ValueError):
    """Raised for malformed fixture data, unknown selectors, or missing prerequisites."""


def _read_json_member(member: str) -> dict:
    try:
        with zipfile.ZipFile(ORDER_BUNDLE_PATH) as archive:
            return json.loads(archive.read(member).decode("utf-8"))
    except (OSError, KeyError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise OrderFixtureError(f"Cannot read {member} from {ORDER_BUNDLE_PATH}: {error}") from error


def load_order_fixture_data() -> dict:
    """Read and structurally validate only the business rows, never the bundled SQL."""
    data = _read_json_member(ORDER_DATA_MEMBER)
    if data.get("fixture_version") != "commerce-orders-boundary-v1":
        raise OrderFixtureError("Unsupported order fixture version")
    if data.get("fixture_clock_utc") != FIXTURE_TIME:
        raise OrderFixtureError("Order fixture clock does not match the evaluation clock")

    for name, table in ORDER_TABLES.items():
        rows = data.get(name)
        if not isinstance(rows, list):
            raise OrderFixtureError(f"{name} must be a list")
        expected_fields = set(table.c.keys())
        ids: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise OrderFixtureError(f"{name}[{index}] does not match the existing SQLAlchemy columns")
            row_id = row["id"]
            if not isinstance(row_id, str) or not row_id or row_id in ids:
                raise OrderFixtureError(f"{name} contains an empty or duplicate primary key: {row_id!r}")
            ids.add(row_id)

    order_ids = {row["id"] for row in data["orders"]}
    if len(data["orders"]) != 50 or order_ids != {f"O{number}" for number in range(3001, 3051)}:
        raise OrderFixtureError("Expected the 50 independent order IDs O3001–O3050")
    if len(data["order_items"]) != 61 or len(data["shipments"]) != 58 or len(data["refund_requests"]) != 2:
        raise OrderFixtureError("Order fixture row counts do not match the bundle manifest")

    product_ids = set(data.get("requires_existing_products", []))
    fixture_product_ids = {row["product_id"] for row in data["order_items"]}
    if not fixture_product_ids.issubset(product_ids):
        raise OrderFixtureError("Order items reference products absent from the prerequisite manifest")
    if not all(row["order_id"] in order_ids for name in ("order_items", "shipments", "refund_requests") for row in data[name]):
        raise OrderFixtureError("A child fixture row references an order outside O3001–O3050")
    return data


def load_order_oracle() -> dict:
    """Load gold labels for evaluation code only; the Agent never receives this object."""
    oracle = _read_json_member(ORDER_ORACLE_MEMBER)
    data = load_order_fixture_data()
    expected_ids = {row["id"] for row in data["orders"]}
    oracle_ids = {row.get("order_id") for row in oracle.get("cases", [])}
    if (
        oracle.get("fixture_version") != data["fixture_version"]
        or oracle.get("fixture_clock_utc") != data["fixture_clock_utc"]
        or len(oracle.get("cases", [])) != 50
        or oracle_ids != expected_ids
    ):
        raise OrderFixtureError("Oracle rows do not match the order fixture version, clock, and IDs")
    return oracle


def _select_order_ids(selector: str, data: dict, oracle: dict) -> set[str]:
    if not selector.startswith(ORDER_FIXTURE_PREFIX):
        raise OrderFixtureError(f"Unknown order fixture selector: {selector}")
    selection = selector.removeprefix(ORDER_FIXTURE_PREFIX)
    all_ids = {row["id"] for row in data["orders"]}
    if selection == "all":
        return all_ids
    if selection in all_ids:
        return {selection}
    scenario_ids = {row["order_id"] for row in oracle["cases"] if row.get("scenario") == selection}
    if scenario_ids:
        return scenario_ids
    raise OrderFixtureError(f"Unknown boundary order ID or scenario: {selection}")


def import_order_fixture(db: DB, selector: str) -> dict:
    """Insert a selected order scenario after its users, tenants, and products exist."""
    data = load_order_fixture_data()
    oracle = load_order_oracle()
    selected_ids = _select_order_ids(selector, data, oracle)
    selected = {
        "orders": [row for row in data["orders"] if row["id"] in selected_ids],
        "order_items": [row for row in data["order_items"] if row["order_id"] in selected_ids],
        "shipments": [row for row in data["shipments"] if row["order_id"] in selected_ids],
        "refund_requests": [row for row in data["refund_requests"] if row["order_id"] in selected_ids],
    }

    with db.tx() as connection:
        tenant_ids = set(connection.scalars(select(tenants.c.id)))
        user_rows = connection.execute(select(users.c.id, users.c.tenant_id)).all()
        user_tenants = {user_id: tenant_id for user_id, tenant_id in user_rows}
        existing_product_ids = set(connection.scalars(select(products.c.id)))
        missing_tenants = {row["tenant_id"] for row in selected["orders"]} - tenant_ids
        missing_users = {row["user_id"] for row in selected["orders"]} - set(user_tenants)
        missing_products = {row["product_id"] for row in selected["order_items"]} - existing_product_ids
        mismatched_users = {
            row["user_id"]
            for row in selected["orders"]
            if user_tenants.get(row["user_id"]) != row["tenant_id"]
        }
        product_tenants = dict(connection.execute(select(products.c.id, products.c.tenant_id)).all())
        mismatched_products = {
            row["product_id"]
            for row in selected["order_items"]
            if product_tenants.get(row["product_id"])
            != next(order["tenant_id"] for order in selected["orders"] if order["id"] == row["order_id"])
        }
        if missing_tenants or missing_users or missing_products or mismatched_users or mismatched_products:
            raise OrderFixtureError(
                "Missing or mismatched prerequisites: "
                + repr({
                    "tenants": sorted(missing_tenants),
                    "users": sorted(missing_users),
                    "products": sorted(missing_products),
                    "user_tenant_mismatch": sorted(mismatched_users),
                    "product_tenant_mismatch": sorted(mismatched_products),
                })
            )

        inserted = {}
        for name in ("orders", "order_items", "shipments", "refund_requests"):
            rows = selected[name]
            if not rows:
                inserted[name] = 0
                continue
            result = connection.execute(
                sqlite_insert(ORDER_TABLES[name]).values(rows).on_conflict_do_nothing()
            )
            inserted[name] = max(result.rowcount or 0, 0)

        current_orders = {
            row["id"]: (row["tenant_id"], row["user_id"])
            for row in connection.execute(
                select(orders.c.id, orders.c.tenant_id, orders.c.user_id).where(orders.c.id.in_(selected_ids))
            ).mappings()
        }
        owner_mismatches = [
            row["id"] for row in selected["orders"]
            if current_orders.get(row["id"]) != (row["tenant_id"], row["user_id"])
        ]
        if owner_mismatches:
            raise OrderFixtureError("Existing fixture order IDs belong to another owner: " + ", ".join(owner_mismatches))

    return {
        "selector": selector,
        "order_ids": sorted(selected_ids),
        "total": {name: len(rows) for name, rows in selected.items()},
        "inserted": inserted,
        "skipped_existing": {name: len(selected[name]) - inserted[name] for name in selected},
    }


def seed_order_scenario(db: DB, selector: str, clock=None) -> dict:
    """Use the required seed order: users/tenants/base rows, product fixture, then orders."""
    db.seed(clock)
    product_result = import_product_fixtures(db)
    order_result = import_order_fixture(db, selector)
    return {"products": product_result, "orders": order_result}


def evaluate_order_boundary_oracle(db: DB) -> dict:
    """Compare real Business/MCP-domain behavior to the bundled 50-case gold Oracle."""
    data = load_order_fixture_data()
    oracle = load_order_oracle()
    frozen_time = datetime.fromisoformat(data["fixture_clock_utc"])
    business = Business(db, clock=lambda: frozen_time)
    with db.tx() as connection:
        actor_rows = connection.execute(select(users.c.id, users.c.tenant_id, users.c.role)).all()
        actors = {user_id: Actor(user_id, tenant_id, role) for user_id, tenant_id, role in actor_rows}
        before_refund_ids = set(connection.scalars(select(refund_requests.c.id)))

    source_orders = {row["id"]: row for row in data["orders"]}
    report = {
        "fixture_version": data["fixture_version"],
        "fixture_clock_utc": data["fixture_clock_utc"],
        "cases": [],
        "failures": [],
    }
    def check(case_report: dict, check_name: str, actual, expected) -> None:
        result = {"check": check_name, "actual": actual, "expected": expected, "passed": actual == expected}
        case_report["results"].append(result)
        if not result["passed"]:
            report["failures"].append({"case_id": case_report["case_id"], **result})

    for oracle_case in oracle["cases"]:
        order_id = oracle_case["order_id"]
        source_order = source_orders[order_id]
        owner = actors[oracle_case["owner_user_id"]]
        user_a = actors["user_a"]
        preview_status = None
        conversation_id = business.conversation(owner)["id"]
        try:
            preview_status = business.preview(owner, conversation_id, order_id)["status"]
        except DomainError as error:
            preview_status = "BUSINESS_RULE_FAILED" if error.code == "BUSINESS_RULE_FAILED" else error.code

        try:
            eligibility = business.tool(owner, "check_refund_eligibility", {"order_id": order_id})
            eligibility_actual = eligibility["reason_code"]
        except DomainError as error:
            eligibility_actual = error.code
        try:
            owner_order = business.tool(owner, "get_order", {"order_id": order_id})
            owner_access = "ALLOW"
        except DomainError as error:
            owner_order = {"items": []}
            owner_access = error.code
        try:
            owner_shipments = business.tool(owner, "track_shipment", {"order_id": order_id})
            shipment_access = "ALLOW"
        except DomainError as error:
            owner_shipments = []
            shipment_access = error.code
        source_items = sorted(
            (row for row in data["order_items"] if row["order_id"] == order_id), key=lambda row: row["id"]
        )
        source_shipments = sorted(
            (row for row in data["shipments"] if row["order_id"] == order_id), key=lambda row: row["id"]
        )
        actual_items = sorted(owner_order["items"], key=lambda row: row["id"])
        actual_shipments = sorted(owner_shipments, key=lambda row: row["id"])
        try:
            business.tool(user_a, "get_order", {"order_id": order_id})
            user_a_access = "ALLOW"
        except DomainError as error:
            user_a_access = error.code

        unauthorized = {}
        for outsider_id, outsider in actors.items():
            if outsider_id == owner.user_id:
                continue
            for tool_name in ("get_order", "track_shipment", "check_refund_eligibility"):
                try:
                    business.tool(outsider, tool_name, {"order_id": order_id})
                    unauthorized[f"{outsider_id}:{tool_name}"] = "ALLOW"
                except DomainError as error:
                    unauthorized[f"{outsider_id}:{tool_name}"] = error.code

        with db.tx() as connection:
            after_refund_ids = set(connection.scalars(select(refund_requests.c.id)))
        preview_refund_delta = len(after_refund_ids - before_refund_ids)
        case_report = {
            "case_id": order_id,
            "scenario": oracle_case["scenario"],
            "flags": oracle_case.get("flags", []),
            "description": oracle_case.get("description", ""),
            "oracle": oracle_case,
            "actual": {
                "status": source_order["status"],
                "eligibility_as_owner": eligibility_actual,
                "get_order_as_owner": owner_access,
                "get_order_as_user_a": user_a_access,
                "refund_preview_as_owner": preview_status,
                "order_items": actual_items,
                "shipment_query_as_owner": shipment_access,
                "shipments_as_owner": actual_shipments,
                "non_owner_access": unauthorized,
                "preview_refund_count_delta": preview_refund_delta,
            },
            "results": [],
        }
        check(case_report, "status_preserved", source_order["status"], oracle_case["status"])
        check(
            case_report,
            "eligibility_as_owner",
            eligibility_actual,
            oracle_case["expected_eligibility_as_owner"],
        )
        check(case_report, "get_order_as_owner", owner_access, "ALLOW")
        check(
            case_report,
            "get_order_as_user_a",
            user_a_access,
            oracle_case["expected_get_order_as_user_a"],
        )
        check(
            case_report,
            "refund_preview_as_owner",
            preview_status,
            oracle_case["expected_refund_preview_as_owner"],
        )
        check(case_report, "get_order_items", actual_items, source_items)
        check(case_report, "shipment_query_as_owner", shipment_access, "ALLOW")
        check(case_report, "track_shipment_as_owner", actual_shipments, source_shipments)
        expected_unauthorized = {
            f"{outsider_id}:{tool_name}": "NOT_FOUND"
            for outsider_id in actors
            if outsider_id != owner.user_id
            for tool_name in ("get_order", "track_shipment", "check_refund_eligibility")
        }
        check(case_report, "non_owner_access", unauthorized, expected_unauthorized)
        check(case_report, "preview_created_no_refund", preview_refund_delta, 0)
        report["cases"].append(case_report)

    with db.tx() as connection:
        final_refund_count = connection.scalar(select(func.count()).select_from(refund_requests))
    report["refund_request_count_before"] = len(before_refund_ids)
    report["refund_request_count_after"] = final_refund_count
    report["refund_request_count_delta"] = final_refund_count - len(before_refund_ids)
    return report
