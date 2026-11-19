"""Validated, insert-only loader for the local demo product catalog fixture."""

import json
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.db import products, tenants


PRODUCT_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "kefu_products_seed_20.json"
PRODUCT_FIELDS = {
    "id",
    "tenant_id",
    "name",
    "category",
    "price_cents",
    "stock",
    "attributes_json",
}
PRODUCT_FIXTURE_SIZE = 20


class ProductFixtureError(ValueError):
    """Raised when the authored demo fixture violates the products contract."""


def load_product_fixture(path: Path = PRODUCT_FIXTURE_PATH) -> list[dict]:
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductFixtureError(f"Cannot read product fixture {path}: {error}") from error

    if not isinstance(rows, list) or len(rows) != PRODUCT_FIXTURE_SIZE:
        raise ProductFixtureError(f"Expected a list of exactly {PRODUCT_FIXTURE_SIZE} product rows")

    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        location = f"product fixture row {index}"
        if not isinstance(row, dict) or set(row) != PRODUCT_FIELDS:
            raise ProductFixtureError(f"{location} must contain exactly the existing products columns")
        for key in ("id", "tenant_id", "name", "category", "attributes_json"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ProductFixtureError(f"{location}.{key} must be a non-empty string")
        if row["id"] in seen_ids:
            raise ProductFixtureError(f"Duplicate product id in fixture: {row['id']}")
        seen_ids.add(row["id"])
        if type(row["price_cents"]) is not int or row["price_cents"] <= 0:
            raise ProductFixtureError(f"{location}.price_cents must be a positive integer")
        if type(row["stock"]) is not int or row["stock"] < 0:
            raise ProductFixtureError(f"{location}.stock must be a non-negative integer")
        try:
            attributes = json.loads(row["attributes_json"])
        except json.JSONDecodeError as error:
            raise ProductFixtureError(f"{location}.attributes_json is not valid JSON") from error
        if not isinstance(attributes, dict):
            raise ProductFixtureError(f"{location}.attributes_json must encode a JSON object")

    return rows


def import_product_fixtures(db, path: Path = PRODUCT_FIXTURE_PATH) -> dict:
    """Insert fixture IDs that are missing; existing rows are never updated."""
    rows = load_product_fixture(path)
    with db.tx() as connection:
        known_tenants = set(connection.scalars(select(tenants.c.id)))
        missing_tenants = sorted({row["tenant_id"] for row in rows} - known_tenants)
        if missing_tenants:
            raise ProductFixtureError(f"Fixture references unknown tenants: {', '.join(missing_tenants)}")

        result = connection.execute(
            sqlite_insert(products)
            .values(rows)
            .on_conflict_do_nothing(index_elements=[products.c.id])
        )
        imported_ids = [row["id"] for row in rows]
        existing = connection.execute(
            select(products.c.id, products.c.tenant_id).where(products.c.id.in_(imported_ids))
        ).all()
        expected_tenants = {row["id"]: row["tenant_id"] for row in rows}
        wrong_tenant_ids = [
            product_id
            for product_id, tenant_id in existing
            if expected_tenants[product_id] != tenant_id
        ]
        if wrong_tenant_ids:
            raise ProductFixtureError(
                "Fixture IDs already belong to another tenant: " + ", ".join(sorted(wrong_tenant_ids))
            )

    inserted = max(result.rowcount or 0, 0)
    return {
        "total": len(rows),
        "inserted": inserted,
        "skipped_existing": len(rows) - inserted,
        "ids": [row["id"] for row in rows],
    }


def merge_missing_static_attributes(db, path: Path = PRODUCT_FIXTURE_PATH) -> dict:
    """Fill missing fixture attributes without replacing existing product state.

    Existing static values must agree with the authored fixture. Extra attributes
    are retained. Price, stock, identity, and every other table are left alone.
    """
    rows = load_product_fixture(path)
    changed_ids: list[str] = []
    with db.tx() as connection:
        for expected in rows:
            current = connection.execute(
                select(products).where(products.c.id == expected["id"])
            ).mappings().first()
            if current is None:
                raise ProductFixtureError(f"Cannot merge static attributes; missing product {expected['id']}")

            for field in ("tenant_id", "name", "category"):
                if current[field] != expected[field]:
                    raise ProductFixtureError(
                        f"Product {expected['id']} has conflicting static field {field}"
                    )

            try:
                existing_attributes = json.loads(current["attributes_json"])
                expected_attributes = json.loads(expected["attributes_json"])
            except json.JSONDecodeError as error:
                raise ProductFixtureError(
                    f"Product {expected['id']} has invalid attributes_json"
                ) from error
            if not isinstance(existing_attributes, dict):
                raise ProductFixtureError(f"Product {expected['id']} attributes_json must be an object")

            conflicts = [
                key for key, value in existing_attributes.items()
                if key in expected_attributes and expected_attributes[key] != value
            ]
            if conflicts:
                raise ProductFixtureError(
                    f"Product {expected['id']} has conflicting static attributes: {', '.join(sorted(conflicts))}"
                )

            merged = dict(expected_attributes)
            merged.update(existing_attributes)
            if merged != existing_attributes:
                connection.execute(
                    update(products)
                    .where(products.c.id == expected["id"])
                    .values(attributes_json=json.dumps(merged, ensure_ascii=False, separators=(",", ":")))
                )
                changed_ids.append(expected["id"])

    return {
        "total": len(rows),
        "updated_static_attributes": len(changed_ids),
        "ids": changed_ids,
    }
