import json

import pytest
from sqlalchemy import func, select

from app.agent import Agent
from app.business import Actor, Business
from app.config import settings
from app.db import DB, order_items, orders, products
from app.fakes import FakeModels
from app.mcp import Gateway, create_mcp
from app.product_fixtures import PRODUCT_FIELDS, PRODUCT_FIXTURE_PATH, import_product_fixtures, load_product_fixture
from app.rag import RAG


def test_product_fixture_contract_and_integrity():
    rows = load_product_fixture()

    assert len(rows) == 20
    assert set(products.c.keys()) == PRODUCT_FIELDS
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(type(row["price_cents"]) is int and row["price_cents"] > 0 for row in rows)
    assert all(type(row["stock"]) is int and row["stock"] >= 0 for row in rows)
    assert all(isinstance(json.loads(row["attributes_json"]), dict) for row in rows)
    assert {row["id"] for row in rows if row["stock"] == 0} == {
        "yt-mini-tws",
        "yt-run-bone",
        "yt-ultra-anc",
        "yt-outdoor-speaker",
    }
    assert PRODUCT_FIXTURE_PATH.is_file()


@pytest.mark.asyncio
async def test_import_is_idempotent_preserves_existing_data_and_is_tenant_scoped(tmp_path):
    cfg = settings.model_copy(
        update={
            "sqlite_url": "sqlite:///" + str(tmp_path / "products.db"),
            "checkpoint_db_path": str(tmp_path / "checkpoints.db"),
            "chroma_path": str(tmp_path / "chroma"),
            "bm25_index_path": str(tmp_path / "bm25"),
            "memory_enabled": False,
            "mcp_service_secret": "test-product-fixture-secret",
        }
    )
    db = DB(cfg.sqlite_url)
    db.seed()
    business = Business(db)
    tenant_a = Actor("user_a", "a")
    tenant_b = Actor("user_b", "b")
    with db.tx() as connection:
        connection.execute(products.update().where(products.c.id == "headphones-a").values(stock=3))
        original_a = dict(connection.execute(select(products).where(products.c.id == "headphones-a")).mappings().one())
        original_b = dict(connection.execute(select(products).where(products.c.id == "headphones-b")).mappings().one())
        original_orders = [dict(row) for row in connection.execute(select(orders).order_by(orders.c.id)).mappings()]
        original_items = [dict(row) for row in connection.execute(select(order_items).order_by(order_items.c.id)).mappings()]

    first = import_product_fixtures(db)
    second = import_product_fixtures(db)
    assert first["total"] == 20 and first["inserted"] == 19 and first["skipped_existing"] == 1
    assert second["total"] == 20 and second["inserted"] == 0 and second["skipped_existing"] == 20

    with db.tx() as connection:
        a_row = dict(connection.execute(select(products).where(products.c.id == "headphones-a")).mappings().one())
        b_row = dict(connection.execute(select(products).where(products.c.id == "headphones-b")).mappings().one())
        current_orders = [dict(row) for row in connection.execute(select(orders).order_by(orders.c.id)).mappings()]
        current_items = [dict(row) for row in connection.execute(select(order_items).order_by(order_items.c.id)).mappings()]
        row_count = connection.scalar(select(func.count()).select_from(products))
        distinct_ids = connection.scalar(select(func.count(func.distinct(products.c.id))))
    assert a_row == original_a and a_row["stock"] == 3
    assert b_row == original_b
    assert current_orders == original_orders
    assert current_items == original_items
    assert row_count == distinct_ids == 21

    server = create_mcp(business, cfg.mcp_service_secret, testing=True)
    gateway = Gateway(business, cfg, server)
    a_result = await gateway.call(tenant_a, "search_products", {"keyword": "", "limit": 20}, "product-consultation")
    b_result = await gateway.call(tenant_b, "search_products", {"keyword": "", "limit": 20}, "product-consultation")
    assert a_result["ok"] and len(a_result["data"]) == 20
    assert [row["id"] for row in a_result["data"]] == sorted(row["id"] for row in a_result["data"])
    assert {row["tenant_id"] for row in a_result["data"]} == {"a"}
    assert {row["id"] for row in a_result["data"]} == {row["id"] for row in load_product_fixture()}
    assert b_result["ok"] and [row["id"] for row in b_result["data"]] == ["headphones-b"]
    assert {row["tenant_id"] for row in b_result["data"]} == {"b"}
    no_stock = {row["id"]: row["stock"] for row in a_result["data"] if row["stock"] == 0}
    assert no_stock == {
        "yt-mini-tws": 0,
        "yt-run-bone": 0,
        "yt-ultra-anc": 0,
        "yt-outdoor-speaker": 0,
    }

    # A recommendation can see stock data but must not present sold-out items as choices.
    rag = RAG(db, FakeModels(), cfg)
    agent = Agent(business, gateway, cfg, rag)
    try:
        recommendation = await agent.chat(
            tenant_a,
            business.conversation(tenant_a)["id"],
            "推荐一款耳机，预算不超过1000元",
            mode="deterministic",
        )
        assert recommendation["tool_actions"][0]["name"] == "search_products"
        assert "库存 0 件" not in recommendation["answer"]
        assert "云听 Mini 轻巧耳机" not in recommendation["answer"]
        assert "云听 Run 骨传导运动耳机" not in recommendation["answer"]
    finally:
        await agent.close()
        db.engine.dispose()
