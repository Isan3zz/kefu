import json
from pathlib import Path

import pytest
from sqlalchemy import select, update
import yaml

from app.db import document_chunks, documents, order_items, orders, products
from app.dataset import load_dataset
from app.product_fixtures import import_product_fixtures, merge_missing_static_attributes
from app.rag import chunks
from scripts.run_rag_v3_experiments import validate_static_product_manuals


ROOT = Path(__file__).resolve().parents[1]
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


def _read_doc(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, front, body = text.split("---", 2)
    return yaml.safe_load(front), body.strip()


def _doc_id_from_source(source):
    return Path(source).stem


def test_rag_v3_import_counts_frontmatter_and_chunk_ids():
    manifest = json.loads((ROOT / "datasets/rag_v3_manifest.json").read_text(encoding="utf-8"))
    chunk_map = json.loads((ROOT / "datasets/rag_v3_gold_evidence_chunk_map.json").read_text(encoding="utf-8"))
    paths = sorted((ROOT / "knowledge/expansion_v3").rglob("*.md"))
    assert len(paths) == manifest["docs"] == 50
    assert manifest["by_category"] == {
        "manuals": 20,
        "policies": 8,
        "troubleshooting": 8,
        "logistics": 6,
        "special": 8,
    }
    ids = set()
    total_chunks = 0
    for path in paths:
        metadata, body = _read_doc(path)
        assert {"id", "title", "version", "tenant_id", "status", "visibility", "effective_at"} <= set(metadata)
        assert metadata["id"].startswith("kbv3_")
        assert metadata["id"] not in ids
        ids.add(metadata["id"])
        parts = chunks(body)
        total_chunks += len(parts)
        actual_chunk_ids = []
        import hashlib
        for index, part in enumerate(parts):
            content_hash = hashlib.sha256(part.encode()).hexdigest()
            actual_chunk_ids.append(
                hashlib.sha256(f"{metadata['id']}:{metadata['version']}:{index}:{content_hash}".encode())
                .hexdigest()[:24]
            )
        assert chunk_map[metadata["id"]]["chunk_ids"] == actual_chunk_ids
    assert total_chunks == manifest["total_chunks"] == 93
    assert len(ids) == len(chunk_map) == 50
    original_ids = set()
    for path in (ROOT / "knowledge").rglob("*.md"):
        if "expansion_v3" in path.parts:
            continue
        metadata, _ = _read_doc(path)
        original_ids.add(metadata["id"])
    assert len(original_ids) == 30
    assert not (ids & original_ids)


def test_rag_v3_gold_contracts_and_tenant_b_adapter():
    _, tenant_a = load_dataset(ROOT / "datasets/rag_v3_positive_51_tenant_a_ready.jsonl")
    _, tenant_scoped = load_dataset(ROOT / "datasets/rag_v3_positive_52_tenant_scoped.jsonl")
    _, tenant_b = load_dataset(ROOT / "datasets/rag_v3_positive_01_tenant_b_scoped.jsonl")
    extended = [
        json.loads(line)
        for line in (ROOT / "datasets/rag_v3_gold_60_extended.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(tenant_a) == 51
    assert len(tenant_scoped) == 52
    assert len(tenant_b) == 1 and tenant_b[0]["actor_user_id"] == "user_b"
    assert tenant_scoped[-1]["actor_user_id"] == "user_b"
    assert len(extended) == 60
    assert all(case.get("gold_chunks") for case in extended if case["kind"] != "negative")
    negatives = [case for case in extended if case["kind"] == "negative"]
    assert len(negatives) == 8
    assert all(not case.get("gold_chunks") for case in negatives)
    expected_behaviors = {
        "abstain_or_clarify",
        "reject_invalid_premise_with_current_policy",
        "deny_unauthorized_knowledge",
        "refuse_unsafe_action",
    }
    assert {case["expected_behavior"] for case in negatives} == expected_behaviors
    multi_doc = [case for case in extended if case.get("kind") == "multi_doc"]
    assert [case["case_id"] for case in multi_doc] == [f"kbv3-{i:03d}" for i in range(43, 51)]
    assert all(len(case["gold_documents"]) >= 2 for case in multi_doc)
    extended_by_id = {case["case_id"]: case for case in extended}
    for case in tenant_scoped:
        oracle = extended_by_id[case["case_id"]]
        assert case["expected"]["gold_chunks"] == oracle["gold_chunks"]
        if oracle["tenant_id"] == "b":
            assert case["actor_user_id"] == "user_b"
        else:
            assert oracle["tenant_id"] == "a"


def test_static_product_attribute_merge_is_additive_and_keeps_orders(env):
    with env["db"].tx() as connection:
        connection.execute(products.update().where(products.c.id == "headphones-a").values(price_cents=31415, stock=7))
        before_orders = [dict(row) for row in connection.execute(select(orders).order_by(orders.c.id)).mappings()]
        before_items = [dict(row) for row in connection.execute(select(order_items).order_by(order_items.c.id)).mappings()]
        before_b = dict(connection.execute(select(products).where(products.c.id == "headphones-b")).mappings().one())

    import_product_fixtures(env["db"])
    first = merge_missing_static_attributes(env["db"])
    second = merge_missing_static_attributes(env["db"])
    assert first["updated_static_attributes"] == 1
    assert first["ids"] == ["headphones-a"]
    assert second["updated_static_attributes"] == 0
    assert validate_static_product_manuals(env["db"]) == {
        "products_checked": 20,
        "manuals_checked": 20,
        "static_fields_match": True,
    }

    with env["db"].tx() as connection:
        product_a = dict(connection.execute(select(products).where(products.c.id == "headphones-a")).mappings().one())
        product_b = dict(connection.execute(select(products).where(products.c.id == "headphones-b")).mappings().one())
        after_orders = [dict(row) for row in connection.execute(select(orders).order_by(orders.c.id)).mappings()]
        after_items = [dict(row) for row in connection.execute(select(order_items).order_by(order_items.c.id)).mappings()]
    assert product_a["price_cents"] == 31415 and product_a["stock"] == 7
    assert product_b == before_b
    assert after_orders == before_orders
    assert after_items == before_items


@pytest.mark.parametrize("mode", MODES)
def test_v3_expired_future_internal_draft_and_foreign_chunks_are_filtered(env, mode):
    rag = env["rag"]
    rag.build(ROOT / "knowledge")
    query = "2024 15 天退款 2030 卫星音频 CASE-STAFF-ALPHA 租户 B 专属配对服务"
    results = rag.search(query, "a", mode=mode, k=40)
    forbidden = {
        "kbv3_special_old_refund_archived",
        "kbv3_special_old_policy_expired",
        "kbv3_special_future_refund",
        "kbv3_special_unpublished_release",
        "kbv3_special_staff_only",
        "kbv3_special_tenant_b_only",
    }
    assert results
    assert not ({_doc_id_from_source(result["source"]) for result in results} & forbidden)


@pytest.mark.parametrize("mode", MODES)
def test_v3_stale_document_version_is_filtered(env, mode):
    rag = env["rag"]
    rag.build(ROOT / "knowledge")
    with env["db"].tx() as connection:
        old_chunk_ids = set(connection.scalars(
            select(document_chunks.c.chunk_id).where(document_chunks.c.document_id == "kbv3_product_headphones_a")
        ))
        connection.execute(
            update(documents).where(documents.c.id == "kbv3_product_headphones_a").values(version="2")
        )
    results = rag.search("YT-BASE 是否支持主动降噪和双设备连接", "a", mode=mode, k=40)
    assert results
    assert not ({result["chunk_id"] for result in results} & old_chunk_ids)


@pytest.mark.parametrize("mode", MODES)
def test_v3_tenant_b_only_document_never_appears_for_tenant_a(env, mode):
    rag = env["rag"]
    rag.build(ROOT / "knowledge")
    with env["db"].tx() as connection:
        tenant_b_chunks = set(connection.scalars(
            select(document_chunks.c.chunk_id).where(document_chunks.c.document_id == "kbv3_special_tenant_b_only")
        ))
    result_a = rag.search("租户 b 基础耳机蓝牙配对 清理旧配对重新搜索", "a", mode=mode, k=40)
    assert not ({result["chunk_id"] for result in result_a} & tenant_b_chunks)
    if mode == "bm25":
        result_b = rag.search("租户 b 基础耳机蓝牙配对 清理旧配对重新搜索", "b", mode=mode, k=40)
        assert {result["chunk_id"] for result in result_b} & tenant_b_chunks


def test_v3_multi_document_gold_queries_retrieve_multiple_sources(env):
    rag = env["rag"]
    rag.build(ROOT / "knowledge")
    rows = [
        json.loads(line)
        for line in (ROOT / "datasets/rag_v3_gold_60_extended.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    multi_doc_cases = [row for row in rows if row.get("kind") == "multi_doc"]
    coverage = {}
    for case in multi_doc_cases:
        results = rag.search(case["query"], case["tenant_id"], mode="hybrid_rerank", k=10)
        actual_documents = {_doc_id_from_source(result["source"]) for result in results}
        expected_documents = set(case["gold_documents"])
        coverage[case["case_id"]] = {
            "expected_document_ids": sorted(expected_documents),
            "actual_document_ids": sorted(actual_documents),
            "recalled_document_count": len(actual_documents & expected_documents),
        }
    assert coverage["kbv3-043"]["recalled_document_count"] == 3
    assert sum(row["recalled_document_count"] >= 2 for row in coverage.values()) >= 6, coverage


@pytest.mark.parametrize("mode", MODES)
def test_v3_unreleased_product_no_answer_does_not_expose_draft(env, mode):
    rag = env["rag"]
    rag.build(ROOT / "knowledge")
    results = rag.search("YT-SECRET-9 卫星音频耳机什么时候上架", "a", mode=mode, k=10)
    assert all(_doc_id_from_source(result["source"]) != "kbv3_special_unpublished_release" for result in results)
    assert all("YT-SECRET-9" not in result["content"] for result in results)


@pytest.mark.parametrize("mode", MODES)
def test_v3_empty_visible_corpus_returns_no_answer(env, mode):
    rag = env["rag"]
    rag.build(ROOT / "knowledge")
    with env["db"].tx() as connection:
        connection.execute(update(documents).values(status="withdrawn"))
    assert rag.search("这份资料里没有答案", "a", mode=mode) == []
