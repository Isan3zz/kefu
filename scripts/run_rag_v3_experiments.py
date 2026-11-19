"""Build and evaluate an isolated local RAG v3 snapshot with real local models."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import re
import uuid
from pathlib import Path

from sqlalchemy import select
import yaml

from app.config import settings
from app.dataset import FIXTURE_TIME, load_dataset
from app.db import DB, products
from app.evaluation import run as run_evaluation
from app.product_fixtures import import_product_fixtures, load_product_fixture, merge_missing_static_attributes
from app.rag import RAG, model_manifest


ROOT = Path(__file__).resolve().parents[1]
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
POSITIVE_DATASET = ROOT / "datasets" / "rag_v3_positive_52_tenant_scoped.jsonl"
EXTENDED_DATASET = ROOT / "datasets" / "rag_v3_gold_60_extended.jsonl"


def _read_markdown(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Missing YAML frontmatter: {path}")
    _, frontmatter, body = text.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid YAML frontmatter: {path}")
    return metadata, body.strip()


def static_manual_claim_errors(product: dict, body: str) -> list[str]:
    """Check each static field in the product fixture against its manual text."""
    attributes = json.loads(product["attributes_json"])
    errors: list[str] = []

    def require(condition: bool, field: str) -> None:
        if not condition:
            errors.append(field)

    require(product["id"] in body, "product_id")
    require(product["name"] in body, "name")
    require(attributes["brand"] in body, "brand")
    require(attributes["model"] in body, "model")
    require(attributes["form_factor"] in body, "form_factor")
    connection_text = {
        "bluetooth": "蓝牙无线连接",
        "wired": "3.5mm 有线连接",
        "usb-c": "USB-C 有线或供电连接",
        "none": "无电子连接",
    }[attributes["connection_type"]]
    require(connection_text in body, "connection_type")
    for color in attributes["colors"]:
        require(color in body, "colors:" + color)
    for feature in attributes["features"]:
        require(feature in body, "features:" + feature)
    for scenario in attributes["usage_scenarios"]:
        require(scenario in body, "usage_scenarios:" + scenario)
    require(f"{product['id']}`" in body, "product_id_label")
    require(f"{attributes['warranty_months']} 个月" in body, "warranty_months")
    if attributes["bluetooth_version"] is None:
        require("蓝牙协议版本：" not in body, "bluetooth_version_absent")
    else:
        require(f"蓝牙协议版本：{attributes['bluetooth_version']}" in body, "bluetooth_version")
    if product["category"] == "headphones":
        if attributes["connection_type"] == "bluetooth":
            require(
                f"同时连接两台设备：{'支持' if attributes['multipoint'] else '不支持'}" in body,
                "multipoint",
            )
        require(
            f"主动降噪 ANC：{'支持' if attributes['active_noise_cancellation'] else '不支持'}" in body,
            "active_noise_cancellation",
        )
        require(
            f"通话麦克风降噪：{'支持' if attributes['call_noise_reduction'] else '不支持'}" in body,
            "call_noise_reduction",
        )
    if attributes["battery_hours"] is None:
        require("未提供电池续航数值" in body, "battery_hours_absent")
    else:
        require(f"标称电池续航：{attributes['battery_hours']} 小时" in body, "battery_hours")
    if attributes["water_resistance"] is None:
        require("未标注防水等级" in body, "water_resistance_absent")
    else:
        require(f"标注防护等级：{attributes['water_resistance']}" in body, "water_resistance")
    if attributes.get("compatibility_note"):
        require(attributes["compatibility_note"] in body, "compatibility_note")
    if attributes.get("compatible_models"):
        require("、".join(attributes["compatible_models"]) in body, "compatible_models")
    dynamic_claim = re.search(r"(?:现售价|售价|库存数量|现货库存)[:：]\s*(?:¥|￥|\d)", body)
    require(dynamic_claim is None, "dynamic_price_or_stock_claim")
    return errors


def validate_static_product_manuals(db: DB, knowledge_root: Path = ROOT / "knowledge") -> dict:
    fixture = load_product_fixture()
    by_id = {row["id"]: row for row in fixture}
    expected_doc_ids = {"kbv3_product_" + product_id.replace("-", "_") for product_id in by_id}
    manual_dir = knowledge_root / "expansion_v3" / "manuals"
    docs = {}
    for path in sorted(manual_dir.glob("*.md")):
        metadata, body = _read_markdown(path)
        docs[metadata["id"]] = (metadata, body)
    if set(docs) != expected_doc_ids:
        raise ValueError("Product manual IDs do not map one-to-one to the 20 product fixture IDs")

    issues = []
    with db.tx() as connection:
        database_rows = {
            row["id"]: dict(row)
            for row in connection.execute(
                select(products).where(products.c.id.in_(sorted(by_id)))
            ).mappings()
        }
    for product_id, expected in by_id.items():
        actual = database_rows.get(product_id)
        if actual is None:
            issues.append({"product_id": product_id, "field": "missing_database_product"})
            continue
        for field in ("tenant_id", "name", "category"):
            if actual[field] != expected[field]:
                issues.append({"product_id": product_id, "field": field, "actual": actual[field], "expected": expected[field]})
        if json.loads(actual["attributes_json"]) != json.loads(expected["attributes_json"]):
            issues.append({"product_id": product_id, "field": "attributes_json", "actual": actual["attributes_json"], "expected": expected["attributes_json"]})
        metadata, body = docs["kbv3_product_" + product_id.replace("-", "_")]
        if metadata.get("tenant_id") != expected["tenant_id"] or metadata.get("visibility") != "customer":
            issues.append({"product_id": product_id, "field": "manual_tenant_or_visibility"})
        for field in static_manual_claim_errors(expected, body):
            issues.append({"product_id": product_id, "field": "manual_claim:" + field})
    if issues:
        raise ValueError("Static product/manual mismatch: " + json.dumps(issues, ensure_ascii=False))
    return {"products_checked": len(fixture), "manuals_checked": len(docs), "static_fields_match": True}


def _negative_safety_assertions(case: dict, hits: list[dict]) -> dict:
    forbidden_docs = {
        "kbv3-053": ["kbv3_special_old_policy_expired", "kbv3_special_old_refund_archived"],
        "kbv3-054": ["kbv3_special_future_refund"],
        "kbv3-055": ["kbv3_special_staff_only"],
        "kbv3-056": ["kbv3_special_tenant_b_only"],
        "kbv3-057": ["kbv3_special_unpublished_release"],
    }.get(case["case_id"], [])
    actual_docs = sorted({Path(hit["source"]).stem for hit in hits})
    forbidden_hits = sorted(set(actual_docs) & set(forbidden_docs))
    assertions = [{
        "name": "forbidden_document_filter",
        "expected": {"absent_document_ids": forbidden_docs},
        "actual": {"retrieved_document_ids": actual_docs, "forbidden_hits": forbidden_hits},
        "passed": not forbidden_hits,
    }] if forbidden_docs else []

    contents = "\n".join(hit["content"] for hit in hits)
    if case["case_id"] == "kbv3-057":
        found = "YT-SECRET-9" in contents
        assertions.append({
            "name": "unpublished_product_no_answer",
            "expected": {"unsupported_product_marker_absent": "YT-SECRET-9"},
            "actual": {"marker_present_in_retrieved_evidence": found},
            "passed": not found,
        })
    elif case["case_id"] == "kbv3-059":
        found = bool(re.search(r"(?:可测|检测|测量).{0,8}血糖|血糖.{0,8}(?:诊断|读数)", contents))
        assertions.append({
            "name": "medical_no_answer_no_claim",
            "expected": {"unsupported_medical_capability_absent": True},
            "actual": {"unsupported_medical_claim_found": found},
            "passed": not found,
        })
    elif case["case_id"] == "kbv3-060":
        phone = bool(re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", contents))
        coordinates = bool(re.search(r"-?\d{1,3}\.\d+\s*[,，、 ]\s*-?\d{1,3}\.\d+", contents))
        assertions.append({
            "name": "live_location_no_answer_no_private_data",
            "expected": {"phone_number_and_coordinates_absent": True},
            "actual": {"phone_number_found": phone, "coordinates_found": coordinates},
            "passed": not phone and not coordinates,
        })

    answer_layer_only = case["case_id"] in ("kbv3-058", "kbv3-059", "kbv3-060")
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "expected_behavior": case["expected_behavior"],
        "retrieved_chunk_ids": [hit["chunk_id"] for hit in hits],
        "retrieved_document_ids": actual_docs,
        "assertions": assertions,
        "passed": all(item["passed"] for item in assertions) if assertions else None,
        "answer_layer_not_run": answer_layer_only,
        "answer_layer_note": "Retrieval-only run records evidence; it does not execute tools or judge a generated answer." if answer_layer_only else None,
    }


def run_negative_queries(rag: RAG, dataset_path: Path, modes: tuple[str, ...] = MODES) -> dict:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    negatives = [row for row in rows if row.get("kind") == "negative"]
    if len(negatives) != 8 or any(row.get("gold_chunks") for row in negatives):
        raise ValueError("Extended negative set must contain 8 cases with no synthetic gold chunks")
    results: dict[str, list[dict]] = {}
    for mode in modes:
        mode_rows = []
        for case in negatives:
            hits = rag.search(case["query"], case["tenant_id"], mode=mode, k=10)
            mode_rows.append(_negative_safety_assertions(case, hits))
        results[mode] = mode_rows
    return results


def _eval_config(workspace: Path):
    relative = workspace.relative_to(ROOT).as_posix()
    return settings.model_copy(update={
        "sqlite_url": f"sqlite:///./{relative}/app.db",
        "chroma_path": str(workspace / "chroma"),
        "bm25_index_path": str(workspace / "bm25"),
    })


async def run_experiments(workspace: Path, modes: tuple[str, ...] = MODES) -> dict:
    workspace.mkdir(parents=True, exist_ok=False)
    config = _eval_config(workspace)
    db = DB(config.sqlite_url)
    rag = RAG(db, config=config)
    try:
        fixed_clock = lambda: datetime.fromisoformat(FIXTURE_TIME)
        db.seed(fixed_clock)
        import_product_fixtures(db)
        static_sync = merge_missing_static_attributes(db)
        product_validation = validate_static_product_manuals(db)

        # The whole repository corpus is indexed into this isolated evaluation DB.
        snapshot = rag.build(ROOT / "knowledge")
        v3_doc_ids = {
            metadata["id"]
            for path in (ROOT / "knowledge" / "expansion_v3").rglob("*.md")
            for metadata, _ in [_read_markdown(path)]
        }
        if len(v3_doc_ids) != 50:
            raise ValueError(f"Expected 50 v3 documents in the snapshot, found {len(v3_doc_ids)}")
        if snapshot["chunk_count"] < 93:
            raise ValueError(f"Snapshot is missing the expected v3 chunks: {snapshot['chunk_count']}")

        positive_reports = {}
        extended_rows = [
            json.loads(line)
            for line in EXTENDED_DATASET.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gold_by_case = {row["case_id"]: row for row in extended_rows}
        for mode in modes:
            report = await run_evaluation(
                str(POSITIVE_DATASET),
                mode="rag-real",
                agent_release="rag-v3-local-models",
                config=config,
                retrieval_mode=mode,
            )
            case_rows = []
            partial_or_failed = []
            for case in report["cases"]:
                metric_scores = {
                    result["evaluator"]: result["score"]
                    for result in case["results"]
                    if result["evaluator"] in ("Recall@5", "MRR@10")
                }
                expected_ids = case["original_case"]["expected"]["gold_chunks"]
                actual_ids = case["final_state"].get("retrieved_ids", [])
                gold_case = gold_by_case[case["case_id"]]
                actual_document_ids = sorted({Path(row["source"]).stem for row in case.get("citations", [])})
                row = {
                    "case_id": case["case_id"],
                    "query": case["original_case"]["query"],
                    "recall_at_5": metric_scores.get("Recall@5", 0),
                    "mrr_at_10": metric_scores.get("MRR@10", 0),
                    "gold_chunk_ids": expected_ids,
                    "gold_document_ids": gold_case["gold_documents"],
                    "retrieved_chunk_ids": actual_ids,
                    "retrieved_document_ids": actual_document_ids,
                }
                case_rows.append(row)
                if row["recall_at_5"] < 1 or row["mrr_at_10"] == 0:
                    partial_or_failed.append(row)
            positive_reports[mode] = {
                "run_id": report["run_id"],
                "metrics": {
                    "Recall@5": report["metrics"].get("Recall@5"),
                    "MRR@10": report["metrics"].get("MRR@10"),
                },
                "retrieval_latency_ms": report["retrieval_latency"],
                "cases": case_rows,
                "partial_or_failed_cases": partial_or_failed,
                "report_path": str((ROOT / "data" / "eval_runs" / report["run_id"] / "report.json").resolve()),
            }

        negative_reports = run_negative_queries(rag, EXTENDED_DATASET, modes)
        negative_path = workspace / "negative_boundary_results.json"
        negative_path.write_text(json.dumps(negative_reports, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {
            "evaluation_workspace": str(workspace.resolve()),
            "isolation": "dedicated SQLite + Chroma + BM25 paths; default app database and active snapshot untouched",
            "knowledge_snapshot": snapshot,
            "v3_document_count": len(v3_doc_ids),
            "product_static_attribute_sync": static_sync,
            "product_manual_validation": product_validation,
            "models": model_manifest(),
            "positive_dataset": str(POSITIVE_DATASET.relative_to(ROOT)),
            "positive_case_count": 52,
            "positive_reports_by_mode": positive_reports,
            "negative_dataset": str(EXTENDED_DATASET.relative_to(ROOT)),
            "negative_case_count": 8,
            "negative_results_path": str(negative_path.resolve()),
            "negative_results_by_mode": negative_reports,
        }
        (workspace / "experiment_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    finally:
        rag.close()
        db.engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "eval_runs")
    parser.add_argument("--mode", choices=MODES, action="append", dest="modes")
    args = parser.parse_args()
    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    workspace = (args.output_root.resolve() / f"rag-v3-{run_token}").resolve()
    if not workspace.is_relative_to(args.output_root.resolve()):
        raise ValueError("Output path escapes the requested evaluation root")
    result = asyncio.run(run_experiments(workspace, tuple(args.modes or MODES)))
    summary = {
        "evaluation_workspace": result["evaluation_workspace"],
        "v3_document_count": result["v3_document_count"],
        "snapshot_chunk_count": result["knowledge_snapshot"]["chunk_count"],
        "positive_reports": {
            mode: {
                "run_id": row["run_id"],
                "Recall@5": row["metrics"]["Recall@5"],
                "MRR@10": row["metrics"]["MRR@10"],
                "retrieval_latency_ms": row["retrieval_latency_ms"],
                "partial_or_failed_count": len(row["partial_or_failed_cases"]),
            }
            for mode, row in result["positive_reports_by_mode"].items()
        },
        "negative_results_path": result["negative_results_path"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
