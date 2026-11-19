"""Import the RAG v3 knowledge and evaluation fixtures without overwriting files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

from app.config import settings
from app.dataset import load_dataset
from app.db import DB
from app.product_fixtures import merge_missing_static_attributes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "kefu_rag_knowledge_v3_bundle.zip"
DATASET_FILES = (
    "rag_v3_gold_60_extended.jsonl",
    "rag_v3_positive_01_tenant_b_adapter_required.jsonl",
    "rag_v3_positive_51_tenant_a_ready.jsonl",
    "rag_v3_positive_52_compatible.jsonl",
)


def _destination(relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe archive path: {relative}")
    target = (ROOT / Path(*rel.parts)).resolve()
    if target != ROOT and ROOT not in target.parents:
        raise ValueError(f"Archive path escapes repository: {relative}")
    return target


def _write_without_overwrite(target: Path, content: bytes) -> str:
    if target.exists():
        if not target.is_file() or target.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite different existing file: {target}")
        return "already_present"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".importing")
    try:
        temporary.write_bytes(content)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "imported"


def _adapt_tenant_cases(raw: bytes, tenant_by_case: dict[str, str]) -> bytes:
    adapted = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tenant = tenant_by_case.get(row["case_id"], "a")
        if tenant == "b":
            row["actor_user_id"] = "user_b"
        adapted.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    payload = ("\n".join(adapted) + "\n").encode("utf-8")
    return payload


def import_bundle(archive_path: Path = DEFAULT_ARCHIVE) -> dict:
    archive_path = Path(archive_path).resolve()
    imported: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate member paths")
        manifest = json.loads(archive.read("manifest.json"))
        source = archive.read("source/kefu_products_seed_20.json")
        source_sha = hashlib.sha256(source).hexdigest()
        if source_sha != manifest["source_product_sha256"]:
            raise ValueError("Bundle product fixture does not match its manifest")
        local_source = (ROOT / "kefu_products_seed_20.json").read_bytes()
        if hashlib.sha256(local_source).hexdigest() != source_sha:
            raise ValueError("Bundle product fixture differs from the imported project fixture")

        knowledge_members = sorted(
            name for name in names
            if name.startswith("knowledge/expansion_v3/") and name.endswith(".md")
        )
        if len(knowledge_members) != 50 or manifest["docs"] != 50:
            raise ValueError(f"Expected 50 expansion documents, found {len(knowledge_members)}")
        for member in knowledge_members:
            relative = member.removeprefix("knowledge/")
            target = _destination(f"knowledge/{relative}")
            imported[str(target.relative_to(ROOT))] = _write_without_overwrite(
                target, archive.read(member)
            )

        for name in DATASET_FILES:
            member = f"datasets/{name}"
            target = _destination(member)
            imported[str(target.relative_to(ROOT))] = _write_without_overwrite(
                target, archive.read(member)
            )

        for member, target_name in (
            ("manifest.json", "datasets/rag_v3_manifest.json"),
            ("gold_evidence_chunk_map.json", "datasets/rag_v3_gold_evidence_chunk_map.json"),
        ):
            target = _destination(target_name)
            imported[str(target.relative_to(ROOT))] = _write_without_overwrite(
                target, archive.read(member)
            )

        extended = [
            json.loads(line) for line in archive.read("datasets/rag_v3_gold_60_extended.jsonl")
            .decode("utf-8").splitlines() if line.strip()
        ]
        tenant_by_case = {row["case_id"]: row["tenant_id"] for row in extended}
        for source_name, target_name in (
            ("rag_v3_positive_52_compatible.jsonl", "rag_v3_positive_52_tenant_scoped.jsonl"),
            ("rag_v3_positive_01_tenant_b_adapter_required.jsonl", "rag_v3_positive_01_tenant_b_scoped.jsonl"),
        ):
            payload = _adapt_tenant_cases(
                archive.read(f"datasets/{source_name}"), tenant_by_case
            )
            target = _destination(f"datasets/{target_name}")
            imported[str(target.relative_to(ROOT))] = _write_without_overwrite(target, payload)

    # Validate the ordinary evaluation contract, including explicit tenant B identity.
    _, cases_51 = load_dataset(ROOT / "datasets/rag_v3_positive_51_tenant_a_ready.jsonl")
    _, cases_52 = load_dataset(ROOT / "datasets/rag_v3_positive_52_tenant_scoped.jsonl")
    _, cases_b = load_dataset(ROOT / "datasets/rag_v3_positive_01_tenant_b_scoped.jsonl")
    if len(cases_51) != 51 or len(cases_52) != 52 or len(cases_b) != 1:
        raise ValueError("Imported positive dataset counts do not match the bundle contract")
    if cases_52[-1]["actor_user_id"] != "user_b" or cases_b[0]["actor_user_id"] != "user_b":
        raise ValueError("Tenant B queries must run under user_b")

    return {
        "archive": str(archive_path),
        "documents": len(knowledge_members),
        "datasets": {"tenant_a": len(cases_51), "tenant_scoped": len(cases_52), "tenant_b": len(cases_b)},
        "files": imported,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--sync-static-product-attributes",
        action="store_true",
        help="Only fill missing static product attributes in the configured local SQLite database.",
    )
    args = parser.parse_args()
    result = import_bundle(args.archive)
    if args.sync_static_product_attributes:
        if not settings.sqlite_url.startswith("sqlite:///") or ":memory:" in settings.sqlite_url:
            raise ValueError("Static fixture enrichment is limited to a local file-backed SQLite database")
        db = DB(settings.sqlite_url)
        try:
            result["static_product_attributes"] = merge_missing_static_attributes(db)
        finally:
            db.engine.dispose()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
