"""Run the 50-row order boundary Oracle against an isolated temporary SQLite DB."""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from app.config import ROOT
from app.dataset import FIXTURE_TIME
from app.db import DB, uid
from app.order_fixtures import ORDER_FIXTURE_ALL, evaluate_order_boundary_oracle, seed_order_scenario


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="kefu-order-boundary-") as folder:
        db = DB("sqlite:///" + str(Path(folder) / "boundary.db"))
        seed = seed_order_scenario(
            db,
            ORDER_FIXTURE_ALL,
            lambda: datetime.fromisoformat(FIXTURE_TIME),
        )
        report = evaluate_order_boundary_oracle(db)
        report["seed"] = seed
        report["checks"] = sum(len(case["results"]) for case in report["cases"])
        report["passed_checks"] = sum(
            result["passed"] for case in report["cases"] for result in case["results"]
        )
        db.engine.dispose()

    output_dir = ROOT / "data" / "eval_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uid('order-boundary-')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Order boundary Oracle: {report['passed_checks']}/{report['checks']} checks passed; "
        f"{len(report['cases'])} cases; {len(report['failures'])} failures."
    )
    print(f"Report: {output_path}")
    if report["failures"]:
        print(json.dumps(report["failures"], ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
