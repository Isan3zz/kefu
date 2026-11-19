from pathlib import Path

from app.config import ROOT, settings
from app.db import DB
from app.product_fixtures import import_product_fixtures


def validate_fixture_target() -> None:
    if settings.app_env != "local" or not settings.demo_mode:
        raise SystemExit("Refusing to seed simulated products unless APP_ENV=local and DEMO_MODE=true.")
    if not settings.sqlite_url.startswith("sqlite:///"):
        raise SystemExit("Refusing to seed simulated products outside a local SQLite database.")
    configured_path = Path(settings.sqlite_url.removeprefix("sqlite:///"))
    if not configured_path.is_absolute():
        configured_path = ROOT / configured_path
    if configured_path.resolve() != (ROOT / "data" / "app.db").resolve():
        raise SystemExit("Refusing to seed simulated products outside the project data/app.db fixture database.")


def main() -> None:
    validate_fixture_target()
    db = DB(settings.sqlite_url)
    try:
        db.seed()
        imported = import_product_fixtures(db)
        print(
            "Local demo database initialized. "
            f"Product fixture: {imported['inserted']} inserted, "
            f"{imported['skipped_existing']} existing IDs preserved, {imported['total']} checked."
        )
    finally:
        db.engine.dispose()


if __name__ == "__main__":
    main()
