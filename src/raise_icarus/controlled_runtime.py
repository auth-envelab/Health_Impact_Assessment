"""Runtime helpers for descriptive controlled-data modules."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Iterable


def config_value(config: object, key: str, default: object = None) -> object:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def domain_dir(out_dir: str | Path, domain: str) -> Path:
    path = Path(out_dir)
    return path if path.name == domain else path / "results" / domain


def reports_dir(out_dir: str | Path) -> Path:
    path = Path(out_dir)
    return path if path.name == "reports" else path / "reports"


def tables_dir(out_dir: str | Path, section: str = "supplementary") -> Path:
    path = Path(out_dir)
    return path if path.name == section else path / "tables" / section


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_alias(source: str | Path, target: str | Path) -> Path:
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_text_report(path: str | Path, title: str, rows: Iterable[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(title + "\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def csv_to_xlsx(source: str | Path, target: str | Path) -> Path:
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return copy_alias(source, target.with_suffix(".csv"))
    pd.read_csv(source).to_excel(target, index=False)
    return target


def file_status_rows(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        p = Path(path)
        rows.append({"path": p.as_posix(), "exists": p.exists(), "status": "PASS" if p.exists() else "FAIL"})
    return rows
