"""Runtime coordination for manuscript reproducibility."""

from __future__ import annotations

import csv
import importlib
import os
import sys
import traceback
from pathlib import Path

from raise_icarus.stage_contracts import blocked_stage_result, definition_for, stage_definitions, write_stage_plan, write_stage_results

CONTROLLED_STAGE_MODULES = [
    "raise_icarus.denominators",
    "raise_icarus.personal_pm_support",
    "raise_icarus.hia",
    "raise_icarus.yll",
    "raise_icarus.lag_models",
    "raise_icarus.sleep_models",
    "raise_icarus.paired_sensitivity",
    "raise_icarus.tables",
    "raise_icarus.figures",
    "raise_icarus.validation",
]
WORKFLOW_STAGES = [stage.stage_name for stage in stage_definitions()]


def _debug_tracebacks_enabled() -> bool:
    return os.environ.get("RAISE_ICARUS_DEBUG_TRACEBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_repo_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    for path in (str(src_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _exception_note(prefix: str, exc: BaseException) -> str:
    note = f"{prefix}: {exc.__class__.__name__}: {exc}"
    if _debug_tracebacks_enabled():
        return note + "\n" + traceback.format_exc()
    return note


def ensure_workflow_layout(run_dir: str | Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    layout = {
        "logs": run_dir / "logs",
        "reports": run_dir / "reports",
        "figures_main": run_dir / "figures" / "main",
        "figures_supplementary": run_dir / "figures" / "supplementary",
        "tables_main": run_dir / "tables" / "main",
        "tables_supplementary": run_dir / "tables" / "supplementary",
        "results": run_dir / "results",
        "candidate": run_dir / "public_release_candidate",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    for name in ["denominators", "exposure", "hia", "yll", "lag_models", "sleep", "paired_sensitivity", "tables", "equity", "validation"]:
        (layout["results"] / name).mkdir(parents=True, exist_ok=True)
    return layout


def _load_stage(module_name: str):
    stage = definition_for(module_name)
    _ensure_repo_import_paths()
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return stage, None, _exception_note("module import failed", exc)
    if hasattr(module, "stage_definition"):
        try:
            stage = module.stage_definition()
        except Exception as exc:
            return stage, module, _exception_note("stage contract failed", exc)
    return stage, module, ""


def run_controlled_workflow(harmonized_zip: str | Path, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False):
    run_dir = Path(run_dir)
    ensure_workflow_layout(run_dir)
    _copy_manifest_if_available(run_dir)
    results = []
    for module_name in CONTROLLED_STAGE_MODULES:
        stage, module, load_note = _load_stage(module_name)
        if module is None or load_note:
            results.append(blocked_stage_result(stage, run_dir, load_note or "stage module unavailable"))
            continue
        if not hasattr(module, "run_stage"):
            results.append(blocked_stage_result(stage, run_dir, "stage module has no run_stage function"))
            continue
        try:
            results.append(module.run_stage(harmonized_zip=harmonized_zip, run_dir=run_dir, n_samples=n_samples, dry_run=dry_run))
        except Exception as exc:
            results.append(blocked_stage_result(stage, run_dir, _exception_note("stage execution failed", exc)))
    _copy_manifest_if_available(run_dir)
    write_stage_results(run_dir / "reports" / "controlled_stage_results.csv", results)
    write_stage_plan(run_dir / "reports" / "controlled_stage_plan.csv", [definition_for(name) for name in CONTROLLED_STAGE_MODULES])
    _write_report(run_dir / "reports" / "controlled_stage_report.txt", results, dry_run)
    return (0 if all(result.status.startswith("PASS") for result in results) else 3), results


def _write_report(path: Path, results, dry_run: bool) -> None:
    failed = [result for result in results if not result.status.startswith("PASS")]
    lines = [
        "Controlled Stage Runtime Report",
        "",
        f"dry_run: {str(dry_run).lower()}",
        f"stages_recorded: {len(results)}",
        f"stages_blocked_or_failed: {len(failed)}",
        "gate_status: " + ("PASS" if not failed else "FAIL"),
    ]
    if failed:
        lines += ["", "Blocked stages:"]
        lines += [f"- {result.stage_name}: {result.status}; {result.notes}" for result in failed]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_descriptive_runtime_blocker(run_dir: Path) -> Path:
    report_dir = Path(run_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [{
        "stage": definition_for(module).stage_name,
        "module": module,
        "status": "BLOCKED_CONTROLLED_DATA_EXECUTION_REQUIRED",
        "notes": "Descriptive stage contract is present; execute only in controlled-data mode.",
    } for module in CONTROLLED_STAGE_MODULES]
    csv_path = report_dir / "descriptive_runtime_blocker.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "module", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    text_path = report_dir / "descriptive_runtime_blocker_report.txt"
    text_path.write_text(
        "Descriptive Runtime Blocker Report\n\n"
        "PUBLIC_RUNTIME_DEPENDENCY_STATUS = DESCRIPTIVE_MODULES_PRESENT\n"
        "Controlled-data execution is blocked until explicitly launched by the user.\n",
        encoding="utf-8",
    )
    return text_path

def _copy_manifest_if_available(run_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sources = [
        repo_root / "local_outputs" / "manuscript_reproducibility" / "manuscript_item_manifest_detected.yaml",
        repo_root / "configs" / "manuscript_item_manifest_detected.yaml",
    ]
    target = run_dir / "reports" / "manuscript_item_manifest_detected.yaml"
    if target.exists():
        return
    for source in sources:
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            return

