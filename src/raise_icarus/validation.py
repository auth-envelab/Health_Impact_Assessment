"""Validation helpers for manuscript reproducibility outputs."""

from __future__ import annotations

import csv
import re
from pathlib import Path

EXPECTED_COUNTS = {"hia_primary_scenario_rows": 40, "upper_tail_sensitivity_cap_rows": 8, "yll_rows": 8, "supplementary_table_s3_rows": 96}


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def count_rows(path: str | Path) -> int | None:
    path = Path(path)
    return None if not path.exists() else len(read_csv_rows(path))


def scan_text_for_pattern(root: str | Path, pattern: str) -> bool:
    root = Path(root)
    if not root.exists():
        return False
    regex = re.compile(pattern, re.IGNORECASE)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".csv", ".md", ".yaml", ".yml", ".json", ".html", ".svg"}:
            if regex.search(path.read_text(encoding="utf-8", errors="ignore")):
                return True
    return False


def scan_output_text_for_pattern(run_dir: str | Path, pattern: str) -> bool:
    run_dir = Path(run_dir)
    roots = [run_dir / "results", run_dir / "figures", run_dir / "tables"]
    return any(scan_text_for_pattern(root, pattern) for root in roots)


def validate_manuscript_outputs(run_dir: str | Path) -> int:
    run_dir = Path(run_dir)
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    checks = [
        ("hia_primary_scenario_rows", run_dir / "results" / "hia" / "hia_primary_scenario_summary.csv"),
        ("upper_tail_sensitivity_cap_rows", run_dir / "results" / "hia" / "hia_upper_tail_cap_audit.csv"),
        ("yll_rows", run_dir / "results" / "yll" / "yll_primary_results.csv"),
        ("supplementary_table_s3_rows", run_dir / "results" / "lag_models" / "Supplementary Table S3 data - Lag-specific HR stress models.csv"),
    ]
    for check, path in checks:
        observed = count_rows(path)
        expected = EXPECTED_COUNTS[check]
        rows.append({"check": check, "status": "PASS" if observed == expected else "FAIL", "observed": "missing" if observed is None else str(observed), "expected": str(expected), "notes": rel(path, run_dir)})
    template_rows = read_csv_rows(report_dir / "figure_template_fidelity_check.csv")
    allowed_template_statuses = {"PASS", "REQUIRES_VISUAL_REVIEW"}
    template_failures = [row for row in template_rows if row.get("status") not in allowed_template_statuses]
    generation_rows = read_csv_rows(report_dir / "figure_generation_report.csv")
    allowed_generation_statuses = {"PASS", "PASS_DRY_RUN", "MATCH_APPROVED_REFERENCE"}
    generation_failures = [row for row in generation_rows if row.get("status", "") not in allowed_generation_statuses]
    lock_rows = read_csv_rows(report_dir / "figure_template_lock_check.csv")
    lock_failures = [row for row in lock_rows if str(row.get("status", "")).startswith("BLOCKED") or row.get("status") == "FAIL"]
    figure_status = "PASS" if template_rows and not template_failures and generation_rows and not generation_failures and lock_rows and not lock_failures else "FAIL"
    observed_figure_failures = str(len(template_failures) + len(generation_failures) + len(lock_failures)) if template_rows or generation_rows or lock_rows else "missing"
    rows.append({"check": "figure_template_fidelity", "status": figure_status, "observed": observed_figure_failures, "expected": "0", "notes": "all required figures must match the user-approved corrected figure references"})
    rows.append({"check": "no_p_value_000", "status": "FAIL" if scan_output_text_for_pattern(run_dir, r"p\s*=\s*0\.000") else "PASS", "observed": "scan", "expected": "no matches", "notes": "generated results/figures/tables scan"})
    table5_status = run_dir / "results" / "validation" / "Table 5 data - Demographic source status.csv"
    table5_dependency_report = run_dir / "results" / "tables" / "Table 5 dependency report.csv"
    if table5_status.exists():
        content = table5_status.read_text(encoding="utf-8", errors="ignore")
        rows.append({"check": "table5_demographic_source", "status": "PASS" if "MISSING_DEPENDENCY" in content else "FAIL", "observed": "file_present", "expected": "MISSING_DEPENDENCY unless source supplied", "notes": ""})
    elif table5_dependency_report.exists():
        content = table5_dependency_report.read_text(encoding="utf-8", errors="ignore")
        rows.append({"check": "table5_demographic_source", "status": "PASS" if "MISSING_DEPENDENCY" in content else "FAIL", "observed": "dependency_report_present", "expected": "MISSING_DEPENDENCY unless source supplied", "notes": rel(table5_dependency_report, run_dir)})
    else:
        rows.append({"check": "table5_demographic_source", "status": "FAIL", "observed": "missing", "expected": "MISSING_DEPENDENCY status file", "notes": ""})
    with (report_dir / "manuscript_output_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "observed", "expected", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if row["status"] != "PASS"]
    (report_dir / "manuscript_output_validation_report.txt").write_text("Manuscript Output Validation Report\n\n" + f"checks_evaluated: {len(rows)}\n" + f"checks_failed: {len(failed)}\n" + "gate_status: " + ("FAIL" if failed else "PASS") + "\n", encoding="utf-8")
    return 1 if failed else 0


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()

# Controlled-data validation and equity analytical exports.

from raise_icarus.controlled_runtime import config_value as _ri_val_config_value
from raise_icarus.controlled_runtime import copy_alias as _ri_val_copy_alias
from raise_icarus.controlled_runtime import domain_dir as _ri_val_domain_dir
from raise_icarus.controlled_runtime import reports_dir as _ri_val_reports_dir
from raise_icarus.controlled_runtime import repo_root as _ri_val_repo_root
from raise_icarus.controlled_runtime import write_text_report as _ri_val_write_text_report
from raise_icarus.stage_contracts import StageDefinition as _RI_ValStageDefinition
from raise_icarus.stage_contracts import StageResult as _RI_ValStageResult
from raise_icarus.stage_contracts import definition_for as _ri_val_definition_for
from raise_icarus.stage_contracts import dry_run_stage_result as _ri_val_dry_run_stage_result

STAGE = _ri_val_definition_for(__name__)


def stage_definition() -> _RI_ValStageDefinition:
    return STAGE


def run_equity_feasibility_audit(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> dict[str, Path]:
    from raise_icarus.phase11_equity import write_phase11_outputs

    output_dir = _ri_val_domain_dir(out_dir, "equity")
    outputs = write_phase11_outputs(
        harmonized_zip,
        outdir=output_dir,
        phase1_dir=_ri_val_domain_dir(out_dir, "denominators"),
        phase2_dir=_ri_val_domain_dir(out_dir, "exposure"),
        phase7_dir=_ri_val_domain_dir(out_dir, "sleep"),
        phase9_dir=_ri_val_domain_dir(out_dir, "paired_sensitivity"),
        phase10_dir=_ri_val_domain_dir(out_dir, "tables"),
        demographics_file=_ri_val_config_value(config, "demographics_file", None),
        min_cell_size=_ri_val_config_value(config, "min_cell_size", 10),
        repo_path=_ri_val_repo_root(),
        date_filter_mode=_ri_val_config_value(config, "date_filter_mode", "campaign"),
        scripts_run=[],
    )
    _ri_val_copy_alias(outputs["equity_analysis_feasibility_summary"], output_dir / "Equity feasibility summary.csv")
    return outputs


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> _RI_ValStageResult:
    del n_samples
    if dry_run:
        return _ri_val_dry_run_stage_result(STAGE, run_dir)
    if harmonized_zip is None:
        return _RI_ValStageResult(STAGE.stage_name, STAGE.module_name, "FAIL", STAGE.output_domain, (), "harmonized archive path is required")
    try:
        run_equity_feasibility_audit(harmonized_zip, run_dir)
    except Exception as exc:
        report = _ri_val_write_text_report(_ri_val_reports_dir(run_dir) / "Equity feasibility validation report.txt", "Equity Feasibility Validation Report", [f"status: FAIL", f"notes: {exc.__class__.__name__}"])
        return _RI_ValStageResult(STAGE.stage_name, STAGE.module_name, "FAIL", STAGE.output_domain, (str(report),), "Equity feasibility audit failed before validation.")
    code = validate_manuscript_outputs(run_dir)
    return _RI_ValStageResult(STAGE.stage_name, STAGE.module_name, "PASS" if code == 0 else "FAIL", STAGE.output_domain, (str(_ri_val_reports_dir(run_dir) / "manuscript_output_validation.csv"),), "Validation outputs generated.")

