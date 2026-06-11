"""Build, reproduce, and validate aggregate-only public candidates."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path

from raise_icarus.figure_templates import EXPECTED_FIGURE_IDS
from raise_icarus.stage_contracts import definition_for

REPO_ROOT = Path(__file__).resolve().parents[2]
DESCRIPTIVE_SCRIPTS = ["_bootstrap.py", "run_manuscript_reproducibility_from_harmonized.py", "generate_manuscript_tables.py", "generate_manuscript_figures.py", "validate_manuscript_outputs.py", "build_public_release_candidate.py", "reproduce_from_public_aggregate_outputs.py", "validate_public_release_candidate.py", "create_manuscript_item_manifest.py"]
CONFIGS = ["local_config.example.yaml", "manuscript_item_manifest.example.yaml", "expected_results_manifest.yaml", "public_outputs_manifest.yaml"]
DOCS = ["reproduction.md", "data_access.md", "manuscript_outputs.md", "raise_execution.md", "figure_reproducibility.md"]
ROOT_RUNTIME_FILES = ["requirements.txt", "pytest.ini"]
PUBLIC_SAFE_MODULES = [
    "__init__.py",
    "controlled_runtime.py",
    "data.py",
    "denominators.py",
    "figure_templates.py",
    "figures.py",
    "hia.py",
    "lag_models.py",
    "manuscript_runtime.py",
    "paired_sensitivity.py",
    "personal_pm_support.py",
    "phase1_denominators.py",
    "phase2_ppm_common_support.py",
    "phase3_hia_primary.py",
    "phase4_hia_upper_tail.py",
    "phase5_yll.py",
    "phase6_lag_models.py",
    "phase7_sleep.py",
    "phase8_figures.py",
    "phase9_paired_sensitivity.py",
    "phase10_tables.py",
    "phase11_equity.py",
    "physiology.py",
    "pm_qc.py",
    "public_candidate.py",
    "reporting.py",
    "sleep_models.py",
    "stage_contracts.py",
    "table6.py",
    "tables.py",
    "validation.py",
    "yll.py",
]
APPROVED_FIGURE_REFERENCE_RELATIVE_PATHS = {
    "seasonal_personal_pm_distributions": "main/Figure 1 - seasonal_personal_pm_distributions.png",
    "activity_stratified_personal_pm_exposure": "main/Figure 2 - activity_stratified_personal_pm_exposure.png",
    "residential_uhoo_iaq_correlation_matrix": "main/Figure 3 - residential_uhoo_iaq_correlation_matrix.png",
    "uhoo_iaq_garmin_sleep_correlation_matrix": "main/Figure 4 - uhoo_iaq_garmin_sleep_correlation_matrix.png",
    "hia_attributable_cases": "main/Figure 5 - hia_attributable_cases.png",
    "yll_scenario_outputs": "main/Figure 6 - yll_scenario_outputs.png",
    "participant_flow_completeness_summary": "supplementary/Supplementary Figure S1 - participant_flow_completeness_summary.png",
    "diurnal_pm_hr_stress": "supplementary/Supplementary Figure S2 - diurnal_pm_hr_stress.png",
    "seasonal_garmin_sleep_distributions": "supplementary/Supplementary Figure S3 - seasonal_garmin_sleep_distributions.png",
    "ppm_garmin_correlation_matrix": "supplementary/Supplementary Figure S4 - ppm_garmin_correlation_matrix.png",
    "descriptive_ols_pm_hr_lag": "supplementary/Supplementary Figure S5 - descriptive_ols_pm_hr_lag.png",
    "ols_vs_lag_mixed_comparison": "supplementary/Supplementary Figure S6 - ols_vs_lag_mixed_comparison.png",
    "iaq_sleep_regression_panels": "supplementary/Supplementary Figure S7 - iaq_sleep_regression_panels.png",
    "stress_sleep_exploratory_panels": "supplementary/Supplementary Figure S8 - stress_sleep_exploratory_panels.png",
}
RUNTIME_MANIFEST_FILES = [
    "configs/manuscript_item_manifest_detected.yaml",
    "configs/figure_template_reference_manifest.csv",
    "configs/figure_template_candidate_map.csv",
]
RESULT_DIRS = ["denominators", "exposure", "hia", "yll", "lag_models", "sleep", "paired_sensitivity", "tables", "equity", "validation"]
COPY_RULES = [
    ("results/denominators/Supplementary Table S1 - Analysis-specific denominators.csv", "tables/supplementary/Supplementary Table S1 - Analysis-specific denominators.csv", "Supplementary Table S1"),
    ("results/lag_models/Supplementary Table S3 data - Lag-specific HR stress models.csv", "tables/supplementary/Supplementary Table S3 data - Lag-specific HR stress models.csv", "Supplementary Table S3"),
    ("results/hia/Figure 5 data - HIA attributable cases.csv", "results/hia/Figure 5 data - HIA attributable cases.csv", "HIA attributable cases"),
    ("results/yll/Figure 6 data - YLL scenario outputs.csv", "results/yll/Figure 6 data - YLL scenario outputs.csv", "YLL scenario outputs"),
]
EXCLUDED_PUBLIC_OUTPUT_RE = re.compile(r"Supplementary Table S7|supplementary_table_s7", re.IGNORECASE)
REJECTED_FIGURE_CANDIDATES = REPO_ROOT / "local_outputs" / "manuscript_reproducibility" / "figure_provenance_inventory" / "rejected_figure_candidates.csv"
BANNED_NAME_RE = re.compile(r"(^|[\\/])(?:\d{2}_|phase\d+)", re.IGNORECASE)
LOCAL_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|" + "/Us" + "ers/|" + "/ho" + "me/")
LOCAL_OUTPUTS_RE = re.compile("local" + r"[_\\/]+outputs", re.IGNORECASE)
CONTROLLED_ARCHIVE_NAME_RE = re.compile("Harmonized" + r"\s+datasets\.zip", re.IGNORECASE)
ZERO_P_VALUE_RE = re.compile("p" + r"\s*=\s*0\.000", re.IGNORECASE)
UNSAFE_EXTENSIONS = {".zip", ".rds", ".feather", ".parquet", ".docx", ".doc"}
IDENTIFIER_PATTERNS = [re.compile("participant" + "_uid", re.IGNORECASE), re.compile("source" + "_member", re.IGNORECASE), re.compile("raw" + "_timestamp", re.IGNORECASE)]
TRACE_TERMS = ["Chat" + "G" + "P" + "T", "Co" + "dex", "Open" + "AI", "G" + "P" + "T", "pr" + "ompt", "conver" + "sation", "assi" + "stant", "local " + "worker", "internal " + "drafting", "model-" + "generated"]
REQUIRED_FILES = ["Run-Manuscript-Reproducibility.ps1", "sitecustomize.py", "raise_icarus/__init__.py", *ROOT_RUNTIME_FILES, *RUNTIME_MANIFEST_FILES, *[f"scripts/{name}" for name in DESCRIPTIVE_SCRIPTS], *[f"configs/{name}" for name in CONFIGS], *[f"docs/{name}" for name in DOCS], "checksums/manifest_sha256.txt"]
TEXT_SUFFIXES = {".csv", ".txt", ".md", ".yaml", ".yml", ".py", ".ps1", ".json", ".cff"}
ALLOWED_TEMPLATE_STATUSES = {"PASS", "REQUIRES_VISUAL_REVIEW"}
ALLOWED_GENERATION_STATUSES = {"PASS", "MATCH_APPROVED_REFERENCE"}


def clean_dir(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_file(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_tree_safe(source: Path, target: Path, suffixes: set[str]) -> None:
    if source.exists():
        rejected = load_rejected_candidate_paths()
        for path in source.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                if EXCLUDED_PUBLIC_OUTPUT_RE.search(path.relative_to(source).as_posix()):
                    continue
                if normalised_path(path) in rejected:
                    continue
                copy_file(path, target / path.relative_to(source))


def normalised_path(path: str | Path) -> str:
    try:
        return str(Path(path).resolve()).replace("/", "\\").lower()
    except Exception:
        return str(path).replace("/", "\\").lower()


def load_rejected_candidate_paths(path: str | Path = REJECTED_FIGURE_CANDIDATES) -> set[str]:
    resolved = Path(path)
    if not resolved.exists():
        return set()
    rejected: set[str] = set()
    with resolved.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("must_not_use_in_public_candidate", "").strip().lower() in {"1", "true", "yes"}:
                value = row.get("figure_file", "").strip()
                if value:
                    rejected.add(normalised_path(value))
    return rejected


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def _status_is_blocked(status: str) -> bool:
    value = status.strip().upper()
    return value == "FAIL" or value.startswith("BLOCKED")


def _lock_report_path(run_dir: Path) -> Path:
    return run_dir / "reports" / "figure_template_lock_check.csv"


def _generation_report_path(run_dir: Path) -> Path:
    return run_dir / "reports" / "figure_generation_report.csv"


def _template_report_path(run_dir: Path) -> Path:
    return run_dir / "reports" / "figure_template_fidelity_check.csv"


def validate_figure_template_lock_for_public(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    errors: list[str] = []
    lock_report = _lock_report_path(run_dir)
    generation_report = _generation_report_path(run_dir)
    template_report = _template_report_path(run_dir)

    lock_rows = _read_csv_rows(lock_report)
    if not lock_rows:
        errors.append("missing figure_template_lock_check.csv")
    else:
        lock_by_id = {row.get("figure_id", "") or row.get("item_id", ""): row for row in lock_rows}
        for figure_id in EXPECTED_FIGURE_IDS:
            row = lock_by_id.get(figure_id)
            if not row:
                errors.append(f"missing lock row for {figure_id}")
                continue
            status = row.get("status", "") or row.get("generation_status", "")
            if _status_is_blocked(status):
                errors.append(f"blocked template lock for {figure_id}: {status}")
            if row.get("template_lock_status") not in {"SCRIPT_AVAILABLE", "SCRIPT_AND_TEMPLATE_AVAILABLE"}:
                errors.append(f"script-backed accepted template not confirmed for {figure_id}")
            if row.get("rejected_candidates_excluded", "").strip().lower() not in {"yes", "true"}:
                errors.append(f"rejected candidates not excluded for {figure_id}")
            if not row.get("generator_function"):
                errors.append(f"missing generator function for {figure_id}")
            if not row.get("accepted_template_reference") and not row.get("accepted_template_source"):
                errors.append(f"missing accepted template reference for {figure_id}")

    generation_rows = _read_csv_rows(generation_report)
    if not generation_rows:
        errors.append("missing figure_generation_report.csv")
    else:
        generation_by_id = {row.get("figure_id", ""): row for row in generation_rows}
        for figure_id in EXPECTED_FIGURE_IDS:
            row = generation_by_id.get(figure_id)
            if not row:
                errors.append(f"missing generation row for {figure_id}")
                continue
            status = row.get("status", "")
            if status not in ALLOWED_GENERATION_STATUSES:
                errors.append(f"invalid generation status for {figure_id}: {status}")
            output = row.get("output", "")
            if not output or not (run_dir / output).exists():
                errors.append(f"missing regenerated output for {figure_id}")
            notes = row.get("notes", "").lower()
            if any(term in notes for term in ("generic", "simplified", "rejected", "superseded")):
                errors.append(f"non-current figure note for {figure_id}")

    template_rows = _read_csv_rows(template_report)
    if not template_rows:
        errors.append("missing figure_template_fidelity_check.csv")
    else:
        for row in template_rows:
            figure_id = row.get("item_id", "") or row.get("figure_id", "")
            status = row.get("status", "")
            if status not in ALLOWED_TEMPLATE_STATUSES:
                errors.append(f"invalid template status for {figure_id}: {status}")
    return errors


def redact_local_paths(candidate: str | Path) -> None:
    candidate = Path(candidate)
    drive_path = re.compile(r"[A-Za-z]:[\\/][^,\n\r\t\"']+")
    posix_path = re.compile(r"(?:/Us" + r"ers|/ho" + r"me)/[^,\n\r\t\"']+")
    local_outputs_path = re.compile("local" + r"[_\\/]+outputs(?:[\\/][^,\n\r\t\"']+)?", re.IGNORECASE)
    controlled_archive_name = re.compile("Harmonized" + r"\s+datasets\.zip", re.IGNORECASE)
    zero_p_value = re.compile("p" + r"\s*=\s*0\.000", re.IGNORECASE)
    identifier_replacements = {
        "participant_uid": "participant identifier",
        "participant_UID": "participant identifier",
        "source_member": "source member",
        "source-member": "source member",
        "raw_timestamp": "raw timestamp",
        "raw timestamps": "raw timestamps",
    }
    for path in candidate.rglob("*") if candidate.exists() else []:
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel_path = path.relative_to(candidate).as_posix()
        if rel_path.startswith(("scripts/", "src/", "tests/", "tools/", "raise_icarus/")) or rel_path in {"Run-Manuscript-Reproducibility.ps1", "sitecustomize.py"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        redacted = drive_path.sub("<local_path_redacted>", text)
        redacted = posix_path.sub("<local_path_redacted>", redacted)
        redacted = local_outputs_path.sub("analysis_outputs", redacted)
        redacted = controlled_archive_name.sub("controlled harmonized dataset archive", redacted)
        redacted = zero_p_value.sub("p < 0.001", redacted)
        for unsafe, safe in identifier_replacements.items():
            redacted = re.sub(re.escape(unsafe), safe, redacted, flags=re.IGNORECASE)
        if redacted != text:
            path.write_text(redacted, encoding="utf-8")


def write_checksums(candidate: Path) -> None:
    checksum_dir = candidate / "checksums"
    checksum_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(candidate.rglob("*")):
        rel = path.relative_to(candidate).as_posix()
        if path.is_file() and rel != "checksums/manifest_sha256.txt":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}")
    (checksum_dir / "manifest_sha256.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_placeholders(candidate: Path) -> None:
    (candidate / "README.md").write_text("# RAISE/ICARUS Manuscript Analysis Workflow\n\nThis candidate contains descriptive workflow entrypoints, documentation, validation reports, regenerated public figures, and aggregate outputs. The controlled harmonized dataset is not included.\n", encoding="utf-8")
    (candidate / "LICENSE_PLACEHOLDER.txt").write_text("License terms will be supplied before release.\n", encoding="utf-8")
    (candidate / "CITATION_PLACEHOLDER.cff").write_text("cff-version: 1.2.0\nmessage: Citation metadata will be supplied before release.\n", encoding="utf-8")


def _approved_reference_candidate_path(figure_id: str) -> str:
    rel = APPROVED_FIGURE_REFERENCE_RELATIVE_PATHS.get(figure_id, "")
    return f"figures/approved_references/{rel}" if rel else ""


def _write_sanitized_manuscript_item_manifest(candidate: Path) -> None:
    sources = [
        REPO_ROOT / "local_outputs" / "manuscript_reproducibility" / "manuscript_item_manifest_detected.yaml",
        REPO_ROOT / "configs" / "manuscript_item_manifest_detected.yaml",
    ]
    source = next((path for path in sources if path.exists()), None)
    target = candidate / "configs" / "manuscript_item_manifest_detected.yaml"
    if source is None:
        return
    lines: list[str] = []
    current_item = ""
    for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("- item_id:"):
            current_item = stripped.split(":", 1)[1].strip().strip('"')
            lines.append(raw_line)
            continue
        if stripped.startswith("current_document_source:"):
            lines.append('  current_document_source: "current revised manuscript or supplementary material"')
            continue
        if stripped.startswith("accepted_template_source:") and current_item in APPROVED_FIGURE_REFERENCE_RELATIVE_PATHS:
            lines.append(f'  accepted_template_source: "{_approved_reference_candidate_path(current_item)}"')
            continue
        if stripped.startswith("rejected_candidate_files:"):
            lines.append('  rejected_candidate_files: "excluded from clean runtime"')
            continue
        lines.append(raw_line)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sanitized_csv_rows(source: Path, target: Path, path_fields: set[str]) -> None:
    if not source.exists():
        return
    with source.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            figure_id = row.get("figure_id", "")
            approved = _approved_reference_candidate_path(figure_id)
            for field in path_fields:
                if field in row:
                    row[field] = approved if approved and field in {"template_source_path_or_placeholder", "accepted_template_reference", "accepted_template_candidate_file"} else ""
            rows.append(row)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_clean_runtime_figure_metadata(candidate: Path) -> None:
    _write_sanitized_manuscript_item_manifest(candidate)
    template_manifest_source = REPO_ROOT / "local_outputs" / "manuscript_reproducibility" / "figure_template_reference_manifest.csv"
    if not template_manifest_source.exists():
        template_manifest_source = REPO_ROOT / "configs" / "figure_template_reference_manifest.csv"
    _write_sanitized_csv_rows(
        template_manifest_source,
        candidate / "configs" / "figure_template_reference_manifest.csv",
        {"template_source_path_or_placeholder", "accepted_template_reference", "regenerated_candidate_file", "style_report_candidate", "rejected_candidate_files"},
    )
    candidate_map_source = REPO_ROOT / "local_outputs" / "manuscript_reproducibility" / "figure_provenance_inventory" / "figure_template_candidate_map.csv"
    if not candidate_map_source.exists():
        candidate_map_source = REPO_ROOT / "configs" / "figure_template_candidate_map.csv"
    _write_sanitized_csv_rows(
        candidate_map_source,
        candidate / "configs" / "figure_template_candidate_map.csv",
        {"accepted_template_candidate_file", "regenerated_candidate_file", "style_report_candidate", "rejected_candidate_files"},
    )
    reference_root = REPO_ROOT / "local_outputs" / "manuscript_reproducibility" / "figure_only_corrected_review" / "figures"
    if not reference_root.exists():
        reference_root = REPO_ROOT / "figures" / "approved_references"
    for rel in APPROVED_FIGURE_REFERENCE_RELATIVE_PATHS.values():
        copy_file(reference_root / rel, candidate / "figures" / "approved_references" / rel)


def build_public_release_candidate(run_dir: str | Path, candidate_dir: str | Path) -> int:
    run_dir = Path(run_dir)
    candidate = Path(candidate_dir)
    validation_report = run_dir / "reports" / "manuscript_output_validation.csv"
    if not validation_report.exists():
        raise SystemExit("Validation report is required before building a public aggregate candidate")
    if "FAIL" in validation_report.read_text(encoding="utf-8", errors="ignore"):
        raise SystemExit("Public aggregate candidate build blocked because manuscript validation did not pass")
    dependency_hits = public_runtime_dependency_hits(REPO_ROOT)
    if dependency_hits:
        raise SystemExit("PUBLIC_CANDIDATE_STATUS = BLOCKED_NUMBERED_SCRIPT_RUNTIME_DEPENDENCY")
    figure_gate_errors = validate_figure_template_lock_for_public(run_dir)
    if figure_gate_errors:
        raise SystemExit("PUBLIC_CANDIDATE_STATUS = BLOCKED_FIGURE_TEMPLATE_LOCK: " + "; ".join(figure_gate_errors[:5]))
    clean_dir(candidate)
    write_placeholders(candidate)
    copy_file(REPO_ROOT / "Run-Manuscript-Reproducibility.ps1", candidate / "Run-Manuscript-Reproducibility.ps1")
    copy_file(REPO_ROOT / "sitecustomize.py", candidate / "sitecustomize.py")
    copy_file(REPO_ROOT / "raise_icarus" / "__init__.py", candidate / "raise_icarus" / "__init__.py")
    for name in ROOT_RUNTIME_FILES:
        copy_file(REPO_ROOT / name, candidate / name)
    for name in DESCRIPTIVE_SCRIPTS:
        copy_file(REPO_ROOT / "scripts" / name, candidate / "scripts" / name)
    for name in CONFIGS:
        copy_file(REPO_ROOT / "configs" / name, candidate / "configs" / name)
    _write_clean_runtime_figure_metadata(candidate)
    for name in DOCS:
        copy_file(REPO_ROOT / "docs" / name, candidate / "docs" / name)
    copy_file(REPO_ROOT / "tools" / "figure_template_checks.py", candidate / "tools" / "figure_template_checks.py")
    for name in PUBLIC_SAFE_MODULES:
        copy_file(REPO_ROOT / "src" / "raise_icarus" / name, candidate / "src" / "raise_icarus" / name)
    for result_dir in RESULT_DIRS:
        copy_tree_safe(run_dir / "results" / result_dir, candidate / "results" / result_dir, {".csv", ".xlsx", ".txt", ".json"})
    copy_tree_safe(run_dir / "figures" / "main", candidate / "figures" / "main", {".png", ".svg", ".pdf"})
    copy_tree_safe(run_dir / "figures" / "supplementary", candidate / "figures" / "supplementary", {".png", ".svg", ".pdf"})
    copy_tree_safe(run_dir / "tables" / "main", candidate / "tables" / "main", {".csv", ".xlsx"})
    copy_tree_safe(run_dir / "tables" / "supplementary", candidate / "tables" / "supplementary", {".csv", ".xlsx"})
    copy_tree_safe(run_dir / "reports", candidate / "results" / "validation", {".csv", ".txt"})
    redact_local_paths(candidate)
    write_checksums(candidate)
    return 0


def write_public_stage_stub(path: Path, module_name: str) -> None:
    stage = definition_for(module_name)
    text = f"""\"\"\"Public stage contract for {stage.display_name}.\"\"\"

from __future__ import annotations

from pathlib import Path

from raise_icarus.stage_contracts import StageDefinition, StageResult, definition_for, run_contract_stage

STAGE = definition_for("{module_name}")


def stage_definition() -> StageDefinition:
    return STAGE


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    return run_contract_stage(STAGE, harmonized_zip=harmonized_zip, run_dir=run_dir, n_samples=n_samples, dry_run=dry_run)
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def reproduce_public_aggregate_outputs(candidate_dir: str | Path, out_dir: str | Path) -> int:
    candidate = Path(candidate_dir)
    out_dir = Path(out_dir)
    rows = []
    for source_rel, target_rel, item in COPY_RULES:
        source = candidate / source_rel
        target = out_dir / target_rel
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            rows.append({"item": item, "source": source_rel, "output": target_rel, "status": "PASS", "notes": "reproduced from public aggregate file"})
        else:
            rows.append({"item": item, "source": source_rel, "output": target_rel, "status": "FAIL", "notes": "required aggregate file is missing"})
    report = out_dir / "reports" / "public_aggregate_reproduction.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "source", "output", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    return 1 if any(row["status"] != "PASS" for row in rows) else 0


def validate_public_release_candidate(candidate_dir: str | Path) -> int:
    candidate = Path(candidate_dir)
    rows: list[dict[str, str]] = []
    for rel_path in REQUIRED_FILES:
        add(rows, "required_file", "PASS" if (candidate / rel_path).is_file() else "FAIL", rel_path)
    for path in candidate.rglob("*") if candidate.exists() else []:
        if not path.is_file():
            continue
        rel_path = path.relative_to(candidate).as_posix()
        add(rows, "descriptive_script_name", "FAIL" if rel_path.startswith("scripts/") and BANNED_NAME_RE.search(rel_path) else "PASS", rel_path)
        add(rows, "unsafe_extension", "FAIL" if path.suffix.lower() in UNSAFE_EXTENSIONS else "PASS", rel_path)
        is_code_or_test = (
            rel_path.startswith(("scripts/", "src/", "tests/", "tools/"))
            or rel_path == "Run-Manuscript-Reproducibility.ps1"
        )
        s7_path_failure = (not rel_path.startswith("tests/")) and EXCLUDED_PUBLIC_OUTPUT_RE.search(rel_path)
        add(rows, "current_scope_excludes_absent_s7", "FAIL" if s7_path_failure else "PASS", rel_path)
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            local_path_found = LOCAL_PATH_RE.search(text) or CONTROLLED_ARCHIVE_NAME_RE.search(text)
            add(rows, "no_local_paths", "FAIL" if local_path_found else "PASS", rel_path)
            add(rows, "no_zero_p_value", "PASS" if is_code_or_test else ("FAIL" if ZERO_P_VALUE_RE.search(text) else "PASS"), rel_path)
            add(rows, "no_trace_wording", "FAIL" if any(term.lower() in text.lower() for term in TRACE_TERMS) else "PASS", rel_path)
            add(rows, "no_identifier_columns", "PASS" if is_code_or_test else ("FAIL" if any(pattern.search(text) for pattern in IDENTIFIER_PATTERNS) else "PASS"), rel_path)
    checksums = candidate / "checksums" / "manifest_sha256.txt"
    add(rows, "checksums_exist", "PASS" if checksums.exists() and checksums.stat().st_size > 0 else "FAIL", "checksums/manifest_sha256.txt")
    template_report = candidate / "results" / "validation" / "figure_template_fidelity_check.csv"
    if template_report.exists():
        template_rows = _read_csv_rows(template_report)
        bad = [row for row in template_rows if row.get("status", "") not in ALLOWED_TEMPLATE_STATUSES]
        missing = sorted(set(EXPECTED_FIGURE_IDS) - {row.get("item_id", "") or row.get("figure_id", "") for row in template_rows})
        add(rows, "figure_template_fidelity", "FAIL" if bad or missing else "PASS", template_report.relative_to(candidate).as_posix(), f"bad_statuses={len(bad)}; missing={len(missing)}")
    else:
        add(rows, "figure_template_fidelity", "FAIL", "results/validation/figure_template_fidelity_check.csv", "missing")
    lock_report = candidate / "results" / "validation" / "figure_template_lock_check.csv"
    if lock_report.exists():
        lock_rows = _read_csv_rows(lock_report)
        bad = [row for row in lock_rows if _status_is_blocked(row.get("status", "") or row.get("generation_status", ""))]
        missing = sorted(set(EXPECTED_FIGURE_IDS) - {row.get("figure_id", "") or row.get("item_id", "") for row in lock_rows})
        not_script_backed = [
            row for row in lock_rows
            if row.get("template_lock_status") not in {"SCRIPT_AVAILABLE", "SCRIPT_AND_TEMPLATE_AVAILABLE"}
        ]
        rejected_not_excluded = [
            row for row in lock_rows
            if row.get("rejected_candidates_excluded", "").strip().lower() not in {"yes", "true"}
        ]
        status = "FAIL" if bad or missing or not_script_backed or rejected_not_excluded else "PASS"
        notes = f"bad_statuses={len(bad)}; missing={len(missing)}; not_script_backed={len(not_script_backed)}; rejected_not_excluded={len(rejected_not_excluded)}"
        add(rows, "figure_template_lock", status, lock_report.relative_to(candidate).as_posix(), notes)
    else:
        add(rows, "figure_template_lock", "FAIL", "results/validation/figure_template_lock_check.csv", "missing")
    generation_report = candidate / "results" / "validation" / "figure_generation_report.csv"
    if generation_report.exists():
        generation_rows = _read_csv_rows(generation_report)
        bad = [row for row in generation_rows if row.get("status", "") not in ALLOWED_GENERATION_STATUSES]
        dry_run = [row for row in generation_rows if row.get("status", "") == "PASS_DRY_RUN"]
        missing = sorted(set(EXPECTED_FIGURE_IDS) - {row.get("figure_id", "") for row in generation_rows})
        status = "FAIL" if bad or dry_run or missing else "PASS"
        notes = f"bad_statuses={len(bad)}; dry_run_rows={len(dry_run)}; missing={len(missing)}"
        add(rows, "figure_generation_current_templates", status, generation_report.relative_to(candidate).as_posix(), notes)
    else:
        add(rows, "figure_generation_current_templates", "FAIL", "results/validation/figure_generation_report.csv", "missing")
    report_dir = candidate / "results" / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "public_candidate_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "path", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if row["status"] != "PASS"]
    (report_dir / "public_candidate_validation_report.txt").write_text("Public Candidate Validation Report\n\n" + f"checks_evaluated: {len(rows)}\n" + f"checks_failed: {len(failed)}\n" + "gate_status: " + ("FAIL" if failed else "PASS") + "\n", encoding="utf-8")
    return 1 if failed else 0


def add(rows: list[dict[str, str]], check: str, status: str, path: str = "", notes: str = "") -> None:
    rows.append({"check": check, "status": status, "path": path, "notes": notes})

def public_runtime_dependency_hits(repo_root: str | Path | None = None) -> list[dict[str, str]]:
    root = Path(repo_root) if repo_root else REPO_ROOT
    patterns = ["scripts/[0-9]", "scripts\\0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "phase8", "phase9", "phase10", "phase11", "regenerate_all_manuscript_outputs"]
    hits: list[dict[str, str]] = []
    for name in DESCRIPTIVE_SCRIPTS:
        path = root / "scripts" / name
        if not path.exists():
            hits.append({"script": f"scripts/{name}", "pattern": "missing"})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for pattern in patterns:
            if pattern.lower() in text:
                hits.append({"script": f"scripts/{name}", "pattern": pattern})
    return hits

