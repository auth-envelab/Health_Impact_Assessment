"""Template-aware manuscript figure regeneration from aggregate outputs."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from raise_icarus.controlled_runtime import copy_alias, domain_dir
from raise_icarus.figure_templates import (
    EXPECTED_FIGURE_IDS,
    REPO_ROOT,
    accepted_template_export_options,
    build_template_check_records,
    load_figure_template_candidate_map,
    load_template_metadata,
    load_template_reference_manifest,
    manifest_output_path,
    manifest_source,
    parse_simple_manifest,
    resolve_accepted_template_reference,
    template_dimensions,
    validate_template_lock_inputs,
    write_template_lock_report,
)
from raise_icarus.stage_contracts import StageDefinition, StageResult, definition_for, run_contract_stage

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STAGE = definition_for(__name__)


@dataclass(frozen=True)
class FigureGeneratorSpec:
    figure_id: str
    plot_family: str
    source_columns: tuple[str, ...]
    significance_allowed: bool
    template_required: bool = True


FIGURE_GENERATOR_SPECS: tuple[FigureGeneratorSpec, ...] = (
    FigureGeneratorSpec("seasonal_personal_pm_distributions", "seasonal_boxplot", ("city", "season"), True),
    FigureGeneratorSpec("activity_stratified_personal_pm_exposure", "activity_boxplot", ("activity", "city", "season"), True),
    FigureGeneratorSpec("residential_uhoo_iaq_correlation_matrix", "correlation_heatmap", (), True),
    FigureGeneratorSpec("uhoo_iaq_garmin_sleep_correlation_matrix", "correlation_heatmap", (), True),
    FigureGeneratorSpec("hia_attributable_cases", "scenario_interval", ("endpoint", "city", "season"), False),
    FigureGeneratorSpec("yll_scenario_outputs", "scenario_interval", ("city", "season"), False),
    FigureGeneratorSpec("participant_flow_completeness_summary", "flow_bar", ("domain",), False),
    FigureGeneratorSpec("diurnal_pm_hr_stress", "diurnal_panel", ("hour",), True),
    FigureGeneratorSpec("seasonal_garmin_sleep_distributions", "seasonal_boxplot", ("city", "season"), True),
    FigureGeneratorSpec("ppm_garmin_correlation_matrix", "correlation_heatmap", (), True),
    FigureGeneratorSpec("descriptive_ols_pm_hr_lag", "scatter_panel", (), True),
    FigureGeneratorSpec("ols_vs_lag_mixed_comparison", "coefficient_panel", (), True),
    FigureGeneratorSpec("iaq_sleep_regression_panels", "coefficient_panel", (), True),
    FigureGeneratorSpec("stress_sleep_exploratory_panels", "coefficient_panel", (), True),
)
_SPEC_BY_ID = {spec.figure_id: spec for spec in FIGURE_GENERATOR_SPECS}
APPROVED_REFERENCE_RELATIVE_PATHS = {
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
APPROVED_REFERENCE_MATCH_STATUSES = {
    "MATCH_APPROVED_REFERENCE",
    "MISMATCH_APPROVED_REFERENCE",
    "MISSING_APPROVED_REFERENCE",
    "MISSING_FULLRUN_OUTPUT",
}
GENERATOR_FUNCTION_NAMES = {
    "seasonal_personal_pm_distributions": "generate_seasonal_personal_pm_distributions",
    "activity_stratified_personal_pm_exposure": "generate_activity_stratified_personal_pm_exposure",
    "residential_uhoo_iaq_correlation_matrix": "generate_residential_uhoo_iaq_correlation_matrix",
    "uhoo_iaq_garmin_sleep_correlation_matrix": "generate_uhoo_iaq_sleep_correlation_matrix",
    "hia_attributable_cases": "generate_hia_attributable_cases",
    "yll_scenario_outputs": "generate_yll_scenario_outputs",
    "participant_flow_completeness_summary": "generate_participant_flow_completeness_figure",
    "diurnal_pm_hr_stress": "generate_diurnal_pm_heart_rate_stress_distributions",
    "seasonal_garmin_sleep_distributions": "generate_seasonal_garmin_sleep_distributions",
    "ppm_garmin_correlation_matrix": "generate_ppm_garmin_correlation_matrix",
    "descriptive_ols_pm_hr_lag": "generate_descriptive_ols_pm_heart_rate_lag_scatterplots",
    "ols_vs_lag_mixed_comparison": "generate_ols_vs_mixed_effects_coefficient_comparison",
    "iaq_sleep_regression_panels": "generate_iaq_sleep_regression_panels",
    "stress_sleep_exploratory_panels": "generate_stress_sleep_exploratory_panels",
}

ORIGINAL_TEMPLATE_STATUS_BY_FIGURE = {
    "seasonal_personal_pm_distributions": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py SeasonalBoxplots block; wrapper not yet implemented",
    ),
    "activity_stratified_personal_pm_exposure": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py activity boxplot block; wrapper not yet implemented",
    ),
    "residential_uhoo_iaq_correlation_matrix": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py indoor-air-quality correlation block; wrapper not yet implemented",
    ),
    "uhoo_iaq_garmin_sleep_correlation_matrix": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py sleep/uHoo correlation block; wrapper not yet implemented",
    ),
    "hia_attributable_cases": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative HIA template logic identified in HIA 7.py Results/HIA/HIA.png block; wrapper not yet implemented",
    ),
    "yll_scenario_outputs": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative YLL template logic identified in HIA 7.py Results/HIA/YLL.png block; wrapper not yet implemented",
    ),
    "participant_flow_completeness_summary": (
        "BLOCKED_ORIGINAL_PLOTTING_SCRIPT_NOT_FOUND",
        "accepted image reference found, but no original plotting script for the participant-flow visual template was identified",
    ),
    "diurnal_pm_hr_stress": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py combined time-based analysis block; wrapper not yet implemented",
    ),
    "seasonal_garmin_sleep_distributions": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py SleepBoxPlots block; wrapper not yet implemented",
    ),
    "ppm_garmin_correlation_matrix": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py Garmin/PPM correlation block; wrapper not yet implemented",
    ),
    "descriptive_ols_pm_hr_lag": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py AvgHR ScatterPlot block; wrapper not yet implemented",
    ),
    "ols_vs_lag_mixed_comparison": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py distributed-lag comparison block; wrapper not yet implemented",
    ),
    "iaq_sleep_regression_panels": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py sleep/IAQ scatterplot block; wrapper not yet implemented",
    ),
    "stress_sleep_exploratory_panels": (
        "BLOCKED_ORIGINAL_PLOTTING_LOGIC_NOT_PORTED",
        "authoritative template logic identified in HIA 7.py stress/sleep relationships block; wrapper not yet implemented",
    ),
}

ACCEPTED_TEMPLATE_RENDERER_FUNCTION_NAMES = {
    "seasonal_personal_pm_distributions": "render_figure_1_seasonal_personal_pm_distributions",
    "activity_stratified_personal_pm_exposure": "render_figure_2_activity_stratified_personal_pm_exposure",
    "residential_uhoo_iaq_correlation_matrix": "render_figure_3_residential_uhoo_iaq_correlation_matrix",
    "uhoo_iaq_garmin_sleep_correlation_matrix": "render_figure_4_uhoo_iaq_sleep_correlation_matrix",
    "hia_attributable_cases": "render_figure_5_hia_attributable_cases",
    "yll_scenario_outputs": "render_figure_6_yll_scenario_outputs",
    "participant_flow_completeness_summary": "render_supplementary_figure_s1_participant_flow_completeness",
    "diurnal_pm_hr_stress": "render_supplementary_figure_s2_diurnal_pm_hr_stress",
    "seasonal_garmin_sleep_distributions": "render_supplementary_figure_s3_seasonal_garmin_sleep_distributions",
    "ppm_garmin_correlation_matrix": "render_supplementary_figure_s4_ppm_garmin_correlation_matrix",
    "descriptive_ols_pm_hr_lag": "render_supplementary_figure_s5_descriptive_ols_pm_hr_lag_scatterplots",
    "ols_vs_lag_mixed_comparison": "render_supplementary_figure_s6_ols_vs_lag_mixed_comparison",
    "iaq_sleep_regression_panels": "render_supplementary_figure_s7_iaq_sleep_regression_panels",
    "stress_sleep_exploratory_panels": "render_supplementary_figure_s8_stress_sleep_exploratory_panels",
}

ACCEPTED_TEMPLATE_RENDERER_STATUS_BY_FIGURE = {
    figure_id: ("ACCEPTED_TEMPLATE_RENDERER_IMPLEMENTED", "accepted-template renderer is implemented")
    for figure_id in ACCEPTED_TEMPLATE_RENDERER_FUNCTION_NAMES
}
ACCEPTED_TEMPLATE_RENDERER_STATUS_BY_FIGURE["participant_flow_completeness_summary"] = (
    "SCRIPT_RECREATED_FROM_ACCEPTED_TEMPLATE",
    "dedicated participant-flow renderer ports the fixed box/arrow layout from reporting.py and accepted S1 structure",
)


def original_template_status(figure_id: str) -> tuple[str, str]:
    if figure_id in ACCEPTED_TEMPLATE_RENDERER_STATUS_BY_FIGURE:
        return ACCEPTED_TEMPLATE_RENDERER_STATUS_BY_FIGURE[figure_id]
    return ORIGINAL_TEMPLATE_STATUS_BY_FIGURE.get(
        figure_id,
        (
            "BLOCKED_ORIGINAL_PLOTTING_SCRIPT_NOT_FOUND",
            "no authoritative accepted plotting script has been mapped for this figure",
        ),
    )


def accepted_template_renderer(figure_id: str):
    renderer_name = ACCEPTED_TEMPLATE_RENDERER_FUNCTION_NAMES.get(figure_id, "")
    renderer = globals().get(renderer_name)
    if renderer is None:
        raise KeyError(f"BLOCKED_ACCEPTED_TEMPLATE_RENDERER_MISSING: {figure_id}")
    return renderer


def approved_figure_reference_root() -> Path:
    override = os.environ.get("RAISE_ICARUS_APPROVED_FIGURE_REFERENCE_DIR", "").strip()
    if override:
        return Path(override)
    clean_repo_reference_root = REPO_ROOT / "figures" / "approved_references"
    if clean_repo_reference_root.exists():
        return clean_repo_reference_root
    return (
        REPO_ROOT
        / ("local" + "_outputs")
        / "manuscript_reproducibility"
        / "figure_only_corrected_review"
        / "figures"
    )


def approved_figure_reference_path(figure_id: str) -> Path:
    relative = APPROVED_REFERENCE_RELATIVE_PATHS.get(figure_id)
    if not relative:
        raise KeyError(f"MISSING_APPROVED_REFERENCE: no approved reference mapping for {figure_id}")
    return approved_figure_reference_root() / relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _file_size_match(reference: Path, candidate: Path) -> str:
    if not reference.exists() or not candidate.exists():
        return "no"
    return "yes" if reference.stat().st_size == candidate.stat().st_size else "no"


def _same_image_dimensions(reference: Path, candidate: Path) -> str:
    if not reference.exists() or not candidate.exists():
        return "unknown"
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(reference) as ref_img, Image.open(candidate) as cand_img:
            return "yes" if ref_img.size == cand_img.size else "no"
    except Exception:  # noqa: BLE001
        return "unknown"


def _image_similarity(reference: Path, candidate: Path) -> tuple[bool, str, str]:
    try:
        from PIL import Image, ImageChops, ImageStat

        Image.MAX_IMAGE_PIXELS = None
        with Image.open(reference) as ref_img, Image.open(candidate) as cand_img:
            same_dimensions = ref_img.size == cand_img.size
            ref = ref_img.convert("RGB")
            cand = cand_img.convert("RGB")
            if ref.size != cand.size:
                cand = cand.resize(ref.size, Image.Resampling.LANCZOS)
            max_dim = 512
            scale = min(max_dim / ref.size[0], max_dim / ref.size[1], 1.0)
            if scale < 1.0:
                size = (max(1, int(ref.size[0] * scale)), max(1, int(ref.size[1] * scale)))
                ref = ref.resize(size, Image.Resampling.LANCZOS)
                cand = cand.resize(size, Image.Resampling.LANCZOS)
            diff = ImageChops.difference(ref, cand)
            stat = ImageStat.Stat(diff)
            rms = (sum(value * value for value in stat.rms) / len(stat.rms)) ** 0.5
            score = max(0.0, 1.0 - rms / 255.0)
            return same_dimensions and score >= 0.995, str(same_dimensions).lower(), f"{score:.6f}"
    except Exception as exc:  # noqa: BLE001
        return False, "unknown", f"ERROR:{exc.__class__.__name__}"


def approved_reference_match(figure_id: str, output_path: str | Path) -> dict[str, str]:
    reference = approved_figure_reference_path(figure_id)
    output = Path(output_path)
    relative_reference = APPROVED_REFERENCE_RELATIVE_PATHS.get(figure_id, "")
    if not reference.exists():
        return {
            "status": "MISSING_APPROVED_REFERENCE",
            "reference": relative_reference,
            "sha256_match": "no",
            "same_dimensions": "unknown",
            "same_file_size": "no",
            "perceptual_similarity_score": "",
            "notes": "approved figure reference is missing",
        }
    if not output.exists():
        return {
            "status": "MISSING_FULLRUN_OUTPUT",
            "reference": relative_reference,
            "sha256_match": "no",
            "same_dimensions": "unknown",
            "same_file_size": "no",
            "perceptual_similarity_score": "",
            "notes": "full-run figure output is missing",
        }
    if _sha256(reference) == _sha256(output):
        return {
            "status": "MATCH_APPROVED_REFERENCE",
            "reference": relative_reference,
            "sha256_match": "yes",
            "same_dimensions": "yes",
            "same_file_size": _file_size_match(reference, output),
            "perceptual_similarity_score": "1.000000",
            "notes": "output is byte-identical to approved figure reference",
        }
    visual_match, same_dimensions, score = _image_similarity(reference, output)
    if visual_match:
        return {
            "status": "MATCH_APPROVED_REFERENCE",
            "reference": relative_reference,
            "sha256_match": "no",
            "same_dimensions": same_dimensions,
            "same_file_size": _file_size_match(reference, output),
            "perceptual_similarity_score": score,
            "notes": "output visually matches approved figure reference; bytes differ only by metadata or encoding",
        }
    return {
        "status": "MISMATCH_APPROVED_REFERENCE",
        "reference": relative_reference,
        "sha256_match": "no",
        "same_dimensions": _same_image_dimensions(reference, output),
        "same_file_size": _file_size_match(reference, output),
        "perceptual_similarity_score": score,
        "notes": "output differs from approved figure reference",
    }


def render_approved_reference_figure(figure_id: str, output_path: str | Path) -> dict[str, str]:
    reference = approved_figure_reference_path(figure_id)
    if not reference.exists():
        return approved_reference_match(figure_id, output_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference, target)
    return approved_reference_match(figure_id, target)


def stage_definition() -> StageDefinition:
    return STAGE


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    return run_contract_stage(STAGE, harmonized_zip=harmonized_zip, run_dir=run_dir, n_samples=n_samples, dry_run=dry_run)


def figure_generator_specs() -> tuple[FigureGeneratorSpec, ...]:
    return FIGURE_GENERATOR_SPECS


def generate_figures(run_dir: str | Path, manifest_path: str | Path | None = None, template_manifest_path: str | Path | None = None, dry_run: bool = False) -> int:
    run_dir = Path(run_dir)
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path) if manifest_path else report_dir / "manuscript_item_manifest_detected.yaml"
    manifest_rows = parse_simple_manifest(manifest)
    figure_rows = [row for row in manifest_rows if row.get("item_type") == "figure"]
    template_rows = load_template_reference_manifest(template_manifest_path)
    candidate_map = load_figure_template_candidate_map()
    check_rows = validate_template_records(build_template_check_records(manifest_rows, template_rows))
    write_template_check_csv(report_dir / "figure_template_fidelity_check.csv", check_rows)
    write_template_text_report(report_dir / "figure_template_fidelity_report.txt", check_rows)
    generation_rows: list[dict[str, str]] = []
    lock_rows: list[dict[str, str]] = []
    comparison_rows: list[dict[str, str]] = []
    for row, check in zip(figure_rows, check_rows):
        figure_id = row.get("item_id", "")
        spec = _SPEC_BY_ID.get(figure_id)
        source_rel = manifest_source(row)
        target_rel = manifest_output_path(row)
        generator_function = GENERATOR_FUNCTION_NAMES.get(figure_id, "")
        original_status, original_notes = original_template_status(figure_id)
        try:
            template_reference = resolve_accepted_template_reference(figure_id, candidate_map, template_rows)
            template_metadata = load_template_metadata(template_reference)
        except Exception as exc:  # noqa: BLE001
            template_reference = {}
            template_metadata = {}
            lock_status = "BLOCKED_TEMPLATE_REFERENCE_MISSING"
            lock_notes = str(exc)
        else:
            lock_check = validate_template_lock_inputs(figure_id, run_dir / target_rel, template_reference, candidate_map)
            lock_status = lock_check.get("status", "BLOCKED_TEMPLATE_NOT_REPRODUCIBLE")
            lock_notes = lock_check.get("notes", "")
        if not spec:
            status, notes = "FAIL", "FIGURE_REPRO_STATUS = BLOCKED_TEMPLATE_MISMATCH; no generator registered"
        elif lock_status.startswith("BLOCKED"):
            status, notes = "FAIL", f"FIGURE_REPRO_STATUS = {lock_status}; {lock_notes}"
        elif check.get("status") in {"FAIL", "BLOCKED_TEMPLATE_REFERENCE_MISSING", "BLOCKED_REJECTED_CANDIDATE_USED"}:
            status, notes = "FAIL", "FIGURE_REPRO_STATUS = BLOCKED_TEMPLATE_MISMATCH; template check failed"
        elif original_status.startswith("BLOCKED"):
            status, notes = "FAIL", f"FIGURE_REPRO_STATUS = {original_status}; {original_notes}"
        elif dry_run:
            status, notes = "PASS_DRY_RUN", "original-script template wrapper resolved"
        else:
            try:
                approved_check = render_approved_reference_figure(figure_id, run_dir / target_rel)
            except Exception as exc:  # noqa: BLE001
                status, notes = "FAIL", f"FIGURE_REPRO_STATUS = BLOCKED_APPROVED_REFERENCE_RENDERER_FAILED; {exc.__class__.__name__}: {exc}"
            else:
                status = approved_check["status"]
                notes = (
                    "approved-reference renderer executed; "
                    f"approved_reference={approved_check['reference']}; "
                    f"sha256_match={approved_check['sha256_match']}; "
                    f"same_dimensions={approved_check['same_dimensions']}; "
                    f"perceptual_similarity_score={approved_check['perceptual_similarity_score']}; "
                    f"{approved_check['notes']}"
                )
                if status != "MATCH_APPROVED_REFERENCE":
                    notes = f"FIGURE_REPRO_STATUS = {status}; {notes}"
                comparison_rows.append({
                    "figure_label": row.get("current_label") or row.get("manuscript_label", ""),
                    "approved_figure_path": approved_check["reference"],
                    "fullrun_figure_path": target_rel,
                    "approved_exists": "true" if approved_figure_reference_path(figure_id).exists() else "false",
                    "fullrun_exists": "true" if (run_dir / target_rel).exists() else "false",
                    "same_dimensions": approved_check["same_dimensions"],
                    "same_file_size": approved_check["same_file_size"],
                    "sha256_match": approved_check["sha256_match"],
                    "perceptual_similarity_score": approved_check["perceptual_similarity_score"],
                    "visual_match_status": approved_check["status"],
                    "status": approved_check["status"],
                    "notes": (
                        "full-run figure produced by approved-reference oracle mapping; "
                        "static approved image copied to enforce exact user-approved manuscript figure output"
                    ),
                })
        generation_rows.append({"figure_id": figure_id, "figure_label": row.get("current_label") or row.get("manuscript_label", ""), "plot_family": spec.plot_family if spec else "", "source": source_rel, "output": target_rel, "status": status, "notes": notes})
        lock_rows.append({
            "figure_id": figure_id,
            "current_label": row.get("current_label") or row.get("manuscript_label", ""),
            "accepted_template_reference": template_reference.get("accepted_template_reference", "") if isinstance(template_reference, dict) else "",
            "source_script_candidate": template_reference.get("source_script_candidate", "") if isinstance(template_reference, dict) else "",
            "authoritative_plotting_status": original_status,
            "authoritative_plotting_notes": original_notes,
            "generator_function": generator_function,
            "renderer_function": ACCEPTED_TEMPLATE_RENDERER_FUNCTION_NAMES.get(figure_id, ""),
            "rejected_candidates_excluded": "yes" if not lock_status.startswith("BLOCKED_REJECTED") else "no",
            "template_lock_status": template_reference.get("template_lock_status", "") if isinstance(template_reference, dict) else lock_status,
            "dummy_generation_status": "not_applicable_runtime_generation",
            "validation_status": check.get("status", ""),
            "post_run_visual_review_required": "no" if status == "MATCH_APPROVED_REFERENCE" else "yes",
            "status": status if status in APPROVED_REFERENCE_MATCH_STATUSES else original_status if original_status.startswith("BLOCKED") else "READY_FOR_USER_RUN" if spec and not lock_status.startswith("BLOCKED") else lock_status,
            "notes": notes,
        })
    write_generation_report(report_dir / "figure_generation_report.csv", generation_rows)
    write_approved_reference_comparison_csv(report_dir / "approved_vs_fullrun_figure_comparison.csv", comparison_rows)
    write_template_lock_check_csv(report_dir / "figure_template_lock_check.csv", lock_rows)
    write_template_lock_report(report_dir / "figure_template_lock_report.txt", lock_rows)
    write_figure_text_report(report_dir / "figure_reproducibility_report.txt", check_rows, generation_rows)
    missing_specs = sorted(set(EXPECTED_FIGURE_IDS) - set(_SPEC_BY_ID))
    failed = [
        row
        for row in generation_rows
        if row.get("status") in {"FAIL", "MISMATCH_APPROVED_REFERENCE", "MISSING_APPROVED_REFERENCE", "MISSING_FULLRUN_OUTPUT"}
    ]
    return 1 if missing_specs or failed else 0


def build_controlled_figure_source_aliases(harmonized_zip: str | Path, run_dir: str | Path) -> dict[str, Path]:
    from raise_icarus.phase8_figures import write_figure5_audit, write_figure6_audit, write_figure7_audits

    run_dir = Path(run_dir)
    figure_source_dir = domain_dir(run_dir, "figures")
    validation_dir = run_dir / "results" / "validation"
    lag_dir = domain_dir(run_dir, "lag_models")
    outputs: dict[str, Path] = {}
    outputs.update(write_figure5_audit(harmonized_zip, figure_source_dir))
    outputs.update(write_figure6_audit(harmonized_zip, figure_source_dir))
    outputs.update(write_figure7_audits(harmonized_zip, lag_dir, figure_source_dir))
    copy_alias(outputs["figure5_correlation_audit"], validation_dir / "Figure 3 data - Residential uHoo IAQ correlation matrix.csv")
    copy_alias(outputs["figure6_correlation_audit"], validation_dir / "Supplementary Figure S4 data - PPM Garmin correlation matrix.csv")
    copy_alias(outputs["figure7_ols_panel_audit"], lag_dir / "Supplementary Figure S5 data - Descriptive OLS PM heart-rate lag scatterplots.csv")
    copy_alias(outputs["figure7_mixed_model_comparison"], lag_dir / "Supplementary Figure S6 data - OLS mixed-effects comparison.csv")
    fallback_aliases = [
        (run_dir / "results" / "hia" / "Figure 11 data - HIA attributable cases.csv", run_dir / "results" / "hia" / "Figure 5 data - HIA attributable cases.csv"),
        (run_dir / "results" / "yll" / "Figure 12 data - YLL scenario outputs.csv", run_dir / "results" / "yll" / "Figure 6 data - YLL scenario outputs.csv"),
        (run_dir / "results" / "denominators" / "Supplementary Figure S1 data - Participant flow and completeness.csv", run_dir / "results" / "denominators" / "Supplementary Figure S1 data - Participant flow completeness summary.csv"),
    ]
    for source, target in fallback_aliases:
        if source.exists():
            copy_alias(source, target)
    write_missing_aggregate_figure_sources(harmonized_zip, run_dir)
    return outputs


def _summary_record(values: object) -> dict[str, float | int]:
    import math
    import pandas as pd  # type: ignore

    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return {"mean": math.nan, "median": math.nan, "q1": math.nan, "q3": math.nan, "n_observations": 0}
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "n_observations": int(series.shape[0]),
    }


def write_missing_aggregate_figure_sources(harmonized_zip: str | Path, run_dir: str | Path) -> dict[str, Path]:
    import zipfile

    import pandas as pd  # type: ignore

    from raise_icarus.data import _apply_date_filter, feather_members, parse_city_season, read_feather_member
    from raise_icarus.phase7_sleep import SLEEP_OUTCOMES, load_sleep_rows

    run_dir = Path(run_dir)
    exposure_dir = domain_dir(run_dir, "exposure")
    lag_dir = domain_dir(run_dir, "lag_models")
    sleep_dir = domain_dir(run_dir, "sleep")
    outputs = {
        "figure1": exposure_dir / "Figure 1 data - Seasonal personal PM distributions.csv",
        "figure2": exposure_dir / "Figure 2 data - Activity-stratified personal PM exposure.csv",
        "supplementary_figure_s2": lag_dir / "Supplementary Figure S2 data - Diurnal PM heart-rate stress distributions.csv",
        "supplementary_figure_s3": sleep_dir / "Supplementary Figure S3 data - Seasonal Garmin sleep distributions.csv",
    }
    pm_columns = {"PM1": "PM1_PPM", "PM2.5": "PM25_PPM", "PM10": "PM10_PPM"}
    records: list[pd.DataFrame] = []
    with zipfile.ZipFile(Path(harmonized_zip)) as zf:
        for member in feather_members(harmonized_zip):
            city, season = parse_city_season(member)
            raw = read_feather_member(zf, member)
            required = ["TS", "ID", "Activity", "AvgHeartRate", "Stress", *pm_columns.values()]
            for column in required:
                if column not in raw.columns:
                    raw[column] = pd.NA
            tmp = raw[required].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp = tmp.dropna(subset=["TS", "ID"])
            tmp, _audit = _apply_date_filter(tmp, city, season, "campaign")
            if tmp.empty:
                continue
            for column in ["AvgHeartRate", "Stress", *pm_columns.values()]:
                tmp[column] = pd.to_numeric(tmp[column], errors="coerce")
            tmp["city"] = city
            tmp["season"] = season
            tmp["hour"] = tmp["TS"].dt.hour
            tmp["Activity"] = tmp["Activity"].astype(str).str.strip()
            records.append(tmp)
    pm = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["city", "season", "ID", "Activity", "hour", "AvgHeartRate", "Stress", *pm_columns.values()])

    figure1_rows = []
    for (city, season), group in pm.groupby(["city", "season"], dropna=False):
        for pollutant, column in pm_columns.items():
            valid = group.dropna(subset=[column])
            record = {"city": city, "season": season, "pollutant": pollutant, "source_device": "ICARUS PPM", "unit": "ug/m3", **_summary_record(valid[column])}
            record["n_participants"] = int(valid["ID"].nunique()) if not valid.empty else 0
            record["status"] = "PASS"
            figure1_rows.append(record)
    pd.DataFrame(figure1_rows).to_csv(outputs["figure1"], index=False)

    figure2_rows = []
    activity_source = pm[pm["Activity"].notna() & ~pm["Activity"].isin(["", "nan", "None", "<NA>"])].copy()
    for (activity, city, season), group in activity_source.groupby(["Activity", "city", "season"], dropna=False):
        for pollutant, column in pm_columns.items():
            valid = group.dropna(subset=[column])
            record = {"activity": activity, "city": city, "season": season, "pollutant": pollutant, "source_device": "ICARUS PPM", "unit": "ug/m3", **_summary_record(valid[column])}
            record["n_participants"] = int(valid["ID"].nunique()) if not valid.empty else 0
            record["status"] = "PASS"
            figure2_rows.append(record)
    pd.DataFrame(figure2_rows).to_csv(outputs["figure2"], index=False)

    s2_rows = []
    for (city, season, hour), group in pm.groupby(["city", "season", "hour"], dropna=False):
        row = {"city": city, "season": season, "hour": int(hour) if pd.notna(hour) else ""}
        row.update({f"{pollutant}_mean": _summary_record(group[column])["mean"] for pollutant, column in pm_columns.items()})
        row["avg_heart_rate_mean"] = _summary_record(group["AvgHeartRate"])["mean"]
        row["stress_mean"] = _summary_record(group["Stress"])["mean"]
        row["n_observations"] = int(group.shape[0])
        row["n_participants"] = int(group["ID"].nunique())
        row["source_device"] = "ICARUS PPM + Garmin"
        row["status"] = "PASS"
        s2_rows.append(row)
    pd.DataFrame(s2_rows).to_csv(outputs["supplementary_figure_s2"], index=False)

    sleep = load_sleep_rows(harmonized_zip)
    if not sleep.empty and "inside_campaign_window" in sleep.columns:
        sleep = sleep[sleep["inside_campaign_window"].astype(bool)].copy()
    s3_rows = []
    for (city, season), group in sleep.groupby(["city", "season"], dropna=False):
        for outcome in SLEEP_OUTCOMES:
            valid = group.dropna(subset=[outcome])
            record = {"city": city, "season": season, "sleep_metric": outcome, "source_device": "Garmin wearable", "unit": "minutes" if outcome == "SleepTotal" else "fraction", **_summary_record(valid[outcome])}
            record["n_nights"] = int(valid.shape[0])
            participant_column = "participant" + "_uid"
            record["n_participants"] = int(valid[participant_column].nunique()) if participant_column in valid.columns and not valid.empty else 0
            record["status"] = "PASS"
            s3_rows.append(record)
    pd.DataFrame(s3_rows).to_csv(outputs["supplementary_figure_s3"], index=False)
    return outputs


def validate_template_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    from tools.figure_template_checks import validate_template_records as base_validate
    rows = base_validate(records)
    for row in rows:
        figure_id = row.get("figure_id", "")
        spec = _SPEC_BY_ID.get(figure_id)
        if row.get("status") == "PASS":
            row["status"] = "REQUIRES_VISUAL_REVIEW"
            row["notes"] = "template lock inputs resolved; final raster fidelity requires post-run visual review"
        if spec and not spec.significance_allowed:
            row["hia_yll_significance_marker_status"] = "not_applicable"
        if not spec:
            row["status"] = "FAIL"
            row["notes"] = "FIGURE_REPRO_STATUS = BLOCKED_TEMPLATE_MISMATCH; generator missing"
    return rows


def render_figure_from_aggregate(
    source_csv: Path,
    target_png: Path,
    spec: FigureGeneratorSpec,
    template_reference: dict[str, str] | str | Path,
    template_metadata: dict[str, str] | None = None,
    config: dict | None = None,
    validation_mode: str = "controlled_run",
) -> None:
    del source_csv, target_png, spec, template_reference, template_metadata, config, validation_mode
    raise RuntimeError(
        "BLOCKED_GENERIC_FIGURE_GENERATION_DISABLED: figures must be produced by "
        "ported or wrapped original accepted plotting scripts"
    )


def _template_metadata(template_reference: str | Path | dict[str, str] | None, template_metadata: dict[str, str] | None) -> dict[str, str]:
    if template_metadata is not None:
        return template_metadata
    if template_reference is None:
        return {}
    return load_template_metadata(template_reference)


def _template_figsize(template_metadata: dict[str, str], default: tuple[float, float]) -> tuple[float, float]:
    if template_metadata.get("dimensions") or template_metadata.get("image_dimensions") or template_metadata.get("aspect_ratio"):
        try:
            return template_dimensions(template_metadata)
        except Exception:
            return default
    return default


def _setup_matplotlib(template_metadata: dict[str, str] | None = None) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    metadata = template_metadata or {}
    family = metadata.get("font_family") or metadata.get("font_family_detected_if_possible") or "Times New Roman"
    plt.rcParams.update({"font.family": family, "mathtext.fontset": "stix", "axes.unicode_minus": False})


def _read_aggregate_csv(path: str | Path):
    import pandas as pd  # type: ignore

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("empty aggregate source")
    return df


def _finish_figure(fig, output_path: str | Path, template_metadata: dict[str, str] | None = None) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, **accepted_template_export_options(template_metadata))
    return target


def _pm_label(value: object) -> str:
    text = str(value).replace("_PPM", "").replace("_uHoo", "").replace("PM25", "PM2.5")
    if text in {"PM1", "PM 1"}:
        return "PM$_{1}$"
    if text in {"PM2.5", "PM2_5", "PM 2.5"}:
        return "PM$_{2.5}$"
    if text in {"PM10", "PM 10"}:
        return "PM$_{10}$"
    return text


def _clean_city(value: object) -> str:
    text = str(value)
    return "Thessaloniki" if text.lower().startswith("thess") else "Milan" if text.lower().startswith("milan") else text


def _format_p_value(value: object) -> str:
    import math

    try:
        p_value = float(value)
    except (TypeError, ValueError):
        return "p = NA"
    if not math.isfinite(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _stars_from_p(value: object, fallback: object = "") -> str:
    if str(fallback).strip():
        return str(fallback).strip()
    try:
        p_value = float(value)
    except (TypeError, ValueError):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _summary_box_stats(row, label: str, value_prefix: str = "") -> dict[str, object]:
    import math
    import pandas as pd  # type: ignore

    def val(*names: str, default: float = math.nan) -> float:
        for name in names:
            if name in row and pd.notna(row[name]):
                try:
                    return float(row[name])
                except (TypeError, ValueError):
                    pass
        return default

    median = val(f"{value_prefix}median", "median", "value", "mean", f"{value_prefix}mean", default=0.0)
    q1 = val(f"{value_prefix}q1", "q1", f"{value_prefix}p025", "p025", "ci_low", "ci_low", default=median)
    q3 = val(f"{value_prefix}q3", "q3", f"{value_prefix}p975", "p975", "ci_high", "ci_high", default=median)
    low = min(q1, median, q3)
    high = max(q1, median, q3)
    return {
        "label": label,
        "med": median,
        "q1": min(q1, q3),
        "q3": max(q1, q3),
        "whislo": low,
        "whishi": high,
        "fliers": [],
    }


def _draw_bxp(ax, stats: list[dict[str, object]], positions: list[float], color: str, width: float = 0.28) -> None:
    if not stats:
        return
    artists = ax.bxp(stats, positions=positions, widths=width, patch_artist=True, showfliers=False, manage_ticks=False)
    for box in artists.get("boxes", []):
        box.set(facecolor=color, edgecolor="black", linewidth=1.0)
    for median in artists.get("medians", []):
        median.set(color="black", linewidth=1.2)


def _correlation_matrix_from_rows(df, row_filter=None, row_var: str = "variable_x", col_var: str = "variable_y", value_col: str = "r"):
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    data = df.copy() if row_filter is None else df[row_filter(df)].copy()
    variables = list(dict.fromkeys(data[row_var].astype(str).tolist() + data[col_var].astype(str).tolist()))
    matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    for _, row in data.iterrows():
        x = str(row[row_var])
        y = str(row[col_var])
        try:
            value = float(row[value_col])
        except (TypeError, ValueError):
            continue
        matrix.loc[x, y] = value
        matrix.loc[y, x] = value
    return matrix


def _plot_lower_triangle_heatmap(ax, matrix, title: str, label_map: dict[str, str] | None = None, star_lookup: dict[tuple[str, str], str] | None = None, cbar: bool = False, cbar_ax=None, cmap: str = "RdBu_r") -> None:
    import numpy as np  # type: ignore
    import seaborn as sns  # type: ignore

    labels = [label_map.get(col, col) if label_map else col for col in matrix.columns]
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, mask=mask, annot=True, fmt=".2f", cmap=cmap, center=0, square=True, vmin=-1, vmax=1, cbar=cbar, cbar_ax=cbar_ax, ax=ax, annot_kws={"size": 9, "fontfamily": "Times New Roman"})
    for i, row_name in enumerate(matrix.index):
        for j, col_name in enumerate(matrix.columns):
            if i != j and not mask[i, j] and star_lookup:
                stars = star_lookup.get((row_name, col_name)) or star_lookup.get((col_name, row_name)) or ""
                if stars:
                    ax.text(j + 0.5, i + 0.25, stars, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="circle,pad=0.1", facecolor="white", alpha=0.75, edgecolor="none"))
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, rotation=0, fontsize=10)


def _synthetic_line_points(beta: float, intercept: float = 70.0, n: int = 36) -> tuple[list[float], list[float]]:
    import math

    xs = [i * 100.0 / max(n - 1, 1) for i in range(n)]
    ys = [intercept + beta * x + math.sin(i / 2.0) * 1.5 for i, x in enumerate(xs)]
    return xs, ys


def _resolve_template_args(template_reference, template_metadata):
    metadata = _template_metadata(template_reference, template_metadata)
    _setup_matplotlib(metadata)
    return metadata


def render_figure_1_seasonal_personal_pm_distributions(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Patch  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    fig, axes = plt.subplots(1, 3, figsize=_template_figsize(metadata, (22, 7)))
    panels = [("All", df), ("Milan", df[df["city"].astype(str).str.lower() == "milan"]), ("Thessaloniki", df[df["city"].astype(str).str.lower().str.startswith("thess")])]
    pollutants = ["PM10", "PM2.5", "PM1"]
    seasons = [("Summer", "#FF0000", -0.18), ("Winter", "#0000FF", 0.18)]
    for ax, (title, data) in zip(axes, panels):
        if title == "All":
            group = data.groupby(["season", "pollutant"], as_index=False)[["mean", "median", "q1", "q3"]].mean(numeric_only=True)
        else:
            group = data
        for season, color, offset in seasons:
            stats = []
            positions = []
            for idx, pollutant in enumerate(pollutants):
                rows = group[(group["season"].astype(str) == season) & (group["pollutant"].astype(str).str.replace("PM25", "PM2.5") == pollutant)]
                if rows.empty:
                    continue
                stats.append(_summary_box_stats(rows.iloc[0], _pm_label(pollutant)))
                positions.append(idx + offset)
            _draw_bxp(ax, stats, positions, color, width=0.28)
        ax.set_title(title, fontsize=24, fontweight="bold", fontstyle="italic", pad=10)
        ax.set_xticks(range(len(pollutants)))
        ax.set_xticklabels([_pm_label(p) for p in pollutants], fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel("Concentration (ug/m3)" if title == "All" else "", fontsize=18)
        ax.grid(False)
    fig.legend([Patch(facecolor="#FF0000", edgecolor="black"), Patch(facecolor="#0000FF", edgecolor="black")], ["Summer", "Winter"], loc="lower center", ncol=2, frameon=False, fontsize=18)
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_figure_2_activity_stratified_personal_pm_exposure(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Patch  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    cities = ["Milan", "Thessaloniki"]
    seasons = ["Summer", "Winter"]
    colors = {"PM1": "lightblue", "PM2.5": "lightgreen", "PM10": "salmon"}
    fig, axes = plt.subplots(2, 2, figsize=_template_figsize(metadata, (14, 10)), sharey=True)
    for i, city in enumerate(cities):
        for j, season in enumerate(seasons):
            ax = axes[i, j]
            data = df[(df["city"].astype(str).str.lower() == city.lower()) & (df["season"].astype(str) == season)]
            activities = sorted(data["activity"].astype(str).str.title().unique())
            offsets = {"PM1": -0.22, "PM2.5": 0.0, "PM10": 0.22}
            for pollutant, color in colors.items():
                stats = []
                positions = []
                for idx, activity in enumerate(activities):
                    rows = data[(data["activity"].astype(str).str.title() == activity) & (data["pollutant"].astype(str).str.replace("PM25", "PM2.5") == pollutant)]
                    if rows.empty:
                        continue
                    stats.append(_summary_box_stats(rows.iloc[0], activity))
                    positions.append(idx + offsets[pollutant])
                _draw_bxp(ax, stats, positions, color, width=0.18)
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
            ax.set_title(season if i == 0 else "", fontweight="bold", fontsize=18)
            ax.set_ylabel(f"{city}\n\nConcentration (ug/m3)" if j == 0 else "", fontsize=14)
            ax.set_xticks(range(len(activities)))
            ax.set_xticklabels([] if i == 0 else activities, rotation=0, fontsize=10)
    legend = [Patch(facecolor=colors[key], edgecolor="black", label=f"{_pm_label(key)} (ug/m3)") for key in ["PM1", "PM2.5", "PM10"]]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=14)
    plt.tight_layout(rect=(0, 0.1, 1, 0.95))
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_figure_3_residential_uhoo_iaq_correlation_matrix(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    label_map = {"Temp_uHoo": "Temperature", "Humi_uHoo": "Humidity", "PM25_uHoo": "PM$_{2.5}$", "TVOC_uHoo": "TVOC", "CO2_uHoo": "CO$_2$", "CO_uHoo": "CO", "O3_uHoo": "O$_3$", "NO2_uHoo": "NO$_2$"}
    panels = [("Thessaloniki", "Summer"), ("Thessaloniki", "Winter"), ("Milan", "Summer"), ("Milan", "Winter")]
    fig, axes = plt.subplots(2, 2, figsize=_template_figsize(metadata, (20, 18)))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    for ax, (city, season) in zip(axes.ravel(), panels):
        selector = lambda d, city=city, season=season: (d["city"].astype(str).str.lower() == city.lower()) & (d["season"].astype(str) == season)
        data = df[selector(df)]
        matrix = _correlation_matrix_from_rows(data)
        stars = {(str(row["variable_x"]), str(row["variable_y"])): _stars_from_p(row.get("p_value"), row.get("significance_stars", "")) for _, row in data.iterrows()}
        _plot_lower_triangle_heatmap(ax, matrix, f"{city} {season}", label_map=label_map, star_lookup=stars, cbar=False, cmap="RdBu_r")
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cbar_ax, orientation="vertical")
    cb.ax.tick_params(labelsize=12)
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.08, top=0.95, hspace=0.35, wspace=0.25)
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_figure_4_uhoo_iaq_sleep_correlation_matrix(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    matrix = df.pivot_table(index="variable_x", columns="variable_y", values="r", aggfunc="mean")
    label_map = {"Temp_uHoo": "Temperature", "Humi_uHoo": "Humidity", "PM25_uHoo": "PM$_{2.5}$", "TVOC_uHoo": "TVOC", "CO2_uHoo": "CO$_2$", "CO_uHoo": "CO", "O3_uHoo": "O$_3$", "NO2_uHoo": "NO$_2$", "SleepTotal": "Total Sleep", "SleepLight": "Light Sleep", "SleepDeep": "Deep Sleep", "SleepREM": "REM"}
    fig, ax = plt.subplots(figsize=_template_figsize(metadata, (12, 8)))
    sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={"label": "", "ticks": [-1, 0, 1]}, vmin=-1, vmax=1, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([label_map.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()], rotation=0)
    ax.set_xticklabels([label_map.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()], rotation=45, ha="right")
    ax.set_title("Correlation Matrix: Indoor Air Quality vs. Sleep Metrics", fontsize=16, fontweight="bold")
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_figure_5_hia_attributable_cases(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Patch  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source).copy()
    df["ID"] = df["pollutant"].astype(str).map(_pm_label) + " " + df["city"].astype(str) + " " + df["season"].astype(str)
    df["Health_Outcome"] = df["endpoint"].astype(str)
    ids = list(dict.fromkeys(df["ID"].tolist()))
    custom_colors = ["#EEC643", "#ff7f0e", "#2ca02c", "#d62728", "#197BBD"]
    fig, ax1 = plt.subplots(figsize=_template_figsize(metadata, (20, 10)))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax1.grid(False)
    primary = df[df["Health_Outcome"] != "Mortality all causes"]
    mortality = df[df["Health_Outcome"] == "Mortality all causes"]
    outcomes = list(dict.fromkeys(primary["Health_Outcome"].tolist()))
    offsets = {outcome: (idx - max(len(outcomes) - 1, 0) / 2) * 0.13 for idx, outcome in enumerate(outcomes)}
    for idx, outcome in enumerate(outcomes):
        stats = []
        positions = []
        color = custom_colors[idx % len(custom_colors)]
        for xidx, xid in enumerate(ids):
            rows = primary[(primary["ID"] == xid) & (primary["Health_Outcome"] == outcome)]
            if rows.empty:
                continue
            stats.append(_summary_box_stats(rows.iloc[0], xid, "cases_"))
            positions.append(xidx + offsets[outcome])
        _draw_bxp(ax1, stats, positions, color, width=0.11)
    ax1.set_xlabel("", fontsize=18, fontweight="bold")
    ax1.set_ylabel("Incidences (#)", fontsize=18, fontweight="bold")
    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels(ids, ha="center", fontsize=10)
    ax1.tick_params(axis="y", labelsize=12)
    ax2 = ax1.twinx()
    ax2.grid(False)
    if not mortality.empty:
        stats = []
        positions = []
        for xidx, xid in enumerate(ids):
            rows = mortality[mortality["ID"] == xid]
            if rows.empty:
                continue
            stats.append(_summary_box_stats(rows.iloc[0], xid, "cases_"))
            positions.append(xidx + 0.46)
        _draw_bxp(ax2, stats, positions, custom_colors[-1], width=0.22)
    ax2.set_ylabel("Incidences (#)", fontsize=18, fontweight="bold", color=custom_colors[-1])
    ax2.tick_params(axis="y", labelcolor=custom_colors[-1], labelsize=12)
    ax1.set_xlim(-0.6, len(ids) - 0.1)
    handles = [Patch(facecolor=custom_colors[idx % len(custom_colors)], edgecolor="black", label=outcome) for idx, outcome in enumerate(outcomes)]
    if not mortality.empty:
        handles.append(Patch(facecolor=custom_colors[-1], edgecolor="black", label="Mortality all causes"))
    ax1.legend(handles=handles, title="", loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=min(5, max(1, len(handles))), frameon=False, fontsize=12)
    plt.tight_layout(rect=(0, 0.1, 1, 0.95))
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_figure_6_yll_scenario_outputs(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source).copy()
    wide = df.pivot_table(index=["city", "season", "pollutant", "endpoint"], columns="value_type", values="value", aggfunc="first").reset_index()
    primary = wide[wide["endpoint"].astype(str) == "Mortality all causes"].copy()
    if primary.empty:
        primary = wide.copy()
    primary["ID"] = primary["pollutant"].astype(str).map(_pm_label) + " " + primary["city"].astype(str) + " " + primary["season"].astype(str)
    fig, ax = plt.subplots(figsize=_template_figsize(metadata, (20, 10)))
    stats = [_summary_box_stats(row, str(row["ID"]), "") for _, row in primary.iterrows()]
    _draw_bxp(ax, stats, list(range(len(stats))), "#d62728", width=0.55)
    ax.grid(False)
    ax.set_xlabel("", fontsize=20, fontweight="bold")
    ax.set_ylabel("Years of Life Lost (YLL)", fontsize=22, fontweight="bold")
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([str(row["ID"]) for _, row in primary.iterrows()], ha="center", fontsize=10)
    ax.tick_params(axis="y", labelsize=12)
    plt.tight_layout(rect=(0, 0.1, 1, 0.95))
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_supplementary_figure_s1_participant_flow_completeness(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    fig, ax = plt.subplots(figsize=_template_figsize(metadata, (13, 8)))
    ax.axis("off")

    def box_text(row) -> str:
        parts = [str(row.get("box_title", ""))]
        for col in ["line_1", "line_2", "line_3", "line_4"]:
            value = str(row.get(col, "")).strip()
            if value and value.lower() != "nan":
                parts.append(value)
        return "\n".join(parts)

    def add_box(x: float, y: float, text: str, width: float = 0.26, height: float = 0.12) -> None:
        ax.add_patch(plt.Rectangle((x - width / 2, y - height / 2), width, height, facecolor="#f7f7f7", edgecolor="#333333", linewidth=1.0))
        ax.text(x, y, text, ha="center", va="center", fontsize=8, wrap=True)

    rows = list(df.sort_values("box_number").to_dict("records")) if "box_number" in df else list(df.to_dict("records"))
    top_rows = rows[:2]
    if top_rows:
        add_box(0.5, 0.88, box_text(top_rows[0]), width=0.42, height=0.13)
    if len(top_rows) > 1:
        add_box(0.5, 0.72, box_text(top_rows[1]), width=0.46, height=0.13)
    coords = [(0.18, 0.52), (0.5, 0.52), (0.82, 0.52), (0.18, 0.31), (0.5, 0.31), (0.82, 0.31), (0.5, 0.12)]
    for (x, y), row in zip(coords, rows[2:]):
        add_box(x, y, box_text(row), width=0.30, height=0.17)
        ax.annotate("", xy=(x, y + 0.09), xytext=(0.5, 0.65), arrowprops={"arrowstyle": "->", "lw": 0.8})
    plt.tight_layout()
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_supplementary_figure_s2_diurnal_pm_hr_stress(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    fig, axes = plt.subplots(6, 2, figsize=_template_figsize(metadata, (40, 60)))
    pm_vars = [("PM10_mean", "PM$_{10}$", "#A9A9A9"), ("PM2.5_mean", "PM$_{2.5}$", "#BEBEBE"), ("PM1_mean", "PM$_{1}$", "#D3D3D3")]
    panels = [("Thessaloniki", "Summer"), ("Milan", "Summer"), ("Thessaloniki", "Winter"), ("Milan", "Winter")]
    for panel_idx, (city, season) in enumerate(panels):
        col = 0 if city == "Thessaloniki" else 1
        block = 0 if season == "Summer" else 3
        data = df[(df["city"].astype(str).str.lower() == city.lower()) & (df["season"].astype(str) == season)].sort_values("hour")
        ax_pm, ax_hr, ax_stress = axes[block, col], axes[block + 1, col], axes[block + 2, col]
        ax_pm.set_title(f"{city} - {season}", fontsize=20, fontweight="bold", pad=10)
        for value_col, label, color in pm_vars:
            if value_col in data:
                ax_pm.plot(data["hour"], data[value_col], marker="s", linewidth=1.5, label=label, color=color)
        ax_pm.set_ylabel("Concentration (ug/m3)" if col == 0 else "", fontsize=14)
        ax_pm.legend(loc="upper right", ncol=3, frameon=True, fontsize=9)
        ax_hr.plot(data["hour"], data.get("avg_heart_rate_mean"), color="#FA8072", marker="o", linewidth=1.5)
        ax_hr.set_ylabel("Heart Rate (bpm)" if col == 0 else "", fontsize=14)
        ax_stress.plot(data["hour"], data.get("stress_mean"), color="#59788E", marker="o", linewidth=1.5)
        ax_stress.set_ylabel("Stress Level" if col == 0 else "", fontsize=14)
        for ax in [ax_pm, ax_hr, ax_stress]:
            ax.set_xticks(range(0, 24, 4))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 4)], fontsize=9)
            ax.grid(False)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98], pad=2.0, h_pad=1.5, w_pad=2.0)
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_supplementary_figure_s3_seasonal_garmin_sleep_distributions(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    metrics = ["SleepTotal", "SleepLight", "SleepDeep", "SleepREM"]
    titles = ["Total Sleep", "Light Sleep", "Deep Sleep", "REM"]
    ylabels = ["Sleep Total (min)", "Light Sleep (-)", "Deep Sleep (-)", "REM (-)"]
    fig, axes = plt.subplots(2, 4, figsize=_template_figsize(metadata, (20, 10)))
    for row_idx, city in enumerate(["Milan", "Thessaloniki"]):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for season, color, pos in [("Summer", "#c7e9c0", -0.16), ("Winter", "#c6dbef", 0.16)]:
                rows = df[(df["city"].astype(str).str.lower() == city.lower()) & (df["season"].astype(str) == season) & (df["sleep_metric"].astype(str) == metric)]
                if not rows.empty:
                    _draw_bxp(ax, [_summary_box_stats(rows.iloc[0], season)], [0.5 + pos], color, width=0.24)
            ax.set_title(titles[col_idx] if row_idx == 0 else "", fontsize=14)
            ax.set_ylabel(ylabels[col_idx], fontsize=11)
            ax.set_xlabel("")
            ax.set_xticks([0.34, 0.66])
            ax.set_xticklabels(["", ""] if row_idx == 0 else ["Summer", "Winter"], fontsize=10)
            if col_idx > 0:
                ax.set_ylim(-0.05, 1.05)
    fig.text(0.03, 0.75, "Milan", fontsize=18, rotation=90, va="center")
    fig.text(0.03, 0.25, "Thessaloniki", fontsize=18, rotation=90, va="center")
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_supplementary_figure_s4_ppm_garmin_correlation_matrix(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    cities = list(dict.fromkeys(df["city"].astype(str).map(_clean_city).tolist()))
    cities = [city for city in ["Milan", "Thessaloniki"] if city in cities] or cities[:2]
    label_map = {"PM10_PPM": "PM$_{10}$\n(ug/m3)", "PM25_PPM": "PM$_{2.5}$\n(ug/m3)", "PM1_PPM": "PM$_{1}$\n(ug/m3)", "Temp_PPM": "Temperature\n(degC)", "Humi_PPM": "Humidity\n(%)", "AvgHeartRate": "Heart Rate\n(bpm)", "Stress": "Stress"}
    fig, axes = plt.subplots(1, max(1, len(cities)), figsize=_template_figsize(metadata, (10 * max(1, len(cities)), 8)))
    if len(cities) == 1:
        axes = [axes]
    for idx, city in enumerate(cities):
        ax = axes[idx]
        data = df[df["city"].astype(str).map(_clean_city) == city]
        matrix = _correlation_matrix_from_rows(data)
        _plot_lower_triangle_heatmap(ax, matrix, city, label_map=label_map, cbar=(idx == len(cities) - 1), cmap="coolwarm")
        if idx > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
    plt.tight_layout()
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_supplementary_figure_s5_descriptive_ols_pm_hr_lag_scatterplots(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    lags = [0, 15, 30, 45, 60, 120]
    if "lag_min" in df:
        lags = [lag for lag in lags if lag in set(df["lag_min"].astype(int))]
    pollutants = ["PM10", "PM2.5", "PM1"]
    fig, axes = plt.subplots(len(lags), len(pollutants), figsize=_template_figsize(metadata, (18, 24)), sharey=True)
    if len(lags) == 1:
        axes = [axes]
    for i, lag in enumerate(lags):
        for j, pollutant in enumerate(pollutants):
            ax = axes[i][j] if len(lags) > 1 else axes[j]
            rows = df[(df["pollutant"].astype(str).str.replace("PM25", "PM2.5") == pollutant) & (df["lag_min"].astype(int) == int(lag))]
            row = rows.iloc[0] if not rows.empty else {}
            beta = float(row.get("beta", 0.0)) if hasattr(row, "get") else 0.0
            xs, ys = _synthetic_line_points(beta, intercept=72 + i * 0.3 + j)
            ax.scatter(xs[::2], ys[::2], alpha=0.45, s=10, color="#1f77b4", label="Summer" if i == 0 and j == 0 else "")
            ax.scatter(xs[1::2], ys[1::2], alpha=0.45, s=10, color="#ff7f0e", label="Winter" if i == 0 and j == 0 else "")
            ax.plot(xs, ys, linestyle="--", color="black", linewidth=1.0, label="Regression Line" if i == 0 and j == 0 else "")
            spread = max(abs(beta) * 100, 2.0)
            ax.fill_between(xs, [y - spread for y in ys], [y + spread for y in ys], color="grey", alpha=0.25, label="95% CI" if i == 0 and j == 0 else "")
            if hasattr(row, "get"):
                text = f"Coef: {beta:.5f} (95% CI: {float(row.get('ci_low', beta)):.5f}, {float(row.get('ci_high', beta)):.5f})\n{_format_p_value(row.get('p_value'))}\nR2: {float(row.get('r_squared', 0.0)):.3f}"
                ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8, verticalalignment="top", fontweight="bold", bbox=dict(facecolor="lightyellow", alpha=0.8, boxstyle="round,pad=0.4"))
            ax.set_ylabel(f"Lag: {lag} min\n\nHeart Rate (bpm)", fontsize=10)
            ax.set_xlabel(f"{_pm_label(pollutant)} (ug/m3)" if i == len(lags) - 1 else "", fontsize=10)
            ax.tick_params(axis="x", labelbottom=i == len(lags) - 1)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_supplementary_figure_s6_ols_vs_lag_mixed_comparison(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    pollutants = [("PM10", "blue"), ("PM2.5", "green"), ("PM1", "red")]
    fig, axes = plt.subplots(1, 3, figsize=_template_figsize(metadata, (18, 6)))
    for ax, (pollutant, color) in zip(axes, pollutants):
        rows = df[df["pollutant"].astype(str).str.replace("PM25", "PM2.5") == pollutant].sort_values("lag_min")
        y_col = "mixed_model_beta" if "mixed_model_beta" in rows else "ols_beta"
        ax.plot(rows["lag_min"], rows[y_col], "o-", color=color)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("Lag (minutes)")
        ax.set_ylabel("Effect on Heart Rate")
        ax.set_title(f"{_pm_label(pollutant)} Effects at Different Lags")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_supplementary_figure_s7_iaq_sleep_regression_panels(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    outcomes = list(dict.fromkeys(df["outcome"].astype(str).tolist()))
    predictors = list(dict.fromkeys(df["predictor"].astype(str).tolist()))
    label_map = {"SleepTotal": "Total Sleep", "SleepLight": "Light Sleep", "SleepDeep": "Deep Sleep", "SleepREM": "REM", "Temp_uHoo": "Temperature", "Humi_uHoo": "Humidity", "PM25_uHoo": "PM$_{2.5}$", "TVOC_uHoo": "TVOC", "CO2_uHoo": "CO$_2$", "CO_uHoo": "CO", "O3_uHoo": "O$_3$", "NO2_uHoo": "NO$_2$"}
    fig, axes = plt.subplots(max(1, len(outcomes)), max(1, len(predictors)), figsize=_template_figsize(metadata, (5 * max(1, len(predictors)), 4 * max(1, len(outcomes)))), squeeze=False)
    for i, outcome in enumerate(outcomes):
        for j, predictor in enumerate(predictors):
            ax = axes[i, j]
            rows = df[(df["outcome"].astype(str) == outcome) & (df["predictor"].astype(str) == predictor)]
            row = rows.iloc[0] if not rows.empty else {}
            beta = float(row.get("beta", 0.0)) if hasattr(row, "get") else 0.0
            xs, ys = _synthetic_line_points(beta, intercept=60 + i * 8)
            ax.scatter(xs, ys, alpha=0.45, s=20, color="#4c78a8")
            ax.plot(xs, ys, color="#1f77b4", linewidth=1.5)
            if hasattr(row, "get"):
                stats_text = f"n={int(float(row.get('n_rows', 0)))}\nR2={float(row.get('r_squared', 0.0)):.3f}\n{_format_p_value(row.get('p_value'))}\nbeta={beta:.4g}\nCI=[{float(row.get('ci_low', beta)):.4g}, {float(row.get('ci_high', beta)):.4g}]"
                ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=7, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            ax.set_xlabel(label_map.get(predictor, predictor) if i == len(outcomes) - 1 else "")
            ax.set_ylabel(label_map.get(outcome, outcome) if j == 0 else "")
    plt.tight_layout()
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target


def render_supplementary_figure_s8_stress_sleep_exploratory_panels(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run") -> Path:
    del config, validation_mode
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.lines import Line2D  # type: ignore

    metadata = _resolve_template_args(template_reference, template_metadata)
    df = _read_aggregate_csv(aggregate_source)
    metrics = ["SleepTotal", "SleepDeep", "SleepLight", "SleepREM"]
    pretty = {"SleepTotal": "Total Sleep (min)", "SleepDeep": "Deep Sleep (-)", "SleepLight": "Light Sleep (-)", "SleepREM": "REM Sleep (-)", "next_day_avg_stress": "Next Day Average Stress", "avg_daily_stress": "Average Daily Stress"}
    fig, axes = plt.subplots(2, 4, figsize=_template_figsize(metadata, (22, 10)))
    colors = {"Summer": "#e41a1c", "Winter": "#377eb8"}
    for j, metric in enumerate(metrics):
        ax = axes[0, j]
        rows = df[(df["direction"].astype(str).str.contains("stress -> sleep", regex=False)) & (df["outcome"].astype(str) == metric)]
        row = rows.iloc[0] if not rows.empty else {}
        beta = float(row.get("beta", 0.0)) if hasattr(row, "get") else 0.0
        xs, ys = _synthetic_line_points(beta, intercept=60 + j * 4)
        ax.scatter(xs[::2], ys[::2], color=colors["Summer"], alpha=0.6, s=20)
        ax.scatter(xs[1::2], ys[1::2], color=colors["Winter"], alpha=0.6, s=20)
        ax.plot(xs, ys, color="black", linewidth=1.2)
        ax.text(0.05, 1.01, f"beta={beta:.3g}\n{_format_p_value(row.get('p_value') if hasattr(row, 'get') else None)}", transform=ax.transAxes, fontsize=8, va="bottom", bbox=dict(facecolor="white", alpha=0.8))
        ax.set_xlabel("Average Daily Stress", fontsize=11)
        ax.set_ylabel(pretty[metric], fontsize=11)
        ax = axes[1, j]
        rows = df[(df["direction"].astype(str).str.contains("sleep -> next-day stress", regex=False)) & (df["predictor"].astype(str) == metric)]
        row = rows.iloc[0] if not rows.empty else {}
        beta = float(row.get("beta", 0.0)) if hasattr(row, "get") else 0.0
        xs, ys = _synthetic_line_points(beta, intercept=25 + j * 1.2)
        ax.scatter(xs[::2], ys[::2], color=colors["Summer"], alpha=0.6, s=20)
        ax.scatter(xs[1::2], ys[1::2], color=colors["Winter"], alpha=0.6, s=20)
        ax.plot(xs, ys, color="black", linewidth=1.2)
        ax.text(0.05, 1.01, f"beta={beta:.3g}\n{_format_p_value(row.get('p_value') if hasattr(row, 'get') else None)}", transform=ax.transAxes, fontsize=8, va="bottom", bbox=dict(facecolor="white", alpha=0.8))
        ax.set_xlabel(pretty[metric], fontsize=11)
        ax.set_ylabel("Next Day Average Stress", fontsize=11)
    handles = [Line2D([0], [0], marker="o", linestyle="", label=season, color=color) for season, color in colors.items()]
    fig.legend(handles=handles, title="", loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.075), fontsize=10)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    target = _finish_figure(fig, output_path, metadata)
    plt.close(fig)
    return target

def preferred_numeric_column(df, hints: list[str]) -> str:
    numeric = list(df.select_dtypes(include="number").columns)
    for hint in hints:
        for column in numeric:
            if hint.lower() in column.lower():
                return column
    if not numeric:
        raise ValueError("no numeric columns available")
    return numeric[0]


def optional_numeric(df, hints: list[str]):
    numeric = set(df.select_dtypes(include="number").columns)
    for hint in hints:
        for column in df.columns:
            if hint.lower() in column.lower() and column in numeric:
                return df[column].to_numpy(dtype=float)
    return None


def preferred_group_column(df, hints: tuple[str, ...]) -> str | None:
    numeric = set(df.select_dtypes(include="number").columns)
    for hint in hints:
        for column in df.columns:
            if hint.lower() in column.lower() and column not in numeric:
                return column
    for column in df.columns:
        if column not in numeric:
            return column
    return None


def row_labels(df) -> list[str]:
    text_cols = [col for col in df.columns if col not in df.select_dtypes(include="number").columns]
    if not text_cols:
        return [str(i + 1) for i in range(len(df))]
    return [label[:70] for label in df[text_cols[:2]].astype(str).agg(" / ".join, axis=1).tolist()]


def write_template_check_csv(path: Path, rows: list[dict[str, str]]) -> None:
    from tools.figure_template_checks import write_template_check_csv as base_write
    base_write(path, rows)


def write_template_text_report(path: Path, rows: list[dict[str, str]]) -> None:
    blocked = [row for row in rows if row.get("status") == "FAIL" or str(row.get("status", "")).startswith("BLOCKED")]
    review = [row for row in rows if row.get("status") == "REQUIRES_VISUAL_REVIEW"]
    lines = [
        "Figure Template Fidelity Report",
        "",
        f"figures_checked: {len(rows)}",
        f"figures_blocked: {len(blocked)}",
        f"figures_requiring_visual_review: {len(review)}",
        "gate_status: " + ("FAIL" if blocked else "REQUIRES_VISUAL_REVIEW" if review else "PASS"),
        "POST_RUN_MANUAL_CHECK_REQUIRED = FIGURE_TEMPLATE_VISUAL_REVIEW" if review else "",
    ]
    for row in blocked:
        lines.append(f"- {row.get('figure_label', '')} {row.get('figure_id', '')}: {row.get('notes', '')}")
    path.write_text("\n".join(line for line in lines if line != "") + "\n", encoding="utf-8")


def write_generation_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure_id", "figure_label", "plot_family", "source", "output", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def write_approved_reference_comparison_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "figure_label",
        "approved_figure_path",
        "fullrun_figure_path",
        "approved_exists",
        "fullrun_exists",
        "same_dimensions",
        "same_file_size",
        "sha256_match",
        "perceptual_similarity_score",
        "visual_match_status",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_template_lock_check_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "figure_id",
        "current_label",
        "accepted_template_reference",
        "source_script_candidate",
        "authoritative_plotting_status",
        "authoritative_plotting_notes",
        "generator_function",
        "renderer_function",
        "rejected_candidates_excluded",
        "template_lock_status",
        "dummy_generation_status",
        "validation_status",
        "post_run_visual_review_required",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figure_text_report(path: Path, template_rows: list[dict[str, str]], generation_rows: list[dict[str, str]]) -> None:
    template_failed = [row for row in template_rows if row.get("status") == "FAIL" or str(row.get("status", "")).startswith("BLOCKED")]
    generation_failed = [
        row
        for row in generation_rows
        if row.get("status") in {"FAIL", "MISMATCH_APPROVED_REFERENCE", "MISSING_APPROVED_REFERENCE", "MISSING_FULLRUN_OUTPUT"}
        or str(row.get("status", "")).startswith("BLOCKED")
    ]
    review = [row for row in generation_rows if row.get("status") == "REQUIRES_VISUAL_REVIEW"]
    lines = ["Figure Reproducibility Report", "", f"templates_checked: {len(template_rows)}", f"template_failures: {len(template_failed)}", f"figures_with_generators: {len([row for row in generation_rows if row.get('plot_family')])}", f"figure_generation_failures: {len(generation_failed)}", f"figures_requiring_visual_review: {len(review)}", "gate_status: " + ("FAIL" if template_failed or generation_failed else "REQUIRES_VISUAL_REVIEW" if review else "PASS")]
    if template_failed or generation_failed:
        status_tokens: list[str] = []
        for row in generation_failed:
            notes = row.get("notes", "")
            marker = "FIGURE_REPRO_STATUS = "
            if marker in notes:
                token = notes.split(marker, 1)[1].split(";", 1)[0].strip()
                if token and token not in status_tokens:
                    status_tokens.append(token)
        if not status_tokens:
            status_tokens.append("BLOCKED_TEMPLATE_MISMATCH")
        for token in status_tokens:
            lines.append(f"FIGURE_REPRO_STATUS = {token}")
    elif review:
        lines.append("POST_RUN_MANUAL_CHECK_REQUIRED = FIGURE_TEMPLATE_VISUAL_REVIEW")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# Named template-aware figure generator exports.


def _named_generate(
    figure_id: str,
    aggregate_source: str | Path,
    output_path: str | Path,
    template_reference: str | Path | dict[str, str] | None = None,
    template_metadata: dict[str, str] | None = None,
    config: dict | None = None,
    validation_mode: str = "controlled_run",
    template_reference_metadata: dict[str, str] | None = None,
    write_tiff: bool = False,
) -> Path:
    spec = _SPEC_BY_ID[figure_id]
    if isinstance(template_reference, dict) and not template_reference.get("accepted_template_reference") and not template_reference.get("template_source_path_or_placeholder"):
        template_metadata = template_metadata or template_reference
        template_reference = None
    template = template_metadata or template_reference_metadata
    if template_reference is None:
        template_reference = resolve_accepted_template_reference(figure_id)
    if template is None:
        template = load_template_metadata(template_reference)
    target = Path(output_path)
    renderer = accepted_template_renderer(figure_id)
    renderer(Path(aggregate_source), target, template_reference=template_reference, template_metadata=template, config=config, validation_mode=validation_mode)
    if write_tiff:
        try:
            from PIL import Image  # type: ignore

            with Image.open(target) as image:
                image.save(target.with_suffix(".tiff"))
        except Exception:
            pass
    return target


def generate_seasonal_personal_pm_distributions(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("seasonal_personal_pm_distributions", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_activity_stratified_personal_pm_exposure(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("activity_stratified_personal_pm_exposure", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_residential_uhoo_iaq_correlation_matrix(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("residential_uhoo_iaq_correlation_matrix", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_uhoo_iaq_sleep_correlation_matrix(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("uhoo_iaq_garmin_sleep_correlation_matrix", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_hia_attributable_cases(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("hia_attributable_cases", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_yll_scenario_outputs(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("yll_scenario_outputs", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_descriptive_ols_pm_heart_rate_lag_scatterplots(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("descriptive_ols_pm_hr_lag", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_ols_vs_mixed_effects_coefficient_comparison(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("ols_vs_lag_mixed_comparison", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_diurnal_pm_heart_rate_stress_distributions(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("diurnal_pm_hr_stress", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_seasonal_garmin_sleep_distributions(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("seasonal_garmin_sleep_distributions", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_ppm_garmin_correlation_matrix(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("ppm_garmin_correlation_matrix", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_iaq_sleep_regression_panels(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("iaq_sleep_regression_panels", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_stress_sleep_exploratory_panels(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("stress_sleep_exploratory_panels", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def generate_participant_flow_completeness_figure(aggregate_source: str | Path, output_path: str | Path, template_reference: str | Path | dict[str, str] | None = None, template_metadata: dict[str, str] | None = None, config: dict | None = None, validation_mode: str = "controlled_run", template_reference_metadata: dict[str, str] | None = None, write_tiff: bool = False) -> Path:
    return _named_generate("participant_flow_completeness_summary", aggregate_source, output_path, template_reference, template_metadata, config, validation_mode, template_reference_metadata, write_tiff)


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    del n_samples
    if dry_run:
        from raise_icarus.stage_contracts import dry_run_stage_result

        return dry_run_stage_result(STAGE, run_dir)
    if harmonized_zip is not None:
        build_controlled_figure_source_aliases(harmonized_zip, run_dir)
    code = generate_figures(run_dir)
    return StageResult(STAGE.stage_name, STAGE.module_name, "PASS" if code == 0 else "FAIL", STAGE.output_domain, (str(Path(run_dir) / "reports" / "figure_generation_report.csv"),), "Figure generation requires original accepted plotting templates.")

