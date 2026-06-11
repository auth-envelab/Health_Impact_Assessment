"""Personal PPM common-support and exposure-source analytical bodies."""

from __future__ import annotations

from pathlib import Path

from raise_icarus.controlled_runtime import config_value, copy_alias, domain_dir, repo_root, reports_dir, write_text_report
from raise_icarus.stage_contracts import StageDefinition, StageResult, definition_for, dry_run_stage_result

STAGE = definition_for(__name__)


def stage_definition() -> StageDefinition:
    return STAGE


def build_personal_pm_common_support(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> dict[str, Path]:
    from raise_icarus.phase2_ppm_common_support import write_phase2_build_outputs

    output_dir = domain_dir(out_dir, "exposure")
    outputs = write_phase2_build_outputs(harmonized_zip, output_dir, date_filter_mode=config_value(config, "date_filter_mode", "campaign"))
    copy_alias(outputs["ppm_common_support_daily_input_audit"], output_dir / "Personal PPM common-support daily input audit.csv")
    copy_alias(outputs["ppm_hierarchy_validation"], output_dir / "PPM hierarchy validation.csv")
    copy_alias(outputs["hia_daily_pm_input_validation"], output_dir / "HIA daily PM input validation.csv")
    return outputs


def validate_personal_pm_hierarchy(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> Path:
    del harmonized_zip, config
    from raise_icarus.phase2_ppm_common_support import write_hierarchy_validation_from_input

    output_dir = domain_dir(out_dir, "exposure")
    path = write_hierarchy_validation_from_input(output_dir, output_dir)
    return copy_alias(path, output_dir / "PPM hierarchy validation.csv")


def validate_hia_exposure_source(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> Path:
    del config
    from raise_icarus.phase2_ppm_common_support import write_hia_exposure_source_audit

    output_dir = domain_dir(out_dir, "exposure")
    path = write_hia_exposure_source_audit(output_dir, harmonized_zip, output_dir, repo_root=repo_root())
    return copy_alias(path, output_dir / "HIA exposure source audit.csv")


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    del n_samples
    if dry_run:
        return dry_run_stage_result(STAGE, run_dir)
    if harmonized_zip is None:
        return StageResult(STAGE.stage_name, STAGE.module_name, "FAIL", STAGE.output_domain, (), "harmonized archive path is required")
    build_personal_pm_common_support(harmonized_zip, run_dir)
    validate_personal_pm_hierarchy(harmonized_zip, run_dir)
    validate_hia_exposure_source(harmonized_zip, run_dir)
    report = write_text_report(reports_dir(run_dir) / "Personal PM support report.txt", "Personal PM Support Report", ["status: PASS"])
    return StageResult(STAGE.stage_name, STAGE.module_name, "PASS", STAGE.output_domain, (str(report),), "Personal PPM support outputs generated.")
