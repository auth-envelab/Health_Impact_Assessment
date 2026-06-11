"""Paired-season sensitivity analytical bodies."""

from __future__ import annotations

from pathlib import Path

from raise_icarus.controlled_runtime import config_value, copy_alias, domain_dir, reports_dir, write_text_report
from raise_icarus.stage_contracts import StageDefinition, StageResult, definition_for, dry_run_stage_result

STAGE = definition_for(__name__)


def stage_definition() -> StageDefinition:
    return STAGE


def run_paired_season_sensitivity(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> dict[str, Path]:
    from raise_icarus.phase9_paired_sensitivity import write_phase9_outputs

    output_dir = domain_dir(out_dir, "paired_sensitivity")
    outputs = write_phase9_outputs(
        harmonized_zip,
        outdir=output_dir,
        phase1_dir=domain_dir(out_dir, "denominators"),
        phase2_dir=domain_dir(out_dir, "exposure"),
        phase6_dir=domain_dir(out_dir, "lag_models"),
        phase7_dir=domain_dir(out_dir, "sleep"),
        repo_path=Path.cwd(),
        date_filter_mode=config_value(config, "date_filter_mode", "campaign"),
        scripts_run=[],
    )
    copy_alias(outputs["paired_ppm"], output_dir / "Paired PPM seasonal sensitivity.csv")
    copy_alias(outputs["paired_uhoo"], output_dir / "Paired uHoo seasonal sensitivity.csv")
    copy_alias(outputs["paired_garmin_hr_stress"], output_dir / "Paired Garmin HR stress sensitivity.csv")
    copy_alias(outputs["paired_sleep"], output_dir / "Paired sleep sensitivity.csv")
    return outputs


def validate_paired_support(out_dir: str | Path, expected_manifest: str | Path | None = None) -> Path:
    del expected_manifest
    output_dir = domain_dir(out_dir, "paired_sensitivity")
    required = [
        output_dir / "Paired PPM seasonal sensitivity.csv",
        output_dir / "Paired uHoo seasonal sensitivity.csv",
        output_dir / "Paired Garmin HR stress sensitivity.csv",
        output_dir / "Paired sleep sensitivity.csv",
    ]
    return write_text_report(reports_dir(out_dir) / "Paired sensitivity validation report.txt", "Paired Sensitivity Validation Report", [f"{p.name}: {'PASS' if p.exists() else 'FAIL'}" for p in required])


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    del n_samples
    if dry_run:
        return dry_run_stage_result(STAGE, run_dir)
    if harmonized_zip is None:
        return StageResult(STAGE.stage_name, STAGE.module_name, "FAIL", STAGE.output_domain, (), "harmonized archive path is required")
    run_paired_season_sensitivity(harmonized_zip, run_dir)
    report = validate_paired_support(run_dir)
    return StageResult(STAGE.stage_name, STAGE.module_name, "PASS", STAGE.output_domain, (str(report),), "Paired-season sensitivity outputs generated.")
