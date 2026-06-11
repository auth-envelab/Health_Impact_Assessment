"""Lag-specific heart-rate and stress analytical bodies."""

from __future__ import annotations

from pathlib import Path

from raise_icarus.controlled_runtime import config_value, copy_alias, csv_to_xlsx, domain_dir, reports_dir, tables_dir, write_text_report
from raise_icarus.stage_contracts import StageDefinition, StageResult, definition_for, dry_run_stage_result

STAGE = definition_for(__name__)


def stage_definition() -> StageDefinition:
    return STAGE


def run_lag_specific_heart_rate_stress_models(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> dict[str, Path]:
    from raise_icarus.phase6_lag_models import run_phase6_models

    output_dir = domain_dir(out_dir, "lag_models")
    outputs = run_phase6_models(
        harmonized_zip,
        outdir=output_dir,
        phase1_dir=domain_dir(out_dir, "denominators"),
        date_filter_mode=config_value(config, "date_filter_mode", "campaign"),
        max_model_rows=config_value(config, "max_model_rows", 10000),
    )
    copy_alias(outputs["supplementary_table_s3_reproduced"], output_dir / "Supplementary Table S3 data - Lag-specific HR stress models.csv")
    return outputs


def build_supplementary_table_s3(out_dir: str | Path, config: object = None) -> Path:
    del config
    from raise_icarus.phase6_lag_models import export_supplementary_table_from_results

    output_dir = domain_dir(out_dir, "lag_models")
    csv_path = export_supplementary_table_from_results(output_dir / "lag_model_results_unadjusted.csv", output_dir / "lag_model_results_adjusted.csv", output_dir)
    descriptive_csv = copy_alias(csv_path, output_dir / "Supplementary Table S3 data - Lag-specific HR stress models.csv")
    return csv_to_xlsx(descriptive_csv, tables_dir(out_dir, "supplementary") / "Supplementary Table S3 - Lag-specific heart-rate and stress models.xlsx")


def validate_lag_model_outputs(out_dir: str | Path, expected_manifest: str | Path | None = None) -> Path:
    del expected_manifest
    from raise_icarus.phase6_lag_models import validate_lag_outputs

    output_dir = domain_dir(out_dir, "lag_models")
    validate_lag_outputs(output_dir / "Supplementary Table S3 data - Lag-specific HR stress models.csv", output_dir / "lag_model_convergence_audit.csv", output_dir)
    return write_text_report(reports_dir(out_dir) / "Lag-model validation report.txt", "Lag-Model Validation Report", ["status: PASS if aggregate checks passed"])


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    del n_samples
    if dry_run:
        return dry_run_stage_result(STAGE, run_dir)
    if harmonized_zip is None:
        return StageResult(STAGE.stage_name, STAGE.module_name, "FAIL", STAGE.output_domain, (), "harmonized archive path is required")
    run_lag_specific_heart_rate_stress_models(harmonized_zip, run_dir)
    table = build_supplementary_table_s3(run_dir)
    report = validate_lag_model_outputs(run_dir)
    return StageResult(STAGE.stage_name, STAGE.module_name, "PASS", STAGE.output_domain, (str(table), str(report)), "Lag model outputs generated.")
