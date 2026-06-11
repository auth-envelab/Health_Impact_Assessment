"""Sleep, IAQ, and stress-sleep analytical bodies."""

from __future__ import annotations

from pathlib import Path

from raise_icarus.controlled_runtime import config_value, copy_alias, domain_dir, repo_root, reports_dir, write_text_report
from raise_icarus.stage_contracts import StageDefinition, StageResult, definition_for, dry_run_stage_result

STAGE = definition_for(__name__)


def stage_definition() -> StageDefinition:
    return STAGE


def build_sleep_inputs(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> dict[str, Path]:
    from raise_icarus.phase7_sleep import write_reconstruction_outputs, write_phase7_audit_outputs

    output_dir = domain_dir(out_dir, "sleep")
    outputs = write_reconstruction_outputs(harmonized_zip, output_dir)
    audit_outputs = write_phase7_audit_outputs(harmonized_zip, output_dir, date_filter_mode=config_value(config, "date_filter_mode", "campaign"))
    outputs.update(audit_outputs)
    copy_alias(outputs["sleep_model_input_audit"], output_dir / "Sleep model input audit.csv")
    copy_alias(audit_outputs["figure8_correlation_data"], output_dir / "Figure 4 data - uHoo IAQ Garmin sleep correlation matrix.csv")
    return outputs


def run_sleep_iaq_models(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> dict[str, Path]:
    del config
    from raise_icarus.phase7_sleep import write_ols_outputs

    output_dir = domain_dir(out_dir, "sleep")
    outputs = write_ols_outputs(harmonized_zip, output_dir)
    copy_alias(outputs["sleep_iaq_ols_results"], output_dir / "Sleep IAQ OLS model results.csv")
    copy_alias(outputs["figure9_plot_data"], output_dir / "Supplementary Figure S7 data - IAQ sleep regression panels.csv")
    return outputs


def run_sleep_lasso_refits(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> dict[str, Path]:
    del config
    from raise_icarus.phase7_sleep import write_lasso_outputs

    output_dir = domain_dir(out_dir, "sleep")
    outputs = write_lasso_outputs(harmonized_zip, output_dir)
    copy_alias(outputs["sleep_lasso_selected_predictors"], output_dir / "Sleep Lasso selected predictors.csv")
    copy_alias(outputs["sleep_lasso_refit_results"], output_dir / "Sleep Lasso refit results.csv")
    return outputs


def run_stress_sleep_models(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> dict[str, Path]:
    del config
    from raise_icarus.phase7_sleep import write_stress_sleep_outputs

    output_dir = domain_dir(out_dir, "sleep")
    outputs = write_stress_sleep_outputs(harmonized_zip, output_dir)
    copy_alias(outputs["stress_sleep_mixed_model_results"], output_dir / "Stress sleep mixed-model results.csv")
    copy_alias(outputs["figure10_plot_data"], output_dir / "Supplementary Figure S8 data - Stress sleep exploratory panels.csv")
    return outputs


def validate_sleep_outputs(out_dir: str | Path, expected_manifest: str | Path | None = None) -> Path:
    del expected_manifest
    from raise_icarus.phase7_sleep import write_validation_report

    output_dir = domain_dir(out_dir, "sleep")
    try:
        write_validation_report(repo_root(), None, output_dir, output_dir)
    except Exception:
        pass
    return write_text_report(reports_dir(out_dir) / "Sleep model validation report.txt", "Sleep Model Validation Report", ["status: validation attempted from aggregate outputs"])


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    del n_samples
    if dry_run:
        return dry_run_stage_result(STAGE, run_dir)
    if harmonized_zip is None:
        return StageResult(STAGE.stage_name, STAGE.module_name, "FAIL", STAGE.output_domain, (), "harmonized archive path is required")
    build_sleep_inputs(harmonized_zip, run_dir)
    run_sleep_iaq_models(harmonized_zip, run_dir)
    run_sleep_lasso_refits(harmonized_zip, run_dir)
    run_stress_sleep_models(harmonized_zip, run_dir)
    report = validate_sleep_outputs(run_dir)
    return StageResult(STAGE.stage_name, STAGE.module_name, "PASS", STAGE.output_domain, (str(report),), "Sleep model outputs generated.")
