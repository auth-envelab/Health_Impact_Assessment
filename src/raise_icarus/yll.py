"""YLL scenario analytical bodies."""

from __future__ import annotations

from pathlib import Path

from raise_icarus.controlled_runtime import config_value, copy_alias, domain_dir, reports_dir, write_text_report
from raise_icarus.stage_contracts import StageDefinition, StageResult, definition_for, dry_run_stage_result

STAGE = definition_for(__name__)


def stage_definition() -> StageDefinition:
    return STAGE


def run_yll_scenarios(harmonized_zip: str | Path, out_dir: str | Path, n_samples: int = 10000, config: object = None) -> dict[str, Path]:
    from raise_icarus.phase5_yll import write_yll_outputs

    output_dir = domain_dir(out_dir, "yll")
    phase2_dir = domain_dir(out_dir, "exposure")
    phase3_dir = domain_dir(out_dir, "hia")
    outputs = write_yll_outputs(
        harmonized_zip,
        phase2_dir=phase2_dir,
        phase3_dir=phase3_dir,
        outdir=output_dir,
        n_samples=n_samples,
        seed=config_value(config, "seed", 20260430),
        date_filter_mode=config_value(config, "date_filter_mode", "campaign"),
    )
    copy_alias(outputs["yll_primary_results"], output_dir / "YLL primary results.csv")
    copy_alias(outputs["figure12_yll_plot_data"], output_dir / "Figure 6 data - YLL scenario outputs.csv")
    copy_alias(outputs["figure12_yll_plot_data"], output_dir / "Figure 12 data - YLL scenario outputs.csv")
    copy_alias(outputs["yll_pm10_pm25_decomposition"], output_dir / "YLL PM10 PM25 decomposition.csv")
    copy_alias(outputs["yll_conversion_ratio_sensitivity"], output_dir / "YLL conversion-ratio sensitivity.csv")
    return outputs


def build_yll_decomposition(out_dir: str | Path, config: object = None) -> Path:
    del config
    from raise_icarus.phase5_yll import write_decomposition_from_results

    output_dir = domain_dir(out_dir, "yll")
    path = write_decomposition_from_results(output_dir / "yll_primary_results.csv", output_dir)
    return copy_alias(path, output_dir / "YLL PM10 PM25 decomposition.csv")


def build_figure12_yll_plot_data(out_dir: str | Path, config: object = None) -> Path:
    del config
    from raise_icarus.phase5_yll import write_figure12_plot_data_from_results

    output_dir = domain_dir(out_dir, "yll")
    path = write_figure12_plot_data_from_results(output_dir / "yll_primary_results.csv", output_dir)
    copy_alias(path, output_dir / "Figure 6 data - YLL scenario outputs.csv")
    return copy_alias(path, output_dir / "Figure 12 data - YLL scenario outputs.csv")


def validate_yll_outputs(out_dir: str | Path, expected_manifest: str | Path | None = None) -> Path:
    del expected_manifest
    output_dir = domain_dir(out_dir, "yll")
    required = [output_dir / "YLL primary results.csv", output_dir / "Figure 6 data - YLL scenario outputs.csv", output_dir / "Figure 12 data - YLL scenario outputs.csv"]
    lines = [f"{path.name}: {'PASS' if path.exists() else 'FAIL'}" for path in required]
    return write_text_report(reports_dir(out_dir) / "YLL validation report.txt", "YLL Validation Report", lines)


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    if dry_run:
        return dry_run_stage_result(STAGE, run_dir)
    if harmonized_zip is None:
        return StageResult(STAGE.stage_name, STAGE.module_name, "FAIL", STAGE.output_domain, (), "harmonized archive path is required")
    run_yll_scenarios(harmonized_zip, run_dir, n_samples=n_samples)
    report = validate_yll_outputs(run_dir)
    return StageResult(STAGE.stage_name, STAGE.module_name, "PASS", STAGE.output_domain, (str(report),), "YLL outputs generated.")
