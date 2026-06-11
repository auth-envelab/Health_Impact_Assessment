"""Shared stage contracts for the manuscript reproducibility workflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class StageDefinition:
    module_name: str
    stage_name: str
    display_name: str
    output_domain: str
    required_inputs: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class StageResult:
    stage_name: str
    module_name: str
    status: str
    output_domain: str
    outputs: tuple[str, ...]
    notes: str

    def to_row(self) -> dict[str, str]:
        return {
            "stage": self.stage_name,
            "module": self.module_name,
            "status": self.status,
            "output_domain": self.output_domain,
            "outputs": "; ".join(self.outputs),
            "notes": self.notes,
        }


DEFAULT_STAGE_DEFINITIONS: tuple[StageDefinition, ...] = (
    StageDefinition("raise_icarus.denominators", "analysis_denominators", "Analysis-specific denominators", "denominators", ("controlled harmonized monitoring archive",), ("results/denominators/Supplementary Table S1 - Analysis-specific denominators.csv", "results/denominators/STROBE checklist.csv", "results/validation/Table 5 data - Demographic source status.csv"), "Derives analysis support counts and demographic-source status."),
    StageDefinition("raise_icarus.personal_pm_support", "personal_pm_common_support", "Personal PPM support and exposure summaries", "exposure", ("controlled harmonized monitoring archive",), ("results/exposure/Figure 1 data - Seasonal personal PM distributions.csv", "results/exposure/Figure 2 data - Activity-stratified personal PM exposure.csv", "results/exposure/Table 6 data - ANOVA personal PM by city and season.csv"), "Builds PPM-only exposure summaries."),
    StageDefinition("raise_icarus.hia", "hia_primary_scenarios", "Scenario-based health impact assessment", "hia", ("controlled harmonized monitoring archive", "HIA parameter tables"), ("results/hia/Table 1 data - Concentration-response functions.csv", "results/hia/Table 2 data - HIA equations summary.csv", "results/hia/Table 3 data - Baseline disease rates.csv", "results/hia/Figure 5 data - HIA attributable cases.csv", "results/hia/hia_primary_scenario_summary.csv", "results/hia/hia_upper_tail_cap_audit.csv"), "Computes attributable-case scenario summaries and sensitivity support."),
    StageDefinition("raise_icarus.yll", "yll_scenario_outputs", "Years of life lost scenario outputs", "yll", ("controlled harmonized monitoring archive", "life-table inputs"), ("results/yll/Table 4 data - YLL life-table inputs.csv", "results/yll/Figure 6 data - YLL scenario outputs.csv", "results/yll/yll_primary_results.csv"), "Computes YLL scenario summaries without significance markers."),
    StageDefinition("raise_icarus.lag_models", "lag_specific_heart_rate_stress_models", "Lag-specific heart-rate and stress models", "lag_models", ("controlled harmonized monitoring archive",), ("results/lag_models/Supplementary Table S3 data - Lag-specific HR stress models.csv", "results/lag_models/Supplementary Figure S5 data - Descriptive OLS PM heart-rate lag scatterplots.csv", "results/lag_models/Supplementary Figure S6 data - OLS vs lag-specific mixed-effects comparison.csv"), "Builds lag-specific PM, heart-rate, and stress model summaries."),
    StageDefinition("raise_icarus.sleep_models", "sleep_iaq_and_stress_sleep_models", "Sleep, IAQ, and stress-sleep models", "sleep", ("controlled harmonized monitoring archive",), ("results/sleep/Figure 4 data - uHoo IAQ Garmin sleep correlation matrix.csv", "results/sleep/Supplementary Figure S3 data - Seasonal Garmin sleep distributions.csv", "results/sleep/Supplementary Figure S7 data - IAQ sleep regression panels.csv", "results/sleep/Supplementary Figure S8 data - Stress sleep exploratory panels.csv"), "Builds Garmin sleep and uHoo IAQ model summaries."),
    StageDefinition("raise_icarus.paired_sensitivity", "paired_season_sensitivity", "Paired-season sensitivity summaries", "paired_sensitivity", ("controlled harmonized monitoring archive",), ("results/paired_sensitivity/paired_season_personal_pm_sensitivity.csv", "results/paired_sensitivity/paired_season_sleep_sensitivity.csv"), "Records paired-participant sensitivity support."),
    StageDefinition("raise_icarus.tables", "manuscript_tables", "Manuscript tables", "tables", ("aggregate table source files", "manuscript item manifest"), ("tables/main", "tables/supplementary"), "Writes public table files using final manuscript labels."),
    StageDefinition("raise_icarus.figures", "manuscript_figures", "Manuscript figures", "figures", ("aggregate figure source files", "accepted figure template references", "manuscript item manifest"), ("figures/main", "figures/supplementary"), "Regenerates figures from aggregate files while applying template records."),
    StageDefinition("raise_icarus.validation", "manuscript_output_validation", "Manuscript output validation", "validation", ("manuscript outputs", "validation manifests"), ("reports/manuscript_output_validation.csv",), "Applies row-count, template, naming, and public-safety gates."),
    StageDefinition("raise_icarus.public_candidate", "public_aggregate_candidate", "Public aggregate candidate", "public_candidate", ("validated aggregate outputs",), ("public_release_candidate",), "Builds and validates an aggregate-only candidate."),
)

_BY_MODULE = {stage.module_name: stage for stage in DEFAULT_STAGE_DEFINITIONS}
_BY_STAGE = {stage.stage_name: stage for stage in DEFAULT_STAGE_DEFINITIONS}


def stage_definitions() -> tuple[StageDefinition, ...]:
    return DEFAULT_STAGE_DEFINITIONS


def definition_for(module_or_stage: str) -> StageDefinition:
    if module_or_stage in _BY_MODULE:
        return _BY_MODULE[module_or_stage]
    if module_or_stage in _BY_STAGE:
        return _BY_STAGE[module_or_stage]
    short = module_or_stage.rsplit(".", 1)[-1]
    for stage in DEFAULT_STAGE_DEFINITIONS:
        if stage.module_name.rsplit(".", 1)[-1] == short:
            return stage
    return StageDefinition(module_or_stage, short, short.replace("_", " ").title(), short, ("controlled harmonized monitoring archive",), ())


def expected_output_paths(stage: StageDefinition, run_dir: Path) -> tuple[str, ...]:
    return tuple((run_dir / rel).as_posix() for rel in stage.expected_outputs)


def dry_run_stage_result(stage: StageDefinition, run_dir: str | Path) -> StageResult:
    run_dir = Path(run_dir)
    return StageResult(stage.stage_name, stage.module_name, "PASS_DRY_RUN", stage.output_domain, expected_output_paths(stage, run_dir), "Stage contract resolved; controlled data were not read.")


def blocked_stage_result(stage: StageDefinition, run_dir: str | Path, notes: str | None = None) -> StageResult:
    run_dir = Path(run_dir)
    return StageResult(stage.stage_name, stage.module_name, "BLOCKED_CONTROLLED_DATA_EXECUTION_REQUIRED", stage.output_domain, expected_output_paths(stage, run_dir), notes or "Controlled-data computation is available only during an explicit controlled run.")


def run_contract_stage(stage: StageDefinition, harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    del harmonized_zip, n_samples
    if dry_run:
        return dry_run_stage_result(stage, run_dir)
    return blocked_stage_result(stage, run_dir, "Descriptive stage contract is implemented; aggregate computation must be enabled in a controlled-data run.")


def write_stage_results(path: str | Path, results: Iterable[StageResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_row() for result in results]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "module", "status", "output_domain", "outputs", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def write_stage_plan(path: str | Path, stages: Iterable[StageDefinition]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{
        "stage": stage.stage_name,
        "module": stage.module_name,
        "display_name": stage.display_name,
        "output_domain": stage.output_domain,
        "required_inputs": "; ".join(stage.required_inputs),
        "expected_outputs": "; ".join(stage.expected_outputs),
        "notes": stage.notes,
    } for stage in stages]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "module", "display_name", "output_domain", "required_inputs", "expected_outputs", "notes"])
        writer.writeheader()
        writer.writerows(rows)
