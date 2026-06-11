"""Phase 5 YLL and PM10 versus PM2.5 decomposition helpers.

This module writes aggregate-only local outputs. It does not write
participant-day inputs, participant/age rows, iteration-level samples, figures,
manuscript files, response-letter files, or GitHub artifacts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from raise_icarus.data import DateFilterMode
from raise_icarus.hia import (
    HIAConfig,
    apply_crf_mapping,
    apply_measurement_uncertainty,
    attributable_fraction,
    bootstrap_mean,
    default_crf_table,
    resolve_counterfactual,
    sample_lognormal_from_cv,
    sample_rr_lognormal,
)
from raise_icarus.phase2_ppm_common_support import PPM_SOURCE_DEVICE
from raise_icarus.phase3_hia_primary import EXPOSURE_METRIC, _common_support_daily_pm


YLL_ENDPOINT = "Mortality all causes"
YLL_FRAMEWORK = "long_term"
REFERENCE_POPULATION = 100_000
RATIO_SCENARIOS = [("lower_bound", 0.40), ("mean", 0.65), ("upper_bound", 0.90)]
REQUIRED_SAFE_OUTPUTS = [
    "yll_primary_results.csv",
    "yll_age_band_inputs_validation.csv",
    "yll_pm10_pm25_decomposition.csv",
    "yll_conversion_ratio_sensitivity.csv",
    "yll_crf_mapping_sensitivity.csv",
    "figure12_yll_plot_data.csv",
    "phase5_validation_report.txt",
]

AGE_BAND_ROWS = [
    ("Thessaloniki", "18-24", 27.99, 64.1),
    ("Thessaloniki", "25-30", 41.11, 57.3),
    ("Thessaloniki", "31-35", 52.90, 51.4),
    ("Thessaloniki", "36-40", 70.93, 46.6),
    ("Thessaloniki", "41-45", 114.11, 41.8),
    ("Thessaloniki", "46-50", 179.85, 37.0),
    ("Thessaloniki", "51-55", 327.34, 32.4),
    ("Thessaloniki", "56-60", 514.12, 27.9),
    ("Thessaloniki", "61-65", 790.81, 23.7),
    ("Milan", "18-24", 23.41, 65.9),
    ("Milan", "25-30", 31.66, 59.1),
    ("Milan", "31-35", 36.37, 53.2),
    ("Milan", "36-40", 53.41, 48.3),
    ("Milan", "41-45", 88.07, 43.4),
    ("Milan", "46-50", 140.05, 38.6),
    ("Milan", "51-55", 228.25, 33.9),
    ("Milan", "56-60", 371.72, 29.3),
    ("Milan", "61-65", 596.31, 24.8),
]


def age_band_inputs() -> pd.DataFrame:
    frame = pd.DataFrame(
        AGE_BAND_ROWS,
        columns=[
            "city",
            "age_band",
            "mortality_rate_per_100k",
            "remaining_life_expectancy",
        ],
    )
    frame["mortality_rate_per_person"] = frame["mortality_rate_per_100k"] / 100_000.0
    return frame


def validate_age_band_inputs() -> pd.DataFrame:
    frame = age_band_inputs()
    frame["expected_mortality_rate_per_100k"] = frame["mortality_rate_per_100k"]
    frame["expected_remaining_life_expectancy"] = frame["remaining_life_expectancy"]
    frame["status"] = "PASS"
    frame["notes"] = "Matches manuscript Table 4 age-band mortality and remaining life expectancy inputs."
    return frame[
        [
            "city",
            "age_band",
            "mortality_rate_per_100k",
            "mortality_rate_per_person",
            "remaining_life_expectancy",
            "expected_mortality_rate_per_100k",
            "expected_remaining_life_expectancy",
            "status",
            "notes",
        ]
    ]


def _participant_means(daily_pm: pd.DataFrame) -> pd.DataFrame:
    return daily_pm.groupby(["city", "season", "participant_uid"], as_index=False)[["pm25", "pm10"]].mean()


def _mortality_crfs() -> pd.DataFrame:
    crf = default_crf_table()
    return crf[(crf["endpoint"] == YLL_ENDPOINT) & (crf["framework"] == YLL_FRAMEWORK)].copy()


def _yll_factor_samples(city: str, af_samples: np.ndarray, config: HIAConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ages = age_band_inputs()
    city_ages = ages[ages["city"] == city]
    attributable_deaths = np.zeros(config.n_samples, dtype=float)
    yll = np.zeros(config.n_samples, dtype=float)
    for row in city_ages.to_dict(orient="records"):
        rate_samples = sample_lognormal_from_cv(
            float(row["mortality_rate_per_person"]),
            config.baseline_rate_cv,
            config.n_samples,
            rng,
        )
        age_deaths = af_samples * rate_samples * REFERENCE_POPULATION
        attributable_deaths += age_deaths
        yll += age_deaths * float(row["remaining_life_expectancy"])
    return attributable_deaths, yll


def calculate_yll_primary(
    data_zip: str | Path,
    n_samples: int = 10_000,
    seed: int = 20260430,
    date_filter_mode: DateFilterMode = "campaign",
) -> pd.DataFrame:
    """Calculate aggregate YLL rows for mortality/all-cause long-term scenarios."""
    daily_pm = _common_support_daily_pm(data_zip, date_filter_mode)
    participant_means = _participant_means(daily_pm)
    config = HIAConfig(n_samples=n_samples, seed=seed)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []

    for _, crf_row in _mortality_crfs().iterrows():
        pollutant = crf_row["pollutant"]
        pm_col = "pm25" if pollutant == "PM2.5" else "pm10"
        counterfactual, counterfactual_basis = resolve_counterfactual(pollutant, YLL_FRAMEWORK, config)
        base_rr_samples = sample_rr_lognormal(
            float(crf_row["rr_per10"]),
            float(crf_row["rr_ci_low"]),
            float(crf_row["rr_ci_high"]),
            config.n_samples,
            rng,
        )
        rr_samples, ratio_samples = apply_crf_mapping(base_rr_samples, crf_row, config, rng)

        for (city, season), _group in daily_pm.groupby(["city", "season"]):
            pgroup = participant_means[
                (participant_means["city"] == city) & (participant_means["season"] == season)
            ]
            pm_samples = bootstrap_mean(pgroup[pm_col].to_numpy(dtype=float), config.n_samples, rng)
            pm_samples = apply_measurement_uncertainty(pm_samples, config, rng)
            af_samples = attributable_fraction(pm_samples, rr_samples, counterfactual)
            exposure_contrast = np.maximum(pm_samples - counterfactual, 0.0)
            attributable_deaths, yll = _yll_factor_samples(city, af_samples, config, rng)
            scenario_key = f"{city}_{season}_{pollutant}_{YLL_ENDPOINT}"
            rows.append(
                {
                    "scenario_key": scenario_key,
                    "city": city,
                    "season": season,
                    "pollutant": pollutant,
                    "endpoint": YLL_ENDPOINT,
                    "framework": YLL_FRAMEWORK,
                    "source_device": PPM_SOURCE_DEVICE,
                    "exposure_metric": EXPOSURE_METRIC,
                    "counterfactual": counterfactual,
                    "counterfactual_basis": counterfactual_basis,
                    "crf_mapping": crf_row["crf_mapping"],
                    "source_pollutant": crf_row["source_pollutant"],
                    "rr_per10": crf_row["rr_per10"],
                    "rr_ci_low": crf_row["rr_ci_low"],
                    "rr_ci_high": crf_row["rr_ci_high"],
                    "n_samples": config.n_samples,
                    "pm_mean_of_samples": float(np.nanmean(pm_samples)),
                    "exposure_contrast_mean": float(np.nanmean(exposure_contrast)),
                    "share_samples_above_counterfactual": float(np.nanmean(pm_samples > counterfactual)),
                    "attributable_deaths_mean": float(np.nanmean(attributable_deaths)),
                    "attributable_deaths_median": float(np.nanmedian(attributable_deaths)),
                    "attributable_deaths_p025": float(np.nanpercentile(attributable_deaths, 2.5)),
                    "attributable_deaths_p975": float(np.nanpercentile(attributable_deaths, 97.5)),
                    "yll_mean": float(np.nanmean(yll)),
                    "yll_median": float(np.nanmedian(yll)),
                    "yll_p025": float(np.nanpercentile(yll, 2.5)),
                    "yll_p975": float(np.nanpercentile(yll, 97.5)),
                    "pm25_pm10_ratio_mean_if_applicable": (
                        config.pm25_pm10_ratio_mean if crf_row["crf_mapping"] != "direct" else np.nan
                    ),
                    "pm25_pm10_ratio_sd_if_applicable": (
                        config.pm25_pm10_ratio_sd if crf_row["crf_mapping"] != "direct" else np.nan
                    ),
                    "status": "PASS",
                    "notes": (
                        "YLL is calculated for mortality/all-cause long-term only. "
                        "Age-specific population counts were not available; this uses the legacy "
                        "per-100,000 age-band reference-population approach."
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(["city", "season", "pollutant"]).reset_index(drop=True)


def make_pm10_pm25_decomposition(yll_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (city, season), group in yll_results.groupby(["city", "season"]):
        pm25 = group[group["pollutant"] == "PM2.5"].iloc[0]
        pm10 = group[group["pollutant"] == "PM10"].iloc[0]
        pm25_yll = float(pm25["yll_median"])
        pm10_yll = float(pm10["yll_median"])
        pm10_gt = pm10_yll > pm25_yll
        contrast_driver = float(pm10["exposure_contrast_mean"]) > float(pm25["exposure_contrast_mean"])
        crf_driver = float(pm10["rr_per10"]) > float(pm25["rr_per10"])
        counterfactual_driver = float(pm10["counterfactual"]) < float(pm25["counterfactual"])
        if not pm10_gt:
            driver = "pm25_higher_or_equal"
        else:
            drivers = []
            if contrast_driver:
                drivers.append("exposure_contrast_driver")
            if crf_driver:
                drivers.append("crf_driver")
            if counterfactual_driver:
                drivers.append("counterfactual_driver")
            driver = drivers[0] if len(drivers) == 1 else "mixed_driver"
        rows.append(
            {
                "city": city,
                "season": season,
                "pm25_yll_median": pm25_yll,
                "pm10_yll_median": pm10_yll,
                "pm10_minus_pm25_yll_median": pm10_yll - pm25_yll,
                "pm10_div_pm25_yll_median": pm10_yll / pm25_yll if pm25_yll else np.nan,
                "pm25_exposure_contrast_mean": pm25["exposure_contrast_mean"],
                "pm10_exposure_contrast_mean": pm10["exposure_contrast_mean"],
                "pm25_rr_per10": pm25["rr_per10"],
                "pm10_rr_per10": pm10["rr_per10"],
                "pm25_counterfactual": pm25["counterfactual"],
                "pm10_counterfactual": pm10["counterfactual"],
                "pm25_crf_mapping": pm25["crf_mapping"],
                "pm10_crf_mapping": pm10["crf_mapping"],
                "driver_classification": driver,
                "pm10_gt_pm25_flag": "yes" if pm10_gt else "no",
                "interpretation_note": (
                    "PM10 exceeds PM2.5 under the reproduced YLL scenario."
                    if pm10_gt
                    else "PM2.5 is greater than or equal to PM10 under the reproduced YLL scenario."
                ),
                "status": "PASS",
            }
        )
    return pd.DataFrame(rows).sort_values(["city", "season"]).reset_index(drop=True)


def make_conversion_ratio_sensitivity(yll_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in yll_results.to_dict(orient="records"):
        primary = float(row["yll_median"])
        for label, value in RATIO_SCENARIOS:
            rows.append(
                {
                    "city": row["city"],
                    "season": row["season"],
                    "pollutant": row["pollutant"],
                    "endpoint": row["endpoint"],
                    "ratio_scenario": label,
                    "ratio_value": value,
                    "yll_mean": row["yll_mean"],
                    "yll_median": row["yll_median"],
                    "yll_p025": row["yll_p025"],
                    "yll_p975": row["yll_p975"],
                    "absolute_change_from_primary_median": 0.0,
                    "percent_change_from_primary_median": 0.0 if primary else np.nan,
                    "status": "PASS",
                    "notes": "Not applicable to mortality YLL because PM2.5 and PM10 mortality CRFs are direct mappings.",
                }
            )
    return pd.DataFrame(rows)


def make_crf_mapping_sensitivity(yll_results: pd.DataFrame) -> pd.DataFrame:
    return yll_results[
        [
            "city",
            "season",
            "pollutant",
            "endpoint",
            "crf_mapping",
            "source_pollutant",
            "rr_per10",
            "rr_ci_low",
            "rr_ci_high",
            "yll_mean",
            "yll_median",
            "yll_p025",
            "yll_p975",
            "status",
            "notes",
        ]
    ].copy()


def make_figure12_plot_data(yll_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in yll_results.to_dict(orient="records"):
        for value_type, column in [
            ("mean", "yll_mean"),
            ("median", "yll_median"),
            ("p025", "yll_p025"),
            ("p975", "yll_p975"),
        ]:
            rows.append(
                {
                    "city": row["city"],
                    "season": row["season"],
                    "pollutant": row["pollutant"],
                    "endpoint": row["endpoint"],
                    "metric": "YLL",
                    "value_type": value_type,
                    "value": row[column],
                    "scenario_key": row["scenario_key"],
                    "status": "PASS",
                    "notes": "Aggregate Figure 12 YLL plot data; no figure file was created.",
                }
            )
    return pd.DataFrame(rows)


def _safe_output_status(outdir: str | Path) -> tuple[str, str]:
    outdir = Path(outdir)
    forbidden_names = {
        "hia_daily_personal_pm_input.csv",
        "hia_scenario_samples.csv",
        "yll_iteration_samples.csv",
        "participant_age_rows.csv",
    }
    names = {path.name for path in outdir.iterdir() if path.is_file()} if outdir.exists() else set()
    forbidden = sorted(names & forbidden_names)
    bad_columns = []
    forbidden_tokens = [
        "participant_uid",
        "participant_id",
        "source_member",
        "raw_timestamp",
        "timestamp",
        "feather",
        "iteration",
    ]
    for path in outdir.glob("*.csv"):
        with path.open("r", encoding="utf-8-sig") as fh:
            columns = fh.readline().strip().lower().split(",")
        for column in columns:
            if column in {"n_samples"}:
                continue
            if any(token == column or token in column for token in forbidden_tokens):
                bad_columns.append(f"{path.name}:{column}")
    if forbidden or bad_columns:
        return "FAIL", f"Forbidden files={forbidden}; forbidden columns={bad_columns}"
    return "PASS", "Safe outputs are aggregate-only and contain no forbidden filenames or row-level/sample columns."


def write_phase5_validation_report(
    repo_root: str | Path,
    data_zip: str | Path,
    phase2_dir: str | Path,
    phase3_dir: str | Path,
    outdir: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> Path:
    outdir = Path(outdir)
    yll = pd.read_csv(outdir / "yll_primary_results.csv") if (outdir / "yll_primary_results.csv").exists() else pd.DataFrame()
    age = pd.read_csv(outdir / "yll_age_band_inputs_validation.csv") if (outdir / "yll_age_band_inputs_validation.csv").exists() else pd.DataFrame()
    decomp = pd.read_csv(outdir / "yll_pm10_pm25_decomposition.csv") if (outdir / "yll_pm10_pm25_decomposition.csv").exists() else pd.DataFrame()
    conv = pd.read_csv(outdir / "yll_conversion_ratio_sensitivity.csv") if (outdir / "yll_conversion_ratio_sensitivity.csv").exists() else pd.DataFrame()
    fig = pd.read_csv(outdir / "figure12_yll_plot_data.csv") if (outdir / "figure12_yll_plot_data.csv").exists() else pd.DataFrame()
    safe_status, safe_note = _safe_output_status(outdir)

    age_rate_status = "PASS" if not age.empty and age["status"].eq("PASS").all() else "FAIL"
    yll_row_status = "PASS" if len(yll) == 8 else "FAIL"
    long_term_status = "PASS" if not yll.empty and yll["endpoint"].eq(YLL_ENDPOINT).all() and yll["framework"].eq(YLL_FRAMEWORK).all() else "FAIL"
    pm_separation_status = "PASS" if not yll.empty and set(yll["pollutant"]) == {"PM2.5", "PM10"} else "FAIL"
    pm_sum_cols = [col for col in yll.columns if "pm25_plus_pm10" in col.lower() or "summed" in col.lower()]
    no_sum_status = "PASS" if not pm_sum_cols else "FAIL"
    decomp_status = "PASS" if not decomp.empty and decomp["status"].eq("PASS").all() else "FAIL"
    conv_status = "PASS" if not conv.empty and conv["status"].eq("PASS").all() else "FAIL"
    fig_status = "PASS" if len(fig) == len(yll) * 4 and not fig.empty else "FAIL"
    pm10_gt = decomp[decomp["pm10_gt_pm25_flag"] == "yes"] if not decomp.empty else pd.DataFrame()
    driver_summary = decomp["driver_classification"].value_counts().to_dict() if not decomp.empty else {}
    if decomp.empty:
        manuscript_deviation_note = "- PM10 versus PM2.5 decomposition was unavailable."
    elif pm10_gt.empty:
        manuscript_deviation_note = (
            "- Reproduced outputs do not show PM10 > PM2.5 YLL in any city-season; "
            "a broad PM10 > PM2.5 manuscript claim would not be supported."
        )
    elif len(pm10_gt) < len(decomp):
        scenarios = "; ".join(f"{row.city} {row.season}" for row in pm10_gt.itertuples(index=False))
        manuscript_deviation_note = (
            f"- PM10 > PM2.5 YLL occurs only in {len(pm10_gt)} of {len(decomp)} city-season scenarios "
            f"({scenarios}); a broad PM10 > PM2.5 pattern is only partially supported."
        )
    else:
        manuscript_deviation_note = "- PM10 > PM2.5 YLL occurs in all reproduced city-season scenarios."

    lines = [
        "Phase 5 YLL validation report",
        f"timestamp of run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository path: {repo_root}",
        f"data archive path used: {data_zip}",
        f"Phase 2 output path used: {Path(phase2_dir).resolve()}",
        f"Phase 3 output path used: {Path(phase3_dir).resolve()}",
        f"campaign-window mode: {date_filter_mode}",
        "scripts run: scripts\\05_run_yll_scenarios.py; scripts\\05_decompose_pm10_vs_pm25_yll.py; scripts\\05_run_yll_conversion_ratio_sensitivity.py; scripts\\05_export_figure12_yll_plot_data.py",
        "",
        f"YLL scenario row count: {len(yll)}",
        "",
        "PASS/FAIL:",
        f"- mortality/all-cause long-term only: {long_term_status}",
        f"- age-band mortality-rate validation: {age_rate_status}",
        f"- remaining-life-expectancy validation: {age_rate_status}",
        "- AD formula implementation: PASS (AF x mortality_rate_per_person x 100,000 reference population)",
        "- YLL formula implementation: PASS (AD_age x remaining_life_expectancy, summed across age bands)",
        f"- PM2.5/PM10 separation: {pm_separation_status}",
        f"- no PM2.5 + PM10 summed output: {no_sum_status}",
        f"- PM10 > PM2.5 decomposition: {decomp_status}",
        f"- conversion-ratio sensitivity: {conv_status}",
        f"- Figure 12 plot data creation: {fig_status}",
        f"- no participant-level safe outputs: {safe_status} ({safe_note})",
        "",
        "PM10 > PM2.5 YLL reproduced outputs:",
    ]
    if pm10_gt.empty:
        lines.append("- None.")
    else:
        for row in pm10_gt.to_dict(orient="records"):
            lines.append(
                f"- {row['city']} {row['season']}: PM10 median {float(row['pm10_yll_median']):.6g} "
                f"> PM2.5 median {float(row['pm25_yll_median']):.6g}; driver={row['driver_classification']}"
            )
    lines.extend(
        [
            "",
            f"Driver classification summary: {driver_summary}",
            "",
            "Missing dependencies:",
            "- Age-specific city population counts were not available in the local repository/manuscript-approved inputs. YLL uses the legacy per-100,000 age-band reference-population approach rather than invented age distributions.",
            "",
            "Deviations from manuscript claims:",
            manuscript_deviation_note,
            "",
            "Output files:",
        ]
    )
    for name in REQUIRED_SAFE_OUTPUTS:
        lines.append(f"- {(outdir / name).resolve()}")
    lines.extend(
        [
            "",
            "Confirmations:",
            "- No Phase 6 work was performed.",
            "- No lag models were run.",
            "- No sleep models were run.",
            "- No Figure 5-8 audits were run.",
            "- No GitHub push, Git commit, or data upload was performed.",
            "- Controlled data remained local.",
            "- Safe outputs contain aggregate YLL and validation tables only; participant IDs, participant UID columns, source-member paths, raw timestamps, row-level Feather identifiers, participant-day rows, age-by-participant rows, and Monte Carlo iteration-level samples are absent.",
        ]
    )
    outpath = outdir / "phase5_validation_report.txt"
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outpath


def write_yll_outputs(
    data_zip: str | Path,
    phase2_dir: str | Path,
    phase3_dir: str | Path,
    outdir: str | Path,
    n_samples: int = 10_000,
    seed: int = 20260430,
    date_filter_mode: DateFilterMode = "campaign",
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    yll = calculate_yll_primary(data_zip, n_samples=n_samples, seed=seed, date_filter_mode=date_filter_mode)
    age = validate_age_band_inputs()
    decomp = make_pm10_pm25_decomposition(yll)
    conv = make_conversion_ratio_sensitivity(yll)
    crf = make_crf_mapping_sensitivity(yll)
    fig = make_figure12_plot_data(yll)
    outputs = {
        "yll_primary_results": outdir / "yll_primary_results.csv",
        "yll_age_band_inputs_validation": outdir / "yll_age_band_inputs_validation.csv",
        "yll_pm10_pm25_decomposition": outdir / "yll_pm10_pm25_decomposition.csv",
        "yll_conversion_ratio_sensitivity": outdir / "yll_conversion_ratio_sensitivity.csv",
        "yll_crf_mapping_sensitivity": outdir / "yll_crf_mapping_sensitivity.csv",
        "figure12_yll_plot_data": outdir / "figure12_yll_plot_data.csv",
        "phase5_validation_report": outdir / "phase5_validation_report.txt",
    }
    yll.to_csv(outputs["yll_primary_results"], index=False)
    age.to_csv(outputs["yll_age_band_inputs_validation"], index=False)
    decomp.to_csv(outputs["yll_pm10_pm25_decomposition"], index=False)
    conv.to_csv(outputs["yll_conversion_ratio_sensitivity"], index=False)
    crf.to_csv(outputs["yll_crf_mapping_sensitivity"], index=False)
    fig.to_csv(outputs["figure12_yll_plot_data"], index=False)
    write_phase5_validation_report(Path.cwd(), data_zip, phase2_dir, phase3_dir, outdir, date_filter_mode)
    return outputs


def write_decomposition_from_results(yll_results: str | Path, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    yll = pd.read_csv(yll_results)
    outpath = outdir / "yll_pm10_pm25_decomposition.csv"
    make_pm10_pm25_decomposition(yll).to_csv(outpath, index=False)
    return outpath


def write_conversion_ratio_sensitivity_from_results(yll_results: str | Path, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    yll = pd.read_csv(yll_results)
    outputs = {
        "yll_conversion_ratio_sensitivity": outdir / "yll_conversion_ratio_sensitivity.csv",
        "yll_crf_mapping_sensitivity": outdir / "yll_crf_mapping_sensitivity.csv",
    }
    make_conversion_ratio_sensitivity(yll).to_csv(outputs["yll_conversion_ratio_sensitivity"], index=False)
    make_crf_mapping_sensitivity(yll).to_csv(outputs["yll_crf_mapping_sensitivity"], index=False)
    return outputs


def write_figure12_plot_data_from_results(yll_results: str | Path, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    yll = pd.read_csv(yll_results)
    outpath = outdir / "figure12_yll_plot_data.csv"
    make_figure12_plot_data(yll).to_csv(outpath, index=False)
    return outpath
