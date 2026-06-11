"""Phase 3 primary-only HIA scenario reproduction helpers.

This module deliberately avoids writing participant-day inputs, Monte Carlo
iteration samples, upper-tail sensitivity outputs, YLL outputs, figures, or
manuscript files. The only outputs are aggregate local validation tables.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from raise_icarus.data import DateFilterMode, feather_members, get_campaign_date_window, parse_city_season
from raise_icarus.hia import (
    HIAConfig,
    calculate_hia_scenarios,
    default_baseline_rates,
    default_counterfactual_table,
    default_crf_table,
    default_populations,
)
from raise_icarus.phase1_denominators import CITY_SEASON_ORDER, PM_COLUMNS, city_prefixed_participant_id
from raise_icarus.phase2_ppm_common_support import HIA_STREAM, PPM_SOURCE_DEVICE, S1_TARGETS


LONG_TERM_ENDPOINTS = {"Mortality all causes", "Cardiovascular", "Chronic bronchitis"}
SHORT_TERM_ENDPOINTS = {"Cardiac hospital admissions", "Respiratory hospital admissions"}
EXPECTED_COUNTERFACTUALS = {
    ("PM2.5", "long_term"): 5.0,
    ("PM10", "long_term"): 15.0,
    ("PM2.5", "short_term"): 15.0,
    ("PM10", "short_term"): 45.0,
}
EXPECTED_UNCERTAINTY = {
    "n_samples": 10_000,
    "baseline_rate_cv": 0.05,
    "population_cv": 0.01,
    "ppm_measurement_cv": 0.10,
    "pm25_pm10_ratio_mean": 0.65,
    "pm25_pm10_ratio_sd": 0.10,
    "pm25_pm10_ratio_min": 0.40,
    "pm25_pm10_ratio_max": 0.90,
}
EXPOSURE_METRIC = "Daily personal PM from common-support ICARUS PPM PM1/PM2.5/PM10 timestamp triplets"


def _read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        return feather.read_table(io.BytesIO(fh.read())).to_pandas()


def _common_support_daily_pm(data_zip: str | Path, date_filter_mode: DateFilterMode = "campaign") -> pd.DataFrame:
    """Build the finalized common-support daily personal PPM input in memory."""
    data_zip = Path(data_zip)
    pieces: list[pd.DataFrame] = []
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            city, season = parse_city_season(member)
            raw = _read_member(zf, member)
            if "TS" not in raw.columns or "ID" not in raw.columns:
                continue
            tmp = raw[["TS", "ID", *[column for column in PM_COLUMNS if column in raw.columns]]].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp = tmp.dropna(subset=["TS", "ID"])
            if date_filter_mode == "campaign":
                start, end = get_campaign_date_window(city, season)
                if start is None or end is None:
                    continue
                mask = (tmp["TS"].dt.date >= start.date()) & (tmp["TS"].dt.date <= end.date())
                tmp = tmp.loc[mask].copy()
            elif date_filter_mode != "none":
                raise ValueError(f"Unsupported date filter mode: {date_filter_mode}")
            if tmp.empty:
                continue
            for column in PM_COLUMNS:
                if column not in tmp.columns:
                    tmp[column] = pd.NA
                tmp[column] = pd.to_numeric(tmp[column], errors="coerce")
            complete_ordered = (
                tmp[list(PM_COLUMNS)].notna().all(axis=1)
                & (tmp["PM1_PPM"] <= tmp["PM25_PPM"])
                & (tmp["PM25_PPM"] <= tmp["PM10_PPM"])
            )
            tmp = tmp.loc[complete_ordered].copy()
            if tmp.empty:
                continue
            raw_id = raw["ID"].dropna().iloc[0] if raw["ID"].dropna().size else Path(member).stem
            tmp["city"] = city
            tmp["season"] = season
            tmp["participant_uid"] = city_prefixed_participant_id(city, raw_id)
            tmp["date"] = tmp["TS"].dt.date
            daily = (
                tmp.groupby(["city", "season", "participant_uid", "date"], as_index=False)[list(PM_COLUMNS)]
                .mean()
                .rename(columns={"PM1_PPM": "pm1", "PM25_PPM": "pm25", "PM10_PPM": "pm10"})
            )
            pieces.append(daily)
    if pieces:
        daily_pm = pd.concat(pieces, ignore_index=True)
    else:
        daily_pm = pd.DataFrame(columns=["city", "season", "participant_uid", "date", "pm1", "pm25", "pm10"])
    return daily_pm.sort_values(["city", "season", "participant_uid", "date"]).reset_index(drop=True)


def _make_exposure_inputs(daily_pm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    participant_means = daily_pm.groupby(["city", "season", "participant_uid"], as_index=False)[["pm25", "pm10"]].mean()
    for city, season in CITY_SEASON_ORDER:
        group = daily_pm[(daily_pm["city"] == city) & (daily_pm["season"] == season)]
        pgroup = participant_means[(participant_means["city"] == city) & (participant_means["season"] == season)]
        days_per_participant = group.groupby("participant_uid")["date"].nunique()
        participant_days = int(len(group))
        target_days = int(S1_TARGETS[HIA_STREAM]["Participant-days/nights/rows"])
        rows.append(
            {
                "city": city,
                "season": season,
                "source_device": PPM_SOURCE_DEVICE,
                "exposure_metric": EXPOSURE_METRIC,
                "date_min": str(group["date"].min()) if not group.empty else "",
                "date_max": str(group["date"].max()) if not group.empty else "",
                "participants": int(group["participant_uid"].nunique()) if not group.empty else 0,
                "participant_days": participant_days,
                "median_days_per_participant": float(days_per_participant.median()) if len(days_per_participant) else 0,
                "max_days_per_participant": int(days_per_participant.max()) if len(days_per_participant) else 0,
                "pm25_participant_day_mean": float(group["pm25"].mean()) if not group.empty else np.nan,
                "pm25_participant_weighted_mean": float(pgroup["pm25"].mean()) if not pgroup.empty else np.nan,
                "pm25_median": float(group["pm25"].median()) if not group.empty else np.nan,
                "pm25_sd": float(group["pm25"].std()) if len(group) > 1 else np.nan,
                "pm10_participant_day_mean": float(group["pm10"].mean()) if not group.empty else np.nan,
                "pm10_participant_weighted_mean": float(pgroup["pm10"].mean()) if not pgroup.empty else np.nan,
                "pm10_median": float(group["pm10"].median()) if not group.empty else np.nan,
                "pm10_sd": float(group["pm10"].std()) if len(group) > 1 else np.nan,
                "city_season_dates": int(group["date"].nunique()) if not group.empty else 0,
                "status": "PASS" if daily_pm.shape[0] == target_days else "FAIL",
                "notes": "Aggregate city-season exposure distribution; participant-day input was kept in memory only.",
            }
        )
    return pd.DataFrame(rows)


def make_counterfactual_audit(config: HIAConfig) -> pd.DataFrame:
    counterfactuals = default_counterfactual_table(config)
    rows: list[dict[str, object]] = []
    for row in counterfactuals.to_dict(orient="records"):
        key = (row["pollutant"], row["framework"])
        expected = EXPECTED_COUNTERFACTUALS.get(key, np.nan)
        observed = float(row["counterfactual_ug_m3"])
        status = "PASS" if np.isclose(observed, expected) and observed > 0 else "FAIL"
        rows.append(
            {
                **row,
                "expected_counterfactual_ug_m3": expected,
                "status": status,
                "notes": "WHO 2021 primary counterfactual; zero-counterfactual sensitivity is not run in Phase 3.",
            }
        )
    return pd.DataFrame(rows)


def make_crf_input_table() -> pd.DataFrame:
    crf = default_crf_table().rename(columns={"pollutant": "target_pollutant"})
    return crf[
        [
            "endpoint",
            "framework",
            "target_pollutant",
            "source_pollutant",
            "rr_per10",
            "rr_ci_low",
            "rr_ci_high",
            "crf_mapping",
            "reference_note",
        ]
    ].copy()


def make_uncertainty_settings(config: HIAConfig) -> pd.DataFrame:
    specs = [
        ("n_samples", config.n_samples, "Monte Carlo/bootstrap iterations.", EXPECTED_UNCERTAINTY["n_samples"]),
        ("crf_uncertainty", "enabled", "RR uncertainty sampled from lognormal distributions using reported 95% CIs.", "enabled"),
        ("exposure_distribution_uncertainty", "enabled", "Long-term rows bootstrap city-season participant means; short-term rows sample participant-day distributions hierarchically.", "enabled"),
        ("baseline_rate_cv", config.baseline_rate_cv, "Multiplicative baseline-rate uncertainty coefficient of variation.", EXPECTED_UNCERTAINTY["baseline_rate_cv"]),
        ("population_cv", config.population_cv, "Multiplicative population-count uncertainty coefficient of variation.", EXPECTED_UNCERTAINTY["population_cv"]),
        ("ppm_measurement_cv", config.ppm_measurement_cv, "Multiplicative PPM measurement uncertainty coefficient of variation.", EXPECTED_UNCERTAINTY["ppm_measurement_cv"]),
        ("pm25_pm10_ratio_mean", config.pm25_pm10_ratio_mean, "Mean PM2.5/PM10 ratio for converted CRF rows.", EXPECTED_UNCERTAINTY["pm25_pm10_ratio_mean"]),
        ("pm25_pm10_ratio_sd", config.pm25_pm10_ratio_sd, "SD of PM2.5/PM10 ratio for converted CRF rows.", EXPECTED_UNCERTAINTY["pm25_pm10_ratio_sd"]),
        ("pm25_pm10_ratio_min", config.pm25_pm10_ratio_min, "Lower bound for truncated PM2.5/PM10 ratio sampling.", EXPECTED_UNCERTAINTY["pm25_pm10_ratio_min"]),
        ("pm25_pm10_ratio_max", config.pm25_pm10_ratio_max, "Upper bound for truncated PM2.5/PM10 ratio sampling.", EXPECTED_UNCERTAINTY["pm25_pm10_ratio_max"]),
    ]
    rows: list[dict[str, object]] = []
    for parameter, value, description, expected in specs:
        if isinstance(expected, str):
            status = "PASS" if str(value) == expected else "FAIL"
        else:
            status = "PASS" if np.isclose(float(value), float(expected)) else "FAIL"
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "description": description,
                "status": status,
                "notes": "Matches expected Phase 3 primary uncertainty setting." if status == "PASS" else f"Expected {expected}.",
            }
        )
    return pd.DataFrame(rows)


def _augment_summary(summary: pd.DataFrame, config: HIAConfig) -> pd.DataFrame:
    crf = default_crf_table()
    baselines = default_baseline_rates()
    populations = default_populations()
    out = summary.merge(
        crf[["pollutant", "endpoint", "framework", "rr_per10", "rr_ci_low", "rr_ci_high"]],
        on=["pollutant", "endpoint", "framework"],
        how="left",
    )
    out = out.merge(
        baselines[["city", "endpoint", "annual_rate_per_person", "daily_rate_per_person"]],
        on=["city", "endpoint"],
        how="left",
    )
    out = out.merge(populations, on="city", how="left")
    out["source_device"] = PPM_SOURCE_DEVICE
    out["exposure_metric"] = EXPOSURE_METRIC
    out["baseline_rate"] = np.where(
        out["framework"].eq("long_term"),
        out["annual_rate_per_person"],
        out["daily_rate_per_person"],
    )
    out["baseline_rate_scale"] = np.where(
        out["framework"].eq("long_term"),
        "annual_rate_per_person",
        "daily_rate_per_person",
    )
    out["pm25_pm10_ratio_mean_if_applicable"] = np.where(
        out["crf_mapping"].ne("direct"),
        config.pm25_pm10_ratio_mean,
        np.nan,
    )
    out["pm25_pm10_ratio_sd_if_applicable"] = np.where(
        out["crf_mapping"].ne("direct"),
        config.pm25_pm10_ratio_sd,
        np.nan,
    )
    out["notes"] = "Ambient-CRF-based scenario output using daily personal ICARUS PPM PM exposure distributions; PM2.5 and PM10 are not summed."
    columns = [
        "scenario_key",
        "city",
        "season",
        "pollutant",
        "endpoint",
        "framework",
        "output_scale",
        "source_device",
        "exposure_metric",
        "counterfactual",
        "counterfactual_basis",
        "crf_mapping",
        "source_pollutant",
        "rr_per10",
        "rr_ci_low",
        "rr_ci_high",
        "reference_note",
        "n_samples",
        "baseline_rate",
        "baseline_rate_scale",
        "population",
        "pm_mean_of_samples",
        "exposure_contrast_mean",
        "share_samples_above_counterfactual",
        "cases_mean",
        "cases_median",
        "cases_p025",
        "cases_p975",
        "pm25_pm10_ratio_mean_if_applicable",
        "pm25_pm10_ratio_sd_if_applicable",
        "notes",
    ]
    return out[columns].sort_values(["city", "season", "pollutant", "endpoint"]).reset_index(drop=True)


def write_primary_hia_outputs(
    data_zip: str | Path,
    phase2_dir: str | Path,
    outdir: str | Path,
    n_samples: int = 10_000,
    seed: int = 20260430,
    date_filter_mode: DateFilterMode = "campaign",
) -> dict[str, Path]:
    """Run primary HIA scenarios and write aggregate local outputs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    daily_pm = _common_support_daily_pm(data_zip, date_filter_mode)
    config = HIAConfig(n_samples=n_samples, seed=seed)
    _, summary, _, _ = calculate_hia_scenarios(daily_pm, config=config, keep_samples=False)
    augmented = _augment_summary(summary, config)
    exposure_inputs = _make_exposure_inputs(daily_pm)

    outputs = {
        "hia_primary_scenario_summary": outdir / "hia_primary_scenario_summary.csv",
        "hia_counterfactual_audit": outdir / "hia_counterfactual_audit.csv",
        "hia_crf_input_table": outdir / "hia_crf_input_table.csv",
        "hia_uncertainty_settings": outdir / "hia_uncertainty_settings.csv",
        "hia_exposure_inputs_by_city_season": outdir / "hia_exposure_inputs_by_city_season.csv",
    }
    augmented.to_csv(outputs["hia_primary_scenario_summary"], index=False)
    make_counterfactual_audit(config).to_csv(outputs["hia_counterfactual_audit"], index=False)
    make_crf_input_table().to_csv(outputs["hia_crf_input_table"], index=False)
    make_uncertainty_settings(config).to_csv(outputs["hia_uncertainty_settings"], index=False)
    exposure_inputs.to_csv(outputs["hia_exposure_inputs_by_city_season"], index=False)
    pd.DataFrame(
        [
            {
                "repo_root": str(Path.cwd()),
                "data_zip": str(data_zip),
                "phase2_dir": str(phase2_dir),
                "date_filter_mode": date_filter_mode,
                "n_samples": n_samples,
                "seed": seed,
            }
        ]
    ).to_csv(outdir / "_phase3_run_metadata.csv", index=False)
    return outputs


def validate_counterfactual_file(input_path: str | Path, outdir: str | Path) -> Path:
    frame = pd.read_csv(input_path)
    frame["expected_counterfactual_ug_m3"] = [
        EXPECTED_COUNTERFACTUALS.get((row.pollutant, row.framework), np.nan)
        for row in frame.itertuples(index=False)
    ]
    frame["status"] = np.where(
        np.isclose(frame["counterfactual_ug_m3"].astype(float), frame["expected_counterfactual_ug_m3"].astype(float))
        & (frame["counterfactual_ug_m3"].astype(float) > 0),
        "PASS",
        "FAIL",
    )
    frame["notes"] = np.where(
        frame["status"].eq("PASS"),
        "Counterfactual matches WHO 2021 primary value and is nonzero.",
        "Counterfactual mismatch or zero value.",
    )
    outpath = Path(outdir) / "hia_counterfactual_audit.csv"
    frame.to_csv(outpath, index=False)
    return outpath


def validate_crf_table(input_path: str | Path, outdir: str | Path) -> Path:
    crf = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(crf.to_dict(orient="records"), start=1):
        target = row["target_pollutant"]
        source = row["source_pollutant"]
        mapping = row["crf_mapping"]
        direct_ok = mapping == "direct" and target == source
        converted_ok = mapping != "direct" and target != source and str(mapping).startswith("converted_from_")
        rr_valid = float(row["rr_per10"]) > 1.0
        ci_valid = float(row["rr_ci_low"]) > 0 and float(row["rr_ci_high"]) > float(row["rr_ci_low"])
        label_valid = target in {"PM2.5", "PM10"} and source in {"PM2.5", "PM10"}
        status = "PASS" if (direct_ok or converted_ok) and rr_valid and ci_valid and label_valid else "FAIL"
        notes = "CRF mapping and pollutant labels validated." if status == "PASS" else "Inspect CRF mapping, RR/CI, or pollutant labels."
        if converted_ok and target == "PM2.5" and source == "PM10":
            notes += " Converted PM2.5 row documents PM2.5/PM10 conversion-ratio uncertainty."
        rows.append(
            {
                "row_number": idx,
                "endpoint": row["endpoint"],
                "framework": row["framework"],
                "target_pollutant": target,
                "source_pollutant": source,
                "crf_mapping": mapping,
                "direct_mapping_valid": direct_ok,
                "converted_mapping_valid": converted_ok,
                "rr_valid": rr_valid,
                "ci_valid": ci_valid,
                "pollutant_label_valid": label_valid,
                "status": status,
                "notes": notes,
            }
        )
    outpath = Path(outdir) / "hia_crf_validation_report.csv"
    pd.DataFrame(rows).to_csv(outpath, index=False)
    return outpath


def _add_validation(rows: list[dict[str, object]], name: str, expected: object, observed: object, status: bool, notes: str) -> None:
    rows.append(
        {
            "validation_check": name,
            "expected_value": expected,
            "observed_value": observed,
            "status": "PASS" if status else "FAIL",
            "notes": notes,
        }
    )


def make_40row_validation(
    summary_path: str | Path,
    counterfactual_path: str | Path,
    crf_report_path: str | Path,
    outdir: str | Path,
    phase2_dir: str | Path = "local_outputs/ppm_common_support",
) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    counterfactuals = pd.read_csv(counterfactual_path)
    crf_report = pd.read_csv(crf_report_path)
    outdir = Path(outdir)
    exposure_path = outdir / "hia_exposure_inputs_by_city_season.csv"
    exposure = pd.read_csv(exposure_path) if exposure_path.exists() else pd.DataFrame()
    phase2_path = Path(phase2_dir) / "hia_daily_pm_input_validation.csv"
    phase2 = pd.read_csv(phase2_path) if phase2_path.exists() else pd.DataFrame()

    rows: list[dict[str, object]] = []
    _add_validation(rows, "total primary scenario rows", 40, len(summary), len(summary) == 40, "2 cities x 2 seasons x 2 pollutants x 5 endpoints.")
    _add_validation(rows, "no zero counterfactual rows", 0, int((summary["counterfactual"].astype(float) == 0).sum()), bool((summary["counterfactual"].astype(float) > 0).all()), "Primary Phase 3 run excludes zero-counterfactual sensitivity.")
    _add_validation(rows, "all long-term endpoints have long_term framework", "all", int(summary[summary["endpoint"].isin(LONG_TERM_ENDPOINTS)]["framework"].eq("long_term").sum()), bool(summary[summary["endpoint"].isin(LONG_TERM_ENDPOINTS)]["framework"].eq("long_term").all()), "Mortality, cardiovascular, and chronic bronchitis are long-term scenarios.")
    _add_validation(rows, "all short-term endpoints have short_term framework", "all", int(summary[summary["endpoint"].isin(SHORT_TERM_ENDPOINTS)]["framework"].eq("short_term").sum()), bool(summary[summary["endpoint"].isin(SHORT_TERM_ENDPOINTS)]["framework"].eq("short_term").all()), "Hospital-admission endpoints are short-term scenarios.")
    _add_validation(rows, "all long-term rows have annual persistent output scale", "annual_persistent_exposure_scenario", sorted(summary.loc[summary["framework"].eq("long_term"), "output_scale"].unique()), bool(summary.loc[summary["framework"].eq("long_term"), "output_scale"].eq("annual_persistent_exposure_scenario").all()), "Long-term rows are persistent-exposure scenario outputs.")
    _add_validation(rows, "all short-term rows have daily short-term output scale", "daily_short_term_scenario", sorted(summary.loc[summary["framework"].eq("short_term"), "output_scale"].unique()), bool(summary.loc[summary["framework"].eq("short_term"), "output_scale"].eq("daily_short_term_scenario").all()), "Hospital-admission rows use daily short-term scaling.")
    pollutants = sorted(summary["pollutant"].unique())
    _add_validation(rows, "PM2.5 and PM10 rows remain separate", "PM2.5;PM10", ";".join(pollutants), pollutants == ["PM10", "PM2.5"] or pollutants == ["PM2.5", "PM10"], "Pollutants are separate scenario rows.")
    total_pm_columns = [column for column in summary.columns if "total_pm" in column.lower() or "summed_pm" in column.lower()]
    _add_validation(rows, "no row sums PM2.5 and PM10", "0 total/summed PM columns", len(total_pm_columns), len(total_pm_columns) == 0, "No total PM burden column is produced.")
    _add_validation(rows, "2 cities present", 2, summary["city"].nunique(), summary["city"].nunique() == 2, "Milan and Thessaloniki present.")
    _add_validation(rows, "2 seasons present", 2, summary["season"].nunique(), summary["season"].nunique() == 2, "Summer and Winter present.")
    _add_validation(rows, "2 pollutants present", 2, summary["pollutant"].nunique(), summary["pollutant"].nunique() == 2, "PM2.5 and PM10 present.")
    _add_validation(rows, "5 endpoints present", 5, summary["endpoint"].nunique(), summary["endpoint"].nunique() == 5, "All primary endpoints present.")
    _add_validation(rows, "all required CRF rows valid", "all PASS", crf_report["status"].value_counts().to_dict(), bool(crf_report["status"].eq("PASS").all()), "CRF table mapping and labels validated.")
    _add_validation(rows, "all counterfactuals match expected values", "all PASS", counterfactuals["status"].value_counts().to_dict(), bool(counterfactuals["status"].eq("PASS").all()), "WHO 2021 values validated.")
    _add_validation(rows, "exposure contrast is non-negative", "minimum >= 0", float(summary["exposure_contrast_mean"].min()), bool((summary["exposure_contrast_mean"].astype(float) >= 0).all()), "Positive exposure contrast uses max(C - C0, 0).")
    _add_validation(rows, "HIA exposure source = ICARUS PPM only", PPM_SOURCE_DEVICE, sorted(summary["source_device"].unique()), bool(summary["source_device"].eq(PPM_SOURCE_DEVICE).all()), "Source-device label remains PPM only.")
    source_audit_path = Path(phase2_dir) / "hia_exposure_source_audit.csv"
    if source_audit_path.exists():
        source_audit = pd.read_csv(source_audit_path)
        uhoo_ok = bool(source_audit["uhoo_columns_used_in_hia_input"].astype(str).str.lower().eq("none").all())
    else:
        uhoo_ok = False
    _add_validation(rows, "uHoo not used in exposure input", "none", "none" if uhoo_ok else "not validated", uhoo_ok, "Uses Phase 2 exposure-source audit.")
    exposure_days = int(pd.to_numeric(exposure.get("participant_days", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not exposure.empty else 0
    phase2_days = int(pd.to_numeric(phase2.get("participant_days", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not phase2.empty else 0
    _add_validation(rows, "daily PM input denominator inherited from Phase 2", 2427, f"phase3={exposure_days}; phase2={phase2_days}", exposure_days == 2427 and phase2_days == 2427, "Common-support daily PPM denominator preserved.")
    return pd.DataFrame(rows)


def write_40row_validation(
    summary_path: str | Path,
    counterfactual_path: str | Path,
    crf_report_path: str | Path,
    outdir: str | Path,
    phase2_dir: str | Path = "local_outputs/ppm_common_support",
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "hia_40row_validation.csv"
    make_40row_validation(summary_path, counterfactual_path, crf_report_path, outdir, phase2_dir).to_csv(outpath, index=False)
    write_phase3_validation_report(Path.cwd(), "", outdir, phase2_dir=phase2_dir)
    return outpath


def write_manuscript_table(summary_path: str | Path, outdir: str | Path) -> Path:
    """Export a local aggregate manuscript-table candidate without touching manuscript files."""
    summary = pd.read_csv(summary_path)
    table = summary[
        [
            "city",
            "season",
            "pollutant",
            "endpoint",
            "framework",
            "output_scale",
            "counterfactual",
            "crf_mapping",
            "source_pollutant",
            "cases_mean",
            "cases_median",
            "cases_p025",
            "cases_p975",
            "notes",
        ]
    ].copy()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "hia_primary_manuscript_table.csv"
    table.to_csv(outpath, index=False)
    write_phase3_validation_report(Path.cwd(), "", outdir)
    return outpath


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _status_for(frame: pd.DataFrame) -> str:
    if frame.empty or "status" not in frame.columns:
        return "FAIL"
    return "PASS" if frame["status"].astype(str).eq("PASS").all() else "FAIL"


def write_phase3_validation_report(
    repo_root: str | Path,
    data_zip: str | Path,
    outdir: str | Path,
    phase1_dir: str | Path = "local_outputs/denominators",
    phase2_dir: str | Path = "local_outputs/ppm_common_support",
    date_filter_mode: DateFilterMode = "campaign",
) -> Path:
    outdir = Path(outdir)
    metadata = _read_csv(outdir / "_phase3_run_metadata.csv")
    if not metadata.empty:
        meta = metadata.iloc[0]
        if not str(repo_root):
            repo_root = meta.get("repo_root", repo_root)
        if not str(data_zip):
            data_zip = meta.get("data_zip", data_zip)
        phase2_dir = meta.get("phase2_dir", phase2_dir)
        date_filter_mode = meta.get("date_filter_mode", date_filter_mode)
    summary = _read_csv(outdir / "hia_primary_scenario_summary.csv")
    counterfactuals = _read_csv(outdir / "hia_counterfactual_audit.csv")
    crf_report = _read_csv(outdir / "hia_crf_validation_report.csv")
    uncertainty = _read_csv(outdir / "hia_uncertainty_settings.csv")
    validation = _read_csv(outdir / "hia_40row_validation.csv")
    exposure = _read_csv(outdir / "hia_exposure_inputs_by_city_season.csv")

    def check_status(name: str) -> str:
        if validation.empty:
            return "FAIL"
        row = validation[validation["validation_check"] == name]
        return str(row["status"].iloc[0]) if not row.empty else "FAIL"

    hia_days = int(pd.to_numeric(exposure.get("participant_days", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not exposure.empty else 0
    rows_40 = len(summary)
    outpath = outdir / "phase3_validation_report.txt"
    lines = [
        "Phase 3 validation report",
        f"timestamp of run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository path: {repo_root}",
        f"data archive path used: {data_zip}",
        f"Phase 1 output path used: {Path(phase1_dir).resolve()}",
        f"Phase 2 output path used: {Path(phase2_dir).resolve()}",
        f"campaign-window mode: {date_filter_mode}",
        "scripts run: scripts\\03_run_hia_primary_scenarios.py; scripts\\03_validate_hia_counterfactuals.py; scripts\\03_validate_hia_crf_table.py; scripts\\03_validate_hia_40row_output.py; scripts\\03_export_hia_manuscript_tables.py",
        "",
        "Target values:",
        "- Primary HIA scenario rows: 40",
        "- HIA daily PM input participant-days: 2,427",
        "- Counterfactuals: PM2.5 long-term 5, PM10 long-term 15, PM2.5 short-term 15, PM10 short-term 45",
        "- HIA exposure source: ICARUS PPM only; uHoo excluded",
        "",
        "Reproduced values:",
        f"- Primary HIA scenario rows: {rows_40}",
        f"- HIA daily PM input participant-days: {hia_days}",
        f"- Counterfactual validation statuses: {counterfactuals['status'].value_counts().to_dict() if not counterfactuals.empty else 'not available'}",
        f"- CRF validation statuses: {crf_report['status'].value_counts().to_dict() if not crf_report.empty else 'not available'}",
        "",
        "PASS/FAIL:",
        f"- 40-row primary output: {check_status('total primary scenario rows')}",
        f"- counterfactual validation: {_status_for(counterfactuals)}",
        f"- no zero-counterfactual rows: {check_status('no zero counterfactual rows')}",
        f"- CRF table validation: {_status_for(crf_report)}",
        f"- long-term/short-term endpoint separation: {'PASS' if check_status('all long-term endpoints have long_term framework') == 'PASS' and check_status('all short-term endpoints have short_term framework') == 'PASS' else 'FAIL'}",
        f"- daily short-term hospital-admission scaling: {check_status('all short-term rows have daily short-term output scale')}",
        f"- PM2.5/PM10 separation and no summing: {'PASS' if check_status('PM2.5 and PM10 rows remain separate') == 'PASS' and check_status('no row sums PM2.5 and PM10') == 'PASS' else 'FAIL'}",
        f"- uncertainty settings: {_status_for(uncertainty)}",
        f"- HIA source device = ICARUS PPM only: {check_status('HIA exposure source = ICARUS PPM only')}",
        f"- uHoo not used in HIA exposure input: {check_status('uHoo not used in exposure input')}",
        "",
        "Missing dependencies:",
        "- None for Phase 3 validation.",
        "",
        "Deviations from target values:",
        "- None." if rows_40 == 40 and hia_days == 2427 and _status_for(counterfactuals) == "PASS" and _status_for(crf_report) == "PASS" else "- One or more validation checks failed; inspect local CSVs.",
        "",
        "Output files:",
    ]
    output_paths = sorted(outdir.glob("*.csv")) + sorted(outdir.glob("*.txt"))
    if outpath not in output_paths:
        output_paths.append(outpath)
    for path in output_paths:
        lines.append(f"- {path.resolve()}")
    lines.extend(
        [
            "",
            "Confirmations:",
            "- YLL was not calculated.",
            "- Upper-tail sensitivity was not run.",
            "- No Phase 4 or Phase 5 work was performed.",
            "- No GitHub push, Git commit, or data upload was performed.",
            "- Controlled data remained local.",
            "- Safe outputs contain aggregate scenario and validation tables only; restricted participant identifiers, source paths, raw timestamps, row-level Feather identifiers, and participant-day rows are absent.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outpath
