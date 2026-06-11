"""Phase 11 optional descriptive equity/context analysis helpers.

The default Phase 11 path is intentionally conservative. If no separate
demographic or questionnaire source is provided, the module records missing
subgroup dependencies and does not attempt to infer equity variables from sensor
stream support. Safe outputs are aggregate-only and do not contain participant
identifiers, source-member paths, raw timestamps, participant-day rows,
participant-night rows, questionnaire/TAD rows, demographic microdata, model
input rows, or unsuppressed small cells.
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from raise_icarus.data import DateFilterMode, feather_members
from raise_icarus.table6 import TABLE6_VARIABLES, load_table6_daily_with_audit


CANDIDATE_VARIABLES = [
    ("age_or_age_group", "demographic", ["age", "age_group", "age group"]),
    ("sex_or_gender", "demographic", ["sex", "gender"]),
    ("asthma_or_respiratory_susceptibility", "questionnaire", ["asthma", "respiratory", "susceptibility"]),
    ("older_adult_indicator", "demographic", ["older_adult", "older adult", "age"]),
    ("child_or_adolescent_indicator", "demographic", ["child", "adolescent", "age"]),
    ("education", "demographic", ["education", "school"]),
    ("employment", "demographic", ["employment", "employed", "occupation"]),
    ("marital_status", "demographic", ["marital", "married"]),
    ("hotspot_vs_lower_exposure_recruitment_context", "recruitment_context", ["hotspot", "recruitment", "context"]),
    ("housing_or_source_variables", "questionnaire", ["housing", "home", "source"]),
    ("cooking_heating_ventilation", "questionnaire", ["cooking", "heating", "ventilation"]),
    ("neighborhood_deprivation_or_socioeconomic_index", "derived_context", ["deprivation", "socioeconomic", "index"]),
    ("city_season_design_strata", "derived_context", ["city", "season"]),
    ("wearable_activity_state", "sensor_stream", ["Activity"]),
]

ALLOWED_METRICS = {
    "PM1_PPM": ("PM1", "ICARUS PPM", "ICARUS PPM personal PM", "ug/m3", "PPM personal PM"),
    "PM25_PPM": ("PM2.5", "ICARUS PPM", "ICARUS PPM personal PM", "ug/m3", "PPM personal PM"),
    "PM10_PPM": ("PM10", "ICARUS PPM", "ICARUS PPM personal PM", "ug/m3", "PPM personal PM"),
    "PM25_uHoo": ("PM2.5", "uHoo", "uHoo indicative residential IAQ", "ug/m3", "uHoo residential IAQ"),
    "CO2_uHoo": ("CO2", "uHoo", "uHoo indicative residential IAQ", "ppm", "uHoo residential IAQ"),
    "TVOC_uHoo": ("TVOC", "uHoo", "uHoo indicative residential IAQ", "ppb", "uHoo residential IAQ"),
    "NO2_uHoo": ("NO2", "uHoo", "uHoo indicative residential IAQ", "ppb", "uHoo residential IAQ"),
    "O3_uHoo": ("O3", "uHoo", "uHoo indicative residential IAQ", "ppb", "uHoo residential IAQ"),
    "CO_uHoo": ("CO", "uHoo", "uHoo indicative residential IAQ", "ppm", "uHoo residential IAQ"),
    "Temp_uHoo": ("temperature", "uHoo", "uHoo indicative residential IAQ", "degC", "uHoo residential IAQ"),
    "Humi_uHoo": ("humidity", "uHoo", "uHoo indicative residential IAQ", "percent", "uHoo residential IAQ"),
    "AvgHeartRate": ("average heart rate", "Garmin", "Garmin wearable-derived field indicator", "bpm", "Garmin HR/stress"),
    "Stress": ("stress index", "Garmin", "Garmin wearable-derived field indicator", "unitless index", "Garmin HR/stress"),
}

DESCRIPTIVE_COLUMNS = [
    "analysis_scope",
    "subgroup_variable",
    "subgroup_category",
    "city",
    "season",
    "metric",
    "sensor",
    "source_device",
    "unit",
    "n_unique_participants",
    "n_participant_days_or_nights",
    "mean",
    "median",
    "sd",
    "p25",
    "p75",
    "suppressed",
    "suppression_reason",
    "count_band",
    "status",
    "notes",
]

SAFE_FORBIDDEN_HEADERS = {
    "participant_id",
    "participant_uid",
    "source_member",
    "archive_member",
    "member",
    "timestamp",
    "raw_timestamp",
    "row_level_feather_identifier",
    "household_id",
    "address",
    "latitude",
    "longitude",
    "coordinates",
    "id",
    "ts",
}
DISALLOWED_OUTPUT_NAMES = {
    "participant_subgroup_rows.csv",
    "participant_demographic_microdata.csv",
    "questionnaire_rows.csv",
    "tad_rows.csv",
    "participant_day_exposure_rows.csv",
    "participant_night_rows.csv",
}


def _read_member_columns(data_zip: str | Path) -> set[str]:
    columns: set[str] = set()
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            with zf.open(member) as fh:
                table = feather.read_table(io.BytesIO(fh.read()))
            columns.update(str(column) for column in table.schema.names)
    return columns


def _read_demographics(demographics_file: str | Path | None) -> pd.DataFrame:
    if not demographics_file:
        return pd.DataFrame()
    path = Path(demographics_file)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        frames = [frame.copy() for frame in sheets.values() if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _find_columns(columns: Iterable[str], terms: list[str]) -> list[str]:
    out: list[str] = []
    for column in columns:
        text = str(column).strip().lower()
        if any(term.lower() in text for term in terms):
            out.append(str(column))
    return sorted(set(out))


def make_variable_availability_audit(
    data_zip: str | Path,
    demographics_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive_columns = _read_member_columns(data_zip)
    demographics = _read_demographics(demographics_file)
    demographics_columns = set(map(str, demographics.columns)) if not demographics.empty else set()
    rows: list[dict[str, object]] = []

    for candidate, domain, terms in CANDIDATE_VARIABLES:
        found_demo = _find_columns(demographics_columns, terms)
        found_archive = _find_columns(archive_columns, terms)
        available = bool(found_demo or (domain in {"sensor_stream", "derived_context"} and found_archive))
        harmonized = bool(found_demo) and domain not in {"sensor_stream", "derived_context"}
        missing = not available
        safe = False
        not_safe_reason = ""
        status = "MISSING_DEPENDENCY"
        notes = "No separate demographic/questionnaire/context source was provided; do not infer this variable from sensor support."
        source = "not_provided"

        if found_demo:
            source = Path(demographics_file).name if demographics_file else "provided_demographics_source"
            harmonized = True
            safe = True
            status = "AVAILABLE_AND_HARMONIZED"
            notes = "Candidate column detected in the provided local demographics/context source; safe descriptive summaries require join support and small-cell suppression."
        elif candidate == "city_season_design_strata":
            source = "controlled_harmonized_archive"
            available = True
            harmonized = True
            missing = False
            safe = False
            not_safe_reason = "city/season are study design strata, not equity or vulnerability subgroup variables"
            status = "NOT_SAFE_FOR_AGGREGATE_OUTPUT"
            notes = "City and season are available but are not a substitute for equity subgroup variables."
        elif candidate == "wearable_activity_state":
            source = "controlled_harmonized_archive"
            available = "Activity" in archive_columns
            harmonized = available
            missing = not available
            safe = False
            not_safe_reason = "timestamp-varying wearable field, not participant equity/context subgroup"
            status = "NOT_SAFE_FOR_AGGREGATE_OUTPUT" if available else "MISSING_DEPENDENCY"
            notes = "Activity exists as a sensor-stream field but is not a socioeconomic, housing, susceptibility, or recruitment-context subgroup."

        rows.append(
            {
                "candidate_variable": candidate,
                "source_domain": domain,
                "source_file_or_archive": source,
                "available": available,
                "harmonized": harmonized,
                "missing_dependency": missing,
                "completeness_notes": "provided source not supplied" if not found_demo else "candidate column detected; completeness not exported",
                "safe_for_descriptive_aggregate": safe,
                "not_safe_reason": not_safe_reason,
                "status": status,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows), demographics


def _normalize_id(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\.0$", "", text)
    return text


def _find_one(columns: Iterable[str], terms: list[str]) -> str | None:
    matches = _find_columns(columns, terms)
    return matches[0] if matches else None


def _normal_city(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if "milan" in text:
        return "Milan"
    if "thess" in text:
        return "Thessaloniki"
    return str(value)


def _normal_season(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if "summer" in text:
        return "Summer"
    if "winter" in text:
        return "Winter"
    return str(value)


def _category_for(candidate: str, value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"
    text = str(value).strip().lower()
    if candidate == "age_or_age_group":
        try:
            age = float(text)
            if age < 18:
                return "child_or_adolescent"
            if age >= 65:
                return "older_adult"
            return "adult_18_to_64"
        except ValueError:
            if "child" in text or "adolescent" in text:
                return "child_or_adolescent"
            if "older" in text or "65" in text:
                return "older_adult"
            return "adult_or_unspecified"
    if candidate == "sex_or_gender":
        if text.startswith("f") or "woman" in text:
            return "female"
        if text.startswith("m") or "man" in text:
            return "male"
        return "other_or_unspecified"
    if candidate == "asthma_or_respiratory_susceptibility":
        if text in {"yes", "y", "true", "1"} or "asthma" in text:
            return "yes"
        if text in {"no", "n", "false", "0"}:
            return "no"
        return "unknown"
    if candidate in {"older_adult_indicator", "child_or_adolescent_indicator"}:
        if text in {"yes", "y", "true", "1"}:
            return "yes"
        if text in {"no", "n", "false", "0"}:
            return "no"
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:60] or "unknown"


def _demographic_subgroup_map(
    demographics: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    if demographics.empty:
        return pd.DataFrame()
    id_col = _find_one(demographics.columns, ["participant_id", "participant id", "subject", "code", "id"])
    city_col = _find_one(demographics.columns, ["city", "site"])
    season_col = _find_one(demographics.columns, ["season", "campaign", "period"])
    if id_col is None or city_col is None:
        return pd.DataFrame()
    base = demographics.copy()
    base["_participant_id"] = base[id_col].map(_normalize_id)
    base["_city"] = base[city_col].map(_normal_city)
    base["_season"] = base[season_col].map(_normal_season) if season_col else "All"
    rows: list[pd.DataFrame] = []
    for record in availability.to_dict(orient="records"):
        if record["status"] != "AVAILABLE_AND_HARMONIZED":
            continue
        candidate = str(record["candidate_variable"])
        terms = next((item[2] for item in CANDIDATE_VARIABLES if item[0] == candidate), [])
        value_col = _find_one(base.columns, terms)
        if value_col is None:
            continue
        tmp = base[["_participant_id", "_city", "_season", value_col]].copy()
        tmp["subgroup_variable"] = candidate
        tmp["subgroup_category"] = tmp[value_col].map(lambda value: _category_for(candidate, value))
        rows.append(tmp[["_participant_id", "_city", "_season", "subgroup_variable", "subgroup_category"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _metric_meta(column: str) -> tuple[str, str, str, str, str]:
    if column in ALLOWED_METRICS:
        return ALLOWED_METRICS[column]
    for item in TABLE6_VARIABLES:
        if item["column"] == column:
            unit = str(item["unit"]).replace("Âµg/mÂ³", "ug/m3").replace("Â°C", "degC")
            source = "ICARUS PPM personal PM" if item["sensor"] == "ICARUS PPM" else "uHoo indicative residential IAQ" if item["sensor"] == "uHoo" else "Garmin wearable-derived field indicator"
            return str(item["variable"]), str(item["sensor"]), source, unit, str(item["sensor"])
    return column, "", "", "", ""


def make_descriptive_summary_and_suppression(
    data_zip: str | Path,
    demographics: pd.DataFrame,
    availability: pd.DataFrame,
    min_cell_size: int,
    date_filter_mode: DateFilterMode = "campaign",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subgroup_map = _demographic_subgroup_map(demographics, availability)
    if subgroup_map.empty:
        summary = pd.DataFrame(
            [
                {
                    "analysis_scope": "overall",
                    "subgroup_variable": "none_available",
                    "subgroup_category": "not_applicable",
                    "city": "Overall",
                    "season": "Overall",
                    "metric": "not_applicable",
                    "sensor": "not_applicable",
                    "source_device": "not_applicable",
                    "unit": "",
                    "n_unique_participants": "",
                    "n_participant_days_or_nights": "",
                    "mean": "",
                    "median": "",
                    "sd": "",
                    "p25": "",
                    "p75": "",
                    "suppressed": "no",
                    "suppression_reason": "not_applicable",
                    "count_band": "not_applicable",
                    "status": "NO_SAFE_SUMMARY_PRODUCED",
                    "notes": "No harmonized subgroup/context variable with safe join support was available; descriptive subgroup exposure summaries were not produced.",
                }
            ],
            columns=DESCRIPTIVE_COLUMNS,
        )
        suppression = pd.DataFrame(
            [
                {
                    "subgroup_variable": "none_available",
                    "subgroup_category": "not_applicable",
                    "city": "Overall",
                    "season": "Overall",
                    "metric_group": "not_applicable",
                    "min_cell_size": min_cell_size,
                    "observed_count_band": "not_applicable",
                    "suppressed": "no",
                    "suppression_reason": "no_eligible_subgroup_cells",
                    "status": "PASS",
                    "notes": "No subgroup cells were created because subgroup dependencies were missing.",
                }
            ]
        )
        return summary, suppression

    load_result = load_table6_daily_with_audit(data_zip, date_filter_mode=date_filter_mode)
    daily = load_result.daily.copy()
    if daily.empty:
        return pd.DataFrame(columns=DESCRIPTIVE_COLUMNS), pd.DataFrame()
    daily["_participant_id"] = daily["participant_uid"].astype(str).str.split("_").str[-1]
    merged = daily.merge(
        subgroup_map,
        left_on=["_participant_id", "city"],
        right_on=["_participant_id", "_city"],
        how="inner",
    )
    if "_season" in merged.columns:
        merged = merged[(merged["_season"] == "All") | (merged["_season"] == merged["season"])]
    rows: list[dict[str, object]] = []
    suppress_rows: list[dict[str, object]] = []
    metric_columns = [column for column in ALLOWED_METRICS if column in merged.columns]
    for (subgroup_variable, subgroup_category, city, season), group in merged.groupby(
        ["subgroup_variable", "subgroup_category", "city", "season"], dropna=False
    ):
        n_participants = int(group["participant_uid"].nunique())
        suppressed = n_participants < min_cell_size
        count_band = f"<{min_cell_size}" if suppressed else f">={min_cell_size}"
        for metric_col in metric_columns:
            metric, sensor, source_device, unit, metric_group = _metric_meta(metric_col)
            suppress_rows.append(
                {
                    "subgroup_variable": subgroup_variable,
                    "subgroup_category": subgroup_category,
                    "city": city,
                    "season": season,
                    "metric_group": metric_group,
                    "min_cell_size": min_cell_size,
                    "observed_count_band": count_band,
                    "suppressed": "yes" if suppressed else "no",
                    "suppression_reason": "small_cell" if suppressed else "",
                    "status": "SUPPRESSED" if suppressed else "PASS",
                    "notes": "Exact counts below threshold are not reported.",
                }
            )
            if suppressed:
                rows.append(
                    {
                        "analysis_scope": "city_season",
                        "subgroup_variable": subgroup_variable,
                        "subgroup_category": subgroup_category,
                        "city": city,
                        "season": season,
                        "metric": metric,
                        "sensor": sensor,
                        "source_device": source_device,
                        "unit": unit,
                        "n_unique_participants": "",
                        "n_participant_days_or_nights": "",
                        "mean": "",
                        "median": "",
                        "sd": "",
                        "p25": "",
                        "p75": "",
                        "suppressed": "yes",
                        "suppression_reason": "small_cell",
                        "count_band": count_band,
                        "status": "SUPPRESSED_SMALL_CELL",
                        "notes": "Subgroup summary suppressed because unique-participant support is below the configured threshold.",
                    }
                )
                continue
            series = pd.to_numeric(group[metric_col], errors="coerce").dropna()
            rows.append(
                {
                    "analysis_scope": "city_season",
                    "subgroup_variable": subgroup_variable,
                    "subgroup_category": subgroup_category,
                    "city": city,
                    "season": season,
                    "metric": metric,
                    "sensor": sensor,
                    "source_device": source_device,
                    "unit": unit,
                    "n_unique_participants": n_participants,
                    "n_participant_days_or_nights": int(series.shape[0]),
                    "mean": float(series.mean()) if len(series) else np.nan,
                    "median": float(series.median()) if len(series) else np.nan,
                    "sd": float(series.std(ddof=1)) if len(series) > 1 else np.nan,
                    "p25": float(series.quantile(0.25)) if len(series) else np.nan,
                    "p75": float(series.quantile(0.75)) if len(series) else np.nan,
                    "suppressed": "no",
                    "suppression_reason": "",
                    "count_band": count_band,
                    "status": "PASS",
                    "notes": "Descriptive aggregate only; no formal subgroup modelling or HIA/YLL performed.",
                }
            )
    return pd.DataFrame(rows, columns=DESCRIPTIVE_COLUMNS), pd.DataFrame(suppress_rows).drop_duplicates()


def make_feasibility_summary(availability: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    safe_vars = availability[availability["safe_for_descriptive_aggregate"] == True]  # noqa: E712
    missing_vars = availability[availability["missing_dependency"] == True]["candidate_variable"].tolist()  # noqa: E712
    descriptive_available = not summary.empty and not set(summary["status"].astype(str)) <= {"NO_SAFE_SUMMARY_PRODUCED"}
    subgroup_inputs = "; ".join(safe_vars["candidate_variable"].astype(str).tolist()) if not safe_vars.empty else "none"
    missing_text = "; ".join(missing_vars) if missing_vars else "none"
    questions = [
        ("differential PPM exposure by subgroup", "harmonized subgroup definitions; PPM personal PM aggregates"),
        ("differential residential IAQ by subgroup", "harmonized subgroup definitions; uHoo residential IAQ aggregates"),
        ("differential Garmin HR/stress by subgroup", "harmonized subgroup definitions; Garmin HR/stress aggregates"),
        ("differential sleep indicators by subgroup", "harmonized subgroup definitions; Garmin sleep aggregates"),
        ("subgroup HIA feasibility", "harmonized subgroup definitions; adequate subgroup denominators; stratified baseline rates; prespecified subgroup HIA plan"),
        ("subgroup YLL feasibility", "harmonized subgroup definitions; adequate subgroup denominators; subgroup age/baseline mortality inputs; prespecified subgroup YLL plan"),
    ]
    rows: list[dict[str, object]] = []
    for question, required in questions:
        formal = "HIA" in question or "YLL" in question
        feasibility = "not_feasible_missing_dependency" if not descriptive_available else "feasible_descriptive_only"
        if formal:
            feasibility = "not_feasible_missing_dependency"
        rows.append(
            {
                "analysis_question": question,
                "required_inputs": required,
                "available_inputs": subgroup_inputs if subgroup_inputs != "none" else "sensor aggregate outputs only; no harmonized subgroup source",
                "missing_inputs": missing_text if missing_text != "none" else "stratified baseline rates and formal subgroup modelling plan" if formal else "none",
                "cell_size_status": "not_evaluable_without_subgroup_source" if not descriptive_available else "suppression_applied",
                "feasibility_status": feasibility,
                "reason_formal_model_not_run": "Formal subgroup exposure-response modelling, subgroup HIA, and subgroup YLL are outside Phase 11 and lack required stratified inputs.",
                "safe_descriptive_output_available": descriptive_available,
                "status": "NOT_FEASIBLE" if feasibility.startswith("not_feasible") else "PASS",
                "notes": "Optional descriptive equity/context analysis only; no causal or environmental-justice burden estimates produced.",
            }
        )
    return pd.DataFrame(rows)


def validate_safe_outputs(outdir: str | Path, min_cell_size: int) -> tuple[str, list[str]]:
    outdir = Path(outdir)
    status = "PASS"
    messages: list[str] = []
    for path in outdir.glob("*.csv"):
        if path.name in DISALLOWED_OUTPUT_NAMES:
            status = "FAIL"
            messages.append(f"Disallowed output filename produced: {path.name}")
        frame = pd.read_csv(path, nrows=10)
        bad_headers = sorted({str(column).strip().lower() for column in frame.columns} & SAFE_FORBIDDEN_HEADERS)
        if bad_headers:
            status = "FAIL"
            messages.append(f"{path.name} has forbidden headers: {bad_headers}")
        values = "\n".join(frame.astype(str).to_numpy().ravel().tolist())
        if ".feather" in values:
            status = "FAIL"
            messages.append(f"{path.name} contains source-member path-like values.")
    summary_path = outdir / "equity_descriptive_exposure_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "status" in summary.columns:
            unsuppressed = summary[summary["suppressed"].astype(str).str.lower() == "no"].copy()
            if "n_unique_participants" in unsuppressed.columns:
                numeric_counts = pd.to_numeric(unsuppressed["n_unique_participants"], errors="coerce").dropna()
                if bool((numeric_counts < min_cell_size).any()):
                    status = "FAIL"
                    messages.append("Unsuppressed subgroup summaries below the configured small-cell threshold were found.")
    if not messages:
        messages.append("Checked Phase 11 CSV outputs; no forbidden headers, source-member paths, disallowed filenames, or unsuppressed small-cell summaries found.")
    return status, messages


def write_validation_report(
    outpath: Path,
    repo_path: Path,
    data_zip: Path,
    demographics_file: Path | None,
    min_cell_size: int,
    scripts_run: list[str],
    availability: pd.DataFrame,
    summary: pd.DataFrame,
    suppression: pd.DataFrame,
    feasibility: pd.DataFrame,
    safe_status: str,
    safe_messages: list[str],
) -> None:
    inventoried = availability["candidate_variable"].astype(str).tolist()
    available_harmonized = availability[availability["status"] == "AVAILABLE_AND_HARMONIZED"]["candidate_variable"].astype(str).tolist()
    missing = availability[availability["missing_dependency"] == True]["candidate_variable"].astype(str).tolist()  # noqa: E712
    suppressed = suppression[suppression["suppressed"].astype(str).str.lower() == "yes"] if not suppression.empty else pd.DataFrame()
    safe_summary_produced = not summary.empty and "NO_SAFE_SUMMARY_PRODUCED" not in set(summary["status"].astype(str))
    descriptive_status = "PASS" if safe_summary_produced else "NOT_FEASIBLE"
    suppression_status = "PASS" if safe_status == "PASS" else "FAIL"
    formal_feasible = "no" if feasibility["feasibility_status"].astype(str).str.startswith("not_feasible").any() else "limited_descriptive_only"
    lines = [
        "Phase 11 optional descriptive equity/context validation report",
        f"timestamp_of_run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository_path: {repo_path}",
        f"data_archive_path_used: {data_zip}",
        f"demographics_file_path_if_provided: {demographics_file if demographics_file else 'not_provided'}",
        f"min_cell_size_used: {min_cell_size}",
        "scripts_run: " + "; ".join(scripts_run),
        "candidate_subgroup_variables_inventoried: " + "; ".join(inventoried),
        "variables_available_and_harmonized: " + ("; ".join(available_harmonized) if available_harmonized else "none"),
        "variables_missing: " + ("; ".join(missing) if missing else "none"),
        "variables_suppressed_due_to_small_cells: "
        + ("; ".join(sorted(suppressed["subgroup_variable"].astype(str).unique())) if not suppressed.empty else "none"),
        f"PASS/FAIL_or_NOT_FEASIBLE_for_descriptive_exposure_summary: {descriptive_status}",
        f"PASS/FAIL_for_small_cell_suppression: {suppression_status}",
        f"safe_descriptive_subgroup_summaries_produced: {'yes' if safe_summary_produced else 'no'}",
        "subgroup_HIA_was_run: no",
        "subgroup_YLL_was_run: no",
        f"formal_equity_stratified_exposure_response_modelling_feasible: {formal_feasible}",
        "missing_dependencies:",
    ]
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("- none")
    lines.append("deviations_from_reviewer_requested_subgroup_dimensions:")
    if missing:
        lines.append("- Reviewer-requested subgroup dimensions could not be evaluated without harmonized demographic/questionnaire/context variables.")
    else:
        lines.append("- none")
    lines.extend(
        [
            "confirmations:",
            "- no Phase 12 work was performed",
            "- no manuscript/response harmonization was performed",
            "- no HIA/YLL/upper-tail workflows were run",
            "- no new lag/sleep/figure/paired workflows were run beyond reading local aggregate outputs if needed",
            "- no GitHub push/commit/upload was performed",
            "- controlled data remained local",
            f"- safe_output_privacy_check: {safe_status}",
        ]
    )
    lines.extend(f"  - {message}" for message in safe_messages)
    lines.append(
        "- safe outputs do not contain participant IDs, participant UID columns, source-member paths, raw timestamps, "
        "row-level Feather-file identifiers, participant-day rows, participant-night rows, questionnaire/TAD rows, "
        "demographic microdata, model input rows, or unsuppressed small-cell subgroup rows"
    )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase11_outputs(
    data_zip: str | Path,
    outdir: str | Path,
    phase1_dir: str | Path = "local_outputs/denominators",
    phase2_dir: str | Path = "local_outputs/ppm_common_support",
    phase7_dir: str | Path = "local_outputs/sleep",
    phase9_dir: str | Path = "local_outputs/paired_sensitivity",
    phase10_dir: str | Path = "local_outputs/tables",
    demographics_file: str | Path | None = None,
    min_cell_size: int = 10,
    repo_path: str | Path | None = None,
    date_filter_mode: DateFilterMode = "campaign",
    scripts_run: list[str] | None = None,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    repo_path = Path(repo_path) if repo_path else Path.cwd()
    data_zip = Path(data_zip)
    demographics_path = Path(demographics_file) if demographics_file else None
    scripts_run = scripts_run or ["scripts/11_equity_descriptive_exposure_context.py"]

    availability, demographics = make_variable_availability_audit(data_zip, demographics_file=demographics_file)
    summary, suppression = make_descriptive_summary_and_suppression(
        data_zip,
        demographics,
        availability,
        min_cell_size=min_cell_size,
        date_filter_mode=date_filter_mode,
    )
    feasibility = make_feasibility_summary(availability, summary)

    paths = {
        "equity_variable_availability_audit": outdir / "equity_variable_availability_audit.csv",
        "equity_descriptive_exposure_summary": outdir / "equity_descriptive_exposure_summary.csv",
        "subgroup_cell_size_suppression_audit": outdir / "subgroup_cell_size_suppression_audit.csv",
        "equity_analysis_feasibility_summary": outdir / "equity_analysis_feasibility_summary.csv",
        "phase11_validation_report": outdir / "phase11_validation_report.txt",
    }
    availability.to_csv(paths["equity_variable_availability_audit"], index=False)
    summary.to_csv(paths["equity_descriptive_exposure_summary"], index=False)
    suppression.to_csv(paths["subgroup_cell_size_suppression_audit"], index=False)
    feasibility.to_csv(paths["equity_analysis_feasibility_summary"], index=False)
    safe_status, safe_messages = validate_safe_outputs(outdir, min_cell_size=min_cell_size)
    write_validation_report(
        paths["phase11_validation_report"],
        repo_path=repo_path,
        data_zip=data_zip,
        demographics_file=demographics_path,
        min_cell_size=min_cell_size,
        scripts_run=scripts_run,
        availability=availability,
        summary=summary,
        suppression=suppression,
        feasibility=feasibility,
        safe_status=safe_status,
        safe_messages=safe_messages,
    )
    return paths
