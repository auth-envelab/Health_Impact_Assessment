"""Reporting guideline outputs for the RAISE/ICARUS manuscript revision."""

from __future__ import annotations

import io
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import statsmodels.formula.api as smf

from raise_icarus.data import DateFilterMode, feather_members, parse_city_season
from raise_icarus.physiology import (
    LAG_MINUTES,
    OUTCOME_COLUMNS,
    PM_COLUMNS,
    add_lagged_pm_terms,
    load_physiology_data,
)


CITY_SEASON_ORDER: tuple[tuple[str, str], ...] = (
    ("Milan", "Summer"),
    ("Milan", "Winter"),
    ("Thessaloniki", "Summer"),
    ("Thessaloniki", "Winter"),
)

TABLE5_REFERENCE_COUNTS: dict[tuple[str, str], int] = {
    ("Milan", "Summer"): 63,
    ("Milan", "Winter"): 89,
    ("Thessaloniki", "Summer"): 38,
    ("Thessaloniki", "Winter"): 87,
}


def parse_count(value: object) -> int:
    """Parse integer counts that may contain commas."""
    if pd.isna(value):
        return 0
    text = str(value).replace(",", "").strip()
    if text == "":
        return 0
    return int(float(text))


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a simple Markdown table without optional dependencies."""
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(str(row[col]) for col in cols) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, divider, *body])


def summarize_archive_inventory(data_zip: str | Path) -> pd.DataFrame:
    """Count harmonized feather members by city-season."""
    rows = []
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            city, season = parse_city_season(member)
            with zf.open(member) as fh:
                table = feather.read_table(io.BytesIO(fh.read()), columns=["ID"])
            participant_ids = table.to_pandas()["ID"].dropna().astype(str).str.replace(r"\.0$", "", regex=True)
            rows.append(
                {
                    "city": city,
                    "season": season,
                    "source_member": member,
                    "participant_id": participant_ids.iloc[0] if not participant_ids.empty else Path(member).stem,
                }
            )
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        return pd.DataFrame(columns=["city", "season", "participant_season_records", "unique_participants"])
    return (
        inventory.groupby(["city", "season"], as_index=False)
        .agg(
            participant_season_records=("source_member", "nunique"),
            unique_participants=("participant_id", "nunique"),
        )
        .sort_values(["city", "season"])
    )


def load_supplementary_s1(repo_root: Path) -> pd.DataFrame:
    """Load existing Supplementary Table S1 participant-completeness audit."""
    path = repo_root / "outputs" / "harmonization_qc" / "supplementary_table_s1_participant_completeness.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def s1_to_long_counts(s1: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide Supplementary S1 audit to long stream rows."""
    if s1.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    mapping = {
        ("Milan", "Summer"): "Milan summer n",
        ("Milan", "Winter"): "Milan winter n",
        ("Thessaloniki", "Summer"): "Thessaloniki summer n",
        ("Thessaloniki", "Winter"): "Thessaloniki winter n",
    }
    paired_cols = {"Milan": "Milan paired n", "Thessaloniki": "Thessaloniki paired n"}
    for record in s1.to_dict(orient="records"):
        stream = record["Analysis/data stream"]
        participant_days = parse_count(record.get("Participant-days if applicable"))
        for (city, season), count_col in mapping.items():
            rows.append(
                {
                    "analysis_stream": stream,
                    "city": city,
                    "season": season,
                    "valid_participants": parse_count(record.get(count_col)),
                    "paired_summer_winter_participants": parse_count(record.get(paired_cols[city])),
                    "participant_days_or_rows_total": participant_days,
                    "primary_denominator": record.get("Primary denominator", ""),
                    "source_file": record.get("Source file", ""),
                    "notes": record.get("Notes", ""),
                    "count_status": "from_existing_supplementary_audit",
                }
            )
    return pd.DataFrame(rows)


def hia_daily_pm_counts(path: str | Path) -> pd.DataFrame:
    """Recompute HIA daily PM support from finalized daily input."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for city, season in CITY_SEASON_ORDER:
        group = df[(df["city"] == city) & (df["season"] == season)]
        season_ids = set(group["participant_uid"].dropna().astype(str))
        other_season = "Winter" if season == "Summer" else "Summer"
        other = df[(df["city"] == city) & (df["season"] == other_season)]
        paired = len(season_ids & set(other["participant_uid"].dropna().astype(str)))
        rows.append(
            {
                "analysis_stream": "HIA daily PM input",
                "city": city,
                "season": season,
                "valid_participants": len(season_ids),
                "paired_summer_winter_participants": paired,
                "participant_days_or_rows_total": len(df),
                "city_season_participant_days": len(group),
                "primary_denominator": "Campaign-window daily participant PM records used in the common-support scenario HIA",
                "source_file": str(path),
                "notes": "Recomputed from finalized HIA daily PM input; PM1, PM2.5, and PM10 retained separately.",
                "count_status": "recomputed_from_hia_daily_input",
            }
        )
    return pd.DataFrame(rows)


def make_participant_flow_counts(data_zip: str | Path, repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create participant flow/count rows and discrepancy audit."""
    s1 = load_supplementary_s1(repo_root)
    stream_counts = s1_to_long_counts(s1)
    hia_counts = hia_daily_pm_counts(repo_root / "outputs" / "hia_common_support_scenario" / "hia_daily_personal_pm_input.csv")
    if not hia_counts.empty:
        stream_counts = stream_counts[stream_counts["analysis_stream"] != "HIA daily PM input"]
        stream_counts = pd.concat([stream_counts, hia_counts], ignore_index=True)

    inventory = summarize_archive_inventory(data_zip)
    inventory_rows = []
    discrepancy_rows = []
    for _, row in inventory.iterrows():
        key = (row["city"], row["season"])
        expected = TABLE5_REFERENCE_COUNTS.get(key)
        observed = int(row["participant_season_records"])
        inventory_rows.append(
            {
                "analysis_stream": "Harmonized archive inventory",
                "city": row["city"],
                "season": row["season"],
                "valid_participants": int(row["unique_participants"]),
                "paired_summer_winter_participants": pd.NA,
                "participant_days_or_rows_total": int(inventory["participant_season_records"].sum()),
                "city_season_participant_days": pd.NA,
                "primary_denominator": "Harmonized feather participant-season files",
                "source_file": str(data_zip),
                "notes": "One harmonized feather member is treated as one participant-season monitoring record; not an independent participant group.",
                "count_status": "recomputed_from_data_zip",
            }
        )
        inventory_rows.append(
            {
                "analysis_stream": "Table 5 participant-season monitoring records reference",
                "city": row["city"],
                "season": row["season"],
                "valid_participants": expected,
                "paired_summer_winter_participants": pd.NA,
                "participant_days_or_rows_total": sum(TABLE5_REFERENCE_COUNTS.values()),
                "city_season_participant_days": pd.NA,
                "primary_denominator": "Manuscript/Table 5 participant-season monitoring records",
                "source_file": "reviewer-task-provided Table 5 target counts",
                "notes": "Reference count included for reconciliation; recruitment/dropout before monitoring is not reconstructable from the harmonized zip alone.",
                "count_status": "reference_not_recomputed_from_zip",
            }
        )
        if expected is not None and observed != expected:
            discrepancy_rows.append(
                {
                    "check": "Table 5 participant-season count versus harmonized archive inventory",
                    "city": row["city"],
                    "season": row["season"],
                    "expected_or_reference": expected,
                    "recomputed": observed,
                    "status": "DISCREPANCY",
                    "notes": "The harmonized archive contains 70 feather files for this city-season; Table 5 reference counts cannot be reconstructed from available monitoring files.",
                }
            )

    participant_flow = pd.concat([pd.DataFrame(inventory_rows), stream_counts], ignore_index=True)
    discrepancy_audit = pd.DataFrame(discrepancy_rows)
    return participant_flow, discrepancy_audit


def write_participant_flow_diagram(flow_counts: pd.DataFrame, discrepancy_audit: pd.DataFrame, outpath: str | Path) -> None:
    """Write a PNG participant-flow/completeness diagram."""
    outpath = Path(outpath)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis("off")

    def add_box(x: float, y: float, text: str, width: float = 0.25, height: float = 0.1) -> None:
        ax.add_patch(
            plt.Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                facecolor="#f7f7f7",
                edgecolor="#333333",
                linewidth=1,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=8, wrap=True)

    def stream_text(stream: str) -> str:
        rows = flow_counts[flow_counts["analysis_stream"] == stream]
        if rows.empty:
            return f"{stream}\nnot available"
        parts = [stream]
        for city in ["Milan", "Thessaloniki"]:
            c = rows[rows["city"] == city]
            if c.empty:
                continue
            summer = int(c.loc[c["season"] == "Summer", "valid_participants"].iloc[0])
            winter = int(c.loc[c["season"] == "Winter", "valid_participants"].iloc[0])
            paired = c["paired_summer_winter_participants"].dropna()
            paired_text = f", paired {int(paired.iloc[0])}" if not paired.empty else ""
            parts.append(f"{city}: S {summer}, W {winter}{paired_text}")
        total = parse_count(rows["participant_days_or_rows_total"].dropna().iloc[0]) if rows["participant_days_or_rows_total"].notna().any() else 0
        if total:
            parts.append(f"participant-days/rows: {total}")
        return "\n".join(parts)

    inventory = flow_counts[flow_counts["analysis_stream"] == "Harmonized archive inventory"]
    inv_total = parse_count(inventory["participant_days_or_rows_total"].iloc[0]) if not inventory.empty else 0
    add_box(0.5, 0.9, f"Harmonized archive inventory\n{inv_total} participant-season files\n70 per city-season", width=0.38)
    add_box(0.5, 0.74, "Recruitment/dropout before monitoring\nnot reconstructable from available harmonized files", width=0.42)

    streams = [
        "PPM personal PM analysis",
        "PPM personal PM common-support analysis",
        "uHoo residential IAQ analysis",
        "Garmin heart-rate/stress analysis",
        "Garmin sleep data availability",
        "Garmin sleep + residential IAQ complete-case model input",
        "HIA daily PM input",
    ]
    coords = [(0.18, 0.55), (0.5, 0.55), (0.82, 0.55), (0.18, 0.34), (0.5, 0.34), (0.82, 0.34), (0.5, 0.16)]
    for (x, y), stream in zip(coords, streams):
        add_box(x, y, stream_text(stream), width=0.29, height=0.17)
        ax.annotate("", xy=(x, y + 0.09), xytext=(0.5, 0.69), arrowprops={"arrowstyle": "->", "lw": 0.8})

    if not discrepancy_audit.empty:
        ax.text(
            0.5,
            0.03,
            "Note: Table 5 reference count (277 participant-season monitoring records) is included in CSV outputs but not reconstructable from the harmonized zip inventory.",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_strobe_checklist() -> pd.DataFrame:
    """Return a STROBE-style checklist with manuscript section locations."""
    rows = [
        ("1a", "Title and abstract", "Indicate the study design with a commonly used term.", "Addressed", "Title/Abstract", "Describe as observational panel study with repeated measures."),
        ("1b", "Title and abstract", "Provide an informative and balanced summary.", "Addressed", "Title/Abstract", "Summarize exposure monitoring, field indicators, HIA scenario framing, and limitations."),
        ("2", "Background/rationale", "Explain scientific background and rationale.", "Addressed", "Introduction", "Indoor/personal exposure and health-relevance rationale."),
        ("3", "Objectives", "State specific objectives and hypotheses.", "Addressed", "Introduction", "Revised objectives for descriptive exposure, physiology indicators, and scenario HIA."),
        ("4", "Study design", "Present key elements of study design early.", "Addressed", "Methods: Study design/sample/population", "Observational repeated-measures campaign design."),
        ("5", "Setting", "Describe setting, locations, dates, and follow-up.", "Addressed", "Methods: Study design/sample/population", "City-season monitoring windows and campaign dates."),
        ("6a", "Participants", "Give eligibility criteria and participant selection.", "Partly addressed", "Methods: Study design/sample/population; Supplementary Figure S1", "Monitoring-file support is reproducible; pre-monitoring recruitment/dropout is not reconstructable from harmonized data."),
        ("6b", "Participants", "For matched studies, give matching criteria.", "Not applicable", "Methods: Study design/sample/population", "Not a matched case-control or cohort matching design; paired season support is reported descriptively."),
        ("7", "Variables", "Clearly define outcomes, exposures, predictors, and confounders.", "Addressed", "Methods: Sensors and data collection; Methods: Statistical analysis", "PM fractions, Garmin field indicators, activity, city, season, hour, weekend."),
        ("8", "Data sources/measurement", "Give sources of data and measurement details.", "Addressed", "Methods: Sensors and data collection", "Harmonized sensor streams and no committed participant data."),
        ("9", "Bias", "Describe efforts to address potential sources of bias.", "Partly addressed", "Methods: Statistical analysis; Limitations", "Campaign windows, participant clustering, completeness tables, cautious interpretation."),
        ("10", "Study size", "Explain how study size was arrived at.", "Addressed", "Methods: Study design/sample/population; Limitations", "Secondary analysis of completed ICARUS campaign data; no formal power claim for post hoc physiological effects."),
        ("11", "Quantitative variables", "Explain handling of quantitative variables.", "Addressed", "Methods: Statistical analysis; Methods: HIA", "PM fractions separate; lagged PM per original unit/per 10 ug/m3; explicit counterfactuals for HIA."),
        ("12a", "Statistical methods", "Describe all statistical methods.", "Addressed", "Methods: Statistical analysis; Methods: HIA", "Lag-specific mixed models, HIA scenario calculations, sensitivity analyses."),
        ("12b", "Statistical methods", "Describe methods for confounding control.", "Addressed", "Methods: Statistical analysis", "City, season, hour, weekend, activity where available; participant random intercepts."),
        ("12c", "Statistical methods", "Explain how missing data were addressed.", "Addressed", "Methods: Statistical analysis; Supplementary Table S1", "Complete-case/support denominators by stream; no forward/backward filling for primary HIA or primary lag physiology estimates."),
        ("12d", "Statistical methods", "For cohort studies, address loss to follow-up.", "Partly addressed", "Supplementary Figure S1; Limitations", "Dropout before monitoring cannot be reconstructed from available harmonized files."),
        ("12e", "Statistical methods", "Describe sensitivity analyses.", "Addressed", "Methods: Statistical analysis; Methods: HIA; Supplementary Table S2", "Moving-average PM, interaction specifications, HIA upper-tail sensitivity."),
        ("13a", "Participants/results", "Report numbers at each stage.", "Addressed", "Results; Supplementary Figure S1; Supplementary Table S1", "Participant-flow/completeness outputs report stream-specific support."),
        ("13b", "Participants/results", "Give reasons for non-participation at each stage.", "Partly addressed", "Supplementary Figure S1; Limitations", "Sensor/data completeness reported; recruitment dropout unavailable."),
        ("13c", "Participants/results", "Consider use of a flow diagram.", "Addressed", "Supplementary Figure S1", "Generated participant_flow_diagram.png."),
        ("14a", "Descriptive data", "Give characteristics of study participants.", "Partly addressed", "Results; Supplementary Table S1", "Sensor/data support by city-season; no randomized or representative-sample claim."),
        ("14b", "Descriptive data", "Indicate number with missing data for each variable.", "Addressed", "Supplementary Table S1", "Participant-level and participant-day/model-row denominators."),
        ("14c", "Descriptive data", "Summarize follow-up time.", "Addressed", "Methods: Study design/sample/population; Results", "Campaign windows and participant-days/nights."),
        ("15", "Outcome data", "Report numbers of outcome events or summary measures.", "Addressed", "Results; Supplementary Table S3", "Garmin heart rate/stress model support and estimates."),
        ("16a", "Main results", "Give unadjusted and adjusted estimates with precision.", "Addressed", "Supplementary Table S3", "Unadjusted and adjusted physiology estimates with beta, 95% CI, p value, n rows, participants."),
        ("16b", "Main results", "Report category boundaries where continuous variables categorized.", "Not applicable", "Methods: Statistical analysis", "Primary PM terms are continuous and not categorized."),
        ("16c", "Main results", "Translate relative risks into absolute risk where relevant.", "Not applicable", "Methods: HIA", "Physiology models are field-indicator associations; HIA scenario outputs are separate."),
        ("17", "Other analyses", "Report other analyses done.", "Addressed", "Supplementary Table S2; Supplementary Table S3", "Sensitivity and interaction specifications are documented."),
        ("18", "Key results", "Summarize key results with objectives.", "Addressed", "Discussion", "Cautious interpretation of weak physiological associations and scenario HIA."),
        ("19", "Limitations", "Discuss limitations.", "Addressed", "Limitations", "Missingness, non-representative sample, repeated minute-level p values, sensor limitations."),
        ("20", "Interpretation", "Give cautious overall interpretation.", "Addressed", "Discussion; Limitations", "No overinterpretation of dense repeated observations."),
        ("21", "Generalizability", "Discuss generalizability.", "Addressed", "Limitations", "No randomized or population-representative claim."),
        ("22", "Funding", "Give funding and role of funders.", "Addressed", "Declarations/Funding/Data availability", "Funding/declarations section."),
        ("Data", "Data availability", "Describe data and code availability.", "Addressed", "Declarations/Funding/Data availability", "Harmonized participant data not committed; scripts accept --data-zip."),
    ]
    return pd.DataFrame(rows, columns=["item", "topic", "recommendation", "status", "manuscript_location", "notes"])


def write_strobe_markdown(checklist: pd.DataFrame, outpath: str | Path) -> None:
    """Write STROBE checklist markdown."""
    text = [
        "# STROBE Checklist For Revised RAISE/ICARUS Manuscript",
        "",
        "Locations use manuscript section names rather than page numbers.",
        "",
        markdown_table(checklist),
        "",
    ]
    Path(outpath).write_text("\n".join(text), encoding="utf-8")


def _deterministic_thin(data: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(data) <= max_rows:
        return data.copy()
    keys = pd.util.hash_pandas_object(data[["participant_uid", "timestamp"]], index=False)
    return data.assign(_sample_key=keys).nsmallest(max_rows, "_sample_key").drop(columns=["_sample_key"])


def _fit_unadjusted(data: pd.DataFrame, outcome_col: str, exposure_col: str, max_model_rows: int) -> dict[str, object]:
    needed = ["participant_uid", "timestamp", outcome_col, exposure_col]
    model_data = data.loc[:, needed].dropna().copy()
    model_data["participant_uid"] = model_data["participant_uid"].astype(str)
    model_data["pm_term"] = model_data[exposure_col] / 10.0
    model_data = _deterministic_thin(model_data, max_model_rows)
    metadata = {
        "n_rows": len(model_data),
        "n_participants": int(model_data["participant_uid"].nunique()) if not model_data.empty else 0,
    }
    if len(model_data) < 200 or metadata["n_participants"] < 5:
        return {
            **metadata,
            "estimate_per_10ug_m3": np.nan,
            "standard_error_per_10ug_m3": np.nan,
            "ci_lower_per_10ug_m3": np.nan,
            "ci_upper_per_10ug_m3": np.nan,
            "p_value": np.nan,
            "convergence_status": "not_fit",
            "notes": "Insufficient rows or participant groups.",
        }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = smf.mixedlm(f"{outcome_col} ~ pm_term", data=model_data, groups=model_data["participant_uid"], re_formula="1").fit(
                reml=False,
                method=["lbfgs", "cg"],
                maxiter=100,
                disp=False,
            )
            estimate = result.params.get("pm_term", np.nan)
            se = result.bse.get("pm_term", np.nan)
            p_value = result.pvalues.get("pm_term", np.nan)
            status = "converged" if getattr(result, "converged", False) else "failed"
            warning_text = " | ".join(str(item.message) for item in caught)
            return {
                **metadata,
                "estimate_per_10ug_m3": estimate,
                "standard_error_per_10ug_m3": se,
                "ci_lower_per_10ug_m3": estimate - 1.96 * se if pd.notna(se) else np.nan,
                "ci_upper_per_10ug_m3": estimate + 1.96 * se if pd.notna(se) else np.nan,
                "p_value": p_value,
                "convergence_status": status,
                "notes": warning_text,
            }
        except Exception as exc:  # pragma: no cover - real-data edge cases
            return {
                **metadata,
                "estimate_per_10ug_m3": np.nan,
                "standard_error_per_10ug_m3": np.nan,
                "ci_lower_per_10ug_m3": np.nan,
                "ci_upper_per_10ug_m3": np.nan,
                "p_value": np.nan,
                "convergence_status": "failed",
                "notes": f"{type(exc).__name__}: {exc}",
            }


def adjusted_from_physiology_results(physiology_results: str | Path) -> pd.DataFrame:
    """Convert adjusted physiology workflow output to Supplementary Table S3 schema."""
    df = pd.read_csv(physiology_results)
    df = df[(df["model_kind"] == "primary_lag_specific") & (df["coefficient"] == "pm_term")].copy()
    rows = []
    for row in df.to_dict(orient="records"):
        estimate10 = float(row["estimate_per_10ug_m3"])
        se10 = float(row["std_error"])
        rows.append(
            {
                "outcome": row["outcome"],
                "pollutant": row["pollutant"],
                "lag_minutes": parse_count(row["lag_minutes"]),
                "model_type": "adjusted",
                "beta": estimate10 / 10.0,
                "standard_error": se10 / 10.0,
                "ci_lower": float(row["ci_low_wald"]) / 10.0,
                "ci_upper": float(row["ci_high_wald"]) / 10.0,
                "p_value": row["p_value"],
                "n_rows": parse_count(row["n_rows_used"]),
                "n_participants": parse_count(row["n_participants_used"]),
                "convergence_status": "converged" if str(row["converged"]).lower() == "true" else "failed",
                "notes": "Adjusted lag-specific mixed-effects model; beta is per 1 ug/m3, beta_per_10ug_m3 also provided.",
                "beta_per_10ug_m3": estimate10,
                "standard_error_per_10ug_m3": se10,
                "ci_lower_per_10ug_m3": row["ci_low_wald"],
                "ci_upper_per_10ug_m3": row["ci_high_wald"],
            }
        )
    return pd.DataFrame(rows)


def fit_unadjusted_physiology_estimates(
    data_zip: str | Path,
    date_filter_mode: DateFilterMode,
    max_model_rows: int = 10_000,
) -> pd.DataFrame:
    """Fit unadjusted lag-specific physiology models for reporting."""
    load_result = load_physiology_data(data_zip, date_filter_mode=date_filter_mode)
    data = add_lagged_pm_terms(load_result.data)
    rows = []
    for outcome_label, outcome_col in OUTCOME_COLUMNS.items():
        for pollutant_label, pm_col in PM_COLUMNS.items():
            for lag in LAG_MINUTES:
                exposure_col = f"{pm_col}_lag_{lag}min"
                fit = _fit_unadjusted(data, outcome_col, exposure_col, max_model_rows)
                estimate10 = fit["estimate_per_10ug_m3"]
                se10 = fit["standard_error_per_10ug_m3"]
                rows.append(
                    {
                        "outcome": outcome_label,
                        "pollutant": pollutant_label,
                        "lag_minutes": lag,
                        "model_type": "unadjusted",
                        "beta": estimate10 / 10.0 if pd.notna(estimate10) else np.nan,
                        "standard_error": se10 / 10.0 if pd.notna(se10) else np.nan,
                        "ci_lower": fit["ci_lower_per_10ug_m3"] / 10.0 if pd.notna(fit["ci_lower_per_10ug_m3"]) else np.nan,
                        "ci_upper": fit["ci_upper_per_10ug_m3"] / 10.0 if pd.notna(fit["ci_upper_per_10ug_m3"]) else np.nan,
                        "p_value": fit["p_value"],
                        "n_rows": fit["n_rows"],
                        "n_participants": fit["n_participants"],
                        "convergence_status": fit["convergence_status"],
                        "notes": "Unadjusted mixed-effects model with participant random intercept; beta is per 1 ug/m3.",
                        "beta_per_10ug_m3": estimate10,
                        "standard_error_per_10ug_m3": se10,
                        "ci_lower_per_10ug_m3": fit["ci_lower_per_10ug_m3"],
                        "ci_upper_per_10ug_m3": fit["ci_upper_per_10ug_m3"],
                    }
                )
    return pd.DataFrame(rows)


def make_physiology_s3(
    data_zip: str | Path,
    date_filter_mode: DateFilterMode,
    physiology_results: str | Path,
    max_model_rows: int = 10_000,
) -> pd.DataFrame:
    """Combine unadjusted and adjusted physiology estimates for Supplementary Table S3."""
    adjusted = adjusted_from_physiology_results(physiology_results)
    unadjusted = fit_unadjusted_physiology_estimates(data_zip, date_filter_mode, max_model_rows=max_model_rows)
    combined = pd.concat([unadjusted, adjusted], ignore_index=True)
    return combined.sort_values(["outcome", "pollutant", "lag_minutes", "model_type"]).reset_index(drop=True)


def write_reporting_qc(
    outpath: str | Path,
    flow_counts: pd.DataFrame,
    discrepancies: pd.DataFrame,
    strobe: pd.DataFrame,
    physiology_s3: pd.DataFrame,
) -> None:
    """Write reporting QC markdown."""
    hia = flow_counts[flow_counts["analysis_stream"] == "HIA daily PM input"]
    lines = [
        "# Reporting Outputs QC",
        "",
        "These outputs are STROBE-style reporting aids for the revised observational repeated-measures manuscript.",
        "",
        "They do not imply a randomized or population-representative sample. Table 5-style counts are participant-season monitoring records, not independent participant groups.",
        "",
        "Primary HIA exposure inputs and primary lag-specific physiology estimates do not use forward/backward filling.",
        "",
        "Physiological outcomes are Garmin-derived field indicators, not clinical endpoints; p values from dense repeated minute-level observations should be interpreted with effect sizes, confidence intervals, and participant clustering.",
        "",
        "## HIA Daily PM Input Check",
        "",
        markdown_table(hia[["city", "season", "valid_participants", "paired_summer_winter_participants", "city_season_participant_days"]]) if not hia.empty else "_HIA daily PM input not found._",
        "",
        "## Discrepancies / Unreconstructable Counts",
        "",
        markdown_table(discrepancies) if not discrepancies.empty else "_No discrepancies flagged._",
        "",
        "Recruitment dropout before monitoring cannot be reconstructed from the harmonized monitoring archive alone.",
        "",
        "## Output Row Counts",
        "",
        f"- Participant-flow rows: {len(flow_counts)}",
        f"- STROBE checklist rows: {len(strobe)}",
        f"- Supplementary Table S3 rows: {len(physiology_s3)}",
        "",
    ]
    Path(outpath).write_text("\n".join(lines), encoding="utf-8")
