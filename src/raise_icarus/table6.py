"""Table 6 descriptive statistics and analytical sample summaries.

This module creates participant-day descriptive summaries for the manuscript's
Table 6 from the harmonized ICARUS feather archive. It deliberately reports
analysis-specific denominators because the PPM, Garmin, and uHoo streams have
different completeness.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .data import (
    DateFilterMode,
    _apply_date_filter,
    feather_members,
    parse_city_season,
    read_feather_member,
)


@dataclass(frozen=True)
class Table6LoadResult:
    """Container for Table 6 daily input and filtering audit."""

    daily: pd.DataFrame
    date_filter_audit: pd.DataFrame


TABLE6_VARIABLES: list[dict[str, str]] = [
    {
        "column": "AvgHeartRate",
        "variable": "Average heart rate",
        "sensor": "Garmin",
        "unit": "bpm",
        "interpretation": "wearable-derived field indicator",
    },
    {
        "column": "Stress",
        "variable": "Stress level",
        "sensor": "Garmin",
        "unit": "unitless index",
        "interpretation": "wearable-derived field indicator",
    },
    {
        "column": "PM1_PPM",
        "variable": "PM1",
        "sensor": "ICARUS PPM",
        "unit": "µg/m³",
        "interpretation": "personal PM exposure",
    },
    {
        "column": "PM25_PPM",
        "variable": "PM2.5",
        "sensor": "ICARUS PPM",
        "unit": "µg/m³",
        "interpretation": "personal PM exposure",
    },
    {
        "column": "PM10_PPM",
        "variable": "PM10",
        "sensor": "ICARUS PPM",
        "unit": "µg/m³",
        "interpretation": "personal PM exposure",
    },
    {
        "column": "Temp_uHoo",
        "variable": "Temperature",
        "sensor": "uHoo",
        "unit": "°C",
        "interpretation": "uncalibrated residential IAQ indicator",
    },
    {
        "column": "Humi_uHoo",
        "variable": "Relative humidity",
        "sensor": "uHoo",
        "unit": "%",
        "interpretation": "uncalibrated residential IAQ indicator",
    },
    {
        "column": "PM25_uHoo",
        "variable": "PM2.5",
        "sensor": "uHoo",
        "unit": "µg/m³",
        "interpretation": "uncalibrated residential IAQ indicator",
    },
    {
        "column": "TVOC_uHoo",
        "variable": "TVOC",
        "sensor": "uHoo",
        "unit": "ppb",
        "interpretation": "uncalibrated residential IAQ indicator",
    },
    {
        "column": "CO2_uHoo",
        "variable": "CO2",
        "sensor": "uHoo",
        "unit": "ppm",
        "interpretation": "uncalibrated residential IAQ indicator",
    },
    {
        "column": "CO_uHoo",
        "variable": "CO",
        "sensor": "uHoo",
        "unit": "ppm",
        "interpretation": "uncalibrated residential IAQ indicator",
    },
    {
        "column": "O3_uHoo",
        "variable": "O3",
        "sensor": "uHoo",
        "unit": "ppb",
        "interpretation": "uncalibrated residential IAQ indicator",
    },
    {
        "column": "NO2_uHoo",
        "variable": "NO2",
        "sensor": "uHoo",
        "unit": "ppb",
        "interpretation": "uncalibrated residential IAQ indicator",
    },
]


def _normalise_participant_id(series: pd.Series) -> pd.Series:
    """Convert participant IDs to compact strings without trailing .0."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def load_table6_daily_with_audit(
    data_zip: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> Table6LoadResult:
    """Load participant-day data used for Table 6.

    Rows are restricted to the configured city-season campaign windows by
    default. Duplicate participant-timestamp rows are averaged before daily
    aggregation so duplicated rows do not receive excess weight.
    """
    data_zip = Path(data_zip)
    value_columns = [item["column"] for item in TABLE6_VARIABLES]
    required_columns = {"TS", "ID"}
    records: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    with zipfile.ZipFile(data_zip) as zf:
        members = feather_members(data_zip)
        if not members:
            raise FileNotFoundError(f"No .feather files found in {data_zip}")

        for member in members:
            city, season = parse_city_season(member)
            df = read_feather_member(zf, member)

            missing_required = required_columns - set(df.columns)
            if missing_required:
                raise ValueError(f"{member} is missing required columns: {sorted(missing_required)}")

            available_value_columns = [col for col in value_columns if col in df.columns]
            tmp = df[["TS", "ID", *available_value_columns]].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp = tmp.dropna(subset=["TS", "ID"])

            for col in available_value_columns:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

            # Treat physiologically implausible Garmin heart-rate values as missing.
            if "AvgHeartRate" in tmp.columns:
                invalid_hr = (tmp["AvgHeartRate"] < 40) | (tmp["AvgHeartRate"] > 200)
                tmp.loc[invalid_hr, "AvgHeartRate"] = np.nan

            # Ensure all value columns exist, even if absent from a file.
            for col in value_columns:
                if col not in tmp.columns:
                    tmp[col] = np.nan

            tmp, audit = _apply_date_filter(tmp, city, season, date_filter_mode)
            audit["archive_member"] = member
            audit_rows.append(audit)

            if tmp.empty:
                continue

            tmp["city"] = city
            tmp["season"] = season
            tmp["participant_id"] = _normalise_participant_id(tmp["ID"])
            tmp["participant_uid"] = city + "_" + tmp["participant_id"]
            tmp["participant_season_uid"] = city + "_" + season + "_" + tmp["participant_id"]
            tmp["source_member"] = member

            # Avoid giving repeated timestamp rows excess weight.
            tmp = (
                tmp.groupby(
                    ["city", "season", "participant_uid", "participant_season_uid", "source_member", "TS"],
                    as_index=False,
                )[value_columns]
                .mean()
            )
            tmp["date"] = tmp["TS"].dt.date

            daily = (
                tmp.groupby(
                    ["city", "season", "participant_uid", "participant_season_uid", "source_member", "date"],
                    as_index=False,
                )[value_columns]
                .mean()
            )
            records.append(daily)

    if records:
        daily = pd.concat(records, ignore_index=True)
    else:
        daily = pd.DataFrame(
            columns=[
                "city",
                "season",
                "participant_uid",
                "participant_season_uid",
                "source_member",
                "date",
                *value_columns,
            ]
        )

    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        audit_df = audit_df[
            [
                "archive_member",
                "city",
                "season",
                "date_filter_mode",
                "campaign_window",
                "rows_before_filter",
                "rows_after_filter",
                "rows_removed_by_filter",
                "unique_dates_before_filter",
                "unique_dates_after_filter",
                "date_min_before_filter",
                "date_max_before_filter",
                "date_min_after_filter",
                "date_max_after_filter",
            ]
        ].sort_values(["city", "season", "archive_member"])

    return Table6LoadResult(daily=daily, date_filter_audit=audit_df)


def _summarize_group(group: pd.DataFrame, group_label: str, city: str, season: str) -> list[dict[str, object]]:
    """Summarize all Table 6 variables for one group."""
    rows: list[dict[str, object]] = []
    for order, meta in enumerate(TABLE6_VARIABLES, start=1):
        col = meta["column"]
        non_missing = group.loc[group[col].notna()].copy()
        series = non_missing[col]
        rows.append(
            {
                "group_label": group_label,
                "city": city,
                "season": season,
                "variable_order": order,
                "variable": meta["variable"],
                "sensor": meta["sensor"],
                "unit": meta["unit"],
                "interpretation": meta["interpretation"],
                "column": col,
                "participant_season_records": int(non_missing["participant_season_uid"].nunique()),
                "unique_participants": int(non_missing["participant_uid"].nunique()),
                "participant_days": int(series.shape[0]),
                "mean": float(series.mean()) if len(series) else np.nan,
                "median": float(series.median()) if len(series) else np.nan,
                "sd": float(series.std(ddof=1)) if len(series) > 1 else np.nan,
                "min": float(series.min()) if len(series) else np.nan,
                "max": float(series.max()) if len(series) else np.nan,
                "p25": float(series.quantile(0.25)) if len(series) else np.nan,
                "p75": float(series.quantile(0.75)) if len(series) else np.nan,
                "p95": float(series.quantile(0.95)) if len(series) else np.nan,
            }
        )
    return rows


def summarize_table6(daily: pd.DataFrame) -> pd.DataFrame:
    """Create long-form descriptive statistics for Table 6."""
    rows: list[dict[str, object]] = []
    group_order = {
        ("Milan", "Summer"): 1,
        ("Milan", "Winter"): 2,
        ("Thessaloniki", "Summer"): 3,
        ("Thessaloniki", "Winter"): 4,
    }

    for (city, season), group in daily.groupby(["city", "season"], sort=False):
        label = f"{city} {season}"
        rows.extend(_summarize_group(group, label, city, season))

    rows.extend(_summarize_group(daily, "Overall", "Overall", "Overall"))

    out = pd.DataFrame(rows)
    out["group_order"] = out.apply(
        lambda r: group_order.get((r["city"], r["season"]), 5),
        axis=1,
    )
    out = out.sort_values(["variable_order", "group_order"]).reset_index(drop=True)
    return out


def make_table6_sample_summary(stats_long: pd.DataFrame) -> pd.DataFrame:
    """Extract variable-specific analytical sample sizes from long stats."""
    cols = [
        "group_label",
        "city",
        "season",
        "variable_order",
        "variable",
        "sensor",
        "unit",
        "interpretation",
        "participant_season_records",
        "unique_participants",
        "participant_days",
    ]
    return stats_long[cols].copy()


def _format_stat(mean: float, median: float, sd: float) -> str:
    if pd.isna(mean):
        return "NA"
    return f"{mean:.2f}; {median:.2f}; {sd:.2f}" if not pd.isna(sd) else f"{mean:.2f}; {median:.2f}; NA"


def make_table6_manuscript_wide(stats_long: pd.DataFrame) -> pd.DataFrame:
    """Create a compact manuscript-ready wide table.

    Each city-season cell is formatted as `mean; median; SD`.
    Denominators are provided as participant-season records / participant-days.
    """
    rows = []
    groups = ["Milan Summer", "Milan Winter", "Thessaloniki Summer", "Thessaloniki Winter", "Overall"]

    for (order, variable, sensor, unit, interpretation), subset in stats_long.groupby(
        ["variable_order", "variable", "sensor", "unit", "interpretation"], sort=True
    ):
        row: dict[str, object] = {
            "Variable": variable,
            "Sensor": sensor,
            "Unit": unit,
            "Interpretation": interpretation,
        }
        for label in groups:
            g = subset.loc[subset["group_label"] == label]
            if g.empty:
                row[label] = "NA"
                row[f"{label} n"] = "0/0"
            else:
                rec = g.iloc[0]
                row[label] = _format_stat(rec["mean"], rec["median"], rec["sd"])
                row[f"{label} n"] = f"{int(rec['participant_season_records'])}/{int(rec['participant_days'])}"
        rows.append(row)

    return pd.DataFrame(rows)


def write_table6_outputs(
    daily: pd.DataFrame,
    date_filter_audit: pd.DataFrame,
    outdir: str | Path,
) -> dict[str, Path]:
    """Write Table 6 outputs and return paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stats_long = summarize_table6(daily)
    sample_summary = make_table6_sample_summary(stats_long)
    manuscript_wide = make_table6_manuscript_wide(stats_long)

    outputs = {
        "daily_input": outdir / "table6_daily_participant_day_input.csv",
        "date_filter_audit": outdir / "table6_date_filter_audit.csv",
        "descriptive_long": outdir / "table6_descriptive_statistics_long.csv",
        "sample_summary": outdir / "table6_analytical_sample_summary.csv",
        "manuscript_wide": outdir / "table6_manuscript_wide.csv",
    }

    daily.to_csv(outputs["daily_input"], index=False)
    date_filter_audit.to_csv(outputs["date_filter_audit"], index=False)
    stats_long.to_csv(outputs["descriptive_long"], index=False)
    sample_summary.to_csv(outputs["sample_summary"], index=False)
    manuscript_wide.to_csv(outputs["manuscript_wide"], index=False)
    return outputs
