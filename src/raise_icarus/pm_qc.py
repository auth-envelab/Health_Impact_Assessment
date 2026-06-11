"""PM fraction hierarchy QA/QC utilities for ICARUS PPM data.

The ICARUS portable particulate monitor (PPM) reports PM1, PM2.5, and PM10.
Because these are cumulative particle-size fractions, the physically expected
ordering is PM1 <= PM2.5 <= PM10. This module audits violations of that ordering
at timestamp and participant-day levels without altering the primary analyses.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from .data import DateFilterMode, _apply_date_filter, feather_members, parse_city_season

RAW_PM_COLUMNS = ["PM1_PPM", "PM25_PPM", "PM10_PPM"]
PM_COLUMNS = ["pm1", "pm25", "pm10"]
GROUP_COLS = ["city", "season", "participant_uid", "participant_season_uid", "source_member", "date"]


def _read_pm_feather_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """Read only timestamp, ID and PPM fraction columns from a feather member."""
    columns = ["TS", "ID", *RAW_PM_COLUMNS]
    with zf.open(member) as fh:
        table = feather.read_table(io.BytesIO(fh.read()), columns=columns)
    return table.to_pandas()

@dataclass(frozen=True)
class PMQCAuditResult:
    timestamp_sample: pd.DataFrame
    timestamp_summary: pd.DataFrame
    daily_all_available: pd.DataFrame
    daily_complete_case: pd.DataFrame
    daily_monotonic_candidate: pd.DataFrame
    daily_summary: pd.DataFrame
    daily_violation_details: pd.DataFrame
    participant_audit_top100: pd.DataFrame
    sensitivity_means: pd.DataFrame
    complete_case_retention: pd.DataFrame
    date_filter_audit: pd.DataFrame


def _normalise_participant_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    complete = out[PM_COLUMNS].notna().all(axis=1)
    out["complete_pm_triplet"] = complete
    out["pm1_gt_pm25"] = complete & (out["pm1"] > out["pm25"])
    out["pm25_gt_pm10"] = complete & (out["pm25"] > out["pm10"])
    out["pm1_gt_pm10"] = complete & (out["pm1"] > out["pm10"])
    out["any_hierarchy_violation"] = out[["pm1_gt_pm25", "pm25_gt_pm10", "pm1_gt_pm10"]].any(axis=1)
    out["pm1_minus_pm25"] = np.where(out["pm1_gt_pm25"], out["pm1"] - out["pm25"], np.nan)
    out["pm25_minus_pm10"] = np.where(out["pm25_gt_pm10"], out["pm25"] - out["pm10"], np.nan)
    out["pm1_minus_pm10"] = np.where(out["pm1_gt_pm10"], out["pm1"] - out["pm10"], np.nan)
    out["pm25_gt_pm10_relative_to_pm10"] = np.where(
        out["pm25_gt_pm10"] & (out["pm10"] > 0),
        (out["pm25"] - out["pm10"]) / out["pm10"],
        np.nan,
    )
    out["pm25_gt_pm10_relative_to_pm25"] = np.where(
        out["pm25_gt_pm10"] & (out["pm25"] > 0),
        (out["pm25"] - out["pm10"]) / out["pm25"],
        np.nan,
    )
    out["pm25_gt_pm10_gt_1ug"] = out["pm25_gt_pm10"] & ((out["pm25"] - out["pm10"]) > 1.0)
    out["pm25_gt_pm10_gt_5ug"] = out["pm25_gt_pm10"] & ((out["pm25"] - out["pm10"]) > 5.0)
    out["pm25_gt_pm10_rel_gt_10pct"] = out["pm25_gt_pm10_relative_to_pm10"] > 0.10
    return out


def _empty_stat() -> dict[str, Any]:
    return {
        "total_records": 0,
        "complete_pm_triplet_records": 0,
        "pm25_gt_pm10_records": 0,
        "pm25_gt_pm10_gt_1ug_records": 0,
        "pm25_gt_pm10_gt_5ug_records": 0,
        "pm25_gt_pm10_rel_gt_10pct_records": 0,
        "pm1_gt_pm25_records": 0,
        "pm1_gt_pm10_records": 0,
        "any_hierarchy_violation_records": 0,
        "pm25_minus_pm10_values": [],
        "pm25_gt_pm10_relative_values": [],
        "pm1_minus_pm25_values": [],
        "pm1_minus_pm10_values": [],
        "participants_with_pm25_gt_pm10": set(),
    }


def _update_stat(stat: dict[str, Any], flagged: pd.DataFrame) -> None:
    complete = flagged.loc[flagged["complete_pm_triplet"]]
    pm25_gt_pm10 = complete.loc[complete["pm25_gt_pm10"]]
    pm1_gt_pm25 = complete.loc[complete["pm1_gt_pm25"]]
    pm1_gt_pm10 = complete.loc[complete["pm1_gt_pm10"]]
    stat["total_records"] += len(flagged)
    stat["complete_pm_triplet_records"] += len(complete)
    stat["pm25_gt_pm10_records"] += len(pm25_gt_pm10)
    stat["pm25_gt_pm10_gt_1ug_records"] += int(complete["pm25_gt_pm10_gt_1ug"].sum())
    stat["pm25_gt_pm10_gt_5ug_records"] += int(complete["pm25_gt_pm10_gt_5ug"].sum())
    stat["pm25_gt_pm10_rel_gt_10pct_records"] += int(complete["pm25_gt_pm10_rel_gt_10pct"].sum())
    stat["pm1_gt_pm25_records"] += len(pm1_gt_pm25)
    stat["pm1_gt_pm10_records"] += len(pm1_gt_pm10)
    stat["any_hierarchy_violation_records"] += int(complete["any_hierarchy_violation"].sum())
    stat["pm25_minus_pm10_values"].extend(pm25_gt_pm10["pm25_minus_pm10"].dropna().astype(float).tolist())
    stat["pm25_gt_pm10_relative_values"].extend(pm25_gt_pm10["pm25_gt_pm10_relative_to_pm10"].dropna().astype(float).tolist())
    stat["pm1_minus_pm25_values"].extend(pm1_gt_pm25["pm1_minus_pm25"].dropna().astype(float).tolist())
    stat["pm1_minus_pm10_values"].extend(pm1_gt_pm10["pm1_minus_pm10"].dropna().astype(float).tolist())
    stat["participants_with_pm25_gt_pm10"].update(pm25_gt_pm10["participant_uid"].unique().tolist())


def _pct(num: int | float, den: int | float) -> float:
    return float(num / den * 100) if den else np.nan


def _quant(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else np.nan


def _stats_to_frame(stats_by_group: dict[tuple[str, str], dict[str, Any]], level: str, dataset: str) -> pd.DataFrame:
    rows = []
    for (city, season), stat in sorted(stats_by_group.items()):
        complete = stat["complete_pm_triplet_records"]
        rows.append(
            {
                "level": level,
                "dataset": dataset,
                "city": city,
                "season": season,
                "total_records": stat["total_records"],
                "complete_pm_triplet_records": complete,
                "complete_pm_triplet_percent": _pct(complete, stat["total_records"]),
                "pm25_gt_pm10_records": stat["pm25_gt_pm10_records"],
                "pm25_gt_pm10_percent_complete": _pct(stat["pm25_gt_pm10_records"], complete),
                "pm25_gt_pm10_gt_1ug_records": stat["pm25_gt_pm10_gt_1ug_records"],
                "pm25_gt_pm10_gt_5ug_records": stat["pm25_gt_pm10_gt_5ug_records"],
                "pm25_gt_pm10_rel_gt_10pct_records": stat["pm25_gt_pm10_rel_gt_10pct_records"],
                "pm25_minus_pm10_median_if_inverted": _quant(stat["pm25_minus_pm10_values"], 0.50),
                "pm25_minus_pm10_p95_if_inverted": _quant(stat["pm25_minus_pm10_values"], 0.95),
                "pm25_minus_pm10_max_if_inverted": max(stat["pm25_minus_pm10_values"]) if stat["pm25_minus_pm10_values"] else np.nan,
                "pm25_gt_pm10_relative_median_if_inverted": _quant(stat["pm25_gt_pm10_relative_values"], 0.50),
                "pm1_gt_pm25_records": stat["pm1_gt_pm25_records"],
                "pm1_gt_pm25_percent_complete": _pct(stat["pm1_gt_pm25_records"], complete),
                "pm1_minus_pm25_max_if_inverted": max(stat["pm1_minus_pm25_values"]) if stat["pm1_minus_pm25_values"] else np.nan,
                "pm1_gt_pm10_records": stat["pm1_gt_pm10_records"],
                "pm1_gt_pm10_percent_complete": _pct(stat["pm1_gt_pm10_records"], complete),
                "pm1_minus_pm10_max_if_inverted": max(stat["pm1_minus_pm10_values"]) if stat["pm1_minus_pm10_values"] else np.nan,
                "any_hierarchy_violation_records": stat["any_hierarchy_violation_records"],
                "any_hierarchy_violation_percent_complete": _pct(stat["any_hierarchy_violation_records"], complete),
                "participants_with_pm25_gt_pm10": len(stat["participants_with_pm25_gt_pm10"]),
            }
        )
    return pd.DataFrame(rows)


def _summarize_daily_dataset(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    stats: dict[tuple[str, str], dict[str, Any]] = {("Overall", "Overall"): _empty_stat()}
    for (city, season), group in df.groupby(["city", "season"], sort=True):
        stats.setdefault((city, season), _empty_stat())
        _update_stat(stats[(city, season)], group)
        _update_stat(stats[("Overall", "Overall")], group)
    return _stats_to_frame(stats, "participant_day", dataset_label)


def _make_daily_violation_details(daily: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    cols = [
        "city", "season", "participant_uid", "participant_season_uid", "source_member", "date",
        "pm1", "pm25", "pm10", "pm1_gt_pm25", "pm25_gt_pm10", "pm1_gt_pm10",
        "pm1_minus_pm25", "pm25_minus_pm10", "pm1_minus_pm10",
        "pm25_gt_pm10_relative_to_pm10", "pm25_gt_pm10_relative_to_pm25",
        "pm25_gt_pm10_gt_1ug", "pm25_gt_pm10_gt_5ug", "pm25_gt_pm10_rel_gt_10pct",
    ]
    out = daily.loc[daily["any_hierarchy_violation"], cols].copy()
    out.insert(0, "daily_dataset", dataset_label)
    return out.sort_values(["city", "season", "participant_uid", "date"]).reset_index(drop=True)


def _make_participant_audit(df: pd.DataFrame, level: str, top_n: int = 100) -> pd.DataFrame:
    rows = []
    for keys, group in df.loc[df["complete_pm_triplet"]].groupby(
        ["city", "season", "participant_uid", "participant_season_uid", "source_member"]
    ):
        city, season, uid, ps_uid, source = keys
        inv = group.loc[group["pm25_gt_pm10"]]
        rows.append(
            {
                "level": level,
                "city": city,
                "season": season,
                "participant_uid": uid,
                "participant_season_uid": ps_uid,
                "source_member": source,
                "complete_pm_triplet_records": int(len(group)),
                "pm25_gt_pm10_records": int(len(inv)),
                "pm25_gt_pm10_percent": _pct(len(inv), len(group)),
                "pm25_minus_pm10_median_if_inverted": float(inv["pm25_minus_pm10"].median()) if len(inv) else np.nan,
                "pm25_minus_pm10_p95_if_inverted": float(inv["pm25_minus_pm10"].quantile(0.95)) if len(inv) else np.nan,
                "pm25_minus_pm10_max_if_inverted": float(inv["pm25_minus_pm10"].max()) if len(inv) else np.nan,
                "pm25_gt_pm10_gt_5ug_records": int(group["pm25_gt_pm10_gt_5ug"].sum()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["pm25_gt_pm10_records", "pm25_gt_pm10_percent", "pm25_minus_pm10_max_if_inverted"],
        ascending=[False, False, False],
    ).head(top_n)


def _make_sensitivity_means(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, df in datasets.items():
        groups = [("Overall", "Overall", df)]
        groups.extend([(city, season, group) for (city, season), group in df.groupby(["city", "season"], sort=True)])
        for city, season, group in groups:
            row = {
                "dataset": label,
                "city": city,
                "season": season,
                "participant_days": int(len(group)),
                "participants": int(group["participant_uid"].nunique()) if "participant_uid" in group else 0,
            }
            for col in PM_COLUMNS:
                row[f"{col}_mean"] = float(group[col].mean()) if len(group[col].dropna()) else np.nan
                row[f"{col}_median"] = float(group[col].median()) if len(group[col].dropna()) else np.nan
                row[f"{col}_sd"] = float(group[col].std()) if len(group[col].dropna()) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _make_retention_summary(primary: pd.DataFrame, complete_case: pd.DataFrame) -> pd.DataFrame:
    keys = ["city", "season", "participant_uid", "source_member", "date"]
    p = primary[keys].copy()
    p["in_primary"] = 1
    c = complete_case[keys].copy()
    c["in_complete_case"] = 1
    merged = p.merge(c, on=keys, how="left")
    merged["in_complete_case"] = merged["in_complete_case"].fillna(0).astype(int)
    rows = []
    groups = [("Overall", "Overall", merged)]
    groups.extend([(city, season, group) for (city, season), group in merged.groupby(["city", "season"], sort=True)])
    for city, season, group in groups:
        rows.append(
            {
                "city": city,
                "season": season,
                "primary_participant_days": int(len(group)),
                "complete_case_participant_days": int(group["in_complete_case"].sum()),
                "retention_percent": _pct(int(group["in_complete_case"].sum()), len(group)),
            }
        )
    return pd.DataFrame(rows)


def run_pm_qc_audit(data_zip: str | Path, date_filter_mode: DateFilterMode = "campaign") -> PMQCAuditResult:
    """Run the full PM fraction hierarchy audit."""
    data_zip = Path(data_zip)
    timestamp_stats: dict[tuple[str, str], dict[str, Any]] = {("Overall", "Overall"): _empty_stat()}
    timestamp_sample_parts: list[pd.DataFrame] = []
    participant_timestamp_rows: list[pd.DataFrame] = []
    daily_all_parts: list[pd.DataFrame] = []
    daily_cc_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            city, season = parse_city_season(member)
            tmp = _read_pm_feather_member(zf, member)
            missing = {"TS", "ID", *RAW_PM_COLUMNS} - set(tmp.columns)
            if missing:
                raise ValueError(f"{member} is missing PM QC columns: {sorted(missing)}")

            tmp = tmp[["TS", "ID", *RAW_PM_COLUMNS]].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp = tmp.dropna(subset=["TS", "ID"])
            for col in RAW_PM_COLUMNS:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            tmp, audit = _apply_date_filter(tmp, city, season, date_filter_mode)
            audit["archive_member"] = member
            audit_rows.append(audit)
            if tmp.empty:
                continue

            tmp = tmp.rename(columns={"PM1_PPM": "pm1", "PM25_PPM": "pm25", "PM10_PPM": "pm10"})
            tmp["city"] = city
            tmp["season"] = season
            tmp["participant_id"] = _normalise_participant_id(tmp["ID"])
            tmp["participant_uid"] = city + "_" + tmp["participant_id"]
            tmp["participant_season_uid"] = city + "_" + season + "_" + tmp["participant_id"]
            tmp["source_member"] = member
            tmp = tmp.groupby(
                ["city", "season", "participant_uid", "participant_season_uid", "source_member", "TS"],
                as_index=False,
            )[PM_COLUMNS].mean()
            tmp["date"] = tmp["TS"].dt.date
            tmp = _add_flags(tmp)

            timestamp_stats.setdefault((city, season), _empty_stat())
            _update_stat(timestamp_stats[(city, season)], tmp)
            _update_stat(timestamp_stats[("Overall", "Overall")], tmp)
            if sum(len(x) for x in timestamp_sample_parts) < 10000:
                needed = 10000 - sum(len(x) for x in timestamp_sample_parts)
                timestamp_sample_parts.append(tmp.head(needed))

            participant_timestamp_rows.append(_make_participant_audit(tmp, "timestamp_deduplicated", top_n=1_000_000))

            daily_all = tmp.groupby(GROUP_COLS, as_index=False)[PM_COLUMNS].mean()
            daily_all = _add_flags(daily_all)
            daily_all_parts.append(daily_all)

            complete_ts = tmp.loc[tmp["complete_pm_triplet"]]
            daily_cc = complete_ts.groupby(GROUP_COLS, as_index=False)[PM_COLUMNS].mean()
            daily_cc = _add_flags(daily_cc)
            daily_cc_parts.append(daily_cc)

    timestamp_summary = _stats_to_frame(timestamp_stats, "timestamp_deduplicated", "timestamp_deduplicated")
    timestamp_sample = pd.concat(timestamp_sample_parts, ignore_index=True) if timestamp_sample_parts else pd.DataFrame()
    daily_all = pd.concat(daily_all_parts, ignore_index=True) if daily_all_parts else pd.DataFrame()
    daily_cc = pd.concat(daily_cc_parts, ignore_index=True) if daily_cc_parts else pd.DataFrame()

    daily_mono = daily_all.copy()
    daily_mono = daily_mono.rename(columns={"pm1": "pm1_original", "pm25": "pm25_original", "pm10": "pm10_original"})
    daily_mono["pm1"] = daily_mono["pm1_original"]
    daily_mono["pm25"] = np.fmax(daily_mono["pm25_original"], daily_mono["pm1_original"])
    daily_mono["pm10"] = np.fmax(daily_mono["pm10_original"], daily_mono["pm25"])
    daily_mono["pm25_adjustment_ug_m3"] = daily_mono["pm25"] - daily_mono["pm25_original"]
    daily_mono["pm10_adjustment_ug_m3"] = daily_mono["pm10"] - daily_mono["pm10_original"]
    daily_mono = _add_flags(daily_mono)

    daily_summary = pd.concat(
        [
            _summarize_daily_dataset(daily_all, "all_available_fraction_means"),
            _summarize_daily_dataset(daily_cc, "complete_case_timestamp_means"),
            _summarize_daily_dataset(daily_mono, "monotonic_sensitivity_candidate"),
        ],
        ignore_index=True,
    )
    daily_violation_details = pd.concat(
        [
            _make_daily_violation_details(daily_all, "all_available_fraction_means"),
            _make_daily_violation_details(daily_cc, "complete_case_timestamp_means"),
        ],
        ignore_index=True,
    )
    participant_audit = pd.concat(
        [
            pd.concat(participant_timestamp_rows, ignore_index=True) if participant_timestamp_rows else pd.DataFrame(),
            _make_participant_audit(daily_all, "participant_day_all_available"),
            _make_participant_audit(daily_cc, "participant_day_complete_case"),
        ],
        ignore_index=True,
    )
    if not participant_audit.empty:
        participant_audit = participant_audit.sort_values(
            ["pm25_gt_pm10_records", "pm25_gt_pm10_percent", "pm25_minus_pm10_max_if_inverted"],
            ascending=[False, False, False],
        ).head(100)

    sensitivity_means = _make_sensitivity_means(
        {
            "all_available_fraction_means": daily_all,
            "complete_case_timestamp_means": daily_cc,
            "monotonic_sensitivity_candidate": daily_mono,
        }
    )
    retention = _make_retention_summary(daily_all, daily_cc)
    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        audit_df = audit_df[
            [
                "archive_member", "city", "season", "date_filter_mode", "campaign_window",
                "rows_before_filter", "rows_after_filter", "rows_removed_by_filter",
                "unique_dates_before_filter", "unique_dates_after_filter",
                "date_min_before_filter", "date_max_before_filter",
                "date_min_after_filter", "date_max_after_filter",
            ]
        ].sort_values(["city", "season", "archive_member"])

    return PMQCAuditResult(
        timestamp_sample=timestamp_sample,
        timestamp_summary=timestamp_summary,
        daily_all_available=daily_all,
        daily_complete_case=daily_cc,
        daily_monotonic_candidate=daily_mono,
        daily_summary=daily_summary,
        daily_violation_details=daily_violation_details,
        participant_audit_top100=participant_audit,
        sensitivity_means=sensitivity_means,
        complete_case_retention=retention,
        date_filter_audit=audit_df,
    )


def write_outputs(result: PMQCAuditResult, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "timestamp_summary": outdir / "pm_fraction_qc_timestamp_summary.csv",
        "daily_summary": outdir / "pm_fraction_qc_daily_summary.csv",
        "daily_all_available": outdir / "pm_fraction_qc_daily_all_available.csv",
        "daily_complete_case": outdir / "pm_fraction_qc_daily_complete_case.csv",
        "daily_monotonic_candidate": outdir / "pm_fraction_qc_daily_monotonic_sensitivity_candidate.csv",
        "daily_violation_details": outdir / "pm_fraction_qc_daily_violation_details.csv",
        "participant_audit_top100": outdir / "pm_fraction_qc_participant_audit_top100.csv",
        "sensitivity_means": outdir / "pm_fraction_qc_sensitivity_means.csv",
        "complete_case_retention": outdir / "pm_fraction_qc_complete_case_retention.csv",
        "timestamp_input_sample": outdir / "pm_fraction_qc_timestamp_input_sample.csv",
        "date_filter_audit": outdir / "pm_fraction_qc_date_filter_audit.csv",
    }
    result.timestamp_summary.to_csv(outputs["timestamp_summary"], index=False)
    result.daily_summary.to_csv(outputs["daily_summary"], index=False)
    result.daily_all_available.to_csv(outputs["daily_all_available"], index=False)
    result.daily_complete_case.to_csv(outputs["daily_complete_case"], index=False)
    result.daily_monotonic_candidate.to_csv(outputs["daily_monotonic_candidate"], index=False)
    result.daily_violation_details.to_csv(outputs["daily_violation_details"], index=False)
    result.participant_audit_top100.to_csv(outputs["participant_audit_top100"], index=False)
    result.sensitivity_means.to_csv(outputs["sensitivity_means"], index=False)
    result.complete_case_retention.to_csv(outputs["complete_case_retention"], index=False)
    result.timestamp_sample.to_csv(outputs["timestamp_input_sample"], index=False)
    result.date_filter_audit.to_csv(outputs["date_filter_audit"], index=False)
    return outputs
