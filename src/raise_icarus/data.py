"""Data loading utilities for the RAISE/ICARUS analysis workflow."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.feather as feather

PM_COLUMNS = ["PM25_PPM", "PM10_PPM"]
DateFilterMode = Literal["campaign", "none"]

# Inclusive date windows used to restrict each city-season folder to the relevant
# ICARUS monitoring period. These windows remove off-period tails in a few files
# that continue for months outside the intended campaign period.
DEFAULT_CAMPAIGN_DATE_WINDOWS: dict[tuple[str, str], tuple[str, str]] = {
    ("Milan", "Summer"): ("2019-06-11", "2019-07-01"),
    ("Milan", "Winter"): ("2019-01-09", "2019-02-17"),
    ("Thessaloniki", "Summer"): ("2019-06-01", "2019-07-29"),
    ("Thessaloniki", "Winter"): ("2018-12-16", "2019-01-24"),
}


@dataclass(frozen=True)
class DataLoadResult:
    """Container returned by data loaders when audit information is requested."""

    daily_pm: pd.DataFrame
    date_filter_audit: pd.DataFrame


def parse_city_season(zip_member_name: str) -> tuple[str, str]:
    """Infer city and season from a zipped feather file path.

    Expected archive folders are named like `MilanSummer/001.feather` or
    `ThessalonikiWinter/033.feather`.
    """
    folder = Path(zip_member_name).parts[0]
    match = re.fullmatch(r"(Milan|Thessaloniki)(Summer|Winter)", folder)
    if not match:
        raise ValueError(f"Cannot infer city/season from archive member: {zip_member_name}")
    return match.group(1), match.group(2)


def feather_members(data_zip: str | Path) -> list[str]:
    """Return feather file names inside a zip archive."""
    data_zip = Path(data_zip)
    with zipfile.ZipFile(data_zip) as zf:
        return sorted([name for name in zf.namelist() if name.endswith(".feather")])


def read_feather_member(zf: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    """Read a feather member from an open zip archive.

    The implementation uses pyarrow directly instead of pandas.read_feather to
    avoid pandas/pyarrow extension-type compatibility issues on some systems.
    """
    with zf.open(member_name) as fh:
        table = feather.read_table(io.BytesIO(fh.read()))
    return table.to_pandas()


def get_campaign_date_window(city: str, season: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return the inclusive campaign date window for a city-season pair."""
    start_end = DEFAULT_CAMPAIGN_DATE_WINDOWS.get((city, season))
    if start_end is None:
        return None, None
    start, end = start_end
    return pd.Timestamp(start), pd.Timestamp(end)


def _apply_date_filter(
    df: pd.DataFrame,
    city: str,
    season: str,
    mode: DateFilterMode,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply date filtering and return the filtered frame plus audit metadata."""
    before_rows = len(df)
    before_dates = df["TS"].dt.date.nunique() if before_rows else 0
    before_min = df["TS"].min().date() if before_rows else pd.NaT
    before_max = df["TS"].max().date() if before_rows else pd.NaT

    start, end = get_campaign_date_window(city, season)
    if mode == "campaign":
        if start is None or end is None:
            raise ValueError(f"No campaign date window configured for {city} / {season}")
        date_series = df["TS"].dt.date
        mask = (date_series >= start.date()) & (date_series <= end.date())
        out = df.loc[mask].copy()
        window_label = f"{start.date()} to {end.date()}"
    elif mode == "none":
        out = df.copy()
        window_label = "not applied"
    else:
        raise ValueError(f"Unsupported date filter mode: {mode}")

    after_rows = len(out)
    after_dates = out["TS"].dt.date.nunique() if after_rows else 0
    after_min = out["TS"].min().date() if after_rows else pd.NaT
    after_max = out["TS"].max().date() if after_rows else pd.NaT
    audit = {
        "city": city,
        "season": season,
        "date_filter_mode": mode,
        "campaign_window": window_label,
        "rows_before_filter": before_rows,
        "rows_after_filter": after_rows,
        "rows_removed_by_filter": before_rows - after_rows,
        "unique_dates_before_filter": before_dates,
        "unique_dates_after_filter": after_dates,
        "date_min_before_filter": before_min,
        "date_max_before_filter": before_max,
        "date_min_after_filter": after_min,
        "date_max_after_filter": after_max,
    }
    return out, audit


def load_daily_personal_pm_with_audit(
    data_zip: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> DataLoadResult:
    """Load daily personal PM means and date-filter audit information.

    The function keeps the data stream needed for HIA only: timestamp, ID, city,
    season, PM2.5 and PM10 from the ICARUS portable particulate monitor (PPM).
    Duplicate participant-timestamp rows are averaged before daily aggregation so
    repeated rows do not receive excess weight.

    By default, rows are restricted to predefined campaign date windows for each
    city-season pair. This removes off-period tails in files that continue for
    months outside the relevant seasonal monitoring period.
    """
    data_zip = Path(data_zip)
    records: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    with zipfile.ZipFile(data_zip) as zf:
        members = feather_members(data_zip)
        if not members:
            raise FileNotFoundError(f"No .feather files found in {data_zip}")

        for member in members:
            city, season = parse_city_season(member)
            df = read_feather_member(zf, member)
            missing = {"TS", "ID", *PM_COLUMNS} - set(df.columns)
            if missing:
                raise ValueError(f"{member} is missing required columns: {sorted(missing)}")

            tmp = df[["TS", "ID", *PM_COLUMNS]].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            for col in PM_COLUMNS:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

            tmp = tmp.dropna(subset=["TS", "ID"])
            tmp, audit = _apply_date_filter(tmp, city, season, date_filter_mode)
            audit["archive_member"] = member
            audit_rows.append(audit)

            if tmp.empty:
                continue

            tmp["city"] = city
            tmp["season"] = season
            tmp["participant_id"] = tmp["ID"].astype(str).str.replace(r"\.0$", "", regex=True)
            tmp["participant_uid"] = city + "_" + tmp["participant_id"]
            tmp["source_member"] = member

            # Avoid giving repeated timestamp rows excess weight.
            tmp = (
                tmp.groupby(["city", "season", "participant_uid", "source_member", "TS"], as_index=False)[PM_COLUMNS]
                .mean()
            )
            tmp["date"] = tmp["TS"].dt.date
            daily = (
                tmp.groupby(["city", "season", "participant_uid", "source_member", "date"], as_index=False)[PM_COLUMNS]
                .mean()
                .rename(columns={"PM25_PPM": "pm25", "PM10_PPM": "pm10"})
            )
            records.append(daily)

    if records:
        daily_pm = pd.concat(records, ignore_index=True)
        daily_pm = daily_pm.dropna(subset=["pm25", "pm10"], how="all")
    else:
        daily_pm = pd.DataFrame(
            columns=["city", "season", "participant_uid", "source_member", "date", "pm25", "pm10"]
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
    return DataLoadResult(daily_pm=daily_pm, date_filter_audit=audit_df)


def load_daily_personal_pm(
    data_zip: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> pd.DataFrame:
    """Load daily personal PM means from the harmonized zipped feather archive."""
    return load_daily_personal_pm_with_audit(data_zip, date_filter_mode=date_filter_mode).daily_pm


def summarize_hia_input_completeness(daily_pm: pd.DataFrame) -> pd.DataFrame:
    """Summarize participant and participant-day coverage for HIA input data."""
    rows = []
    for (city, season), group in daily_pm.groupby(["city", "season"]):
        days_per_participant = group.groupby("participant_uid")["date"].nunique()
        rows.append(
            {
                "city": city,
                "season": season,
                "date_min": group["date"].min(),
                "date_max": group["date"].max(),
                "participants_any_pm": group["participant_uid"].nunique(),
                "participant_days_any_pm": len(group),
                "median_days_per_participant": float(days_per_participant.median()) if len(days_per_participant) else 0,
                "max_days_per_participant": int(days_per_participant.max()) if len(days_per_participant) else 0,
                "participants_pm25": group.loc[group["pm25"].notna(), "participant_uid"].nunique(),
                "participant_days_pm25": int(group["pm25"].notna().sum()),
                "participants_pm10": group.loc[group["pm10"].notna(), "participant_uid"].nunique(),
                "participant_days_pm10": int(group["pm10"].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["city", "season"]).reset_index(drop=True)
