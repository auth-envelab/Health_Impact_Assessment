"""Phase 1 denominator and STROBE-support reproducibility helpers.

The helpers in this module intentionally write aggregate-only outputs. They use
participant identifiers in memory only to count unique and paired seasonal
support; no participant-level rows are exported.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from raise_icarus.data import (
    DateFilterMode,
    feather_members,
    get_campaign_date_window,
    parse_city_season,
)
from raise_icarus.table6 import load_table6_daily_with_audit, make_table6_sample_summary, summarize_table6


CITY_SEASON_ORDER: tuple[tuple[str, str], ...] = (
    ("Milan", "Summer"),
    ("Milan", "Winter"),
    ("Thessaloniki", "Summer"),
    ("Thessaloniki", "Winter"),
)

S1_STREAM_ORDER: tuple[str, ...] = (
    "PPM personal PM analysis",
    "PPM personal PM common-support analysis",
    "uHoo residential IAQ analysis",
    "Garmin heart-rate/stress analysis",
    "Garmin sleep data availability",
    "Garmin sleep + residential IAQ complete-case model input",
    "HIA daily PM input",
)

PM_COLUMNS: tuple[str, ...] = ("PM1_PPM", "PM25_PPM", "PM10_PPM")
UHOO_COLUMNS: tuple[str, ...] = (
    "Temp_uHoo",
    "Humi_uHoo",
    "PM25_uHoo",
    "TVOC_uHoo",
    "CO2_uHoo",
    "CO_uHoo",
    "O3_uHoo",
    "NO2_uHoo",
)
GARMIN_COLUMNS: tuple[str, ...] = ("AvgHeartRate", "Stress")

TABLE5_TARGETS: dict[tuple[str, str], int] = {
    ("Milan", "Summer"): 63,
    ("Milan", "Winter"): 89,
    ("Thessaloniki", "Summer"): 38,
    ("Thessaloniki", "Winter"): 87,
}
TABLE5_OVERALL_TARGET = 277

S1_TARGETS: dict[str, dict[str, int]] = {
    "PPM personal PM analysis": {
        "Milan summer n": 54,
        "Milan winter n": 45,
        "Milan paired n": 32,
        "Thessaloniki summer n": 51,
        "Thessaloniki winter n": 57,
        "Thessaloniki paired n": 34,
        "Participant-days/nights/rows": 2431,
    },
    "PPM personal PM common-support analysis": {
        "Milan summer n": 54,
        "Milan winter n": 45,
        "Milan paired n": 32,
        "Thessaloniki summer n": 51,
        "Thessaloniki winter n": 57,
        "Thessaloniki paired n": 34,
        "Participant-days/nights/rows": 2427,
    },
    "uHoo residential IAQ analysis": {
        "Milan summer n": 28,
        "Milan winter n": 38,
        "Milan paired n": 25,
        "Thessaloniki summer n": 21,
        "Thessaloniki winter n": 18,
        "Thessaloniki paired n": 16,
        "Participant-days/nights/rows": 1710,
    },
    "Garmin heart-rate/stress analysis": {
        "Milan summer n": 55,
        "Milan winter n": 59,
        "Milan paired n": 37,
        "Thessaloniki summer n": 58,
        "Thessaloniki winter n": 40,
        "Thessaloniki paired n": 25,
        "Participant-days/nights/rows": 2488,
    },
    "Garmin sleep data availability": {
        "Milan summer n": 44,
        "Milan winter n": 50,
        "Milan paired n": 26,
        "Thessaloniki summer n": 55,
        "Thessaloniki winter n": 33,
        "Thessaloniki paired n": 20,
        "Participant-days/nights/rows": 1090,
    },
    "Garmin sleep + residential IAQ complete-case model input": {
        "Milan summer n": 8,
        "Milan winter n": 23,
        "Milan paired n": 6,
        "Thessaloniki summer n": 3,
        "Thessaloniki winter n": 6,
        "Thessaloniki paired n": 1,
        "Participant-days/nights/rows": 212,
    },
    "HIA daily PM input": {
        "Milan summer n": 54,
        "Milan winter n": 45,
        "Milan paired n": 32,
        "Thessaloniki summer n": 51,
        "Thessaloniki winter n": 57,
        "Thessaloniki paired n": 34,
        "Participant-days/nights/rows": 2427,
    },
}

PRIMARY_DENOMINATORS: dict[str, str] = {
    "PPM personal PM analysis": "Campaign-window participant-days with any non-missing personal PPM PM fraction",
    "PPM personal PM common-support analysis": "Campaign-window participant-days with complete ordered PM1/PM2.5/PM10 timestamp support",
    "uHoo residential IAQ analysis": "Campaign-window participant-days with any non-missing residential uHoo IAQ variable",
    "Garmin heart-rate/stress analysis": "Campaign-window participant-days with Garmin average heart-rate or stress support",
    "Garmin sleep data availability": "Participant-night sleep-stage availability for Total, Light, Deep, and REM sleep",
    "Garmin sleep + residential IAQ complete-case model input": "Participant-night complete cases for sleep outcomes plus residential uHoo IAQ predictors",
    "HIA daily PM input": "Campaign-window daily personal PPM PM common-support input used for scenario HIA",
}

STREAM_NOTES: dict[str, str] = {
    "PPM personal PM analysis": "Valid support is any non-missing ICARUS PPM PM1, PM2.5, or PM10 value inside the campaign window.",
    "PPM personal PM common-support analysis": "Valid support requires complete ordered PM1 <= PM2.5 <= PM10 triplets before daily aggregation.",
    "uHoo residential IAQ analysis": "Valid support is any non-missing uHoo residential IAQ value inside the campaign window.",
    "Garmin heart-rate/stress analysis": "Valid support is non-missing average heart rate or stress after applying the current heart-rate plausibility filter.",
    "Garmin sleep data availability": "Sleep stages are reconstructed from harmonized rows; sleep efficiency is unavailable and is not counted.",
    "Garmin sleep + residential IAQ complete-case model input": "Complete cases require sleep Total/Light/Deep/REM plus all currently available uHoo residential IAQ predictors.",
    "HIA daily PM input": "Matches the common-support daily personal PPM input used for scenario-based HIA.",
}


@dataclass
class StreamAggregate:
    """Aggregate counters for one analysis stream."""

    participant_sets: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: {key: set() for key in CITY_SEASON_ORDER}
    )
    unit_counts: dict[tuple[str, str], int] = field(
        default_factory=lambda: {key: 0 for key in CITY_SEASON_ORDER}
    )

    def add(self, city: str, season: str, participant_id: str, units: int) -> None:
        if units <= 0:
            return
        key = (city, season)
        self.participant_sets[key].add(participant_id)
        self.unit_counts[key] += int(units)

    def n_participants(self, city: str, season: str) -> int:
        return len(self.participant_sets[(city, season)])

    def paired_n(self, city: str) -> int:
        return len(self.participant_sets[(city, "Summer")] & self.participant_sets[(city, "Winter")])

    def total_units(self) -> int:
        return int(sum(self.unit_counts.values()))


def city_prefixed_participant_id(city: str, value: object) -> str:
    """Standardize participant ID for in-memory paired-season counting."""
    text = "" if pd.isna(value) else str(value).strip()
    if "_" in text:
        text = text.split("_")[-1]
    text = re.sub(r"\.0$", "", text)
    match = re.search(r"\d+", text)
    if match:
        token = f"{int(match.group()):03d}"
    else:
        token = text.upper()
    prefix = "M" if city == "Milan" else "T" if city == "Thessaloniki" else ""
    return f"{prefix}{token}" if prefix else token


def _read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        return feather.read_table(io.BytesIO(fh.read())).to_pandas()


def _campaign_mask(ts: pd.Series, city: str, season: str, mode: DateFilterMode) -> pd.Series:
    valid_ts = ts.notna()
    if mode == "none":
        return valid_ts
    start, end = get_campaign_date_window(city, season)
    if start is None or end is None:
        return pd.Series(False, index=ts.index)
    return valid_ts & (ts.dt.date >= start.date()) & (ts.dt.date <= end.date())


def _numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _valid_date_units(df: pd.DataFrame, mask: pd.Series) -> int:
    if not bool(mask.any()) or "_date" not in df.columns:
        return 0
    return int(df.loc[mask, "_date"].dropna().nunique())


def _archive_inventory_row(member: str, df: pd.DataFrame) -> dict[str, object]:
    city, season = parse_city_season(member)
    raw_id = df["ID"].dropna().iloc[0] if "ID" in df.columns and df["ID"].dropna().size else Path(member).stem
    return {
        "city": city,
        "season": season,
        "participant_id": city_prefixed_participant_id(city, raw_id),
        "member": member,
    }


def _add_high_frequency_streams(
    aggregates: dict[str, StreamAggregate],
    city: str,
    season: str,
    participant_id: str,
    df: pd.DataFrame,
) -> None:
    ppm = _numeric_columns(df, PM_COLUMNS)
    ppm_any = ppm[list(PM_COLUMNS)].notna().any(axis=1)
    aggregates["PPM personal PM analysis"].add(city, season, participant_id, _valid_date_units(ppm, ppm_any))

    ppm_common = (
        ppm[list(PM_COLUMNS)].notna().all(axis=1)
        & (ppm["PM1_PPM"] <= ppm["PM25_PPM"])
        & (ppm["PM25_PPM"] <= ppm["PM10_PPM"])
    )
    common_units = _valid_date_units(ppm, ppm_common)
    aggregates["PPM personal PM common-support analysis"].add(city, season, participant_id, common_units)
    aggregates["HIA daily PM input"].add(city, season, participant_id, common_units)

    uhoo = _numeric_columns(df, UHOO_COLUMNS)
    uhoo_any = uhoo[list(UHOO_COLUMNS)].notna().any(axis=1)
    aggregates["uHoo residential IAQ analysis"].add(city, season, participant_id, _valid_date_units(uhoo, uhoo_any))

    garmin = _numeric_columns(df, GARMIN_COLUMNS)
    if "AvgHeartRate" in garmin.columns:
        invalid_hr = garmin["AvgHeartRate"].notna() & ((garmin["AvgHeartRate"] < 40) | (garmin["AvgHeartRate"] > 200))
        garmin.loc[invalid_hr, "AvgHeartRate"] = np.nan
    garmin_any = garmin[list(GARMIN_COLUMNS)].notna().any(axis=1)
    aggregates["Garmin heart-rate/stress analysis"].add(city, season, participant_id, _valid_date_units(garmin, garmin_any))


def _add_sleep_streams(
    aggregates: dict[str, StreamAggregate],
    city: str,
    season: str,
    df: pd.DataFrame,
    sleep_inside_outside: dict[str, int],
) -> None:
    if "Sleep" not in df.columns or "TS" not in df.columns or "ID" not in df.columns:
        return
    sleep = df[["TS", "ID", "Sleep", *[col for col in UHOO_COLUMNS if col in df.columns]]].copy()
    sleep["TS"] = pd.to_datetime(sleep["TS"], errors="coerce")
    sleep["date"] = sleep["TS"].dt.date
    sleep["Sleep"] = sleep["Sleep"].astype(str).str.strip().str.lower()
    sleep = sleep[sleep["Sleep"].isin(["deep", "light", "rem"])].dropna(subset=["ID", "date"])
    if sleep.empty:
        return
    sleep = _numeric_columns(sleep, UHOO_COLUMNS)
    grouped = (
        sleep.groupby(["ID", "date"], dropna=False)
        .agg(
            **{column: (column, "mean") for column in UHOO_COLUMNS},
            Sleep=("Sleep", lambda s: s.value_counts().to_dict()),
        )
        .reset_index()
    )
    grouped["SleepTotal"] = grouped["Sleep"].apply(lambda value: sum(value.values()) if isinstance(value, dict) else 0)
    grouped = grouped[grouped["SleepTotal"] > 0].copy()
    if grouped.empty:
        return

    start, end = get_campaign_date_window(city, season)
    date_series = pd.to_datetime(grouped["date"], errors="coerce")
    inside = date_series.between(start, end, inclusive="both") if start is not None and end is not None else pd.Series(False, index=grouped.index)
    sleep_inside_outside["inside_campaign_sleep_rows"] += int(inside.sum())
    sleep_inside_outside["outside_campaign_sleep_rows"] += int((~inside).sum())

    grouped["participant_id"] = [city_prefixed_participant_id(city, value) for value in grouped["ID"]]
    grouped["complete_uhoo_predictor_row"] = grouped[list(UHOO_COLUMNS)].notna().all(axis=1)
    for row in grouped.itertuples(index=False):
        aggregates["Garmin sleep data availability"].add(city, season, row.participant_id, 1)
        if bool(row.complete_uhoo_predictor_row):
            aggregates["Garmin sleep + residential IAQ complete-case model input"].add(city, season, row.participant_id, 1)


def compute_phase1_denominators(
    data_zip: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compute aggregate denominator outputs from the harmonized zip archive."""
    data_zip = Path(data_zip)
    aggregates = {stream: StreamAggregate() for stream in S1_STREAM_ORDER}
    inventory_rows: list[dict[str, object]] = []
    sleep_window_counts = {"inside_campaign_sleep_rows": 0, "outside_campaign_sleep_rows": 0}

    with zipfile.ZipFile(data_zip) as zf:
        members = feather_members(data_zip)
        if not members:
            raise FileNotFoundError(f"No .feather files found in {data_zip}")
        for member in members:
            city, season = parse_city_season(member)
            raw = _read_member(zf, member)
            inventory_rows.append(_archive_inventory_row(member, raw))
            raw_id = raw["ID"].dropna().iloc[0] if "ID" in raw.columns and raw["ID"].dropna().size else Path(member).stem
            participant_id = city_prefixed_participant_id(city, raw_id)

            if "TS" in raw.columns:
                ts = pd.to_datetime(raw["TS"], errors="coerce")
                in_window = raw.loc[_campaign_mask(ts, city, season, date_filter_mode)].copy()
                in_window["_date"] = ts.loc[in_window.index].dt.date
            else:
                in_window = pd.DataFrame()

            _add_high_frequency_streams(aggregates, city, season, participant_id, in_window)
            _add_sleep_streams(aggregates, city, season, raw, sleep_window_counts)

    inventory = pd.DataFrame(inventory_rows)
    archive_counts = inventory.groupby(["city", "season"])["member"].nunique().to_dict()

    strobe_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []

    for city, season in CITY_SEASON_ORDER:
        archive_n = int(archive_counts.get((city, season), 0))
        flow_rows.append(
            {
                "flow_stage": "harmonized_archive_inventory",
                "city": city,
                "season": season,
                "n_records": archive_n,
                "n_participants": archive_n,
                "n_days_or_nights_or_rows": "",
                "notes": "Raw harmonized feather member count; not a demographic enrollment or dropout count.",
            }
        )
        flow_rows.append(
            {
                "flow_stage": "table5_demographic_reference",
                "city": city,
                "season": season,
                "n_records": TABLE5_TARGETS[(city, season)],
                "n_participants": TABLE5_TARGETS[(city, season)],
                "n_days_or_nights_or_rows": "",
                "notes": "Current manuscript Table 5 target; not regenerated from harmonized sensor rows.",
            }
        )

    for stream in S1_STREAM_ORDER:
        aggregate = aggregates[stream]
        for city, season in CITY_SEASON_ORDER:
            n_participants = aggregate.n_participants(city, season)
            unit_count = int(aggregate.unit_counts[(city, season)])
            paired = aggregate.paired_n(city)
            strobe_rows.append(
                {
                    "analysis_stream": stream,
                    "city": city,
                    "season": season,
                    "n_participants": n_participants,
                    "n_participant_days_or_nights_or_rows": unit_count,
                    "paired_n_if_city_level": paired,
                    "primary_denominator": PRIMARY_DENOMINATORS[stream],
                    "source_basis": "controlled harmonized dataset archive",
                    "notes": STREAM_NOTES[stream],
                }
            )
            flow_rows.append(
                {
                    "flow_stage": stream,
                    "city": city,
                    "season": season,
                    "n_records": int(archive_counts.get((city, season), 0)),
                    "n_participants": n_participants,
                    "n_days_or_nights_or_rows": unit_count,
                    "notes": STREAM_NOTES[stream],
                }
            )

            if stream == "Garmin sleep + residential IAQ complete-case model input":
                available_rows = aggregates["Garmin sleep data availability"].unit_counts[(city, season)]
                missing_rule = "Participant-night complete cases among reconstructed sleep-stage rows."
            else:
                available_rows = int(archive_counts.get((city, season), 0))
                missing_rule = "Participant-season support among raw harmonized archive members."
            complete_rows = unit_count if stream == "Garmin sleep + residential IAQ complete-case model input" else n_participants
            missing = max(int(available_rows) - int(complete_rows), 0)
            complete_fraction = (complete_rows / available_rows) if available_rows else np.nan
            missing_rows.append(
                {
                    "analysis_stream": stream,
                    "city": city,
                    "season": season,
                    "available_rows": int(available_rows),
                    "complete_rows": int(complete_rows),
                    "missing_rows": int(missing),
                    "complete_fraction": complete_fraction,
                    "primary_missingness_rule": missing_rule,
                    "notes": STREAM_NOTES[stream],
                }
            )

        for city in ("Milan", "Thessaloniki"):
            paired_rows.append(
                {
                    "analysis_stream": stream,
                    "city": city,
                    "summer_n": aggregate.n_participants(city, "Summer"),
                    "winter_n": aggregate.n_participants(city, "Winter"),
                    "paired_n": aggregate.paired_n(city),
                    "paired_definition": "Unique city-prefixed participants with valid stream support in both summer and winter.",
                    "notes": STREAM_NOTES[stream],
                }
            )

    metadata = {
        "archive_members": int(len(inventory)),
        "sleep_rows_inside_campaign": int(sleep_window_counts["inside_campaign_sleep_rows"]),
        "sleep_rows_outside_campaign": int(sleep_window_counts["outside_campaign_sleep_rows"]),
        "date_filter_mode": date_filter_mode,
    }
    return (
        pd.DataFrame(strobe_rows),
        pd.DataFrame(flow_rows),
        pd.DataFrame(paired_rows),
        pd.DataFrame(missing_rows),
        metadata,
    )


def write_phase1_denominator_outputs(
    data_zip: str | Path,
    outdir: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> dict[str, Path]:
    """Write aggregate Phase 1 denominator outputs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    strobe, flow, paired, missingness, metadata = compute_phase1_denominators(data_zip, date_filter_mode)
    outputs = {
        "strobe_denominators_by_stream": outdir / "strobe_denominators_by_stream.csv",
        "participant_flow_counts": outdir / "participant_flow_counts.csv",
        "paired_seasonal_support": outdir / "paired_seasonal_support.csv",
        "missingness_by_stream_city_season": outdir / "missingness_by_stream_city_season.csv",
    }
    strobe.to_csv(outputs["strobe_denominators_by_stream"], index=False)
    flow.to_csv(outputs["participant_flow_counts"], index=False)
    paired.to_csv(outputs["paired_seasonal_support"], index=False)
    missingness.to_csv(outputs["missingness_by_stream_city_season"], index=False)
    pd.DataFrame([metadata]).to_csv(outdir / "_phase1_denominator_metadata.csv", index=False)
    return outputs


def _stream_wide_counts(strobe: pd.DataFrame, stream: str) -> dict[str, int]:
    rows = strobe[strobe["analysis_stream"] == stream]
    values: dict[str, int] = {}
    for city, season in CITY_SEASON_ORDER:
        row = rows[(rows["city"] == city) & (rows["season"] == season)]
        key = f"{city} {season.lower()} n"
        if row.empty:
            values[key] = 0
        else:
            values[key] = int(row["n_participants"].iloc[0])
    for city in ("Milan", "Thessaloniki"):
        city_rows = rows[rows["city"] == city]
        key = f"{city} paired n"
        values[key] = int(city_rows["paired_n_if_city_level"].dropna().iloc[0]) if not city_rows.empty else 0
    values["Participant-days/nights/rows"] = int(rows["n_participant_days_or_nights_or_rows"].sum()) if not rows.empty else 0
    return values


def _target_status(stream: str, values: dict[str, int]) -> str:
    target = S1_TARGETS[stream]
    mismatches = [key for key, expected in target.items() if int(values.get(key, -1)) != int(expected)]
    return "PASS" if not mismatches else "FAIL: " + "; ".join(mismatches)


def make_supplementary_table_s1(strobe_csv: str | Path) -> pd.DataFrame:
    """Create the requested reproduced Supplementary Table S1 CSV."""
    strobe = pd.read_csv(strobe_csv)
    rows: list[dict[str, object]] = []
    for stream in S1_STREAM_ORDER:
        values = _stream_wide_counts(strobe, stream)
        stream_rows = strobe[strobe["analysis_stream"] == stream]
        source_basis = stream_rows["source_basis"].dropna().iloc[0] if not stream_rows.empty else ""
        primary = stream_rows["primary_denominator"].dropna().iloc[0] if not stream_rows.empty else PRIMARY_DENOMINATORS[stream]
        notes = stream_rows["notes"].dropna().iloc[0] if not stream_rows.empty else STREAM_NOTES[stream]
        rows.append(
            {
                "Analysis/data stream": stream,
                "Milan summer n": values["Milan summer n"],
                "Milan winter n": values["Milan winter n"],
                "Milan paired n": values["Milan paired n"],
                "Thessaloniki summer n": values["Thessaloniki summer n"],
                "Thessaloniki winter n": values["Thessaloniki winter n"],
                "Thessaloniki paired n": values["Thessaloniki paired n"],
                "Participant-days/nights/rows": values["Participant-days/nights/rows"],
                "Primary denominator": primary,
                "Source file": source_basis,
                "Notes": notes,
                "target_match_status": _target_status(stream, values),
            }
        )
    return pd.DataFrame(rows)


def write_supplementary_table_s1(strobe_csv: str | Path, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "supplementary_table_s1_reproduced.csv"
    make_supplementary_table_s1(strobe_csv).to_csv(outpath, index=False)
    return outpath


def make_supplementary_figure_s1_counts(strobe_csv: str | Path) -> pd.DataFrame:
    """Create aggregate count rows for Supplementary Figure S1 reproduction."""
    strobe = pd.read_csv(strobe_csv)
    rows: list[dict[str, object]] = [
        {
            "panel": "Participant flow",
            "box_number": 1,
            "box_title": "Pre-monitoring recruitment/dropout",
            "line_1": "Not reconstructable from harmonized sensor data",
            "line_2": "Do not infer enrollment or dropout counts",
            "line_3": "Report analysis support by stream",
            "line_4": "STROBE limitation documented",
            "target_match_status": "NOT_RECONSTRUCTABLE",
        },
        {
            "panel": "Participant flow",
            "box_number": 2,
            "box_title": "Table 5 demographic records",
            "line_1": "Overall participant-season records: 277",
            "line_2": "Milan: summer 63; winter 89",
            "line_3": "Thessaloniki: summer 38; winter 87",
            "line_4": "Demographic source not present in harmonized zip",
            "target_match_status": "REFERENCE_ONLY",
        },
    ]
    box = 3
    for stream in S1_STREAM_ORDER:
        values = _stream_wide_counts(strobe, stream)
        rows.append(
            {
                "panel": "Analysis support",
                "box_number": box,
                "box_title": stream,
                "line_1": f"Milan: summer {values['Milan summer n']}; winter {values['Milan winter n']}; paired {values['Milan paired n']}",
                "line_2": f"Thessaloniki: summer {values['Thessaloniki summer n']}; winter {values['Thessaloniki winter n']}; paired {values['Thessaloniki paired n']}",
                "line_3": f"Participant-days/nights/rows: {values['Participant-days/nights/rows']}",
                "line_4": PRIMARY_DENOMINATORS[stream],
                "target_match_status": _target_status(stream, values),
            }
        )
        box += 1
    return pd.DataFrame(rows)


def write_supplementary_figure_s1_counts(strobe_csv: str | Path, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "supplementary_figure_s1_counts.csv"
    make_supplementary_figure_s1_counts(strobe_csv).to_csv(outpath, index=False)
    return outpath


def make_table5_denominator_check(data_zip: str | Path) -> pd.DataFrame:
    """Document Table 5 denominator status without fabricating demographic data."""
    data_zip = Path(data_zip)
    archive_counts: dict[tuple[str, str], int] = defaultdict(int)
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            archive_counts[parse_city_season(member)] += 1
    rows: list[dict[str, object]] = []
    for city, season in CITY_SEASON_ORDER:
        archive_n = int(archive_counts.get((city, season), 0))
        rows.append(
            {
                "city": city,
                "season": season,
                "target_demographic_records": TABLE5_TARGETS[(city, season)],
                "reproduced_demographic_records": "",
                "status": "MISSING_DEPENDENCY",
                "notes": (
                    "The harmonized sensor archive has no participant-level demographic source for this count. "
                    f"Raw harmonized archive inventory for this city-season is {archive_n}, which is not a demographic denominator."
                ),
            }
        )
    rows.append(
        {
            "city": "Overall",
            "season": "All",
            "target_demographic_records": TABLE5_OVERALL_TARGET,
            "reproduced_demographic_records": "",
            "status": "MISSING_DEPENDENCY",
            "notes": "Pre-monitoring recruitment/dropout and Table 5 demographic records cannot be reconstructed from the harmonized sensor archive alone.",
        }
    )
    return pd.DataFrame(rows)


def make_table6_denominator_check(data_zip: str | Path, date_filter_mode: DateFilterMode = "campaign") -> pd.DataFrame:
    """Recompute Table 6 variable-specific denominator logic from the archive."""
    result = load_table6_daily_with_audit(data_zip, date_filter_mode=date_filter_mode)
    stats = summarize_table6(result.daily)
    sample = make_table6_sample_summary(stats)
    target_map = {
        "ICARUS PPM": "PPM personal PM analysis",
        "uHoo": "uHoo residential IAQ analysis",
    }
    rows: list[dict[str, object]] = []
    for record in sample.to_dict(orient="records"):
        if record["city"] == "Overall":
            target = ""
            status = "PASS_RECOMPUTED"
        else:
            target_stream = target_map.get(str(record["sensor"]))
            if target_stream:
                target = S1_TARGETS[target_stream][f"{record['city']} {str(record['season']).lower()} n"]
                status = "PASS" if int(record["participant_season_records"]) == int(target) else "FAIL"
            else:
                target = ""
                status = "PASS_RECOMPUTED"
        rows.append(
            {
                "variable": record["variable"],
                "sensor": record["sensor"],
                "city": record["city"],
                "season": record["season"],
                "participant_season_records": int(record["participant_season_records"]),
                "unique_participants": int(record["unique_participants"]),
                "participant_days": int(record["participant_days"]),
                "target_if_available": target,
                "status": status,
                "notes": "Recomputed from campaign-window participant-day Table 6 loader; no participant-level rows exported.",
            }
        )
    return pd.DataFrame(rows)


def write_table5_table6_checks(
    data_zip: str | Path,
    denominator_dir: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> dict[str, Path]:
    denominator_dir = Path(denominator_dir)
    denominator_dir.mkdir(parents=True, exist_ok=True)
    table5 = make_table5_denominator_check(data_zip)
    table6 = make_table6_denominator_check(data_zip, date_filter_mode)
    outputs = {
        "table5_denominator_check": denominator_dir / "table5_denominator_check.csv",
        "table6_denominator_check": denominator_dir / "table6_denominator_check.csv",
    }
    table5.to_csv(outputs["table5_denominator_check"], index=False)
    table6.to_csv(outputs["table6_denominator_check"], index=False)
    return outputs


def _status_summary(frame: pd.DataFrame, column: str = "target_match_status") -> str:
    if frame.empty or column not in frame.columns:
        return "not available"
    counts = frame[column].astype(str).value_counts().to_dict()
    return "; ".join(f"{key}: {value}" for key, value in counts.items())


def write_phase1_validation_report(
    repo_root: str | Path,
    data_zip: str | Path,
    denominator_dir: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> Path:
    """Write the requested Phase 1 validation report."""
    repo_root = Path(repo_root)
    data_zip = Path(data_zip)
    denominator_dir = Path(denominator_dir)
    s1_path = denominator_dir / "supplementary_table_s1_reproduced.csv"
    figure_path = denominator_dir / "supplementary_figure_s1_counts.csv"
    table5_path = denominator_dir / "table5_denominator_check.csv"
    table6_path = denominator_dir / "table6_denominator_check.csv"
    outpath = denominator_dir / "phase1_validation_report.txt"

    s1 = pd.read_csv(s1_path) if s1_path.exists() else pd.DataFrame()
    figure = pd.read_csv(figure_path) if figure_path.exists() else pd.DataFrame()
    table5 = pd.read_csv(table5_path) if table5_path.exists() else pd.DataFrame()
    table6 = pd.read_csv(table6_path) if table6_path.exists() else pd.DataFrame()

    lines = [
        "Phase 1 validation report",
        f"timestamp of run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository path: {repo_root}",
        f"data archive path used: {data_zip}",
        f"campaign-window mode: {date_filter_mode}",
        "",
        "Target values and reproduced values by Supplementary Table S1 stream:",
    ]
    if s1.empty:
        lines.append("- Supplementary Table S1 output not found.")
    else:
        for row in s1.to_dict(orient="records"):
            stream = row["Analysis/data stream"]
            target = S1_TARGETS[stream]
            reproduced = {
                "Milan summer n": int(row["Milan summer n"]),
                "Milan winter n": int(row["Milan winter n"]),
                "Milan paired n": int(row["Milan paired n"]),
                "Thessaloniki summer n": int(row["Thessaloniki summer n"]),
                "Thessaloniki winter n": int(row["Thessaloniki winter n"]),
                "Thessaloniki paired n": int(row["Thessaloniki paired n"]),
                "Participant-days/nights/rows": int(row["Participant-days/nights/rows"]),
            }
            lines.append(f"- {stream}: {row['target_match_status']}; target={target}; reproduced={reproduced}")

    lines.extend(
        [
            "",
            "PASS/FAIL by stream:",
            f"- Supplementary Table S1: {_status_summary(s1)}",
            f"- Supplementary Figure S1 counts: {_status_summary(figure)}",
            f"- Table 5 denominator check: {_status_summary(table5, 'status')}",
            f"- Table 6 denominator check: {_status_summary(table6, 'status')}",
            "",
            "Missing dependencies:",
        ]
    )
    if not table5.empty and (table5["status"] == "MISSING_DEPENDENCY").any():
        lines.append("- Table 5 participant-level demographic source is not present in the harmonized sensor archive; demographic records cannot be regenerated from the zip alone.")
    else:
        lines.append("- None flagged for Phase 1 denominator streams.")
    if not figure.empty and (figure["target_match_status"] == "NOT_RECONSTRUCTABLE").any():
        lines.append("- Pre-monitoring recruitment/dropout counts are not reconstructable from harmonized sensor data and were not invented.")

    lines.extend(
        [
            "",
            "Values that could not be reproduced and why:",
        ]
    )
    if table5.empty:
        lines.append("- Table 5 check output not found.")
    else:
        missing_table5 = table5[table5["status"] == "MISSING_DEPENDENCY"]
        if missing_table5.empty:
            lines.append("- None.")
        else:
            for row in missing_table5.to_dict(orient="records"):
                lines.append(f"- Table 5 {row['city']} {row['season']}: target {row['target_demographic_records']}; {row['notes']}")

    lines.extend(
        [
            "",
            "Output files:",
        ]
    )
    output_paths = sorted(denominator_dir.glob("*.csv")) + sorted(denominator_dir.glob("*.txt"))
    if outpath not in output_paths:
        output_paths.append(outpath)
    for path in output_paths:
        lines.append(f"- {path.resolve()}")
    lines.extend(
        [
            "",
            "Confirmations:",
            "- No GitHub push was performed.",
            "- No Git commit was performed.",
            "- No data upload was performed.",
            "- Controlled data remain local.",
            "- Outputs under local_outputs/denominators are aggregate-only and local-only.",
        ]
    )

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outpath
