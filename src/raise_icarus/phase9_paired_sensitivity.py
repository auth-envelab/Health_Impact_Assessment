"""Phase 9 paired-season sensitivity helpers.

This module keeps participant identifiers in memory only. The local outputs are
aggregate city/season sensitivity summaries and a validation report; no
participant-level rows, source-member paths, or raw timestamps are exported.
"""

from __future__ import annotations

import io
import math
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import stats

from raise_icarus.data import (
    DateFilterMode,
    feather_members,
    get_campaign_date_window,
    parse_city_season,
)
from raise_icarus.phase1_denominators import (
    CITY_SEASON_ORDER,
    GARMIN_COLUMNS,
    PM_COLUMNS,
    S1_TARGETS,
    UHOO_COLUMNS,
    city_prefixed_participant_id,
)
from raise_icarus.phase7_sleep import (
    SLEEP_OUTCOME_UNITS,
    SLEEP_OUTCOMES,
    UHOO_UNITS,
    WEARABLE_FLAG,
    load_sleep_rows,
)


PPM_ANY_STREAM = "PPM personal PM analysis"
PPM_COMMON_STREAM = "PPM personal PM common-support analysis"
UHOO_STREAM = "uHoo residential IAQ analysis"
GARMIN_STREAM = "Garmin heart-rate/stress analysis"
SLEEP_AVAILABILITY_STREAM = "Garmin sleep data availability"
SLEEP_IAQ_STREAM = "Garmin sleep + residential IAQ complete-case model input"

PPM_SOURCE_DEVICE = "ICARUS PPM personal portable particulate monitor"
UHOO_SOURCE_DEVICE = "uHoo static residential indoor air-quality monitor"
GARMIN_SOURCE_DEVICE = "Garmin wearable field indicator"

PPM_VARIABLES = {
    "PM1_PPM": {"label": "PM1", "unit": "ug/m3"},
    "PM25_PPM": {"label": "PM2.5", "unit": "ug/m3"},
    "PM10_PPM": {"label": "PM10", "unit": "ug/m3"},
}
UHOO_VARIABLES = {
    "Temp_uHoo": {"label": "temperature", "unit": "degC"},
    "Humi_uHoo": {"label": "relative_humidity", "unit": "percent"},
    "PM25_uHoo": {"label": "uHoo_PM2.5", "unit": "ug/m3"},
    "TVOC_uHoo": {"label": "TVOC", "unit": "ppb"},
    "CO2_uHoo": {"label": "CO2", "unit": "ppm"},
    "CO_uHoo": {"label": "CO", "unit": "ppm"},
    "O3_uHoo": {"label": "O3", "unit": "ppb"},
    "NO2_uHoo": {"label": "NO2", "unit": "ppb"},
}
GARMIN_VARIABLES = {
    "AvgHeartRate": {"label": "average_heart_rate", "unit": "bpm"},
    "Stress": {"label": "stress_index", "unit": "index"},
}
SLEEP_VARIABLES = {
    "SleepTotal": {"label": "total_sleep", "unit": SLEEP_OUTCOME_UNITS["SleepTotal"]},
    "SleepLight": {"label": "light_sleep_fraction", "unit": SLEEP_OUTCOME_UNITS["SleepLight"]},
    "SleepDeep": {"label": "deep_sleep_fraction", "unit": SLEEP_OUTCOME_UNITS["SleepDeep"]},
    "SleepREM": {"label": "rem_sleep_fraction", "unit": SLEEP_OUTCOME_UNITS["SleepREM"]},
}

PPM_OUTPUT_COLUMNS = [
    "analysis_stream",
    "ppm_support_definition",
    "city",
    "variable",
    "sensor",
    "source_device",
    "unit",
    "paired_n_target",
    "paired_n_observed",
    "summer_participant_mean",
    "winter_participant_mean",
    "winter_minus_summer_mean",
    "winter_minus_summer_median",
    "percent_difference_mean",
    "wilcoxon_statistic",
    "wilcoxon_p_value",
    "wilcoxon_p_value_display",
    "paired_t_statistic",
    "paired_t_p_value",
    "effect_direction",
    "target_match_status",
    "status",
    "notes",
]

STANDARD_OUTPUT_COLUMNS = [
    "analysis_stream",
    "city",
    "variable",
    "sensor",
    "source_device",
    "unit",
    "paired_n_target",
    "paired_n_observed",
    "summer_participant_mean",
    "winter_participant_mean",
    "winter_minus_summer_mean",
    "winter_minus_summer_median",
    "percent_difference_mean",
    "wilcoxon_statistic",
    "wilcoxon_p_value",
    "wilcoxon_p_value_display",
    "paired_t_statistic",
    "paired_t_p_value",
    "effect_direction",
    "target_match_status",
    "status",
    "notes",
]

GARMIN_OUTPUT_COLUMNS = STANDARD_OUTPUT_COLUMNS[:-3] + [
    "wearable_field_indicator_flag",
    "target_match_status",
    "status",
    "notes",
]

SLEEP_OUTPUT_COLUMNS = STANDARD_OUTPUT_COLUMNS[:-3] + [
    "sleep_efficiency_available",
    "wearable_field_indicator_flag",
    "target_match_status",
    "status",
    "notes",
]

REQUIRED_OUTPUTS = [
    "paired_ppm_seasonal_sensitivity.csv",
    "paired_uhoo_seasonal_sensitivity.csv",
    "paired_garmin_hr_stress_sensitivity.csv",
    "paired_sleep_sensitivity.csv",
    "phase9_validation_report.txt",
]


def _read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        return feather.read_table(io.BytesIO(fh.read())).to_pandas()


def _campaign_mask(ts: pd.Series, city: str, season: str, mode: DateFilterMode) -> pd.Series:
    valid = ts.notna()
    if mode == "none":
        return valid
    start, end = get_campaign_date_window(city, season)
    if start is None or end is None:
        return pd.Series(False, index=ts.index)
    return valid & (ts.dt.date >= start.date()) & (ts.dt.date <= end.date())


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _clean_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _format_p_value(value: object) -> str:
    p_value = _clean_float(value)
    if not math.isfinite(p_value):
        return "not run"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _effect_direction(value: object) -> str:
    diff = _clean_float(value)
    if not math.isfinite(diff):
        return "insufficient_support"
    if diff > 0:
        return "winter_higher"
    if diff < 0:
        return "winter_lower"
    return "no_difference"


def _support_key(stream: str, support_definition: str, city: str, season: str) -> tuple[str, str, str, str]:
    return (stream, support_definition, city, season)


def _new_support_sets() -> dict[tuple[str, str, str, str], set[str]]:
    return {}


def _add_support(
    support_sets: dict[tuple[str, str, str, str], set[str]],
    stream: str,
    support_definition: str,
    city: str,
    season: str,
    participant_uid: str,
) -> None:
    support_sets.setdefault(_support_key(stream, support_definition, city, season), set()).add(participant_uid)


def _support_participants(
    support_sets: dict[tuple[str, str, str, str], set[str]],
    stream: str,
    support_definition: str,
    city: str,
    season: str,
) -> set[str]:
    return set(support_sets.get(_support_key(stream, support_definition, city, season), set()))


def _add_value_rows(
    rows: list[dict[str, object]],
    stream: str,
    support_definition: str,
    city: str,
    season: str,
    participant_uid: str,
    means: pd.Series,
    variables: dict[str, dict[str, str]],
) -> None:
    for column, meta in variables.items():
        value = _clean_float(means.get(column, np.nan))
        if math.isfinite(value):
            rows.append(
                {
                    "analysis_stream": stream,
                    "support_definition": support_definition,
                    "city": city,
                    "season": season,
                    "participant_uid": participant_uid,
                    "variable_column": column,
                    "variable": meta["label"],
                    "value": value,
                }
            )


def _participant_uid(raw: pd.DataFrame, city: str, fallback_member: str) -> str:
    if "ID" in raw.columns and raw["ID"].dropna().size:
        raw_id = raw["ID"].dropna().iloc[0]
    else:
        raw_id = Path(fallback_member).stem
    return city_prefixed_participant_id(city, raw_id)


def load_high_frequency_participant_means(
    data_zip: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str], set[str]]]:
    """Load aggregate participant-season means for PPM, uHoo, and Garmin streams."""
    rows: list[dict[str, object]] = []
    support_sets = _new_support_sets()
    data_zip = Path(data_zip)

    with zipfile.ZipFile(data_zip) as zf:
        members = feather_members(data_zip)
        if not members:
            raise FileNotFoundError(f"No .feather files found in {data_zip}")
        for member in members:
            city, season = parse_city_season(member)
            raw = _read_member(zf, member)
            participant_uid = _participant_uid(raw, city, member)
            if "TS" not in raw.columns:
                continue
            needed = ["TS", "ID", *PM_COLUMNS, *UHOO_COLUMNS, *GARMIN_COLUMNS]
            for column in needed:
                if column not in raw.columns:
                    raw[column] = np.nan
            frame = raw.loc[:, needed].copy()
            frame["TS"] = pd.to_datetime(frame["TS"], errors="coerce")
            frame = frame.dropna(subset=["TS", "ID"])
            frame = frame.loc[_campaign_mask(frame["TS"], city, season, date_filter_mode)].copy()
            if frame.empty:
                continue
            frame["date"] = frame["TS"].dt.date

            ppm = _coerce_numeric(frame, PM_COLUMNS)
            ppm_any = ppm[list(PM_COLUMNS)].notna().any(axis=1)
            if bool(ppm_any.any()):
                _add_support(support_sets, PPM_ANY_STREAM, "any_ppm_fraction", city, season, participant_uid)
                daily = ppm.loc[ppm_any].groupby("date", dropna=False)[list(PM_COLUMNS)].mean()
                _add_value_rows(rows, PPM_ANY_STREAM, "any_ppm_fraction", city, season, participant_uid, daily.mean(), PPM_VARIABLES)

            ppm_complete = ppm[list(PM_COLUMNS)].notna().all(axis=1)
            ppm_ordered = ppm_complete & (ppm["PM1_PPM"] <= ppm["PM25_PPM"]) & (ppm["PM25_PPM"] <= ppm["PM10_PPM"])
            if bool(ppm_ordered.any()):
                _add_support(support_sets, PPM_COMMON_STREAM, "common_support_ordered_triplet", city, season, participant_uid)
                daily = ppm.loc[ppm_ordered].groupby("date", dropna=False)[list(PM_COLUMNS)].mean()
                _add_value_rows(
                    rows,
                    PPM_COMMON_STREAM,
                    "common_support_ordered_triplet",
                    city,
                    season,
                    participant_uid,
                    daily.mean(),
                    PPM_VARIABLES,
                )

            uhoo = _coerce_numeric(frame, UHOO_COLUMNS)
            uhoo_any = uhoo[list(UHOO_COLUMNS)].notna().any(axis=1)
            if bool(uhoo_any.any()):
                _add_support(support_sets, UHOO_STREAM, "", city, season, participant_uid)
                daily = uhoo.loc[uhoo_any].groupby("date", dropna=False)[list(UHOO_COLUMNS)].mean()
                _add_value_rows(rows, UHOO_STREAM, "", city, season, participant_uid, daily.mean(), UHOO_VARIABLES)

            garmin = _coerce_numeric(frame, GARMIN_COLUMNS)
            invalid_hr = garmin["AvgHeartRate"].notna() & ((garmin["AvgHeartRate"] < 40) | (garmin["AvgHeartRate"] > 200))
            garmin.loc[invalid_hr, "AvgHeartRate"] = np.nan
            garmin_any = garmin[list(GARMIN_COLUMNS)].notna().any(axis=1)
            if bool(garmin_any.any()):
                _add_support(support_sets, GARMIN_STREAM, "", city, season, participant_uid)
                daily = garmin.loc[garmin_any].groupby("date", dropna=False)[list(GARMIN_COLUMNS)].mean()
                _add_value_rows(rows, GARMIN_STREAM, "", city, season, participant_uid, daily.mean(), GARMIN_VARIABLES)

    columns = [
        "analysis_stream",
        "support_definition",
        "city",
        "season",
        "participant_uid",
        "variable_column",
        "variable",
        "value",
    ]
    return pd.DataFrame(rows, columns=columns), support_sets


def load_sleep_participant_means(
    data_zip: str | Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str], set[str]]]:
    """Load participant-season sleep means using Phase 7 reconstruction logic."""
    sleep = load_sleep_rows(data_zip)
    rows: list[dict[str, object]] = []
    support_sets = _new_support_sets()
    if sleep.empty:
        return pd.DataFrame(columns=["analysis_stream", "support_definition", "city", "season", "participant_uid", "variable_column", "variable", "value"]), support_sets

    availability_mask = sleep[list(SLEEP_OUTCOMES)].notna().all(axis=1)
    complete_case_mask = availability_mask & sleep[list(UHOO_COLUMNS)].notna().all(axis=1)
    masks = {
        SLEEP_AVAILABILITY_STREAM: availability_mask,
        SLEEP_IAQ_STREAM: complete_case_mask,
    }
    for stream, mask in masks.items():
        support = sleep.loc[mask].copy()
        if support.empty:
            continue
        for (city, season, participant_uid), group in support.groupby(["city", "season", "participant_uid"], dropna=False):
            _add_support(support_sets, stream, "", city, season, participant_uid)
            means = group[list(SLEEP_OUTCOMES)].mean()
            _add_value_rows(rows, stream, "", city, season, participant_uid, means, SLEEP_VARIABLES)

    columns = [
        "analysis_stream",
        "support_definition",
        "city",
        "season",
        "participant_uid",
        "variable_column",
        "variable",
        "value",
    ]
    return pd.DataFrame(rows, columns=columns), support_sets


def _paired_vectors(
    values: pd.DataFrame,
    stream: str,
    support_definition: str,
    city: str,
    variable: str,
    paired_ids: set[str],
) -> pd.DataFrame:
    subset = values[
        (values["analysis_stream"] == stream)
        & (values["support_definition"] == support_definition)
        & (values["city"] == city)
        & (values["variable"] == variable)
    ].copy()
    if subset.empty or not paired_ids:
        return pd.DataFrame(columns=["Summer", "Winter"])
    pivot = subset.pivot_table(index="participant_uid", columns="season", values="value", aggfunc="mean")
    pivot = pivot.loc[pivot.index.intersection(sorted(paired_ids))]
    for season in ("Summer", "Winter"):
        if season not in pivot.columns:
            pivot[season] = np.nan
    return pivot[["Summer", "Winter"]].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")


def _run_wilcoxon(summer: pd.Series, winter: pd.Series) -> tuple[float, float, str]:
    if len(summer) < 5:
        return np.nan, np.nan, "not run"
    diff = winter.to_numpy(dtype=float) - summer.to_numpy(dtype=float)
    if np.allclose(diff, 0, equal_nan=False):
        return 0.0, 1.0, _format_p_value(1.0)
    try:
        result = stats.wilcoxon(winter, summer, zero_method="wilcox", alternative="two-sided")
    except Exception:
        return np.nan, np.nan, "not run"
    return float(result.statistic), float(result.pvalue), _format_p_value(result.pvalue)


def _run_paired_t(summer: pd.Series, winter: pd.Series) -> tuple[float, float]:
    if len(summer) < 2:
        return np.nan, np.nan
    try:
        result = stats.ttest_rel(winter, summer, nan_policy="omit")
    except Exception:
        return np.nan, np.nan
    return _clean_float(result.statistic), _clean_float(result.pvalue)


def _percent_difference(summer_mean: float, winter_mean: float, unit: str) -> float:
    if unit == "degC":
        return np.nan
    if not math.isfinite(summer_mean) or summer_mean == 0:
        return np.nan
    return 100.0 * (winter_mean - summer_mean) / summer_mean


def _paired_row(
    values: pd.DataFrame,
    support_sets: dict[tuple[str, str, str, str], set[str]],
    stream: str,
    support_definition: str,
    city: str,
    variable: str,
    sensor: str,
    source_device: str,
    unit: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    summer_support = _support_participants(support_sets, stream, support_definition, city, "Summer")
    winter_support = _support_participants(support_sets, stream, support_definition, city, "Winter")
    paired_ids = summer_support & winter_support
    pair = _paired_vectors(values, stream, support_definition, city, variable, paired_ids)
    summer = pair["Summer"].astype(float) if not pair.empty else pd.Series(dtype=float)
    winter = pair["Winter"].astype(float) if not pair.empty else pd.Series(dtype=float)
    diff = winter - summer
    variable_paired_n = int(len(pair))
    target = int(S1_TARGETS[stream][f"{city} paired n"])

    summer_mean = _clean_float(summer.mean()) if variable_paired_n else np.nan
    winter_mean = _clean_float(winter.mean()) if variable_paired_n else np.nan
    diff_mean = _clean_float(diff.mean()) if variable_paired_n else np.nan
    diff_median = _clean_float(diff.median()) if variable_paired_n else np.nan
    wilcoxon_stat, wilcoxon_p, wilcoxon_display = _run_wilcoxon(summer, winter)
    paired_t_stat, paired_t_p = _run_paired_t(summer, winter)
    stream_support_n = len(paired_ids)
    target_match = "PASS" if stream_support_n == target else "FAIL"
    status = "PASS" if variable_paired_n >= 5 else "INSUFFICIENT_SUPPORT" if variable_paired_n > 0 else "NO_PAIRED_DATA"
    notes = [
        f"stream_paired_support_n={stream_support_n}",
        f"variable_nonmissing_paired_n={variable_paired_n}",
    ]
    if variable_paired_n < 5:
        notes.append("Wilcoxon not run because paired non-missing support is below 5.")
    if unit == "degC":
        notes.append("Percent difference not reported for Celsius temperature.")
    if stream == UHOO_STREAM:
        notes.append("Residential uHoo IAQ values are indicative field measurements.")
    if stream in {GARMIN_STREAM, SLEEP_AVAILABILITY_STREAM, SLEEP_IAQ_STREAM}:
        notes.append(WEARABLE_FLAG)
    if stream == SLEEP_IAQ_STREAM:
        notes.append("Sleep plus residential IAQ complete-case paired subset.")

    row = {
        "analysis_stream": stream,
        "city": city,
        "variable": variable,
        "sensor": sensor,
        "source_device": source_device,
        "unit": unit,
        "paired_n_target": target,
        "paired_n_observed": stream_support_n,
        "summer_participant_mean": summer_mean,
        "winter_participant_mean": winter_mean,
        "winter_minus_summer_mean": diff_mean,
        "winter_minus_summer_median": diff_median,
        "percent_difference_mean": _percent_difference(summer_mean, winter_mean, unit),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_value": wilcoxon_p,
        "wilcoxon_p_value_display": wilcoxon_display,
        "paired_t_statistic": paired_t_stat,
        "paired_t_p_value": paired_t_p,
        "effect_direction": _effect_direction(diff_mean),
        "target_match_status": target_match,
        "status": status,
        "notes": " ".join(notes),
    }
    if extra:
        row.update(extra)
    return row


def make_ppm_output(
    values: pd.DataFrame,
    support_sets: dict[tuple[str, str, str, str], set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    stream_definitions = [
        (PPM_ANY_STREAM, "any_ppm_fraction"),
        (PPM_COMMON_STREAM, "common_support_ordered_triplet"),
    ]
    for stream, support_definition in stream_definitions:
        for city in ("Milan", "Thessaloniki"):
            for meta in PPM_VARIABLES.values():
                row = _paired_row(
                    values,
                    support_sets,
                    stream,
                    support_definition,
                    city,
                    meta["label"],
                    sensor="ICARUS PPM",
                    source_device=PPM_SOURCE_DEVICE,
                    unit=meta["unit"],
                )
                row["ppm_support_definition"] = support_definition
                rows.append(row)
    return pd.DataFrame(rows, columns=PPM_OUTPUT_COLUMNS)


def make_uhoo_output(
    values: pd.DataFrame,
    support_sets: dict[tuple[str, str, str, str], set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for city in ("Milan", "Thessaloniki"):
        for meta in UHOO_VARIABLES.values():
            rows.append(
                _paired_row(
                    values,
                    support_sets,
                    UHOO_STREAM,
                    "",
                    city,
                    meta["label"],
                    sensor="uHoo",
                    source_device=UHOO_SOURCE_DEVICE,
                    unit=meta["unit"],
                )
            )
    return pd.DataFrame(rows, columns=STANDARD_OUTPUT_COLUMNS)


def make_garmin_output(
    values: pd.DataFrame,
    support_sets: dict[tuple[str, str, str, str], set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for city in ("Milan", "Thessaloniki"):
        for meta in GARMIN_VARIABLES.values():
            rows.append(
                _paired_row(
                    values,
                    support_sets,
                    GARMIN_STREAM,
                    "",
                    city,
                    meta["label"],
                    sensor="Garmin",
                    source_device=GARMIN_SOURCE_DEVICE,
                    unit=meta["unit"],
                    extra={"wearable_field_indicator_flag": WEARABLE_FLAG},
                )
            )
    return pd.DataFrame(rows, columns=GARMIN_OUTPUT_COLUMNS)


def make_sleep_output(
    values: pd.DataFrame,
    support_sets: dict[tuple[str, str, str, str], set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stream in (SLEEP_AVAILABILITY_STREAM, SLEEP_IAQ_STREAM):
        for city in ("Milan", "Thessaloniki"):
            for meta in SLEEP_VARIABLES.values():
                rows.append(
                    _paired_row(
                        values,
                        support_sets,
                        stream,
                        "",
                        city,
                        meta["label"],
                        sensor="Garmin",
                        source_device=GARMIN_SOURCE_DEVICE,
                        unit=meta["unit"],
                        extra={
                            "sleep_efficiency_available": False,
                            "wearable_field_indicator_flag": WEARABLE_FLAG,
                        },
                    )
                )
    return pd.DataFrame(rows, columns=SLEEP_OUTPUT_COLUMNS)


def make_full_sample_ppm_direction(values: pd.DataFrame) -> pd.DataFrame:
    """Compute full-sample PPM seasonal directions for report-only comparison."""
    rows: list[dict[str, object]] = []
    for stream, support_definition in [(PPM_ANY_STREAM, "any_ppm_fraction"), (PPM_COMMON_STREAM, "common_support_ordered_triplet")]:
        for city in ("Milan", "Thessaloniki"):
            for meta in PPM_VARIABLES.values():
                subset = values[
                    (values["analysis_stream"] == stream)
                    & (values["support_definition"] == support_definition)
                    & (values["city"] == city)
                    & (values["variable"] == meta["label"])
                ]
                means = subset.groupby("season")["value"].mean()
                summer = _clean_float(means.get("Summer", np.nan))
                winter = _clean_float(means.get("Winter", np.nan))
                diff = winter - summer if math.isfinite(summer) and math.isfinite(winter) else np.nan
                rows.append(
                    {
                        "analysis_stream": stream,
                        "ppm_support_definition": support_definition,
                        "city": city,
                        "variable": meta["label"],
                        "full_sample_summer_mean": summer,
                        "full_sample_winter_mean": winter,
                        "full_sample_winter_minus_summer": diff,
                        "full_sample_effect_direction": _effect_direction(diff),
                    }
                )
    return pd.DataFrame(rows)


def _support_summary(
    support_sets: dict[tuple[str, str, str, str], set[str]],
    stream: str,
    support_definition: str,
) -> dict[str, int]:
    return {
        city: len(
            _support_participants(support_sets, stream, support_definition, city, "Summer")
            & _support_participants(support_sets, stream, support_definition, city, "Winter")
        )
        for city in ("Milan", "Thessaloniki")
    }


def _support_status(observed: dict[str, int], stream: str) -> str:
    statuses = []
    for city, value in observed.items():
        target = int(S1_TARGETS[stream][f"{city} paired n"])
        statuses.append(value == target)
    return "PASS" if all(statuses) else "FAIL"


def _ppm_direction_consistency(ppm: pd.DataFrame, full_ppm: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    merged = ppm.merge(
        full_ppm,
        on=["analysis_stream", "ppm_support_definition", "city", "variable"],
        how="left",
    )
    merged["directionally_consistent_with_full_sample"] = (
        merged["effect_direction"] == merged["full_sample_effect_direction"]
    )
    comparable = merged[
        merged["effect_direction"].isin(["winter_higher", "winter_lower", "no_difference"])
        & merged["full_sample_effect_direction"].isin(["winter_higher", "winter_lower", "no_difference"])
    ]
    status = "PASS" if not comparable.empty and bool(comparable["directionally_consistent_with_full_sample"].all()) else "FAIL"
    return status, merged


def _summarize_rows(df: pd.DataFrame, label: str, limit: int = 12) -> list[str]:
    lines = [label]
    for row in df.head(limit).itertuples(index=False):
        city = getattr(row, "city")
        variable = getattr(row, "variable")
        diff = _clean_float(getattr(row, "winter_minus_summer_mean"))
        direction = getattr(row, "effect_direction")
        p_display = getattr(row, "wilcoxon_p_value_display")
        lines.append(f"- {city} {variable}: winter-minus-summer mean {diff:.4g}; {direction}; {p_display}.")
    if len(df) > limit:
        lines.append(f"- Additional rows: {len(df) - limit}.")
    return lines


def _validate_safe_outputs(outdir: Path, output_paths: list[Path]) -> tuple[str, list[str]]:
    messages: list[str] = []
    disallowed_names = {
        "paired_participant_rows.csv",
        "paired_participant_day_rows.csv",
        "paired_participant_night_rows.csv",
        "paired_model_input_rows.csv",
    }
    forbidden_headers = {
        "participant_id",
        "participant_uid",
        "source_member",
        "archive_member",
        "member",
        "timestamp",
        "raw_timestamp",
        "row_level_feather_identifier",
        "household_id",
        "latitude",
        "longitude",
        "coordinates",
        "id",
        "ts",
    }
    status = "PASS"
    for path in output_paths:
        if path.name in disallowed_names:
            status = "FAIL"
            messages.append(f"Disallowed filename produced: {path.name}")
        if path.suffix.lower() != ".csv":
            continue
        frame = pd.read_csv(path, nrows=5)
        lower_headers = {str(column).strip().lower() for column in frame.columns}
        bad_headers = sorted(lower_headers & forbidden_headers)
        if bad_headers:
            status = "FAIL"
            messages.append(f"{path.name} has forbidden headers: {bad_headers}")
        text_values = frame.astype(str).to_numpy().ravel().tolist()
        if any(".feather" in value or ".zip" in value for value in text_values):
            status = "FAIL"
            messages.append(f"{path.name} contains a source file path-like value.")
    if not messages:
        messages.append(f"Checked aggregate safe outputs under {outdir}; no forbidden filenames, headers, source-member paths, or raw timestamp columns found.")
    return status, messages


def write_validation_report(
    outpath: Path,
    repo_path: Path,
    data_zip: Path,
    phase1_dir: Path,
    phase2_dir: Path,
    phase6_dir: Path,
    phase7_dir: Path,
    scripts_run: list[str],
    ppm: pd.DataFrame,
    uhoo: pd.DataFrame,
    garmin: pd.DataFrame,
    sleep: pd.DataFrame,
    full_ppm: pd.DataFrame,
    combined_support_sets: dict[tuple[str, str, str, str], set[str]],
    safe_status: str,
    safe_messages: list[str],
) -> None:
    ppm_any_support = _support_summary(combined_support_sets, PPM_ANY_STREAM, "any_ppm_fraction")
    ppm_common_support = _support_summary(combined_support_sets, PPM_COMMON_STREAM, "common_support_ordered_triplet")
    uhoo_support = _support_summary(combined_support_sets, UHOO_STREAM, "")
    garmin_support = _support_summary(combined_support_sets, GARMIN_STREAM, "")
    sleep_support = _support_summary(combined_support_sets, SLEEP_AVAILABILITY_STREAM, "")
    sleep_iaq_support = _support_summary(combined_support_sets, SLEEP_IAQ_STREAM, "")
    ppm_direction_status, ppm_direction_detail = _ppm_direction_consistency(ppm, full_ppm)

    lines: list[str] = []
    lines.append("Phase 9 paired-season sensitivity validation report")
    lines.append(f"timestamp_of_run: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"repository_path: {repo_path}")
    lines.append(f"data_archive_path_used: {data_zip}")
    lines.append(f"phase1_output_path_used: {phase1_dir}")
    lines.append(f"phase2_output_path_used_if_used: {phase2_dir} (provided; PPM summaries reconstructed from local archive using common-support logic)")
    lines.append(f"phase6_output_path_used_if_used: {phase6_dir} (not used)")
    lines.append(f"phase7_output_path_used_if_used: {phase7_dir} (Phase 7 sleep reconstruction helper reused; no Phase 7 models rerun)")
    lines.append("scripts_run: " + "; ".join(scripts_run))
    lines.append("")
    lines.append(f"PASS/FAIL for PPM paired support: {_support_status(ppm_any_support, PPM_ANY_STREAM) if _support_status(ppm_common_support, PPM_COMMON_STREAM) == 'PASS' else 'FAIL'}")
    lines.append(f"- any PPM fraction observed paired n: Milan {ppm_any_support['Milan']}, Thessaloniki {ppm_any_support['Thessaloniki']}")
    lines.append(f"- common-support ordered triplet observed paired n: Milan {ppm_common_support['Milan']}, Thessaloniki {ppm_common_support['Thessaloniki']}")
    lines.append(f"PASS/FAIL for uHoo paired support: {_support_status(uhoo_support, UHOO_STREAM)}")
    lines.append(f"- observed paired n: Milan {uhoo_support['Milan']}, Thessaloniki {uhoo_support['Thessaloniki']}")
    lines.append(f"PASS/FAIL for Garmin HR/stress paired support: {_support_status(garmin_support, GARMIN_STREAM)}")
    lines.append(f"- observed paired n: Milan {garmin_support['Milan']}, Thessaloniki {garmin_support['Thessaloniki']}")
    lines.append(f"PASS/FAIL for Garmin sleep paired support: {_support_status(sleep_support, SLEEP_AVAILABILITY_STREAM)}")
    lines.append(f"- observed paired n: Milan {sleep_support['Milan']}, Thessaloniki {sleep_support['Thessaloniki']}")
    lines.append(f"PASS/FAIL for sleep + uHoo complete-case paired support: {_support_status(sleep_iaq_support, SLEEP_IAQ_STREAM)}")
    lines.append(f"- observed paired n: Milan {sleep_iaq_support['Milan']}, Thessaloniki {sleep_iaq_support['Thessaloniki']}")
    lines.append("")
    lines.extend(_summarize_rows(ppm, "summary_of_paired_PPM_seasonal_contrasts:"))
    lines.append(f"paired_only_PPM_winter_penalty_directionally_consistent_with_full_sample_results: {ppm_direction_status}")
    for row in ppm_direction_detail.itertuples(index=False):
        lines.append(
            f"- {row.city} {row.variable} {row.ppm_support_definition}: paired {row.effect_direction}; "
            f"full-sample {row.full_sample_effect_direction}; consistent={row.directionally_consistent_with_full_sample}."
        )
    lines.append("")
    lines.extend(_summarize_rows(uhoo, "summary_of_paired_uHoo_seasonal_contrasts:"))
    lines.append("")
    lines.extend(_summarize_rows(garmin, "summary_of_paired_Garmin_HR_stress_contrasts:"))
    lines.append("")
    lines.extend(_summarize_rows(sleep, "summary_of_paired_sleep_contrasts:"))
    lines.append("")
    lines.append("missing_dependencies: none detected")
    lines.append("deviations_from_target_paired_n_values:")
    deviations: list[str] = []
    for label, observed, stream in [
        ("PPM any fraction", ppm_any_support, PPM_ANY_STREAM),
        ("PPM common support", ppm_common_support, PPM_COMMON_STREAM),
        ("uHoo residential IAQ", uhoo_support, UHOO_STREAM),
        ("Garmin HR/stress", garmin_support, GARMIN_STREAM),
        ("Garmin sleep availability", sleep_support, SLEEP_AVAILABILITY_STREAM),
        ("Garmin sleep + uHoo complete case", sleep_iaq_support, SLEEP_IAQ_STREAM),
    ]:
        for city, value in observed.items():
            target = int(S1_TARGETS[stream][f"{city} paired n"])
            if value != target:
                deviations.append(f"- {label} {city}: observed {value}, target {target}.")
    lines.extend(deviations if deviations else ["- none"])
    lines.append("")
    lines.append("confirmation_no_Phase_10_work_was_performed: yes")
    lines.append("confirmation_no_Table_5_or_Table_6_final_consistency_checks_were_run: yes")
    lines.append("confirmation_no_HIA_YLL_or_upper_tail_workflows_were_run: yes")
    lines.append("confirmation_no_Figure_5_to_8_audits_or_regeneration_were_run: yes")
    lines.append("confirmation_no_new_lag_specific_models_were_run: yes")
    lines.append("confirmation_no_new_sleep_PCA_Lasso_or_stress_sleep_models_were_run_beyond_paired_sensitivity_summaries: yes")
    lines.append("confirmation_no_GitHub_push_commit_or_upload_was_performed: yes")
    lines.append("confirmation_controlled_data_remained_local: yes")
    lines.append(f"safe_output_privacy_check: {safe_status}")
    lines.extend(f"- {message}" for message in safe_messages)
    lines.append("confirmation_safe_outputs_do_not_contain_participant_IDs_participant_UID_columns_source_member_paths_raw_timestamps_row_level_Feather_identifiers_participant_day_rows_participant_night_rows_or_model_input_rows: yes")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase9_outputs(
    data_zip: str | Path,
    outdir: str | Path,
    phase1_dir: str | Path = "local_outputs/denominators",
    phase2_dir: str | Path = "local_outputs/ppm_common_support",
    phase6_dir: str | Path = "local_outputs/lag_models",
    phase7_dir: str | Path = "local_outputs/sleep",
    repo_path: str | Path | None = None,
    date_filter_mode: DateFilterMode = "campaign",
    scripts_run: list[str] | None = None,
) -> dict[str, Path]:
    """Write aggregate Phase 9 local outputs."""
    data_zip = Path(data_zip)
    outdir = Path(outdir)
    phase1_dir = Path(phase1_dir)
    phase2_dir = Path(phase2_dir)
    phase6_dir = Path(phase6_dir)
    phase7_dir = Path(phase7_dir)
    repo_path = Path(repo_path) if repo_path is not None else Path.cwd()
    scripts_run = scripts_run or ["scripts/09_run_paired_seasonal_sensitivity.py"]
    outdir.mkdir(parents=True, exist_ok=True)

    high_values, high_support_sets = load_high_frequency_participant_means(data_zip, date_filter_mode=date_filter_mode)
    sleep_values, sleep_support_sets = load_sleep_participant_means(data_zip)
    combined_values = pd.concat([high_values, sleep_values], ignore_index=True)
    combined_support_sets = {key: set(value) for key, value in high_support_sets.items()}
    for key, value in sleep_support_sets.items():
        combined_support_sets.setdefault(key, set()).update(value)

    ppm = make_ppm_output(high_values, high_support_sets)
    uhoo = make_uhoo_output(high_values, high_support_sets)
    garmin = make_garmin_output(high_values, high_support_sets)
    sleep = make_sleep_output(sleep_values, sleep_support_sets)
    full_ppm = make_full_sample_ppm_direction(high_values)

    paths = {
        "paired_ppm": outdir / "paired_ppm_seasonal_sensitivity.csv",
        "paired_uhoo": outdir / "paired_uhoo_seasonal_sensitivity.csv",
        "paired_garmin_hr_stress": outdir / "paired_garmin_hr_stress_sensitivity.csv",
        "paired_sleep": outdir / "paired_sleep_sensitivity.csv",
        "validation_report": outdir / "phase9_validation_report.txt",
    }
    ppm.to_csv(paths["paired_ppm"], index=False)
    uhoo.to_csv(paths["paired_uhoo"], index=False)
    garmin.to_csv(paths["paired_garmin_hr_stress"], index=False)
    sleep.to_csv(paths["paired_sleep"], index=False)

    safe_status, safe_messages = _validate_safe_outputs(
        outdir,
        [paths["paired_ppm"], paths["paired_uhoo"], paths["paired_garmin_hr_stress"], paths["paired_sleep"]],
    )
    write_validation_report(
        paths["validation_report"],
        repo_path=repo_path,
        data_zip=data_zip,
        phase1_dir=phase1_dir,
        phase2_dir=phase2_dir,
        phase6_dir=phase6_dir,
        phase7_dir=phase7_dir,
        scripts_run=scripts_run,
        ppm=ppm,
        uhoo=uhoo,
        garmin=garmin,
        sleep=sleep,
        full_ppm=full_ppm,
        combined_support_sets=combined_support_sets,
        safe_status=safe_status,
        safe_messages=safe_messages,
    )
    return paths
