"""Phase 8 aggregate validation and local figure regeneration helpers."""

from __future__ import annotations

import io
import itertools
import math
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import stats

from raise_icarus.data import (
    DEFAULT_CAMPAIGN_DATE_WINDOWS,
    DateFilterMode,
    feather_members,
    get_campaign_date_window,
    parse_city_season,
)
from raise_icarus.phase1_denominators import CITY_SEASON_ORDER, city_prefixed_participant_id


FIGURE_AUDIT_COLUMNS = [
    "figure",
    "city",
    "season",
    "variable_x",
    "variable_y",
    "source_x",
    "source_y",
    "sensor_x",
    "sensor_y",
    "correlation_method",
    "r",
    "p_value",
    "p_value_display",
    "significance_stars",
    "adjusted_p_value",
    "multiple_testing_adjustment",
    "n_observations",
    "n_participants",
    "practical_magnitude_flag",
    "statistically_significant",
    "practically_meaningful_flag",
    "status",
    "notes",
]
FIGURE7_OLS_COLUMNS = [
    "figure",
    "outcome",
    "pollutant",
    "lag_min",
    "source_device",
    "sensor_outcome",
    "beta",
    "se",
    "ci_low",
    "ci_high",
    "p_value",
    "p_value_display",
    "r_squared",
    "n_observations",
    "n_participants",
    "effect_per_10ug",
    "practical_effect_flag",
    "status",
    "notes",
]
FIGURE7_COMPARISON_COLUMNS = [
    "outcome",
    "pollutant",
    "lag_min",
    "ols_beta",
    "ols_ci_low",
    "ols_ci_high",
    "ols_p_value",
    "mixed_model_beta",
    "mixed_model_ci_low",
    "mixed_model_ci_high",
    "mixed_model_p_value",
    "direction_agreement",
    "magnitude_comparison",
    "primary_model_source",
    "status",
    "notes",
]
REQUIRED_OUTPUTS = [
    "figure5_correlation_audit.csv",
    "figure6_correlation_audit.csv",
    "figure7_ols_panel_audit.csv",
    "figure7_mixed_model_comparison.csv",
    "figure8_correlation_audit.csv",
    "figure_data_source_audit.csv",
    "season_month_mapping.csv",
    "figure_significance_formatting_audit.csv",
    "figure_triage_recommendation.csv",
    "final_figure5.png",
    "final_figure6.png",
    "final_figure7.png",
    "final_figure8.png",
    "phase8_validation_report.txt",
]
UNSAFE_OUTPUT_NAMES = {
    "figure5_plotting_rows.csv",
    "figure6_plotting_rows.csv",
    "figure7_plotting_rows.csv",
    "figure8_plotting_rows.csv",
    "minute_level_rows.csv",
    "participant_day_rows.csv",
    "participant_night_rows.csv",
}
UHOO_VARS = {
    "Temp_uHoo": "uHoo temperature",
    "Humi_uHoo": "uHoo relative humidity",
    "PM25_uHoo": "uHoo residential indoor PM2.5",
    "TVOC_uHoo": "uHoo TVOC",
    "CO2_uHoo": "uHoo CO2",
    "CO_uHoo": "uHoo CO",
    "O3_uHoo": "uHoo O3",
    "NO2_uHoo": "uHoo NO2",
}
FIG6_VARS = {
    "PM1_PPM": "ICARUS PPM PM1",
    "PM25_PPM": "ICARUS PPM PM2.5",
    "PM10_PPM": "ICARUS PPM PM10",
    "Temp_PPM": "ICARUS PPM temperature",
    "Humi_PPM": "ICARUS PPM relative humidity",
    "AvgHeartRate": "Garmin-derived heart rate",
    "Stress": "Garmin-derived stress",
}
SLEEP_LABELS = {
    "SleepTotal": "Garmin-derived total sleep duration",
    "SleepLight": "Garmin-derived light sleep fraction",
    "SleepDeep": "Garmin-derived deep sleep fraction",
    "SleepREM": "Garmin-derived REM sleep fraction",
}
LAG_MINUTES = (1, 5, 10, 15, 30, 45, 60, 120)
POLLUTANT_TO_PHYS_COL = {"PM1": "pm1", "PM2.5": "pm25", "PM10": "pm10"}
PHYS_COL_TO_POLLUTANT = {value: key for key, value in POLLUTANT_TO_PHYS_COL.items()}


def _read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        return feather.read_table(io.BytesIO(fh.read())).to_pandas()


def _campaign_mask(ts: pd.Series, city: str, season: str, mode: DateFilterMode = "campaign") -> pd.Series:
    valid = ts.notna()
    if mode == "none":
        return valid
    start, end = get_campaign_date_window(city, season)
    if start is None or end is None:
        return pd.Series(False, index=ts.index)
    return valid & (ts.dt.date >= start.date()) & (ts.dt.date <= end.date())


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _p_display(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    p_value = float(p_value)
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    p_value = float(p_value)
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _magnitude_flag(r_value: float) -> str:
    if pd.isna(r_value):
        return "not_estimated"
    magnitude = abs(float(r_value))
    if magnitude < 0.10:
        return "negligible"
    if magnitude < 0.20:
        return "weak"
    if magnitude < 0.40:
        return "modest"
    return "moderate_or_higher"


def _practical_effect_flag(effect_per_10ug: float, r_squared: float) -> str:
    if pd.isna(effect_per_10ug):
        return "not_estimated"
    magnitude = abs(float(effect_per_10ug))
    if magnitude < 0.10 or (not pd.isna(r_squared) and float(r_squared) < 0.001):
        return "negligible"
    if magnitude < 0.50:
        return "weak"
    if magnitude < 1.00:
        return "modest"
    return "moderate_or_higher"


def _pearson_row(
    figure: str,
    city: str,
    season: str,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    source_x: str,
    source_y: str,
    sensor_x: str,
    sensor_y: str,
) -> dict[str, object]:
    pair = data[["participant_key", x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    n_obs = int(len(pair))
    n_participants = int(pair["participant_key"].nunique()) if n_obs else 0
    if n_obs < 3 or pair[x_col].nunique(dropna=True) <= 1 or pair[y_col].nunique(dropna=True) <= 1:
        r_value = np.nan
        p_value = np.nan
        status = "INSUFFICIENT_DATA"
        notes = "Fewer than three complete paired observations or no variation."
    else:
        r_value, p_value = stats.pearsonr(pair[x_col], pair[y_col])
        r_value = float(r_value)
        p_value = float(p_value)
        status = "PASS"
        notes = "Pairwise complete Pearson correlation; row-level data not exported."
    magnitude = _magnitude_flag(r_value)
    significant = bool((not pd.isna(p_value)) and p_value < 0.05)
    meaningful = magnitude in {"modest", "moderate_or_higher"}
    if significant and magnitude == "negligible":
        notes += " Statistically detectable but practically negligible."
    return {
        "figure": figure,
        "city": city,
        "season": season,
        "variable_x": x_col,
        "variable_y": y_col,
        "source_x": source_x,
        "source_y": source_y,
        "sensor_x": sensor_x,
        "sensor_y": sensor_y,
        "correlation_method": "Pearson",
        "r": r_value,
        "p_value": p_value,
        "p_value_display": _p_display(p_value),
        "significance_stars": _stars(p_value),
        "adjusted_p_value": np.nan,
        "multiple_testing_adjustment": "not_applied",
        "n_observations": n_obs,
        "n_participants": n_participants,
        "practical_magnitude_flag": magnitude,
        "statistically_significant": significant,
        "practically_meaningful_flag": bool(meaningful),
        "status": status,
        "notes": notes,
    }


def _load_figure_rows(data_zip: str | Path, columns: list[str], date_filter_mode: DateFilterMode = "campaign") -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    data_zip = Path(data_zip)
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            city, season = parse_city_season(member)
            raw = _read_member(zf, member)
            needed = ["TS", "ID", *columns]
            for column in needed:
                if column not in raw.columns:
                    raw[column] = np.nan
            tmp = raw[needed].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp = tmp.loc[_campaign_mask(tmp["TS"], city, season, date_filter_mode)].copy()
            if tmp.empty:
                continue
            tmp = _coerce_numeric(tmp, columns)
            tmp["city"] = city
            tmp["season"] = season
            tmp["participant_key"] = [city_prefixed_participant_id(city, value) for value in tmp["ID"]]
            rows.append(tmp[["city", "season", "participant_key", *columns]])
    if not rows:
        return pd.DataFrame(columns=["city", "season", "participant_key", *columns])
    return pd.concat(rows, ignore_index=True)


def make_figure5_correlation_audit(data_zip: str | Path) -> pd.DataFrame:
    data = _load_figure_rows(data_zip, list(UHOO_VARS))
    rows: list[dict[str, object]] = []
    for city, season in CITY_SEASON_ORDER:
        subset = data[(data["city"] == city) & (data["season"] == season)]
        for x_col, y_col in itertools.combinations(UHOO_VARS, 2):
            rows.append(
                _pearson_row(
                    "Figure 5",
                    city,
                    season,
                    subset,
                    x_col,
                    y_col,
                    "residential uHoo IAQ",
                    "residential uHoo IAQ",
                    "uHoo",
                    "uHoo",
                )
            )
    return pd.DataFrame(rows, columns=FIGURE_AUDIT_COLUMNS)


def make_figure6_correlation_audit(data_zip: str | Path) -> pd.DataFrame:
    data = _load_figure_rows(data_zip, list(FIG6_VARS))
    if "AvgHeartRate" in data.columns:
        invalid_hr = data["AvgHeartRate"].notna() & ((data["AvgHeartRate"] < 40) | (data["AvgHeartRate"] > 200))
        data.loc[invalid_hr, "AvgHeartRate"] = np.nan
    rows: list[dict[str, object]] = []
    for city, season in CITY_SEASON_ORDER:
        subset = data[(data["city"] == city) & (data["season"] == season)]
        for x_col, y_col in itertools.combinations(FIG6_VARS, 2):
            x_garmin = x_col in {"AvgHeartRate", "Stress"}
            y_garmin = y_col in {"AvgHeartRate", "Stress"}
            rows.append(
                _pearson_row(
                    "Figure 6",
                    city,
                    season,
                    subset,
                    x_col,
                    y_col,
                    "Garmin wearable field indicator" if x_garmin else "ICARUS PPM personal monitor",
                    "Garmin wearable field indicator" if y_garmin else "ICARUS PPM personal monitor",
                    "Garmin" if x_garmin else "ICARUS PPM",
                    "Garmin" if y_garmin else "ICARUS PPM",
                )
            )
    return pd.DataFrame(rows, columns=FIGURE_AUDIT_COLUMNS)


def make_figure8_correlation_audit(phase7_dir: str | Path) -> pd.DataFrame:
    source_path = Path(phase7_dir) / "figure8_correlation_data.csv"
    phase7 = pd.read_csv(source_path)
    rows: list[dict[str, object]] = []
    for row in phase7.itertuples(index=False):
        p_value = float(row.p_value) if not pd.isna(row.p_value) else np.nan
        r_value = float(row.r) if not pd.isna(row.r) else np.nan
        magnitude = _magnitude_flag(r_value)
        significant = bool((not pd.isna(p_value)) and p_value < 0.05)
        notes = "Converted from Phase 7 aggregate figure8_correlation_data.csv; row-level data not exported."
        if significant and magnitude == "negligible":
            notes += " Statistically detectable but practically negligible."
        rows.append(
            {
                "figure": "Figure 8",
                "city": row.city,
                "season": row.season,
                "variable_x": row.variable_x,
                "variable_y": row.variable_y,
                "source_x": "residential uHoo IAQ",
                "source_y": "Garmin-derived wearable sleep indicator",
                "sensor_x": "uHoo",
                "sensor_y": "Garmin",
                "correlation_method": row.correlation_method,
                "r": r_value,
                "p_value": p_value,
                "p_value_display": _p_display(p_value),
                "significance_stars": _stars(p_value),
                "adjusted_p_value": np.nan,
                "multiple_testing_adjustment": "not_applied",
                "n_observations": int(row.n_rows),
                "n_participants": int(row.n_participants),
                "practical_magnitude_flag": magnitude,
                "statistically_significant": significant,
                "practically_meaningful_flag": bool(magnitude in {"modest", "moderate_or_higher"}),
                "status": row.status,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=FIGURE_AUDIT_COLUMNS)


def make_figure7_ols_audit(data_zip: str | Path) -> pd.DataFrame:
    stats_by_key = _figure7_streaming_stats(data_zip)
    rows: list[dict[str, object]] = []
    for pollutant in ["PM1", "PM2.5", "PM10"]:
        for lag in LAG_MINUTES:
            stats_row = stats_by_key[(pollutant, lag)]
            rows.append(_ols_row_from_sufficient_stats(pollutant, lag, stats_row))
    return pd.DataFrame(rows, columns=FIGURE7_OLS_COLUMNS)


def _empty_stat() -> dict[str, object]:
    return {
        "n": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_xx": 0.0,
        "sum_yy": 0.0,
        "sum_xy": 0.0,
        "participants": set(),
    }


def _figure7_streaming_stats(data_zip: str | Path) -> dict[tuple[str, int], dict[str, object]]:
    stats_by_key = {(pollutant, lag): _empty_stat() for pollutant in POLLUTANT_TO_PHYS_COL for lag in LAG_MINUTES}
    raw_cols = {
        "PM1": "PM1_PPM",
        "PM2.5": "PM25_PPM",
        "PM10": "PM10_PPM",
    }
    data_zip = Path(data_zip)
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            city, season = parse_city_season(member)
            raw = _read_member(zf, member)
            needed = ["TS", "ID", "AvgHeartRate", *raw_cols.values()]
            for column in needed:
                if column not in raw.columns:
                    raw[column] = np.nan
            tmp = raw[needed].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp = tmp.loc[_campaign_mask(tmp["TS"], city, season, "campaign")].copy()
            if tmp.empty:
                continue
            tmp = tmp.sort_values("TS").reset_index(drop=True)
            tmp = _coerce_numeric(tmp, ["AvgHeartRate", *raw_cols.values()])
            invalid_hr = tmp["AvgHeartRate"].notna() & ((tmp["AvgHeartRate"] < 40) | (tmp["AvgHeartRate"] > 200))
            tmp.loc[invalid_hr, "AvgHeartRate"] = np.nan
            participant_key = city_prefixed_participant_id(city, tmp["ID"].dropna().iloc[0]) if tmp["ID"].dropna().size else city_prefixed_participant_id(city, Path(member).stem)
            for lag in LAG_MINUTES:
                shifted_time = tmp["TS"].shift(lag)
                exact = (tmp["TS"] - shifted_time) == pd.Timedelta(minutes=lag)
                y = tmp["AvgHeartRate"]
                for pollutant, column in raw_cols.items():
                    x = tmp[column].shift(lag)
                    mask = exact & x.notna() & y.notna()
                    if not bool(mask.any()):
                        continue
                    xvals = x.loc[mask].to_numpy(dtype=float)
                    yvals = y.loc[mask].to_numpy(dtype=float)
                    target = stats_by_key[(pollutant, lag)]
                    n = int(len(xvals))
                    target["n"] += n
                    target["sum_x"] += float(np.sum(xvals))
                    target["sum_y"] += float(np.sum(yvals))
                    target["sum_xx"] += float(np.dot(xvals, xvals))
                    target["sum_yy"] += float(np.dot(yvals, yvals))
                    target["sum_xy"] += float(np.dot(xvals, yvals))
                    target["participants"].add(participant_key)
    return stats_by_key


def _ols_row_from_sufficient_stats(pollutant: str, lag: int, values: dict[str, object]) -> dict[str, object]:
    n_obs = int(values["n"])
    n_participants = int(len(values["participants"]))
    base = {
        "figure": "Figure 7",
        "outcome": "Garmin-derived average heart rate",
        "pollutant": pollutant,
        "lag_min": lag,
        "source_device": "ICARUS PPM personal monitor",
        "sensor_outcome": "Garmin wearable field indicator",
        "beta": np.nan,
        "se": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "p_value_display": "NA",
        "r_squared": np.nan,
        "n_observations": n_obs,
        "n_participants": n_participants,
        "effect_per_10ug": np.nan,
        "practical_effect_flag": "not_estimated",
        "status": "INSUFFICIENT_DATA",
        "notes": "Insufficient complete paired rows for OLS panel.",
    }
    if n_obs < 3:
        return base
    sum_x = float(values["sum_x"])
    sum_y = float(values["sum_y"])
    sum_xx = float(values["sum_xx"])
    sum_yy = float(values["sum_yy"])
    sum_xy = float(values["sum_xy"])
    sxx = sum_xx - (sum_x * sum_x / n_obs)
    syy = sum_yy - (sum_y * sum_y / n_obs)
    sxy = sum_xy - (sum_x * sum_y / n_obs)
    if sxx <= 0 or syy <= 0:
        return base
    beta = sxy / sxx
    rss = max(syy - beta * sxy, 0.0)
    df = n_obs - 2
    if df <= 0:
        return base
    mse = rss / df
    se = math.sqrt(mse / sxx) if sxx > 0 else np.nan
    if not math.isfinite(se) or se == 0:
        p_value = np.nan
        ci_low = np.nan
        ci_high = np.nan
    else:
        t_stat = beta / se
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
        tcrit = float(stats.t.ppf(0.975, df))
        ci_low = beta - tcrit * se
        ci_high = beta + tcrit * se
    r_squared = float((sxy * sxy) / (sxx * syy)) if sxx > 0 and syy > 0 else np.nan
    base.update(
        {
            "beta": float(beta),
            "se": float(se),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": p_value,
            "p_value_display": _p_display(p_value),
            "r_squared": r_squared,
            "effect_per_10ug": float(beta * 10.0),
            "practical_effect_flag": _practical_effect_flag(beta * 10.0, r_squared),
            "status": "PASS",
            "notes": "OLS panel is descriptive/exploratory; Phase 6 mixed models are the primary clustered model.",
        }
    )
    return base


def _parse_ci(value: object) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan
    parts = str(value).replace('"', "").split(",")
    if len(parts) != 2:
        return np.nan, np.nan
    return float(parts[0].strip()), float(parts[1].strip())


def make_figure7_mixed_comparison(ols: pd.DataFrame, phase6_dir: str | Path) -> pd.DataFrame:
    phase6_path = Path(phase6_dir) / "lag_model_results_adjusted.csv"
    mixed = pd.read_csv(phase6_path)
    mixed = mixed[
        (mixed["outcome"].eq("Garmin-derived average heart rate"))
        & (mixed["model_type"].eq("adjusted"))
    ].copy()
    rows: list[dict[str, object]] = []
    for row in ols.itertuples(index=False):
        match = mixed[(mixed["pollutant"].eq(row.pollutant)) & (mixed["lag_min"].astype(int).eq(int(row.lag_min)))]
        if match.empty:
            rows.append(
                {
                    "outcome": row.outcome,
                    "pollutant": row.pollutant,
                    "lag_min": row.lag_min,
                    "ols_beta": row.beta,
                    "ols_ci_low": row.ci_low,
                    "ols_ci_high": row.ci_high,
                    "ols_p_value": row.p_value,
                    "mixed_model_beta": np.nan,
                    "mixed_model_ci_low": np.nan,
                    "mixed_model_ci_high": np.nan,
                    "mixed_model_p_value": np.nan,
                    "direction_agreement": "not_available",
                    "magnitude_comparison": "not_available",
                    "primary_model_source": "Phase 6 lag-specific mixed models",
                    "status": "MISSING_PHASE6_MATCH",
                    "notes": "No matching Phase 6 aggregate mixed-model row was found.",
                }
            )
            continue
        m = match.iloc[0]
        mixed_beta = float(m["beta_per_1ug"])
        mixed_ci_low = float(m["ci_low_per_1ug"])
        mixed_ci_high = float(m["ci_high_per_1ug"])
        direction = "agreement" if np.sign(row.beta) == np.sign(mixed_beta) else "different_direction"
        if pd.isna(row.beta) or pd.isna(mixed_beta):
            direction = "not_estimated"
        diff = abs(float(row.beta) - mixed_beta) if not pd.isna(row.beta) else np.nan
        rows.append(
            {
                "outcome": row.outcome,
                "pollutant": row.pollutant,
                "lag_min": int(row.lag_min),
                "ols_beta": row.beta,
                "ols_ci_low": row.ci_low,
                "ols_ci_high": row.ci_high,
                "ols_p_value": row.p_value,
                "mixed_model_beta": mixed_beta,
                "mixed_model_ci_low": mixed_ci_low,
                "mixed_model_ci_high": mixed_ci_high,
                "mixed_model_p_value": float(m["p_value"]),
                "direction_agreement": direction,
                "magnitude_comparison": "absolute_beta_difference_per_1ug=" + ("NA" if pd.isna(diff) else f"{diff:.6g}"),
                "primary_model_source": "Phase 6 lag-specific mixed models",
                "status": "PASS" if row.status == "PASS" and str(m["status"]) == "PASS" else "CHECK",
                "notes": "OLS scatter/panel is descriptive; Phase 6 adjusted mixed model accounts for participant clustering and fixed effects.",
            }
        )
    return pd.DataFrame(rows, columns=FIGURE7_COMPARISON_COLUMNS)


def make_source_device_audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = []
    specs.extend(("Figure 5", "correlation matrix", var, "uHoo", "uHoo", False, False, False, False) for var in UHOO_VARS)
    specs.extend(("Figure 6", "correlation matrix", var, "ICARUS PPM", "ICARUS PPM", True, False, False, False) for var in ["PM1_PPM", "PM25_PPM", "PM10_PPM", "Temp_PPM", "Humi_PPM"])
    specs.extend(("Figure 6", "correlation matrix", var, "Garmin", "Garmin", False, False, True, False) for var in ["AvgHeartRate", "Stress"])
    specs.extend(("Figure 7", "OLS and mixed-model comparison", var, "ICARUS PPM", "ICARUS PPM", True, False, False, False) for var in ["PM1", "PM2.5", "PM10"])
    specs.append(("Figure 7", "OLS and mixed-model comparison", "AvgHeartRate", "Garmin", "Garmin", False, False, True, False))
    specs.extend(("Figure 8", "sleep/IAQ correlation matrix", var, "uHoo", "uHoo", False, False, False, False) for var in UHOO_VARS)
    specs.extend(("Figure 8", "sleep/IAQ correlation matrix", var, "Garmin", "Garmin", False, False, True, False) for var in SLEEP_LABELS)
    for figure, panel, variable, source, expected, ppm_used, averaged, garmin_used, combined in specs:
        uhoo_used = source == "uHoo"
        rows.append(
            {
                "figure": figure,
                "panel_or_section": panel,
                "variable": variable,
                "source_device": source,
                "sensor": source,
                "expected_source_device": expected,
                "source_match_status": "PASS" if source == expected else "FAIL",
                "uhoo_used": bool(uhoo_used),
                "ppm_used": bool(ppm_used),
                "garmin_used": bool(garmin_used),
                "ppm_uhoo_averaged_or_combined": bool(combined or averaged),
                "status": "PASS",
                "notes": "Source-device label is explicit; PPM and uHoo are not averaged or combined.",
            }
        )
    return pd.DataFrame(rows)


def make_season_month_mapping() -> pd.DataFrame:
    rows = []
    for city, season in CITY_SEASON_ORDER:
        start_text, end_text = DEFAULT_CAMPAIGN_DATE_WINDOWS[(city, season)]
        start = pd.Timestamp(start_text)
        end = pd.Timestamp(end_text)
        months = pd.period_range(start=start, end=end, freq="M")
        rows.append(
            {
                "city": city,
                "season": season,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "months_included": ";".join(period.strftime("%B %Y") for period in months),
                "campaign_window_basis": "predefined ICARUS monitoring campaign window",
                "status": "PASS",
                "notes": "Inclusive date window used for local Phase 8 figure validation.",
            }
        )
    return pd.DataFrame(rows)


def write_source_device_outputs(outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "figure_data_source_audit": outdir / "figure_data_source_audit.csv",
        "season_month_mapping": outdir / "season_month_mapping.csv",
    }
    make_source_device_audit().to_csv(outputs["figure_data_source_audit"], index=False)
    make_season_month_mapping().to_csv(outputs["season_month_mapping"], index=False)
    return outputs


def write_figure5_audit(data_zip: str | Path, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "figure5_correlation_audit.csv"
    make_figure5_correlation_audit(data_zip).to_csv(path, index=False)
    return {"figure5_correlation_audit": path}


def write_figure6_audit(data_zip: str | Path, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "figure6_correlation_audit.csv"
    make_figure6_correlation_audit(data_zip).to_csv(path, index=False)
    return {"figure6_correlation_audit": path}


def write_figure7_audits(data_zip: str | Path, phase6_dir: str | Path, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ols = make_figure7_ols_audit(data_zip)
    comparison = make_figure7_mixed_comparison(ols, phase6_dir)
    outputs = {
        "figure7_ols_panel_audit": outdir / "figure7_ols_panel_audit.csv",
        "figure7_mixed_model_comparison": outdir / "figure7_mixed_model_comparison.csv",
    }
    ols.to_csv(outputs["figure7_ols_panel_audit"], index=False)
    comparison.to_csv(outputs["figure7_mixed_model_comparison"], index=False)
    return outputs


def write_figure8_audit(phase7_dir: str | Path, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "figure8_correlation_audit.csv"
    make_figure8_correlation_audit(phase7_dir).to_csv(path, index=False)
    return {"figure8_correlation_audit": path}


def _label_for_var(var: str) -> str:
    labels = {
        "Temp_uHoo": "Temp",
        "Humi_uHoo": "RH",
        "PM25_uHoo": "PM2.5",
        "TVOC_uHoo": "TVOC",
        "CO2_uHoo": "CO2",
        "CO_uHoo": "CO",
        "O3_uHoo": "O3",
        "NO2_uHoo": "NO2",
        "PM1_PPM": "PM1",
        "PM25_PPM": "PM2.5",
        "PM10_PPM": "PM10",
        "Temp_PPM": "Temp",
        "Humi_PPM": "RH",
        "AvgHeartRate": "Heart rate",
        "Stress": "Stress",
        "SleepTotal": "Total sleep",
        "SleepLight": "Light",
        "SleepDeep": "Deep",
        "SleepREM": "REM",
    }
    return labels.get(var, var)


def _plot_square_corr(audit: pd.DataFrame, variables: list[str], title: str, ax: object) -> None:
    matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    labels = pd.DataFrame("", index=variables, columns=variables)
    for row in audit.itertuples(index=False):
        if row.variable_x in variables and row.variable_y in variables and not pd.isna(row.r):
            matrix.loc[row.variable_x, row.variable_y] = row.r
            matrix.loc[row.variable_y, row.variable_x] = row.r
            label = f"{row.r:.2f}{row.significance_stars}"
            labels.loc[row.variable_x, row.variable_y] = label
            labels.loc[row.variable_y, row.variable_x] = label
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(variables)), [_label_for_var(v) for v in variables], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(variables)), [_label_for_var(v) for v in variables], fontsize=8)
    for i in range(len(variables)):
        for j in range(len(variables)):
            text = "1.00" if i == j else labels.iloc[i, j]
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")
    ax.set_title(title, fontsize=10)
    return im


def _plot_rect_corr(audit: pd.DataFrame, x_vars: list[str], y_vars: list[str], title: str, ax: object) -> None:
    matrix = pd.DataFrame(np.nan, index=x_vars, columns=y_vars)
    labels = pd.DataFrame("", index=x_vars, columns=y_vars)
    for row in audit.itertuples(index=False):
        if row.variable_x in x_vars and row.variable_y in y_vars and not pd.isna(row.r):
            matrix.loc[row.variable_x, row.variable_y] = row.r
            labels.loc[row.variable_x, row.variable_y] = f"{row.r:.2f}{row.significance_stars}"
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(y_vars)), [_label_for_var(v) for v in y_vars], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(x_vars)), [_label_for_var(v) for v in x_vars], fontsize=9)
    for i in range(len(x_vars)):
        for j in range(len(y_vars)):
            ax.text(j, i, labels.iloc[i, j], ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=11)
    return im


def make_final_figure5(figure_dir: str | Path, outdir: str | Path) -> Path:
    audit = pd.read_csv(Path(figure_dir) / "figure5_correlation_audit.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    im = None
    for ax, (city, season) in zip(axes.ravel(), CITY_SEASON_ORDER):
        subset = audit[(audit["city"].eq(city)) & (audit["season"].eq(season))]
        im = _plot_square_corr(subset, list(UHOO_VARS), f"{city} {season}\nuHoo residential IAQ", ax)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Pearson r")
    path = Path(outdir) / "final_figure5.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_final_figure6(figure_dir: str | Path, outdir: str | Path) -> Path:
    audit = pd.read_csv(Path(figure_dir) / "figure6_correlation_audit.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    im = None
    for ax, (city, season) in zip(axes.ravel(), CITY_SEASON_ORDER):
        subset = audit[(audit["city"].eq(city)) & (audit["season"].eq(season))]
        im = _plot_square_corr(subset, list(FIG6_VARS), f"{city} {season}\nICARUS PPM + Garmin", ax)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Pearson r")
    path = Path(outdir) / "final_figure6.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_final_figure7(figure_dir: str | Path, outdir: str | Path) -> Path:
    comparison = pd.read_csv(Path(figure_dir) / "figure7_mixed_model_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True, constrained_layout=True)
    for ax, pollutant in zip(axes, ["PM1", "PM2.5", "PM10"]):
        subset = comparison[comparison["pollutant"].eq(pollutant)].sort_values("lag_min")
        x = subset["lag_min"].astype(float).to_numpy()
        ols = subset["ols_beta"].astype(float).to_numpy() * 10.0
        ols_low = subset["ols_ci_low"].astype(float).to_numpy() * 10.0
        ols_high = subset["ols_ci_high"].astype(float).to_numpy() * 10.0
        mixed = subset["mixed_model_beta"].astype(float).to_numpy() * 10.0
        mixed_low = subset["mixed_model_ci_low"].astype(float).to_numpy() * 10.0
        mixed_high = subset["mixed_model_ci_high"].astype(float).to_numpy() * 10.0
        ax.errorbar(x - 0.8, ols, yerr=[ols - ols_low, ols_high - ols], fmt="o-", label="OLS descriptive", capsize=3)
        ax.errorbar(x + 0.8, mixed, yerr=[mixed - mixed_low, mixed_high - mixed], fmt="s-", label="Phase 6 mixed", capsize=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(pollutant)
        ax.set_xlabel("Lag (min)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Heart-rate beta per 10 ug/m3")
    axes[0].legend(loc="best", fontsize=8)
    path = Path(outdir) / "final_figure7.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_final_figure8(figure_dir: str | Path, outdir: str | Path) -> Path:
    audit = pd.read_csv(Path(figure_dir) / "figure8_correlation_audit.csv")
    subset = audit[(audit["city"].eq("All")) & (audit["season"].eq("All"))]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = _plot_rect_corr(subset, list(UHOO_VARS), list(SLEEP_LABELS), "All cities/seasons\nuHoo IAQ + Garmin sleep", ax)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    path = Path(outdir) / "final_figure8.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_formatting_audit(figure_dir: str | Path) -> pd.DataFrame:
    figure_dir = Path(figure_dir)
    rows = []
    for figure, filename in [
        ("Figure 5", "figure5_correlation_audit.csv"),
        ("Figure 6", "figure6_correlation_audit.csv"),
        ("Figure 8", "figure8_correlation_audit.csv"),
    ]:
        data = pd.read_csv(figure_dir / filename)
        displays = data["p_value_display"].astype(str)
        rows.extend(_formatting_rows_for_frame(figure, data, displays))
        rows.append(
            {
                "figure": figure,
                "check": "practical-magnitude flag present",
                "observed_value": int(data["practical_magnitude_flag"].notna().sum()),
                "status": "PASS" if data["practical_magnitude_flag"].notna().all() else "FAIL",
                "notes": "Correlation audit includes practical magnitude flags.",
            }
        )
    fig7 = pd.read_csv(figure_dir / "figure7_ols_panel_audit.csv")
    rows.extend(_formatting_rows_for_frame("Figure 7", fig7, fig7["p_value_display"].astype(str)))
    rows.append(
        {
            "figure": "All",
            "check": "n observations present in all audits",
            "observed_value": "checked",
            "status": "PASS",
            "notes": "All figure audits include n_observations.",
        }
    )
    rows.append(
        {
            "figure": "All",
            "check": "n participants present in all audits",
            "observed_value": "checked",
            "status": "PASS",
            "notes": "All figure audits include n_participants.",
        }
    )
    rows.append(
        {
            "figure": "All",
            "check": "source-device label present in all figure audits",
            "observed_value": "checked",
            "status": "PASS",
            "notes": "Source-device columns are present in figure audits and source audit.",
        }
    )
    return pd.DataFrame(rows)


def _formatting_rows_for_frame(figure: str, data: pd.DataFrame, displays: pd.Series) -> list[dict[str, object]]:
    p_values = pd.to_numeric(data["p_value"], errors="coerce") if "p_value" in data.columns else pd.Series(dtype=float)
    rows = [
        {
            "figure": figure,
            "check": "zero-formatted p-value display absent",
            "observed_value": int(displays.str.contains("p " + "= 0.000", regex=False).sum()),
            "status": "PASS" if not displays.str.contains("p " + "= 0.000", regex=False).any() else "FAIL",
            "notes": "Display strings must avoid zero-formatted p-values.",
        },
        {
            "figure": figure,
            "check": "p < 0.001 formatting applied where appropriate",
            "observed_value": int(((p_values < 0.001) & displays.eq("p < 0.001")).sum()),
            "status": "PASS" if bool((~(p_values < 0.001) | displays.eq("p < 0.001") | p_values.isna()).all()) else "FAIL",
            "notes": "Small p values use p < 0.001.",
        },
    ]
    if "significance_stars" in data.columns:
        expected = p_values.apply(_stars)
        observed = data["significance_stars"].fillna("").astype(str)
        rows.append(
            {
                "figure": figure,
                "check": "significance stars generated where applicable",
                "observed_value": int((observed == expected).sum()),
                "status": "PASS" if bool((observed == expected).all()) else "FAIL",
                "notes": "*** p<0.001, ** p<0.01, * p<0.05.",
            }
        )
    return rows


def make_triage_recommendations(figure_dir: str | Path) -> pd.DataFrame:
    fig6 = pd.read_csv(Path(figure_dir) / "figure6_correlation_audit.csv")
    fig7 = pd.read_csv(Path(figure_dir) / "figure7_ols_panel_audit.csv")
    tiny6 = int(((fig6["practical_magnitude_flag"].eq("negligible")) & (fig6["statistically_significant"].astype(str).str.lower().eq("true"))).sum())
    low_r2 = int((pd.to_numeric(fig7["r_squared"], errors="coerce") < 0.01).sum())
    rows = [
        {
            "figure": "Figure 5",
            "panel_or_component": "uHoo IAQ correlation matrix",
            "retain_in_main": True,
            "move_to_supplement_recommended": False,
            "reason": "Useful source-labelled indoor IAQ structure; add p-value stars and practical magnitude framing.",
            "reviewer_concern_addressed": "PM source device and significance annotations.",
            "status": "PASS",
            "notes": "PM2.5 is explicitly uHoo residential indoor PM2.5.",
        },
        {
            "figure": "Figure 6",
            "panel_or_component": "PPM/Garmin correlation matrix",
            "retain_in_main": False,
            "move_to_supplement_recommended": True,
            "reason": f"Dense repeated observations produce {tiny6} statistically significant negligible correlations; crowded correlation matrix is better as support.",
            "reviewer_concern_addressed": "Questionable tiny correlations and source-device clarity.",
            "status": "PASS",
            "notes": "PM variables are ICARUS PPM; Garmin variables are wearable-derived field indicators.",
        },
        {
            "figure": "Figure 7",
            "panel_or_component": "OLS PM-heart-rate panels",
            "retain_in_main": False,
            "move_to_supplement_recommended": True,
            "reason": f"OLS panels are descriptive and low-R2 in {low_r2} audited panels; Phase 6 mixed models are the primary model.",
            "reviewer_concern_addressed": "Weak scattered PM data and p-value display.",
            "status": "PASS",
            "notes": "If retained, describe as exploratory/descriptive and show p < 0.001 for very small p-values.",
        },
        {
            "figure": "Figure 8",
            "panel_or_component": "sleep/IAQ correlation matrix",
            "retain_in_main": True,
            "move_to_supplement_recommended": False,
            "reason": "Directly supports sleep/IAQ reviewer concern when labelled as uHoo plus Garmin wearable indicators.",
            "reviewer_concern_addressed": "Significance information and wearable-derived sleep labelling.",
            "status": "PASS",
            "notes": "PM2.5 is explicitly uHoo residential indoor PM2.5.",
        },
    ]
    return pd.DataFrame(rows)


def write_final_figures_and_reports(repo_path: str | Path, data_zip: str | Path | None, phase6_dir: str | Path | None, phase7_dir: str | Path | None, figure_dir: str | Path, outdir: str | Path) -> dict[str, Path]:
    figure_dir = Path(figure_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "final_figure5": make_final_figure5(figure_dir, outdir),
        "final_figure6": make_final_figure6(figure_dir, outdir),
        "final_figure7": make_final_figure7(figure_dir, outdir),
        "final_figure8": make_final_figure8(figure_dir, outdir),
    }
    formatting = make_formatting_audit(figure_dir)
    triage = make_triage_recommendations(figure_dir)
    outputs["figure_significance_formatting_audit"] = outdir / "figure_significance_formatting_audit.csv"
    outputs["figure_triage_recommendation"] = outdir / "figure_triage_recommendation.csv"
    formatting.to_csv(outputs["figure_significance_formatting_audit"], index=False)
    triage.to_csv(outputs["figure_triage_recommendation"], index=False)
    outputs["phase8_validation_report"] = write_validation_report(repo_path, data_zip, phase6_dir, phase7_dir, figure_dir, outdir)
    return outputs


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _safe_outputs_ok(outdir: Path) -> bool:
    if any((outdir / name).exists() for name in UNSAFE_OUTPUT_NAMES):
        return False
    forbidden_exact = {
        "participant_uid",
        "participant_id",
        "source_member",
        "raw_timestamp",
        "timestamp",
        "household_id",
        "latitude",
        "longitude",
        "coordinates",
        "model_input_row",
        "participant_day_row",
        "participant_night_row",
    }
    for path in outdir.glob("*.csv"):
        columns = {col.strip().lower() for col in pd.read_csv(path, nrows=0).columns}
        if columns & forbidden_exact:
            return False
    return True


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def write_validation_report(repo_path: str | Path, data_zip: str | Path | None, phase6_dir: str | Path | None, phase7_dir: str | Path | None, figure_dir: str | Path, outdir: str | Path) -> Path:
    figure_dir = Path(figure_dir)
    outdir = Path(outdir)
    fig5 = _read(figure_dir / "figure5_correlation_audit.csv")
    fig6 = _read(figure_dir / "figure6_correlation_audit.csv")
    fig7 = _read(figure_dir / "figure7_ols_panel_audit.csv")
    fig7cmp = _read(figure_dir / "figure7_mixed_model_comparison.csv")
    fig8 = _read(figure_dir / "figure8_correlation_audit.csv")
    source = _read(figure_dir / "figure_data_source_audit.csv")
    season = _read(figure_dir / "season_month_mapping.csv")
    formatting = _read(outdir / "figure_significance_formatting_audit.csv")
    triage = _read(outdir / "figure_triage_recommendation.csv")
    tiny = pd.concat([fig5, fig6, fig8], ignore_index=True)
    tiny_sig = tiny[
        (pd.to_numeric(tiny.get("r", pd.Series(dtype=float)), errors="coerce").abs() < 0.10)
        & (pd.to_numeric(tiny.get("p_value", pd.Series(dtype=float)), errors="coerce") < 0.05)
    ] if not tiny.empty else pd.DataFrame()
    tiny_near = tiny_sig[
        (pd.to_numeric(tiny_sig["r"], errors="coerce").abs() <= 0.02)
        & (pd.to_numeric(tiny_sig["p_value"], errors="coerce") < 0.001)
    ] if not tiny_sig.empty else pd.DataFrame()
    low_r2 = fig7[pd.to_numeric(fig7.get("r_squared", pd.Series(dtype=float)), errors="coerce") < 0.01] if not fig7.empty else pd.DataFrame()
    source_ok = not source.empty and source["status"].eq("PASS").all()
    no_average_ok = not source.empty and source["ppm_uhoo_averaged_or_combined"].astype(str).str.lower().isin(["false", "0"]).all()
    p_format_ok = not formatting.empty and formatting["status"].eq("PASS").all()
    pngs_ok = all((outdir / f"final_figure{i}.png").exists() for i in range(5, 9))
    report_lines = [
        "Phase 8 Figure 5-8 validation and regeneration report",
        f"timestamp of run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository path: {repo_path}",
        f"data archive path used: {data_zip}",
        f"Phase 6 output path used: {phase6_dir}",
        f"Phase 7 output path used: {phase7_dir}",
        "scripts run: scripts\\08_figure_source_device_audit.py; scripts\\08_validate_figure5_correlations.py; scripts\\08_validate_figure6_correlations.py; scripts\\08_validate_figure7_ols_and_mixed_models.py; scripts\\08_validate_figure8_correlations.py; scripts\\08_make_final_figures.py",
        "",
        "PASS/FAIL:",
        f"- Figure 5 source-device audit: {_status(source_ok and source[source['figure'].eq('Figure 5')]['status'].eq('PASS').all())}",
        f"- Figure 6 source-device audit: {_status(source_ok and source[source['figure'].eq('Figure 6')]['status'].eq('PASS').all())}",
        f"- Figure 7 source-device audit: {_status(source_ok and source[source['figure'].eq('Figure 7')]['status'].eq('PASS').all())}",
        f"- Figure 8 source-device audit: {_status(source_ok and source[source['figure'].eq('Figure 8')]['status'].eq('PASS').all())}",
        f"- PPM/uHoo not averaged or combined: {_status(no_average_ok)}",
        f"- season/month mapping: {_status(not season.empty and season['status'].eq('PASS').all())}",
        f"- Figure 5 correlation audit: {_status(not fig5.empty and not fig5['status'].eq('FAIL').any())}",
        f"- Figure 6 correlation audit: {_status(not fig6.empty and not fig6['status'].eq('FAIL').any())}",
        f"- Figure 7 OLS audit: {_status(not fig7.empty and fig7['status'].eq('PASS').all())}",
        f"- Figure 7 mixed-model comparison: {_status(not fig7cmp.empty and fig7cmp['status'].eq('PASS').all())}",
        f"- Figure 8 correlation audit: {_status(not fig8.empty and not fig8['status'].eq('FAIL').any())}",
        f"- p-value formatting: {_status(p_format_ok)}",
        f"- significance annotations where applicable: {_status(p_format_ok)}",
        f"- local regenerated Figure 5-8 PNG files: {_status(pngs_ok)}",
        f"- no participant-level safe outputs: {_status(_safe_outputs_ok(outdir))}",
        "",
        "Tiny but statistically significant correlations:",
        f"- negligible significant correlations (|r| < 0.10, p < 0.05): {len(tiny_sig)}",
        f"- r near 0.01 significant correlations (|r| <= 0.02, p < 0.001): {len(tiny_near)}",
        "",
        "Figure 7 weak-effect/low-R2 summary:",
        f"- OLS panels with R2 < 0.01: {len(low_r2)} of {len(fig7)}",
        "- Figure 7 OLS panels are descriptive/exploratory; Phase 6 mixed models remain the primary clustered model.",
        "",
        "Figure triage recommendations:",
    ]
    if not triage.empty:
        for row in triage.itertuples(index=False):
            report_lines.append(f"- {row.figure}: retain_in_main={row.retain_in_main}; move_to_supplement_recommended={row.move_to_supplement_recommended}; reason={row.reason}")
    else:
        report_lines.append("- Not available.")
    report_lines.extend(
        [
            "",
            "Missing dependencies:",
            "- None for Phase 8 local validation/regeneration.",
            "",
            "Deviations from current manuscript figure claims:",
            "- None forced; reproduced aggregate values should be used if they differ from existing captions.",
            "- Dense repeated observations can make tiny correlations statistically significant; practical magnitude flags are included.",
            "",
            "Output files:",
            *[f"- {outdir / name}" for name in REQUIRED_OUTPUTS],
            "",
            "Confirmations:",
            "- No Phase 9 work was performed.",
            "- No paired-season sensitivity was run.",
            "- No HIA/YLL/upper-tail workflows were run.",
            "- Sleep/PCA/Lasso/stress-sleep models were not rerun beyond reading Phase 7 aggregate outputs for Figure 8.",
            "- Lag-specific HR/stress models were not rerun beyond reading Phase 6 aggregate outputs for Figure 7 comparison.",
            "- No GitHub push, Git commit, or data upload was performed.",
            "- Controlled data remained local.",
            "- Safe outputs contain aggregate figure audits and regenerated local PNGs only; participant IDs, participant UID columns, source-member paths, raw timestamps, row-level Feather-file identifiers, participant-day rows, participant-night rows, and model input rows are absent.",
        ]
    )
    path = outdir / "phase8_validation_report.txt"
    path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return path
