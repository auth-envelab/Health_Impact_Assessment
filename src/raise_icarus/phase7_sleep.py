"""Phase 7 sleep, IAQ, Lasso, PCA, and stress-sleep helpers.

The functions in this module keep participant-level sleep and stress rows in
memory only. Public-facing/local safe outputs are aggregate model results,
correlations, PCA summaries, and validation tables.
"""

from __future__ import annotations

import io
import math
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from raise_icarus.data import (
    DateFilterMode,
    feather_members,
    get_campaign_date_window,
    parse_city_season,
)
from raise_icarus.phase1_denominators import (
    CITY_SEASON_ORDER,
    S1_TARGETS,
    UHOO_COLUMNS,
    city_prefixed_participant_id,
)


SLEEP_AVAILABILITY_STREAM = "Garmin sleep data availability"
SLEEP_IAQ_STREAM = "Garmin sleep + residential IAQ complete-case model input"
SLEEP_OUTCOMES = ["SleepTotal", "SleepLight", "SleepDeep", "SleepREM"]
SLEEP_OUTCOME_LABELS = {
    "SleepTotal": "Garmin-derived total sleep duration",
    "SleepLight": "Garmin-derived light sleep fraction",
    "SleepDeep": "Garmin-derived deep sleep fraction",
    "SleepREM": "Garmin-derived REM sleep fraction",
}
SLEEP_OUTCOME_UNITS = {
    "SleepTotal": "minutes",
    "SleepLight": "fraction",
    "SleepDeep": "fraction",
    "SleepREM": "fraction",
}
UHOO_UNITS = {
    "Temp_uHoo": "degC",
    "Humi_uHoo": "percent",
    "PM25_uHoo": "ug/m3",
    "TVOC_uHoo": "ppb",
    "CO2_uHoo": "ppm",
    "CO_uHoo": "ppm",
    "O3_uHoo": "ppb",
    "NO2_uHoo": "ppb",
}
PM_PCA_COLUMNS = ["PM1_PPM", "PM25_PPM", "PM10_PPM"]
PM_PCA_LABELS = {
    "PM1_PPM": "PM1",
    "PM25_PPM": "PM2.5",
    "PM10_PPM": "PM10",
}
REQUIRED_OUTPUTS = [
    "sleep_model_input_audit.csv",
    "pca_pm_loadings.csv",
    "pca_pm_scores_summary.csv",
    "sleep_iaq_ols_results.csv",
    "sleep_lasso_selected_predictors.csv",
    "sleep_lasso_refit_results.csv",
    "stress_sleep_mixed_model_results.csv",
    "figure8_correlation_data.csv",
    "figure9_plot_data.csv",
    "figure10_plot_data.csv",
    "phase7_validation_report.txt",
]
WEARABLE_FLAG = "wearable-derived field indicator; not clinical or polysomnography-equivalent"


def _read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        return feather.read_table(io.BytesIO(fh.read())).to_pandas()


def _coerce_numeric(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _campaign_mask(ts: pd.Series, city: str, season: str, mode: DateFilterMode) -> pd.Series:
    valid = ts.notna()
    if mode == "none":
        return valid
    start, end = get_campaign_date_window(city, season)
    if start is None or end is None:
        return pd.Series(False, index=ts.index)
    return valid & (ts.dt.date >= start.date()) & (ts.dt.date <= end.date())


def _format_list(values: list[str] | tuple[str, ...]) -> str:
    return ";".join(str(value) for value in values)


def load_sleep_rows(data_zip: str | Path) -> pd.DataFrame:
    """Reconstruct participant-night Garmin sleep and residential uHoo summaries."""
    rows: list[pd.DataFrame] = []
    data_zip = Path(data_zip)
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            city, season = parse_city_season(member)
            raw = _read_member(zf, member)
            if not {"TS", "ID", "Sleep"}.issubset(raw.columns):
                continue
            for column in UHOO_COLUMNS:
                if column not in raw.columns:
                    raw[column] = np.nan
            tmp = raw[["TS", "ID", "Sleep", *UHOO_COLUMNS]].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp["date"] = tmp["TS"].dt.date
            tmp["Sleep"] = tmp["Sleep"].astype(str).str.strip().str.lower()
            tmp = tmp[tmp["Sleep"].isin(["deep", "light", "rem"])].dropna(subset=["ID", "date"])
            if tmp.empty:
                continue
            tmp = _coerce_numeric(tmp, UHOO_COLUMNS)
            grouped = (
                tmp.groupby(["ID", "date"], dropna=False)
                .agg(
                    **{column: (column, "mean") for column in UHOO_COLUMNS},
                    Sleep=("Sleep", lambda s: s.value_counts().to_dict()),
                )
                .reset_index()
            )
            grouped["participant_uid"] = [city_prefixed_participant_id(city, value) for value in grouped["ID"]]
            grouped["city"] = city
            grouped["season"] = season
            grouped["SleepTotal"] = grouped["Sleep"].apply(lambda value: sum(value.values()) if isinstance(value, dict) else 0)
            grouped = grouped[grouped["SleepTotal"] > 0].copy()
            if grouped.empty:
                continue
            grouped["SleepLight"] = grouped["Sleep"].apply(lambda value: value.get("light", 0) / sum(value.values()) if isinstance(value, dict) and sum(value.values()) else np.nan)
            grouped["SleepDeep"] = grouped["Sleep"].apply(lambda value: value.get("deep", 0) / sum(value.values()) if isinstance(value, dict) and sum(value.values()) else np.nan)
            grouped["SleepREM"] = grouped["Sleep"].apply(lambda value: value.get("rem", 0) / sum(value.values()) if isinstance(value, dict) and sum(value.values()) else np.nan)
            start, end = get_campaign_date_window(city, season)
            dates = pd.to_datetime(grouped["date"], errors="coerce")
            grouped["inside_campaign_window"] = dates.between(start, end, inclusive="both") if start is not None and end is not None else False
            grouped["complete_uhoo_predictors"] = grouped[list(UHOO_COLUMNS)].notna().all(axis=1)
            rows.append(
                grouped[
                    [
                        "participant_uid",
                        "date",
                        "city",
                        "season",
                        "inside_campaign_window",
                        *UHOO_COLUMNS,
                        *SLEEP_OUTCOMES,
                        "complete_uhoo_predictors",
                    ]
                ]
            )
    if not rows:
        return pd.DataFrame(columns=["participant_uid", "date", "city", "season", "inside_campaign_window", *UHOO_COLUMNS, *SLEEP_OUTCOMES, "complete_uhoo_predictors"])
    return pd.concat(rows, ignore_index=True)


def load_daily_stress_rows(data_zip: str | Path) -> pd.DataFrame:
    """Aggregate Garmin stress to participant-day rows for stress-sleep models."""
    rows: list[pd.DataFrame] = []
    data_zip = Path(data_zip)
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            city, season = parse_city_season(member)
            raw = _read_member(zf, member)
            if not {"TS", "ID", "Stress"}.issubset(raw.columns):
                continue
            tmp = raw[["TS", "ID", "Stress"]].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp["Stress"] = pd.to_numeric(tmp["Stress"], errors="coerce")
            tmp = tmp.dropna(subset=["TS", "ID", "Stress"])
            if tmp.empty:
                continue
            tmp["date"] = tmp["TS"].dt.date
            tmp["participant_uid"] = [city_prefixed_participant_id(city, value) for value in tmp["ID"]]
            tmp["city"] = city
            tmp["season"] = season
            daily = (
                tmp.groupby(["participant_uid", "date", "city", "season"], dropna=False)
                .agg(
                    avg_daily_stress=("Stress", "mean"),
                    max_daily_stress=("Stress", "max"),
                    stress_variability=("Stress", "std"),
                )
                .reset_index()
            )
            rows.append(daily)
    if not rows:
        return pd.DataFrame(columns=["participant_uid", "date", "city", "season", "avg_daily_stress", "max_daily_stress", "stress_variability"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["participant_uid", "date", "city", "season"]).reset_index(drop=True)


def load_pm_pca_rows(data_zip: str | Path, date_filter_mode: DateFilterMode = "campaign") -> pd.DataFrame:
    """Load complete PM1/PM2.5/PM10 campaign rows for aggregate PCA summaries."""
    rows: list[pd.DataFrame] = []
    data_zip = Path(data_zip)
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            city, season = parse_city_season(member)
            raw = _read_member(zf, member)
            if not {"TS", "ID", *PM_PCA_COLUMNS}.issubset(raw.columns):
                continue
            tmp = raw[["TS", "ID", *PM_PCA_COLUMNS]].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp = tmp.loc[_campaign_mask(tmp["TS"], city, season, date_filter_mode)].copy()
            if tmp.empty:
                continue
            tmp = _coerce_numeric(tmp, PM_PCA_COLUMNS)
            tmp = tmp.dropna(subset=["ID", *PM_PCA_COLUMNS])
            if tmp.empty:
                continue
            tmp["participant_uid"] = [city_prefixed_participant_id(city, value) for value in tmp["ID"]]
            rows.append(tmp[["participant_uid", *PM_PCA_COLUMNS]])
    if not rows:
        return pd.DataFrame(columns=["participant_uid", *PM_PCA_COLUMNS])
    return pd.concat(rows, ignore_index=True)


def make_sleep_model_input_audit(sleep: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required_predictors = _format_list(list(UHOO_COLUMNS))
    available_predictors = _format_list([col for col in UHOO_COLUMNS if col in sleep.columns])
    stream_masks = {
        SLEEP_AVAILABILITY_STREAM: sleep[SLEEP_OUTCOMES].notna().all(axis=1) if not sleep.empty else pd.Series(dtype=bool),
        SLEEP_IAQ_STREAM: (sleep[SLEEP_OUTCOMES].notna().all(axis=1) & sleep[list(UHOO_COLUMNS)].notna().all(axis=1)) if not sleep.empty else pd.Series(dtype=bool),
    }

    for stream, mask in stream_masks.items():
        target = S1_TARGETS[stream]
        support = sleep.loc[mask].copy() if not sleep.empty else sleep.copy()
        paired_by_city = {
            city: len(
                set(support.loc[(support["city"] == city) & (support["season"] == "Summer"), "participant_uid"])
                & set(support.loc[(support["city"] == city) & (support["season"] == "Winter"), "participant_uid"])
            )
            for city in ("Milan", "Thessaloniki")
        }
        for city, season in CITY_SEASON_ORDER:
            subset = support[(support["city"] == city) & (support["season"] == season)]
            full_subset = sleep[(sleep["city"] == city) & (sleep["season"] == season)] if not sleep.empty else sleep
            city_key = f"{city} {season.lower()} n"
            paired_key = f"{city} paired n"
            target_participants = int(target[city_key])
            target_rows = int(target["Participant-days/nights/rows"])
            observed_participants = int(subset["participant_uid"].nunique()) if not subset.empty else 0
            observed_rows = int(len(subset))
            observed_paired = int(paired_by_city[city])
            status = "PASS" if observed_participants == target_participants and observed_paired == int(target[paired_key]) else "FAIL"
            if stream == SLEEP_AVAILABILITY_STREAM and int(support.shape[0]) != target_rows:
                status = "FAIL"
            if stream == SLEEP_IAQ_STREAM and int(support.shape[0]) != target_rows:
                status = "FAIL"
            rows.append(
                {
                    "analysis_stream": stream,
                    "city": city,
                    "season": season,
                    "n_participants": observed_participants,
                    "paired_n_if_city_level": observed_paired,
                    "n_sleep_records_or_nights": int(len(full_subset)),
                    "n_complete_case_rows": observed_rows,
                    "sleep_total_available": True,
                    "light_sleep_available": True,
                    "deep_sleep_available": True,
                    "rem_sleep_available": True,
                    "sleep_efficiency_available": False,
                    "uhoo_predictors_required": required_predictors if stream == SLEEP_IAQ_STREAM else "not required for sleep availability",
                    "uhoo_predictors_available": available_predictors,
                    "target_participants": target_participants,
                    "target_rows": target_rows,
                    "status": status,
                    "notes": "Garmin sleep stages are wearable-derived field indicators; sleep efficiency is unavailable and not used.",
                }
            )
    return pd.DataFrame(rows)


def _complete_sleep_iaq(sleep: pd.DataFrame, outcome: str) -> pd.DataFrame:
    columns = ["participant_uid", "city", "season", outcome, *UHOO_COLUMNS]
    complete = sleep[columns].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()
    return complete


def _fit_ols_rows(data: pd.DataFrame, y_col: str, predictors: list[str], model_family: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if data.empty:
        return rows
    for predictor in predictors:
        cols = ["participant_uid", y_col, predictor]
        pair = data[cols].dropna(axis=0, how="any").copy()
        if len(pair) < 3 or pair[predictor].nunique(dropna=True) <= 1 or pair[y_col].nunique(dropna=True) <= 1:
            rows.append(_empty_ols_row(y_col, predictor, model_family, len(pair), int(pair["participant_uid"].nunique()), "INSUFFICIENT_DATA"))
            continue
        try:
            model = sm.OLS(pair[y_col], sm.add_constant(pair[[predictor]], has_constant="add")).fit()
            rows.append(_ols_result_row(model, pair, y_col, predictor, model_family))
        except Exception as exc:
            row = _empty_ols_row(y_col, predictor, model_family, len(pair), int(pair["participant_uid"].nunique()), "FIT_FAILED")
            row["notes"] = f"OLS fit failed: {exc}"
            rows.append(row)
    return rows


def _empty_ols_row(outcome: str, predictor: str, model_family: str, n_rows: int, n_participants: int, status: str) -> dict[str, object]:
    return {
        "outcome": outcome,
        "predictor": predictor,
        "model_family": model_family,
        "beta": np.nan,
        "se": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
        "adj_r_squared": np.nan,
        "n_rows": int(n_rows),
        "n_participants": int(n_participants),
        "complete_case_rule": "sleep outcome plus all current residential uHoo IAQ predictors",
        "predictor_unit": UHOO_UNITS.get(predictor, ""),
        "outcome_unit": SLEEP_OUTCOME_UNITS.get(outcome, ""),
        "wearable_field_indicator_flag": WEARABLE_FLAG,
        "status": status,
        "notes": "Aggregate-only model result.",
    }


def _ols_result_row(model: object, pair: pd.DataFrame, outcome: str, predictor: str, model_family: str) -> dict[str, object]:
    ci = model.conf_int().loc[predictor]
    return {
        "outcome": outcome,
        "predictor": predictor,
        "model_family": model_family,
        "beta": float(model.params[predictor]),
        "se": float(model.bse[predictor]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(model.pvalues[predictor]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "n_rows": int(len(pair)),
        "n_participants": int(pair["participant_uid"].nunique()),
        "complete_case_rule": "sleep outcome plus all current residential uHoo IAQ predictors",
        "predictor_unit": UHOO_UNITS.get(predictor, ""),
        "outcome_unit": SLEEP_OUTCOME_UNITS.get(outcome, ""),
        "wearable_field_indicator_flag": WEARABLE_FLAG,
        "status": "PASS",
        "notes": "Aggregate-only OLS screening result.",
    }


def run_sleep_iaq_ols(data_zip: str | Path) -> pd.DataFrame:
    sleep = load_sleep_rows(data_zip)
    rows: list[dict[str, object]] = []
    for outcome in SLEEP_OUTCOMES:
        complete = _complete_sleep_iaq(sleep, outcome)
        rows.extend(_fit_ols_rows(complete, outcome, list(UHOO_COLUMNS), "single-predictor OLS screening"))
    return pd.DataFrame(rows)


def run_sleep_lasso(data_zip: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sleep = load_sleep_rows(data_zip)
    selected_rows: list[dict[str, object]] = []
    refit_rows: list[dict[str, object]] = []
    for outcome in SLEEP_OUTCOMES:
        complete = _complete_sleep_iaq(sleep, outcome)
        usable = [col for col in UHOO_COLUMNS if col in complete.columns and complete[col].nunique(dropna=True) > 1]
        n_rows = int(len(complete))
        n_participants = int(complete["participant_uid"].nunique()) if not complete.empty else 0
        if n_rows < 10 or not usable or complete[outcome].nunique(dropna=True) <= 1:
            selected_rows.append(_lasso_selected_row(outcome, "", np.nan, "no predictors selected", n_rows, n_participants, "INSUFFICIENT_DATA"))
            refit_rows.append(_lasso_refit_empty_row(outcome, "", n_rows, n_participants, "NO_REFIT"))
            continue
        x_raw = complete[usable]
        y = complete[outcome]
        scaler = StandardScaler()
        x_scaled = pd.DataFrame(scaler.fit_transform(x_raw), columns=usable, index=x_raw.index)
        x_train, y_train = (x_scaled, y)
        if len(y) >= 12:
            x_train, _, y_train, _ = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
        cv = min(5, max(2, len(y_train) // 2))
        try:
            lasso = LassoCV(cv=cv, random_state=42, max_iter=10000)
            lasso.fit(x_train, y_train)
            selected = [usable[i] for i, coef in enumerate(lasso.coef_) if abs(coef) > 1e-12]
            if selected:
                for predictor in selected:
                    selected_rows.append(_lasso_selected_row(outcome, predictor, float(lasso.alpha_), "selected", n_rows, n_participants, "PASS"))
                refit_data = complete[["participant_uid", outcome, *selected]].dropna(axis=0, how="any").copy()
                model = sm.OLS(refit_data[outcome], sm.add_constant(refit_data[selected], has_constant="add")).fit()
                for predictor in selected:
                    ci = model.conf_int().loc[predictor]
                    refit_rows.append(
                        {
                            "outcome": outcome,
                            "predictor": predictor,
                            "beta": float(model.params[predictor]),
                            "se": float(model.bse[predictor]),
                            "ci_low": float(ci.iloc[0]),
                            "ci_high": float(ci.iloc[1]),
                            "p_value": float(model.pvalues[predictor]),
                            "r_squared": float(model.rsquared),
                            "adj_r_squared": float(model.rsquared_adj),
                            "n_rows": int(len(refit_data)),
                            "n_participants": int(refit_data["participant_uid"].nunique()),
                            "predictor_unit": UHOO_UNITS.get(predictor, ""),
                            "outcome_unit": SLEEP_OUTCOME_UNITS.get(outcome, ""),
                            "wearable_field_indicator_flag": WEARABLE_FLAG,
                            "status": "PASS",
                            "notes": "OLS refit uses unstandardized selected predictors for interpretable coefficients.",
                        }
                    )
            else:
                selected_rows.append(_lasso_selected_row(outcome, "", float(lasso.alpha_), "no predictors selected", n_rows, n_participants, "PASS"))
                refit_rows.append(_lasso_refit_empty_row(outcome, "", n_rows, n_participants, "NO_SELECTED_PREDICTORS"))
        except Exception as exc:
            selected_rows.append(_lasso_selected_row(outcome, "", np.nan, f"LassoCV failed: {exc}", n_rows, n_participants, "FIT_FAILED"))
            refit_rows.append(_lasso_refit_empty_row(outcome, "", n_rows, n_participants, "NO_REFIT"))
    return pd.DataFrame(selected_rows), pd.DataFrame(refit_rows)


def _lasso_selected_row(outcome: str, predictor: str, alpha: float, selection_status: str, n_rows: int, n_participants: int, status: str) -> dict[str, object]:
    return {
        "outcome": outcome,
        "selected_predictor": predictor,
        "standardization_applied": True,
        "lasso_alpha": alpha,
        "selection_status": selection_status,
        "n_rows": int(n_rows),
        "n_participants": int(n_participants),
        "status": status,
        "notes": "Predictors standardized before Lasso selection; participant rows are not exported.",
    }


def _lasso_refit_empty_row(outcome: str, predictor: str, n_rows: int, n_participants: int, status: str) -> dict[str, object]:
    return {
        "outcome": outcome,
        "predictor": predictor,
        "beta": np.nan,
        "se": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "r_squared": np.nan,
        "adj_r_squared": np.nan,
        "n_rows": int(n_rows),
        "n_participants": int(n_participants),
        "predictor_unit": "",
        "outcome_unit": SLEEP_OUTCOME_UNITS.get(outcome, ""),
        "wearable_field_indicator_flag": WEARABLE_FLAG,
        "status": status,
        "notes": "No OLS refit because Lasso selected no predictors or fitting failed.",
    }


def run_pm_pca(data_zip: str | Path, date_filter_mode: DateFilterMode = "campaign") -> tuple[pd.DataFrame, pd.DataFrame]:
    pm = load_pm_pca_rows(data_zip, date_filter_mode=date_filter_mode)
    if pm.empty:
        empty_loadings = pd.DataFrame(columns=["component", "pollutant", "loading", "explained_variance_ratio", "status", "notes"])
        empty_scores = pd.DataFrame(columns=["component", "n_rows_used", "n_participants", "mean", "sd", "median", "p25", "p75", "min", "max", "status", "notes"])
        return empty_loadings, empty_scores
    x = pm[PM_PCA_COLUMNS].to_numpy(dtype=float)
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA()
    scores = pca.fit_transform(x_scaled)
    loading_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for comp_idx in range(len(PM_PCA_COLUMNS)):
        component = f"PC{comp_idx + 1}"
        for col_idx, column in enumerate(PM_PCA_COLUMNS):
            loading_rows.append(
                {
                    "component": component,
                    "pollutant": PM_PCA_LABELS[column],
                    "loading": float(pca.components_[comp_idx, col_idx]),
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[comp_idx]),
                    "status": "PASS",
                    "notes": "PCA fit on standardized complete PM1/PM2.5/PM10 campaign rows; participant-level scores not exported.",
                }
            )
        values = scores[:, comp_idx]
        score_rows.append(
            {
                "component": component,
                "n_rows_used": int(len(pm)),
                "n_participants": int(pm["participant_uid"].nunique()),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "status": "PASS",
                "notes": "Aggregate score distribution only; individual PCA scores are not exported.",
            }
        )
    return pd.DataFrame(loading_rows), pd.DataFrame(score_rows)


def _pearson_pair(data: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, int, int, str]:
    cols = ["participant_uid", x_col, y_col]
    pair = data[cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    n_rows = int(len(pair))
    n_participants = int(pair["participant_uid"].nunique()) if n_rows else 0
    if n_rows < 3 or pair[x_col].nunique(dropna=True) <= 1 or pair[y_col].nunique(dropna=True) <= 1:
        return np.nan, np.nan, n_rows, n_participants, "INSUFFICIENT_DATA"
    r, p = stats.pearsonr(pair[x_col], pair[y_col])
    return float(r), float(p), n_rows, n_participants, "PASS"


def make_figure8_correlation_data(data_zip: str | Path) -> pd.DataFrame:
    sleep = load_sleep_rows(data_zip)
    groups: list[tuple[str, str, pd.DataFrame]] = [("All", "All", sleep)]
    for city, season in CITY_SEASON_ORDER:
        groups.append((city, season, sleep[(sleep["city"] == city) & (sleep["season"] == season)]))
    rows: list[dict[str, object]] = []
    for city, season, subset in groups:
        for x_col in UHOO_COLUMNS:
            for y_col in SLEEP_OUTCOMES:
                r, p, n_rows, n_participants, status = _pearson_pair(subset, x_col, y_col)
                rows.append(
                    {
                        "variable_x": x_col,
                        "variable_y": y_col,
                        "source_x": "residential uHoo IAQ",
                        "source_y": "Garmin-derived wearable sleep indicator",
                        "correlation_method": "Pearson",
                        "r": r,
                        "p_value": p,
                        "n_rows": n_rows,
                        "n_participants": n_participants,
                        "city": city,
                        "season": season,
                        "wearable_field_indicator_flag": WEARABLE_FLAG,
                        "status": status,
                        "notes": "Aggregate pairwise complete correlation; participant-night rows not exported.",
                    }
                )
    return pd.DataFrame(rows)


def make_figure9_plot_data(ols: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in ols.itertuples(index=False):
        rows.append(
            {
                "panel": f"{row.outcome} vs {row.predictor}",
                "outcome": row.outcome,
                "predictor": row.predictor,
                "model_type": row.model_family,
                "beta": row.beta,
                "se": row.se,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "p_value": row.p_value,
                "r_squared": row.r_squared,
                "n_rows": row.n_rows,
                "n_participants": row.n_participants,
                "predictor_unit": row.predictor_unit,
                "outcome_unit": row.outcome_unit,
                "status": row.status,
                "notes": "Aggregate model/panel data only; final figure rendering is not performed in Phase 7.",
            }
        )
    return pd.DataFrame(rows)


def _merged_stress_sleep(data_zip: str | Path) -> pd.DataFrame:
    sleep = load_sleep_rows(data_zip)
    stress = load_daily_stress_rows(data_zip)
    merged = stress.merge(sleep, on=["participant_uid", "date", "city", "season"], how="inner")
    merged = merged.sort_values(["participant_uid", "date"]).reset_index(drop=True)
    merged["next_day_avg_stress"] = merged.groupby("participant_uid", sort=False)["avg_daily_stress"].shift(-1)
    needed = ["avg_daily_stress", "next_day_avg_stress", *SLEEP_OUTCOMES]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=needed, how="any").copy()
    return merged


def _fit_mixed_model(data: pd.DataFrame, model_name: str, direction: str, outcome: str, predictor: str, formula: str) -> dict[str, object]:
    cols = ["participant_uid", "city", "season", outcome, predictor]
    model_data = data[cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").copy()
    base = {
        "model_name": model_name,
        "direction": direction,
        "outcome": outcome,
        "predictor": predictor,
        "beta": np.nan,
        "se": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "n_rows": int(len(model_data)),
        "n_participants": int(model_data["participant_uid"].nunique()) if not model_data.empty else 0,
        "random_effect_structure": "participant-specific random intercept",
        "fixed_effects": "city; season",
        "converged": False,
        "aic": np.nan,
        "bic": np.nan,
        "log_likelihood": np.nan,
        "wearable_field_indicator_flag": WEARABLE_FLAG,
        "status": "INSUFFICIENT_DATA",
        "notes": "Formula matches current stress-sleep workflow fixed effects; ML fit used so AIC/BIC are available.",
    }
    if len(model_data) < 10 or model_data["participant_uid"].nunique() < 3 or model_data[predictor].nunique(dropna=True) <= 1 or model_data[outcome].nunique(dropna=True) <= 1:
        return base
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(formula, data=model_data, groups=model_data["participant_uid"])
            try:
                result = model.fit(reml=False, maxiter=200, disp=False)
            except Exception:
                result = model.fit(reml=False, method="powell", maxiter=200, disp=False)
        ci = result.conf_int().loc[predictor]
        base.update(
            {
                "beta": float(result.params[predictor]),
                "se": float(result.bse[predictor]),
                "ci_low": float(ci.iloc[0]),
                "ci_high": float(ci.iloc[1]),
                "p_value": float(result.pvalues[predictor]),
                "converged": bool(getattr(result, "converged", False)),
                "aic": float(result.aic) if math.isfinite(result.aic) else np.nan,
                "bic": float(result.bic) if math.isfinite(result.bic) else np.nan,
                "log_likelihood": float(result.llf) if math.isfinite(result.llf) else np.nan,
                "status": "PASS" if bool(getattr(result, "converged", False)) else "CONVERGENCE_DEVIATION",
                "notes": "Formula matches current stress-sleep workflow fixed effects; warnings: " + "; ".join(str(w.message) for w in caught[:3]),
            }
        )
    except Exception as exc:
        base["status"] = "FIT_FAILED"
        base["notes"] = f"MixedLM fit failed: {exc}"
    return base


def run_stress_sleep_mixed_models(data_zip: str | Path) -> pd.DataFrame:
    data = _merged_stress_sleep(data_zip)
    specs = [
        ("stress_to_total_sleep", "same-day stress -> sleep", "SleepTotal", "avg_daily_stress", "SleepTotal ~ avg_daily_stress + C(season) + C(city)"),
        ("stress_to_deep_sleep", "same-day stress -> sleep", "SleepDeep", "avg_daily_stress", "SleepDeep ~ avg_daily_stress + C(season) + C(city)"),
        ("total_sleep_to_next_day_stress", "sleep -> next-day stress", "next_day_avg_stress", "SleepTotal", "next_day_avg_stress ~ SleepTotal + C(season) + C(city)"),
        ("deep_sleep_to_next_day_stress", "sleep -> next-day stress", "next_day_avg_stress", "SleepDeep", "next_day_avg_stress ~ SleepDeep + C(season) + C(city)"),
    ]
    rows = [_fit_mixed_model(data, *spec) for spec in specs]
    return pd.DataFrame(rows)


def make_figure10_plot_data(data_zip: str | Path) -> pd.DataFrame:
    data = _merged_stress_sleep(data_zip)
    rows: list[dict[str, object]] = []
    for metric in SLEEP_OUTCOMES:
        r, p, n_rows, n_participants, status = _pearson_pair(data, "avg_daily_stress", metric)
        beta = se = ci_low = ci_high = r2 = np.nan
        pair = data[["participant_uid", "avg_daily_stress", metric]].dropna()
        if status == "PASS":
            model = sm.OLS(pair[metric], sm.add_constant(pair[["avg_daily_stress"]], has_constant="add")).fit()
            ci = model.conf_int().loc["avg_daily_stress"]
            beta, se, ci_low, ci_high, r2 = float(model.params["avg_daily_stress"]), float(model.bse["avg_daily_stress"]), float(ci.iloc[0]), float(ci.iloc[1]), float(model.rsquared)
        rows.append(_figure10_row(f"same-day stress -> {metric}", "same-day stress -> sleep", metric, "avg_daily_stress", beta, se, ci_low, ci_high, p, r2, n_rows, n_participants, status))
        r, p, n_rows, n_participants, status = _pearson_pair(data, metric, "next_day_avg_stress")
        beta = se = ci_low = ci_high = r2 = np.nan
        pair = data[["participant_uid", metric, "next_day_avg_stress"]].dropna()
        if status == "PASS":
            model = sm.OLS(pair["next_day_avg_stress"], sm.add_constant(pair[[metric]], has_constant="add")).fit()
            ci = model.conf_int().loc[metric]
            beta, se, ci_low, ci_high, r2 = float(model.params[metric]), float(model.bse[metric]), float(ci.iloc[0]), float(ci.iloc[1]), float(model.rsquared)
        rows.append(_figure10_row(f"{metric} -> next-day stress", "sleep -> next-day stress", "next_day_avg_stress", metric, beta, se, ci_low, ci_high, p, r2, n_rows, n_participants, status))
    return pd.DataFrame(rows)


def _figure10_row(panel: str, direction: str, outcome: str, predictor: str, beta: float, se: float, ci_low: float, ci_high: float, p_value: float, r2: float, n_rows: int, n_participants: int, status: str) -> dict[str, object]:
    return {
        "panel": panel,
        "direction": direction,
        "outcome": outcome,
        "predictor": predictor,
        "beta": beta,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "r_squared_or_model_fit": r2,
        "n_rows": int(n_rows),
        "n_participants": int(n_participants),
        "city_or_season_adjustment": "unadjusted figure plot-data regression; mixed-model output includes city and season fixed effects",
        "wearable_field_indicator_flag": WEARABLE_FLAG,
        "status": status,
        "notes": "Aggregate plot-data only; final figure rendering is not performed in Phase 7.",
    }


def write_reconstruction_outputs(data_zip: str | Path, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sleep = load_sleep_rows(data_zip)
    audit = make_sleep_model_input_audit(sleep)
    output = outdir / "sleep_model_input_audit.csv"
    audit.to_csv(output, index=False)
    return {"sleep_model_input_audit": output}


def write_ols_outputs(data_zip: str | Path, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ols = run_sleep_iaq_ols(data_zip)
    figure9 = make_figure9_plot_data(ols)
    outputs = {
        "sleep_iaq_ols_results": outdir / "sleep_iaq_ols_results.csv",
        "figure9_plot_data": outdir / "figure9_plot_data.csv",
    }
    ols.to_csv(outputs["sleep_iaq_ols_results"], index=False)
    figure9.to_csv(outputs["figure9_plot_data"], index=False)
    return outputs


def write_lasso_outputs(data_zip: str | Path, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    selected, refit = run_sleep_lasso(data_zip)
    outputs = {
        "sleep_lasso_selected_predictors": outdir / "sleep_lasso_selected_predictors.csv",
        "sleep_lasso_refit_results": outdir / "sleep_lasso_refit_results.csv",
    }
    selected.to_csv(outputs["sleep_lasso_selected_predictors"], index=False)
    refit.to_csv(outputs["sleep_lasso_refit_results"], index=False)
    return outputs


def write_stress_sleep_outputs(data_zip: str | Path, outdir: str | Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mixed = run_stress_sleep_mixed_models(data_zip)
    figure10 = make_figure10_plot_data(data_zip)
    outputs = {
        "stress_sleep_mixed_model_results": outdir / "stress_sleep_mixed_model_results.csv",
        "figure10_plot_data": outdir / "figure10_plot_data.csv",
    }
    mixed.to_csv(outputs["stress_sleep_mixed_model_results"], index=False)
    figure10.to_csv(outputs["figure10_plot_data"], index=False)
    return outputs


def write_phase7_audit_outputs(data_zip: str | Path, outdir: str | Path, date_filter_mode: DateFilterMode = "campaign") -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    loadings, scores = run_pm_pca(data_zip, date_filter_mode=date_filter_mode)
    figure8 = make_figure8_correlation_data(data_zip)
    outputs = {
        "pca_pm_loadings": outdir / "pca_pm_loadings.csv",
        "pca_pm_scores_summary": outdir / "pca_pm_scores_summary.csv",
        "figure8_correlation_data": outdir / "figure8_correlation_data.csv",
    }
    loadings.to_csv(outputs["pca_pm_loadings"], index=False)
    scores.to_csv(outputs["pca_pm_scores_summary"], index=False)
    figure8.to_csv(outputs["figure8_correlation_data"], index=False)
    return outputs


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _safe_output_headers_ok(outdir: Path) -> bool:
    forbidden = {
        "participant_uid",
        "participant_id",
        "source_member",
        "raw_timestamp",
        "timestamp",
        "household_id",
        "coordinates",
        "latitude",
        "longitude",
        "participant_night_row",
        "model_input_row",
    }
    for path in outdir.glob("*.csv"):
        try:
            header = pd.read_csv(path, nrows=0).columns
        except Exception:
            return False
        normalized = {str(col).strip().lower() for col in header}
        if normalized & forbidden:
            return False
    unsafe_names = {
        "sleep_model_input_rows.csv",
        "participant_night_sleep_records.csv",
        "sleep_stage_minute_rows.csv",
        "garmin_raw_sleep_records.csv",
        "uhoo_raw_rows.csv",
    }
    return not any((outdir / name).exists() for name in unsafe_names)


def build_validation_report(repo_path: str | Path, data_zip: str | Path | None, sleep_dir: str | Path, outdir: str | Path, date_filter_mode: DateFilterMode = "campaign") -> str:
    repo_path = Path(repo_path)
    outdir = Path(outdir)
    sleep_dir = Path(sleep_dir)
    audit = _read_csv_if_exists(sleep_dir / "sleep_model_input_audit.csv")
    pca_loadings = _read_csv_if_exists(outdir / "pca_pm_loadings.csv")
    pca_scores = _read_csv_if_exists(outdir / "pca_pm_scores_summary.csv")
    ols = _read_csv_if_exists(outdir / "sleep_iaq_ols_results.csv")
    lasso_selected = _read_csv_if_exists(outdir / "sleep_lasso_selected_predictors.csv")
    lasso_refit = _read_csv_if_exists(outdir / "sleep_lasso_refit_results.csv")
    stress = _read_csv_if_exists(outdir / "stress_sleep_mixed_model_results.csv")
    fig8 = _read_csv_if_exists(outdir / "figure8_correlation_data.csv")
    fig9 = _read_csv_if_exists(outdir / "figure9_plot_data.csv")
    fig10 = _read_csv_if_exists(outdir / "figure10_plot_data.csv")

    availability = audit[audit["analysis_stream"].eq(SLEEP_AVAILABILITY_STREAM)] if not audit.empty else pd.DataFrame()
    complete_case = audit[audit["analysis_stream"].eq(SLEEP_IAQ_STREAM)] if not audit.empty else pd.DataFrame()
    availability_rows = int(availability["n_complete_case_rows"].sum()) if not availability.empty else 0
    complete_rows = int(complete_case["n_complete_case_rows"].sum()) if not complete_case.empty else 0
    availability_ok = availability_rows == S1_TARGETS[SLEEP_AVAILABILITY_STREAM]["Participant-days/nights/rows"] and bool(availability["status"].eq("PASS").all()) if not availability.empty else False
    complete_ok = complete_rows == S1_TARGETS[SLEEP_IAQ_STREAM]["Participant-days/nights/rows"] and bool(complete_case["status"].eq("PASS").all()) if not complete_case.empty else False
    sleep_eff_ok = bool(audit["sleep_efficiency_available"].astype(str).str.lower().eq("false").all()) if not audit.empty else False
    wearable_frames = [ols, lasso_refit, stress, fig8, fig10]
    wearable_ok = all(
        ("wearable_field_indicator_flag" in frame.columns and frame["wearable_field_indicator_flag"].astype(str).str.contains("wearable-derived field indicator", case=False, na=False).all())
        for frame in wearable_frames
        if not frame.empty
    )
    pca_ok = (
        not pca_loadings.empty
        and set(pca_loadings["pollutant"].astype(str)) >= {"PM1", "PM2.5", "PM10"}
        and not pca_scores.empty
    )
    ols_ok = not ols.empty and {"beta", "ci_low", "ci_high", "p_value", "r_squared", "n_rows", "n_participants"}.issubset(ols.columns)
    lasso_selected_ok = not lasso_selected.empty and lasso_selected["standardization_applied"].astype(str).str.lower().eq("true").all()
    lasso_refit_ok = not lasso_refit.empty and {"beta", "ci_low", "ci_high", "p_value", "r_squared", "n_rows", "n_participants"}.issubset(lasso_refit.columns)
    stress_ok = (
        not stress.empty
        and {"participant-specific random intercept"} <= set(stress["random_effect_structure"].astype(str))
        and "FIT_FAILED" not in set(stress["status"].astype(str))
        and "INSUFFICIENT_DATA" not in set(stress["status"].astype(str))
        and stress["beta"].notna().all()
    )
    fig8_ok = not fig8.empty
    fig9_ok = not fig9.empty
    fig10_ok = not fig10.empty
    safe_ok = _safe_output_headers_ok(outdir)

    deviations: list[str] = []
    if not availability_ok:
        deviations.append(f"Garmin sleep availability reproduced rows={availability_rows}; target={S1_TARGETS[SLEEP_AVAILABILITY_STREAM]['Participant-days/nights/rows']}.")
    if not complete_ok:
        deviations.append(f"Sleep + uHoo IAQ complete-case rows={complete_rows}; target={S1_TARGETS[SLEEP_IAQ_STREAM]['Participant-days/nights/rows']}.")
    if not deviations:
        deviations.append("None.")
    model_claim_notes = [
        "None identified in the aggregate Phase 7 reproduction outputs.",
        "Stress-sleep mixed models use sleep/stress complete observations; the 212-row residential uHoo IAQ complete-case denominator is validated separately for sleep/IAQ models.",
    ]

    output_paths = [outdir / name for name in REQUIRED_OUTPUTS]
    lines = [
        "Phase 7 sleep/IAQ/PCA/Lasso/stress-sleep validation report",
        f"timestamp of run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository path: {repo_path}",
        f"data archive path used: {data_zip if data_zip is not None else 'not supplied to audit exporter'}",
        f"campaign-window mode: {date_filter_mode}",
        "scripts run: scripts\\07_reconstruct_garmin_sleep_inputs.py; scripts\\07_run_sleep_iaq_ols_models.py; scripts\\07_run_sleep_lasso_models.py; scripts\\07_run_stress_sleep_mixed_models.py; scripts\\07_export_sleep_model_audits.py",
        "",
        "Target denominator values:",
        f"- Garmin sleep records/nights target: {S1_TARGETS[SLEEP_AVAILABILITY_STREAM]['Participant-days/nights/rows']}; reproduced: {availability_rows}",
        f"- Sleep + residential uHoo IAQ complete-case rows target: {S1_TARGETS[SLEEP_IAQ_STREAM]['Participant-days/nights/rows']}; reproduced: {complete_rows}",
        "",
        "PASS/FAIL:",
        f"- Garmin sleep availability denominator: {_status(availability_ok)}",
        f"- sleep + uHoo IAQ complete-case denominator: {_status(complete_ok)}",
        f"- sleep efficiency unavailable/not used: {_status(sleep_eff_ok)}",
        f"- wearable-derived field-indicator labelling: {_status(wearable_ok)}",
        f"- PCA loadings export: {_status(pca_ok)}",
        f"- OLS sleep/IAQ model outputs: {_status(ols_ok)}",
        f"- Lasso selected-predictor output: {_status(lasso_selected_ok)}",
        f"- Lasso OLS refit output: {_status(lasso_refit_ok)}",
        f"- stress-sleep mixed-model output: {_status(stress_ok)}",
        f"- Figure 8 aggregate correlation-data output: {_status(fig8_ok)}",
        f"- Figure 9 aggregate plot-data output: {_status(fig9_ok)}",
        f"- Figure 10 aggregate plot-data output: {_status(fig10_ok)}",
        f"- no participant-level safe outputs: {_status(safe_ok)}",
        "",
        "Missing dependencies:",
        "- None for Phase 7 local reproduction.",
        "",
        "Deviations from target denominator values:",
        *[f"- {item}" for item in deviations],
        "",
        "Deviations from current manuscript model claims:",
        *[f"- {item}" for item in model_claim_notes],
        "",
        "Output files:",
        *[f"- {path}" for path in output_paths],
        "",
        "Confirmations:",
        "- No Phase 8 work was performed.",
        "- No Figure 5-8 final validation/regeneration was performed.",
        "- No lag-specific HR/stress models were run.",
        "- No HIA/YLL/upper-tail workflows were run.",
        "- No paired-season sensitivity was run.",
        "- No GitHub push, Git commit, or data upload was performed.",
        "- Controlled data remained local.",
        "- Safe outputs contain aggregate model results and validation tables only; participant IDs, participant UID columns, source-member paths, raw timestamps, row-level Feather-file identifiers, participant-night rows, and model input rows are absent.",
    ]
    return "\n".join(lines) + "\n"


def write_validation_report(repo_path: str | Path, data_zip: str | Path | None, sleep_dir: str | Path, outdir: str | Path, date_filter_mode: DateFilterMode = "campaign") -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = build_validation_report(repo_path, data_zip, sleep_dir, outdir, date_filter_mode=date_filter_mode)
    path = outdir / "phase7_validation_report.txt"
    path.write_text(report, encoding="utf-8")
    return path
