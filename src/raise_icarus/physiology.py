"""Lag-specific physiology model utilities for the RAISE/ICARUS workflow."""

from __future__ import annotations

import io
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import statsmodels.formula.api as smf

from raise_icarus.data import (
    DateFilterMode,
    feather_members,
    get_campaign_date_window,
    parse_city_season,
)


LAG_MINUTES: tuple[int, ...] = (1, 5, 10, 15, 30, 45, 60, 120)
PM_COLUMNS: dict[str, str] = {
    "PM1": "pm1",
    "PM2.5": "pm25",
    "PM10": "pm10",
}
OUTCOME_COLUMNS: dict[str, str] = {
    "heart_rate": "heart_rate",
    "stress": "stress",
}
RAW_COLUMNS: tuple[str, ...] = (
    "TS",
    "ID",
    "AvgHeartRate",
    "Stress",
    "Activity",
    "PM1_PPM",
    "PM25_PPM",
    "PM10_PPM",
    "Temp_PPM",
    "Humi_PPM",
    "Temp_uHoo",
    "Humi_uHoo",
)

RESULT_COLUMNS: tuple[str, ...] = (
    "model_kind",
    "outcome",
    "pollutant",
    "lag_minutes",
    "exposure_column",
    "exposure_window",
    "n_rows_available",
    "n_rows_used",
    "n_participants_used",
    "activity_included",
    "max_model_rows",
    "formula",
    "coefficient",
    "estimate_per_10ug_m3",
    "std_error",
    "z_value",
    "p_value",
    "ci_low_wald",
    "ci_high_wald",
    "converged",
    "model_warning",
)

FAILURE_COLUMNS: tuple[str, ...] = (
    "model_kind",
    "outcome",
    "pollutant",
    "lag_minutes",
    "exposure_column",
    "exposure_window",
    "n_rows_available",
    "n_rows_used",
    "n_participants_used",
    "activity_included",
    "max_model_rows",
    "formula",
    "fit_status",
    "message",
)


@dataclass(frozen=True)
class PhysiologyLoadResult:
    """Loaded physiology frame and audit metadata."""

    data: pd.DataFrame
    date_filter_audit: pd.DataFrame
    missing_columns: pd.DataFrame
    heart_rate_qc: pd.DataFrame


@dataclass(frozen=True)
class ModelRunConfig:
    """Configuration for lag-specific physiology model runs."""

    max_model_rows: int = 10_000
    min_model_rows: int = 200
    min_groups: int = 5
    maxiter: int = 100
    fit_interaction_sensitivities: bool = False


def _date_or_nat(value: object) -> object:
    if pd.isna(value):
        return pd.NaT
    return pd.Timestamp(value).date()


def _mode_or_first(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return pd.NA
    mode = values.mode()
    if not mode.empty:
        return mode.iloc[0]
    return values.iloc[0]


def _apply_campaign_filter(
    df: pd.DataFrame,
    city: str,
    season: str,
    mode: DateFilterMode,
) -> tuple[pd.DataFrame, dict[str, object]]:
    before_rows = len(df)
    before_dates = df["timestamp"].dt.date.nunique() if before_rows else 0
    before_min = df["timestamp"].min() if before_rows else pd.NaT
    before_max = df["timestamp"].max() if before_rows else pd.NaT

    start, end = get_campaign_date_window(city, season)
    if mode == "campaign":
        if start is None or end is None:
            raise ValueError(f"No campaign date window configured for {city} / {season}")
        date_series = df["timestamp"].dt.date
        mask = (date_series >= start.date()) & (date_series <= end.date())
        out = df.loc[mask].copy()
        window_label = f"{start.date()} to {end.date()}"
    elif mode == "none":
        out = df.copy()
        window_label = "not applied"
    else:
        raise ValueError(f"Unsupported date filter mode: {mode}")

    after_rows = len(out)
    audit = {
        "city": city,
        "season": season,
        "date_filter_mode": mode,
        "campaign_window": window_label,
        "rows_before_filter": before_rows,
        "rows_after_filter": after_rows,
        "rows_removed_by_filter": before_rows - after_rows,
        "unique_dates_before_filter": before_dates,
        "unique_dates_after_filter": out["timestamp"].dt.date.nunique() if after_rows else 0,
        "date_min_before_filter": _date_or_nat(before_min),
        "date_max_before_filter": _date_or_nat(before_max),
        "date_min_after_filter": _date_or_nat(out["timestamp"].min()) if after_rows else pd.NaT,
        "date_max_after_filter": _date_or_nat(out["timestamp"].max()) if after_rows else pd.NaT,
    }
    return out, audit


def load_physiology_data(
    data_zip: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> PhysiologyLoadResult:
    """Load minute-level wearable and PPM fields for lag-specific models."""
    data_zip = Path(data_zip)
    records: list[pd.DataFrame] = []
    date_audit_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    with zipfile.ZipFile(data_zip) as zf:
        members = feather_members(data_zip)
        if not members:
            raise FileNotFoundError(f"No .feather files found in {data_zip}")

        for member in members:
            city, season = parse_city_season(member)
            with zf.open(member) as fh:
                table = feather.read_table(io.BytesIO(fh.read()))
            raw = table.to_pandas()

            missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
            if missing:
                missing_rows.append(
                    {
                        "archive_member": member,
                        "city": city,
                        "season": season,
                        "missing_columns": ";".join(missing),
                    }
                )
            for column in RAW_COLUMNS:
                if column not in raw.columns:
                    raw[column] = pd.NA

            tmp = raw.loc[:, RAW_COLUMNS].copy()
            tmp = tmp.rename(
                columns={
                    "TS": "timestamp",
                    "ID": "participant_id",
                    "AvgHeartRate": "heart_rate",
                    "Stress": "stress",
                    "Activity": "activity",
                    "PM1_PPM": "pm1",
                    "PM25_PPM": "pm25",
                    "PM10_PPM": "pm10",
                    "Temp_PPM": "temp_ppm",
                    "Humi_PPM": "humidity_ppm",
                    "Temp_uHoo": "temp_uhoo",
                    "Humi_uHoo": "humidity_uhoo",
                }
            )
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
            tmp = tmp.dropna(subset=["timestamp", "participant_id"])
            tmp, audit = _apply_campaign_filter(tmp, city, season, date_filter_mode)
            audit["archive_member"] = member
            date_audit_rows.append(audit)
            if tmp.empty:
                continue

            tmp["city"] = city
            tmp["season"] = season
            tmp["participant_id"] = tmp["participant_id"].astype(str).str.replace(r"\.0$", "", regex=True)
            tmp["participant_uid"] = tmp["city"] + "_" + tmp["participant_id"]
            for column in [
                "heart_rate",
                "stress",
                "pm1",
                "pm25",
                "pm10",
                "temp_ppm",
                "humidity_ppm",
                "temp_uhoo",
                "humidity_uhoo",
            ]:
                tmp[column] = pd.to_numeric(tmp[column], errors="coerce")
            tmp["activity"] = tmp["activity"].astype("string").str.strip()
            tmp.loc[tmp["activity"].isin(["", "nan", "None", "<NA>"]), "activity"] = pd.NA
            records.append(tmp)

    if records:
        data = pd.concat(records, ignore_index=True)
    else:
        data = pd.DataFrame(
            columns=[
                "timestamp",
                "participant_id",
                "heart_rate",
                "stress",
                "activity",
                "pm1",
                "pm25",
                "pm10",
                "temp_ppm",
                "humidity_ppm",
                "temp_uhoo",
                "humidity_uhoo",
                "city",
                "season",
                "participant_uid",
            ]
        )

    invalid_hr = data["heart_rate"].notna() & ((data["heart_rate"] < 40) | (data["heart_rate"] > 200))
    heart_rate_qc = (
        data.assign(invalid_heart_rate=invalid_hr)
        .groupby(["city", "season"], dropna=False)
        .agg(
            rows=("timestamp", "size"),
            heart_rate_nonmissing=("heart_rate", lambda s: int(s.notna().sum())),
            invalid_heart_rate_rows=("invalid_heart_rate", "sum"),
        )
        .reset_index()
    )
    data.loc[invalid_hr, "heart_rate"] = np.nan

    if not data.empty:
        agg_map = {
            "heart_rate": "mean",
            "stress": "mean",
            "pm1": "mean",
            "pm25": "mean",
            "pm10": "mean",
            "temp_ppm": "mean",
            "humidity_ppm": "mean",
            "temp_uhoo": "mean",
            "humidity_uhoo": "mean",
            "activity": _mode_or_first,
            "participant_id": "first",
        }
        data = (
            data.groupby(["city", "season", "participant_uid", "timestamp"], as_index=False)
            .agg(agg_map)
            .sort_values(["city", "season", "participant_uid", "timestamp"])
            .reset_index(drop=True)
        )
        data["hour"] = data["timestamp"].dt.hour.astype(int)
        data["weekend"] = data["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    date_filter_audit = pd.DataFrame(date_audit_rows)
    if not date_filter_audit.empty:
        date_filter_audit = date_filter_audit.sort_values(["city", "season", "archive_member"]).reset_index(drop=True)
    missing_columns = pd.DataFrame(missing_rows)
    return PhysiologyLoadResult(
        data=data,
        date_filter_audit=date_filter_audit,
        missing_columns=missing_columns,
        heart_rate_qc=heart_rate_qc,
    )


def add_lagged_pm_terms(
    data: pd.DataFrame,
    lags: tuple[int, ...] = LAG_MINUTES,
) -> pd.DataFrame:
    """Add prespecified exact-minute PM lag columns within participant/city/season."""
    out = data.sort_values(["city", "season", "participant_uid", "timestamp"]).copy()
    group_cols = ["city", "season", "participant_uid"]
    grouped = out.groupby(group_cols, sort=False)

    for lag in lags:
        shifted_time = grouped["timestamp"].shift(lag)
        exact_lag = (out["timestamp"] - shifted_time) == pd.Timedelta(minutes=lag)
        for pollutant, pm_col in PM_COLUMNS.items():
            lag_col = f"{pm_col}_lag_{lag}min"
            out[lag_col] = grouped[pm_col].shift(lag)
            out.loc[~exact_lag, lag_col] = np.nan
    return out


def add_moving_average_pm_terms(data: pd.DataFrame, window: str = "60min", min_periods: int = 10) -> pd.DataFrame:
    """Add trailing one-hour PM moving averages within participant/city/season."""
    out = data.sort_values(["city", "season", "participant_uid", "timestamp"]).copy()
    pieces: list[pd.DataFrame] = []
    for _, group in out.groupby(["city", "season", "participant_uid"], sort=False):
        rolling = (
            group.set_index("timestamp")[list(PM_COLUMNS.values())]
            .rolling(window, min_periods=min_periods)
            .mean()
        )
        rolling.index = group.index
        pieces.append(rolling)
    if pieces:
        rolled = pd.concat(pieces).sort_index()
        for pm_col in PM_COLUMNS.values():
            out[f"{pm_col}_ma60min"] = rolled[pm_col]
    else:
        for pm_col in PM_COLUMNS.values():
            out[f"{pm_col}_ma60min"] = np.nan
    return out


def activity_support_is_sufficient(data: pd.DataFrame, min_rows_per_level: int = 200) -> bool:
    """Return whether activity can be used as a categorical fixed effect."""
    counts = data["activity"].dropna().value_counts()
    return len(counts) >= 2 and int(counts.iloc[1]) >= min_rows_per_level


def make_specification_table(
    include_activity: bool,
    config: ModelRunConfig,
) -> pd.DataFrame:
    """Return an auditable model-specification table."""
    rows = [
        ("analysis_name", "prespecified lag-specific mixed-effects physiology models"),
        ("formal_dlnm", "No formal distributed lag non-linear model was implemented or claimed."),
        ("spline_exposure_response_basis", "No spline exposure-response basis was estimated."),
        ("spline_lag_basis", "No spline lag basis was estimated."),
        ("cross_basis", "No cross-basis was estimated."),
        ("knot_placement", "No knot placement was performed."),
        ("spline_degree_selection", "No spline-degree or degrees-of-freedom selection was performed."),
        ("smooth_exposure_lag_surface", "No smooth exposure-lag response surface was estimated."),
        ("prespecified_lags_minutes", ",".join(str(lag) for lag in LAG_MINUTES)),
        ("moving_average_sensitivity", "Trailing 60-minute PM moving average with at least 10 valid minutes."),
        ("outcomes", "Average heart rate and stress are Garmin-derived field indicators, not clinical endpoints."),
        ("pollutants", "PM1, PM2.5, and PM10 are modelled separately and are not summed."),
        ("primary_formula", "outcome ~ lagged_PM_per10 + city + season + hour + weekend + activity_if_available"),
        ("random_effects", "Participant-specific random intercepts."),
        ("fixed_effects", "City, season, hour, weekend, and activity where available."),
        ("activity_in_primary_model", str(include_activity)),
        ("sensitivity_lagged_pm_by_season", "Documented specification: outcome ~ lagged_PM_per10 * season + city + hour + weekend + activity_if_available."),
        ("sensitivity_lagged_pm_by_season_by_city", "Documented specification: outcome ~ lagged_PM_per10 * season * city + hour + weekend + activity_if_available."),
        ("sensitivity_lagged_pm_by_season_by_activity", "Documented if activity support is sufficient: outcome ~ lagged_PM_per10 * season * activity + city + hour + weekend."),
        ("interaction_sensitivity_fit_default", str(config.fit_interaction_sensitivities)),
        ("max_model_rows_per_fit", str(config.max_model_rows)),
        ("row_cap_note", "Rows are deterministically thinned per model when max_model_rows_per_fit is positive; set --max-model-rows 0 to fit all available rows."),
        ("p_value_caveat", "Dense repeated minute-level observations can yield small p values; estimates are interpreted cautiously as field-indicator associations."),
    ]
    return pd.DataFrame(rows, columns=["specification_item", "specification_value"])


def _base_formula(outcome: str, include_activity: bool) -> str:
    terms = ["pm_term", "C(city)", "C(season)", "C(hour)", "weekend"]
    if include_activity:
        terms.append("C(activity)")
    return f"{outcome} ~ " + " + ".join(terms)


def _interaction_formula(outcome: str, interaction: Literal["season", "season_city", "season_activity"], include_activity: bool) -> str:
    base_terms = ["C(hour)", "weekend"]
    if interaction == "season":
        exposure_term = "pm_term * C(season)"
        base_terms.insert(0, "C(city)")
        if include_activity:
            base_terms.append("C(activity)")
    elif interaction == "season_city":
        exposure_term = "pm_term * C(season) * C(city)"
        if include_activity:
            base_terms.append("C(activity)")
    elif interaction == "season_activity":
        exposure_term = "pm_term * C(season) * C(activity)"
        base_terms.insert(0, "C(city)")
    else:
        raise ValueError(f"Unsupported interaction: {interaction}")
    return f"{outcome} ~ {exposure_term} + " + " + ".join(base_terms)


def _deterministic_thin(data: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(data) <= max_rows:
        return data.copy()
    keys = pd.util.hash_pandas_object(data[["participant_uid", "timestamp"]], index=False)
    sampled = data.assign(_sample_key=keys).nsmallest(max_rows, "_sample_key")
    return sampled.drop(columns=["_sample_key"]).sort_values(["participant_uid", "timestamp"]).reset_index(drop=True)


def _prepare_model_frame(
    data: pd.DataFrame,
    outcome: str,
    exposure_col: str,
    include_activity: bool,
    config: ModelRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = ["city", "season", "participant_uid", "timestamp", "hour", "weekend", outcome, exposure_col]
    if include_activity:
        needed.append("activity")
    available = data.loc[:, needed].dropna(subset=[outcome, exposure_col, "city", "season", "participant_uid", "hour", "weekend"]).copy()
    if include_activity:
        available = available.dropna(subset=["activity"])
    for column in ["city", "season", "participant_uid"]:
        available[column] = available[column].astype(str)
    if include_activity:
        available["activity"] = available["activity"].astype(str)
    available["pm_term"] = available[exposure_col] / 10.0

    used = _deterministic_thin(available, config.max_model_rows)
    audit = _model_input_counts(available, used)
    return used, audit


def _model_input_counts(available: pd.DataFrame, used: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["city", "season"]
    available_counts = available.groupby(group_cols, dropna=False).size().rename("n_rows_available").reset_index()
    used_counts = used.groupby(group_cols, dropna=False).size().rename("n_rows_used").reset_index()
    out = available_counts.merge(used_counts, on=group_cols, how="outer").fillna(0)
    out["n_rows_available"] = out["n_rows_available"].astype(int)
    out["n_rows_used"] = out["n_rows_used"].astype(int)
    return out


def _fit_single_mixedlm(
    model_data: pd.DataFrame,
    formula: str,
    config: ModelRunConfig,
) -> tuple[object | None, str, str]:
    if len(model_data) < config.min_model_rows:
        return None, "not_fit", f"Insufficient rows: {len(model_data)} < {config.min_model_rows}"
    n_groups = model_data["participant_uid"].nunique()
    if n_groups < config.min_groups:
        return None, "not_fit", f"Insufficient participant groups: {n_groups} < {config.min_groups}"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = smf.mixedlm(formula, data=model_data, groups=model_data["participant_uid"], re_formula="1")
            result = model.fit(reml=False, method=["lbfgs", "cg"], maxiter=config.maxiter, disp=False)
        except Exception as exc:  # pragma: no cover - exercised by real data edge cases
            return None, "failed", f"{type(exc).__name__}: {exc}"
    warning_text = " | ".join(str(item.message) for item in caught)
    if not getattr(result, "converged", False):
        return result, "failed", f"Model did not converge. {warning_text}".strip()
    return result, "fitted", warning_text


def _result_rows(
    result: object,
    metadata: dict[str, object],
    formula: str,
    warning_text: str,
) -> list[dict[str, object]]:
    params = getattr(result, "params", pd.Series(dtype=float))
    bse = getattr(result, "bse", pd.Series(dtype=float))
    pvalues = getattr(result, "pvalues", pd.Series(dtype=float))
    rows = []
    for name, estimate in params.items():
        if name != "pm_term" and not name.startswith("pm_term:"):
            continue
        se = bse.get(name, np.nan)
        ci_low = estimate - 1.96 * se if pd.notna(se) else np.nan
        ci_high = estimate + 1.96 * se if pd.notna(se) else np.nan
        rows.append(
            {
                **metadata,
                "formula": formula,
                "coefficient": name,
                "estimate_per_10ug_m3": estimate,
                "std_error": se,
                "z_value": estimate / se if pd.notna(se) and se != 0 else np.nan,
                "p_value": pvalues.get(name, np.nan),
                "ci_low_wald": ci_low,
                "ci_high_wald": ci_high,
                "converged": getattr(result, "converged", pd.NA),
                "model_warning": warning_text,
            }
        )
    if not rows:
        rows.append(
            {
                **metadata,
                "formula": formula,
                "coefficient": "pm_term",
                "estimate_per_10ug_m3": np.nan,
                "std_error": np.nan,
                "z_value": np.nan,
                "p_value": np.nan,
                "ci_low_wald": np.nan,
                "ci_high_wald": np.nan,
                "converged": getattr(result, "converged", pd.NA),
                "model_warning": f"No PM coefficient found. {warning_text}".strip(),
            }
        )
    return rows


def fit_lagged_physiology_models(
    data: pd.DataFrame,
    config: ModelRunConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit primary lag-specific models and document sensitivity specifications."""
    config = config or ModelRunConfig()
    include_activity = activity_support_is_sufficient(data)
    specification = make_specification_table(include_activity, config)

    result_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    input_audit_rows: list[pd.DataFrame] = []

    model_jobs: list[dict[str, object]] = []
    for outcome_label, outcome_col in OUTCOME_COLUMNS.items():
        for pollutant_label, pm_col in PM_COLUMNS.items():
            for lag in LAG_MINUTES:
                model_jobs.append(
                    {
                        "model_kind": "primary_lag_specific",
                        "outcome": outcome_label,
                        "outcome_column": outcome_col,
                        "pollutant": pollutant_label,
                        "exposure_column": f"{pm_col}_lag_{lag}min",
                        "lag_minutes": lag,
                        "exposure_window": "prespecified_lag",
                        "formula": _base_formula(outcome_col, include_activity),
                        "include_activity": include_activity,
                    }
                )
            model_jobs.append(
                {
                    "model_kind": "moving_average_sensitivity",
                    "outcome": outcome_label,
                    "outcome_column": outcome_col,
                    "pollutant": pollutant_label,
                    "exposure_column": f"{pm_col}_ma60min",
                    "lag_minutes": pd.NA,
                    "exposure_window": "trailing_60min_min10",
                    "formula": _base_formula(outcome_col, include_activity),
                    "include_activity": include_activity,
                }
            )
            for interaction in ["season", "season_city"]:
                model_jobs.append(
                    {
                        "model_kind": f"documented_interaction_{interaction}",
                        "outcome": outcome_label,
                        "outcome_column": outcome_col,
                        "pollutant": pollutant_label,
                        "exposure_column": f"{pm_col}_lag_60min",
                        "lag_minutes": 60,
                        "exposure_window": "documented_sensitivity",
                        "formula": _interaction_formula(outcome_col, interaction, include_activity),
                        "include_activity": include_activity,
                        "document_only": not config.fit_interaction_sensitivities,
                    }
                )
            if include_activity:
                model_jobs.append(
                    {
                        "model_kind": "documented_interaction_season_activity",
                        "outcome": outcome_label,
                        "outcome_column": outcome_col,
                        "pollutant": pollutant_label,
                        "exposure_column": f"{pm_col}_lag_60min",
                        "lag_minutes": 60,
                        "exposure_window": "documented_sensitivity",
                        "formula": _interaction_formula(outcome_col, "season_activity", include_activity),
                        "include_activity": include_activity,
                        "document_only": not config.fit_interaction_sensitivities,
                    }
                )

    for job in model_jobs:
        model_data, audit = _prepare_model_frame(
            data,
            str(job["outcome_column"]),
            str(job["exposure_column"]),
            bool(job["include_activity"]),
            config,
        )
        audit["model_kind"] = job["model_kind"]
        audit["outcome"] = job["outcome"]
        audit["pollutant"] = job["pollutant"]
        audit["lag_minutes"] = job["lag_minutes"]
        audit["exposure_column"] = job["exposure_column"]
        audit["exposure_window"] = job["exposure_window"]
        input_audit_rows.append(audit)

        metadata = {
            "model_kind": job["model_kind"],
            "outcome": job["outcome"],
            "pollutant": job["pollutant"],
            "lag_minutes": job["lag_minutes"],
            "exposure_column": job["exposure_column"],
            "exposure_window": job["exposure_window"],
            "n_rows_available": int(audit["n_rows_available"].sum()),
            "n_rows_used": len(model_data),
            "n_participants_used": int(model_data["participant_uid"].nunique()) if not model_data.empty else 0,
            "activity_included": bool(job["include_activity"]),
            "max_model_rows": config.max_model_rows,
        }

        if job.get("document_only", False):
            failure_rows.append(
                {
                    **metadata,
                    "formula": job["formula"],
                    "fit_status": "documented_not_fit_by_default",
                    "message": "Interaction sensitivity specification documented; rerun with --fit-interaction-sensitivities to fit.",
                }
            )
            continue

        result, status, message = _fit_single_mixedlm(model_data, str(job["formula"]), config)
        if status == "fitted" and result is not None:
            result_rows.extend(_result_rows(result, metadata, str(job["formula"]), message))
        else:
            failure_rows.append(
                {
                    **metadata,
                    "formula": job["formula"],
                    "fit_status": status,
                    "message": message,
                }
            )
            if result is not None:
                result_rows.extend(_result_rows(result, metadata, str(job["formula"]), message))

    input_audit = pd.concat(input_audit_rows, ignore_index=True) if input_audit_rows else pd.DataFrame()
    results = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)
    failures = pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS)
    return input_audit, specification, results, failures


def write_qc_markdown(
    path: str | Path,
    load_result: PhysiologyLoadResult,
    input_audit: pd.DataFrame,
    results: pd.DataFrame,
    failures: pd.DataFrame,
    config: ModelRunConfig,
) -> None:
    """Write a compact QC note for lag-specific physiology models."""
    path = Path(path)
    fitted_models = (
        results[["model_kind", "outcome", "pollutant", "lag_minutes", "exposure_column"]].drop_duplicates()
        if not results.empty
        else pd.DataFrame()
    )
    failed = failures.loc[failures["fit_status"].isin(["failed", "not_fit"])] if not failures.empty else pd.DataFrame()
    documented = failures.loc[failures["fit_status"].eq("documented_not_fit_by_default")] if not failures.empty else pd.DataFrame()
    city_season = (
        load_result.data.groupby(["city", "season"], dropna=False)
        .agg(rows=("timestamp", "size"), participants=("participant_uid", "nunique"))
        .reset_index()
    )

    def markdown_table(frame: pd.DataFrame) -> str:
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

    lines = [
        "# Lag-specific physiology model QC",
        "",
        "This workflow implements prespecified lag-specific mixed-effects models.",
        "It is not a formal DLNM. No spline exposure-response basis, spline lag basis, cross-basis, knot placement, or spline-degree selection is estimated.",
        "",
        "Heart rate and stress are Garmin-derived field indicators, not clinical endpoints. Associations should be interpreted cautiously because the observational repeated minute-level data are dense and weak physiological signals can be sensitive to modelling choices.",
        "",
        f"Prespecified lags: {', '.join(str(lag) for lag in LAG_MINUTES)} minutes.",
        f"Maximum rows per model fit: {config.max_model_rows if config.max_model_rows > 0 else 'all available rows'}.",
        "",
        "## Loaded input",
        "",
        markdown_table(city_season),
        "",
        "## Model counts",
        "",
        f"Fitted model/coefficient rows: {len(results)}",
        f"Unique fitted model specifications: {len(fitted_models)}",
        f"Failed or not-fit model specifications: {len(failed)}",
        f"Documented interaction sensitivity specifications not fit by default: {len(documented)}",
        "",
        "## Output files",
        "",
        "- `physiology_lag_model_input_audit.csv`",
        "- `physiology_lag_model_specification.csv`",
        "- `physiology_lag_model_results.csv`",
        "- `physiology_lag_model_failures.csv`, when failures or documented-only specifications are present",
    ]
    if not input_audit.empty:
        total_used = int(input_audit.drop_duplicates(["model_kind", "outcome", "pollutant", "lag_minutes", "exposure_column"])["n_rows_used"].sum())
        lines.extend(["", f"Total model rows used across fitted/documented specifications: {total_used}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
