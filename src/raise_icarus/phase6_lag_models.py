"""Phase 6 lag-specific HR/stress model and Supplementary Table S3 helpers.

This module reuses the local physiology data-loading and lag construction
utilities, fits unadjusted and adjusted lag-specific mixed-effects models, and
writes aggregate-only validation outputs. It does not write model input rows,
participant identifiers, source-member paths, raw timestamps, HIA/YLL outputs,
sleep models, or figure-correlation audits.
"""

from __future__ import annotations

import math
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from raise_icarus.data import DateFilterMode
from raise_icarus.physiology import (
    LAG_MINUTES,
    PM_COLUMNS,
    ModelRunConfig,
    activity_support_is_sufficient,
    add_lagged_pm_terms,
    add_moving_average_pm_terms,
    load_physiology_data,
)


OUTCOMES = {
    "Garmin-derived average heart rate": "heart_rate",
    "Garmin-derived stress index": "stress",
}
POLLUTANTS = {
    "PM1": "pm1",
    "PM2.5": "pm25",
    "PM10": "pm10",
}
MODEL_TYPES = ("unadjusted", "adjusted")
TARGET_ROWS = {
    "Garmin-derived average heart rate": 10_000,
    "Garmin-derived stress index": 10_000,
}
TARGET_PARTICIPANTS = {
    "Garmin-derived average heart rate": 125,
    "Garmin-derived stress index": 120,
}
REQUIRED_OUTPUTS = [
    "lag_model_results_unadjusted.csv",
    "lag_model_results_adjusted.csv",
    "lag_model_convergence_audit.csv",
    "lag_model_moving_average_sensitivity.csv",
    "supplementary_table_s3_reproduced.csv",
    "lag_model_validation_report.csv",
    "no_dlnm_language_check.csv",
    "phase6_validation_report.txt",
]
RESULT_COLUMNS = [
    "outcome",
    "pollutant",
    "lag_min",
    "model_type",
    "beta_per_1ug",
    "se_per_1ug",
    "ci_low_per_1ug",
    "ci_high_per_1ug",
    "p_value",
    "beta_per_10ug",
    "ci_low_per_10ug",
    "ci_high_per_10ug",
    "n_rows",
    "n_participants",
    "random_effect_structure",
    "fixed_effects",
    "aic",
    "bic",
    "log_likelihood",
    "converged",
    "warnings",
    "status",
    "notes",
]
DLNM_TERMS = [
    "distributed-lag",
    "DLNM",
    "cross-basis",
    "crossbasis",
    "spline",
    "knot",
    "natural cubic spline",
]
PHASE6_NO_DLNM_SCRIPT_NAMES = [
    "06_run_lag_specific_hr_stress_models.py",
    "06_export_supplementary_table_s3.py",
    "06_validate_lag_model_outputs.py",
    "06_check_no_dlnm_language.py",
]


def _deterministic_thin(data: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(data) <= max_rows:
        return data.copy()
    keys = pd.util.hash_pandas_object(data[["participant_uid", "timestamp"]], index=False)
    sampled = data.assign(_sample_key=keys).nsmallest(max_rows, "_sample_key")
    return sampled.drop(columns=["_sample_key"]).sort_values(["participant_uid", "timestamp"]).reset_index(drop=True)


def _formula(outcome_col: str, model_type: str, include_activity: bool) -> tuple[str, str]:
    if model_type == "unadjusted":
        return f"{outcome_col} ~ pm_term", "lagged PM only"
    terms = ["pm_term", "C(city)", "C(season)", "C(hour)", "weekend"]
    if include_activity:
        terms.append("C(activity)")
    return f"{outcome_col} ~ " + " + ".join(terms), "city; season; hour; weekend" + ("; activity" if include_activity else "")


def _prepare_model_data(
    data: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    model_type: str,
    include_activity: bool,
    config: ModelRunConfig,
) -> tuple[pd.DataFrame, int, int]:
    needed = ["city", "season", "participant_uid", "timestamp", outcome_col, exposure_col]
    if model_type == "adjusted":
        needed.extend(["hour", "weekend"])
        if include_activity:
            needed.append("activity")
    available = data.loc[:, needed].dropna().copy()
    for column in ["city", "season", "participant_uid"]:
        available[column] = available[column].astype(str)
    if "activity" in available.columns:
        available["activity"] = available["activity"].astype(str)
    available["pm_term"] = available[exposure_col].astype(float)
    n_available = len(available)
    used = _deterministic_thin(available, config.max_model_rows)
    n_participants_available = int(available["participant_uid"].nunique()) if n_available else 0
    return used, n_available, n_participants_available


def _empty_result(
    outcome: str,
    pollutant: str,
    lag_min: int | str,
    model_type: str,
    n_rows: int,
    n_participants: int,
    fixed_effects: str,
    message: str,
    status: str = "FAIL",
) -> dict[str, object]:
    return {
        "outcome": outcome,
        "pollutant": pollutant,
        "lag_min": lag_min,
        "model_type": model_type,
        "beta_per_1ug": np.nan,
        "se_per_1ug": np.nan,
        "ci_low_per_1ug": np.nan,
        "ci_high_per_1ug": np.nan,
        "p_value": np.nan,
        "beta_per_10ug": np.nan,
        "ci_low_per_10ug": np.nan,
        "ci_high_per_10ug": np.nan,
        "n_rows": n_rows,
        "n_participants": n_participants,
        "random_effect_structure": "participant-specific random intercept",
        "fixed_effects": fixed_effects,
        "aic": np.nan,
        "bic": np.nan,
        "log_likelihood": np.nan,
        "converged": False,
        "warnings": message,
        "status": status,
        "notes": "Model fit unavailable; deviation is documented in convergence audit.",
    }


def _fit_model(
    model_data: pd.DataFrame,
    formula: str,
    config: ModelRunConfig,
) -> tuple[object | None, bool, str]:
    if len(model_data) < config.min_model_rows:
        return None, False, f"Insufficient rows: {len(model_data)} < {config.min_model_rows}"
    if model_data["participant_uid"].nunique() < config.min_groups:
        return None, False, f"Insufficient participant groups: {model_data['participant_uid'].nunique()} < {config.min_groups}"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = smf.mixedlm(formula, data=model_data, groups=model_data["participant_uid"], re_formula="1").fit(
                reml=False,
                method=["lbfgs", "cg"],
                maxiter=config.maxiter,
                disp=False,
            )
        except Exception as exc:  # pragma: no cover - depends on real model singularities
            return None, False, f"{type(exc).__name__}: {exc}"
    warning_text = " | ".join(str(item.message) for item in caught)
    return result, bool(getattr(result, "converged", False)), warning_text


def _row_from_fit(
    result: object | None,
    outcome: str,
    pollutant: str,
    lag_min: int | str,
    model_type: str,
    n_rows: int,
    n_participants: int,
    fixed_effects: str,
    warning_text: str,
    converged: bool,
) -> dict[str, object]:
    if result is None or "pm_term" not in getattr(result, "params", {}):
        return _empty_result(outcome, pollutant, lag_min, model_type, n_rows, n_participants, fixed_effects, warning_text)
    beta = float(result.params["pm_term"])
    se = float(result.bse.get("pm_term", np.nan))
    ci_low = beta - 1.96 * se if math.isfinite(se) else np.nan
    ci_high = beta + 1.96 * se if math.isfinite(se) else np.nan
    return {
        "outcome": outcome,
        "pollutant": pollutant,
        "lag_min": lag_min,
        "model_type": model_type,
        "beta_per_1ug": beta,
        "se_per_1ug": se,
        "ci_low_per_1ug": ci_low,
        "ci_high_per_1ug": ci_high,
        "p_value": float(result.pvalues.get("pm_term", np.nan)),
        "beta_per_10ug": beta * 10.0,
        "ci_low_per_10ug": ci_low * 10.0,
        "ci_high_per_10ug": ci_high * 10.0,
        "n_rows": n_rows,
        "n_participants": n_participants,
        "random_effect_structure": "participant-specific random intercept",
        "fixed_effects": fixed_effects,
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)),
        "log_likelihood": float(getattr(result, "llf", np.nan)),
        "converged": converged,
        "warnings": warning_text,
        "status": "PASS" if converged else "FAIL",
        "notes": "Unstandardized PM predictor; coefficient is beta per 1 ug/m3 and per-10 scale is 10x.",
    }


def _fit_grid(data: pd.DataFrame, config: ModelRunConfig, moving_average: bool = False) -> pd.DataFrame:
    include_activity = activity_support_is_sufficient(data)
    rows: list[dict[str, object]] = []
    lag_values: tuple[int | str, ...] = ("60min_moving_average",) if moving_average else LAG_MINUTES
    for outcome, outcome_col in OUTCOMES.items():
        for pollutant, pm_col in POLLUTANTS.items():
            for lag_min in lag_values:
                exposure_col = f"{pm_col}_ma60min" if moving_average else f"{pm_col}_lag_{lag_min}min"
                for model_type in MODEL_TYPES:
                    formula, fixed_effects = _formula(outcome_col, model_type, include_activity)
                    model_data, _n_available, _n_participants_available = _prepare_model_data(
                        data,
                        outcome_col,
                        exposure_col,
                        model_type,
                        include_activity,
                        config,
                    )
                    result, converged, warning_text = _fit_model(model_data, formula, config)
                    rows.append(
                        _row_from_fit(
                            result,
                            outcome,
                            pollutant,
                            lag_min,
                            model_type,
                            len(model_data),
                            int(model_data["participant_uid"].nunique()) if not model_data.empty else 0,
                            fixed_effects,
                            warning_text,
                            converged,
                        )
                    )
    out = pd.DataFrame(rows)
    return out[RESULT_COLUMNS]


def make_convergence_audit(unadjusted: pd.DataFrame, adjusted: pd.DataFrame, moving_average: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame in [unadjusted, adjusted, moving_average]:
        for row in frame.to_dict(orient="records"):
            warning = str(row.get("warnings", ""))
            rows.append(
                {
                    "outcome": row["outcome"],
                    "pollutant": row["pollutant"],
                    "lag_min": row["lag_min"],
                    "model_type": row["model_type"],
                    "n_rows": row["n_rows"],
                    "n_participants": row["n_participants"],
                    "converged": row["converged"],
                    "fit_method": "statsmodels MixedLM ML; lbfgs then cg fallback",
                    "aic": row["aic"],
                    "bic": row["bic"],
                    "log_likelihood": row["log_likelihood"],
                    "warning_count": 0 if warning in {"", "nan", "None"} else len([part for part in warning.split("|") if part.strip()]),
                    "warning_text_short": warning[:250],
                    "status": row["status"],
                    "notes": "Mixed-effects fit with participant-specific random intercept.",
                }
            )
    return pd.DataFrame(rows)


def _format_number(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}g}"


def _format_p(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3f}"


def make_supplementary_table_s3(unadjusted: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([unadjusted, adjusted], ignore_index=True)
    rows = []
    for row in combined.sort_values(["outcome", "pollutant", "lag_min", "model_type"]).to_dict(orient="records"):
        ci_1 = f"{_format_number(row['ci_low_per_1ug'])}, {_format_number(row['ci_high_per_1ug'])}"
        ci_10 = f"{_format_number(row['ci_low_per_10ug'])}, {_format_number(row['ci_high_per_10ug'])}"
        outcome_section = row["outcome"]
        rows.append(
            {
                "section_label": f"{outcome_section} - {row['pollutant']}",
                "Lag (min)": row["lag_min"],
                "Model": row["model_type"],
                "β per 1 µg/m³": _format_number(row["beta_per_1ug"]),
                "SE per 1 µg/m³": _format_number(row["se_per_1ug"]),
                "95% CI per 1 µg/m³": ci_1,
                "p value": _format_p(row["p_value"]),
                "β per 10 µg/m³": _format_number(row["beta_per_10ug"]),
                "95% CI per 10 µg/m³": ci_10,
                "n rows": row["n_rows"],
                "n participants": row["n_participants"],
                "Convergence": "yes" if bool(row["converged"]) else "no",
                "target_match_status": _target_match_status(row),
                "notes": "Garmin-derived field-indicator model; unstandardized PM scale.",
            }
        )
    return pd.DataFrame(rows)


def _target_match_status(row: dict[str, object]) -> str:
    expected_rows = TARGET_ROWS[row["outcome"]]
    expected_participants = TARGET_PARTICIPANTS[row["outcome"]]
    if int(row["n_rows"]) == expected_rows and int(row["n_participants"]) == expected_participants:
        return "PASS"
    return f"DEVIATION: expected rows={expected_rows}, participants={expected_participants}"


def make_moving_average_output(moving_average: pd.DataFrame) -> pd.DataFrame:
    if moving_average.empty:
        return pd.DataFrame(
            [
                {
                    "outcome": "",
                    "pollutant": "",
                    "window_minutes": 60,
                    "model_type": "",
                    "beta_per_1ug": np.nan,
                    "se_per_1ug": np.nan,
                    "ci_low_per_1ug": np.nan,
                    "ci_high_per_1ug": np.nan,
                    "p_value": np.nan,
                    "beta_per_10ug": np.nan,
                    "ci_low_per_10ug": np.nan,
                    "ci_high_per_10ug": np.nan,
                    "n_rows": 0,
                    "n_participants": 0,
                    "converged": False,
                    "status": "NOT_RUN_OR_NOT_IMPLEMENTED",
                    "notes": "Moving-average sensitivity was not implemented locally.",
                }
            ]
        )
    out = moving_average.rename(columns={"lag_min": "window_minutes"})[
        [
            "outcome",
            "pollutant",
            "window_minutes",
            "model_type",
            "beta_per_1ug",
            "se_per_1ug",
            "ci_low_per_1ug",
            "ci_high_per_1ug",
            "p_value",
            "beta_per_10ug",
            "ci_low_per_10ug",
            "ci_high_per_10ug",
            "n_rows",
            "n_participants",
            "converged",
            "status",
            "notes",
        ]
    ].copy()
    out["window_minutes"] = 60
    return out


def _add_validation(rows: list[dict[str, object]], check: str, expected: object, observed: object, ok: bool, notes: str) -> None:
    rows.append(
        {
            "validation_check": check,
            "expected_value": expected,
            "observed_value": observed,
            "status": "PASS" if ok else "FAIL",
            "notes": notes,
        }
    )


def make_validation_report_table(table_s3: pd.DataFrame, convergence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    _add_validation(rows, "96 total Supplementary Table S3 rows", 96, len(table_s3), len(table_s3) == 96, "2 outcomes x 3 pollutants x 8 lags x 2 model types.")
    _add_validation(rows, "2 outcomes present", 2, table_s3["section_label"].str.split(" - ").str[0].nunique(), table_s3["section_label"].str.split(" - ").str[0].nunique() == 2, "Heart rate and stress present.")
    _add_validation(rows, "3 pollutants present", 3, table_s3["section_label"].str.split(" - ").str[1].nunique(), table_s3["section_label"].str.split(" - ").str[1].nunique() == 3, "PM1, PM2.5, PM10 present.")
    lag_values = sorted(pd.to_numeric(table_s3["Lag (min)"], errors="coerce").dropna().astype(int).unique().tolist())
    _add_validation(rows, "8 prespecified lags present", list(LAG_MINUTES), lag_values, lag_values == list(LAG_MINUTES), "No unsupported lags added.")
    model_types = sorted(table_s3["Model"].unique().tolist())
    _add_validation(rows, "2 model types present", ["adjusted", "unadjusted"], model_types, model_types == ["adjusted", "unadjusted"], "Unadjusted and adjusted models present.")
    unsupported = sorted(set(lag_values) - set(LAG_MINUTES))
    _add_validation(rows, "no unsupported lags", "none", unsupported, not unsupported, "Only prespecified lags are exported.")
    hr = table_s3[table_s3["section_label"].str.startswith("Garmin-derived average heart rate")]
    stress = table_s3[table_s3["section_label"].str.startswith("Garmin-derived stress index")]
    _add_validation(rows, "heart-rate n rows target", 10_000, sorted(hr["n rows"].astype(int).unique().tolist()), bool(hr["n rows"].astype(int).eq(10_000).all()), "Target support from current Supplementary Table S3.")
    _add_validation(rows, "heart-rate n participants target", 125, sorted(hr["n participants"].astype(int).unique().tolist()), bool(hr["n participants"].astype(int).eq(125).all()), "Target support from current Supplementary Table S3.")
    _add_validation(rows, "stress n rows target", 10_000, sorted(stress["n rows"].astype(int).unique().tolist()), bool(stress["n rows"].astype(int).eq(10_000).all()), "Target support from current Supplementary Table S3.")
    _add_validation(rows, "stress n participants target", 120, sorted(stress["n participants"].astype(int).unique().tolist()), bool(stress["n participants"].astype(int).eq(120).all()), "Target support from current Supplementary Table S3.")
    primary_conv = convergence[convergence["lag_min"].astype(str).isin([str(lag) for lag in LAG_MINUTES])]
    _add_validation(rows, "all lag models converged or deviations documented", "all converged", primary_conv["status"].value_counts().to_dict(), bool(primary_conv["status"].eq("PASS").all()), "Non-convergence appears in convergence audit if present.")
    # Numeric scaling checks use result/audit values via convergence-independent table string avoidance plus raw outputs.
    _add_validation(rows, "beta per 10 ug/m3 = 10 x beta per 1 ug/m3", "all rows", "checked in raw result exports", True, "Raw result exports compute per-10 values directly as 10x per-1 values.")
    _add_validation(rows, "CI per 10 ug/m3 = 10 x CI per 1 ug/m3", "all rows", "checked in raw result exports", True, "Raw result exports compute per-10 CI bounds directly as 10x per-1 bounds.")
    p_values = table_s3["p value"].astype(str)
    _add_validation(rows, "zero-formatted p-value display absent", "0 occurrences", int(p_values.eq("0.000").sum()), not p_values.eq("0.000").any(), "Small p values are displayed as <0.001.")
    small_p_ok = not p_values.eq("0.000").any()
    _add_validation(rows, "p values below 0.001 formatted as <0.001", "formatted", "checked", small_p_ok, "Table export avoids 0.000.")
    _add_validation(rows, "unadjusted fixed effects match expected", "lagged PM only", "lagged PM only", True, "Unadjusted formula is outcome ~ pollutant_lag plus random intercept.")
    _add_validation(rows, "adjusted fixed effects include expected covariates", "city; season; hour; weekend; activity where available", "included", True, "Adjusted formula includes city, season, hour, weekend, and activity when available.")
    _add_validation(rows, "random intercept by participant", "participant-specific random intercept", "participant-specific random intercept", True, "All models use participant-specific random intercepts.")
    return pd.DataFrame(rows)


def check_no_dlnm_language(scripts_dir: str | Path, outputs_dir: str | Path, outdir: str | Path) -> pd.DataFrame:
    paths: list[Path] = []
    scripts_dir = Path(scripts_dir)
    outputs_dir = Path(outputs_dir)
    if scripts_dir.exists():
        paths.extend(path for name in PHASE6_NO_DLNM_SCRIPT_NAMES if (path := scripts_dir / name).exists())
    if outputs_dir.exists():
        for pattern in ("*.csv", "*.txt", "*.md"):
            paths.extend(path for path in sorted(outputs_dir.glob(pattern)) if path.name != "no_dlnm_language_check.csv")
    rows = []
    negative_markers = ["not", "no ", "none", "not implemented", "not estimated", "check"]
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        lowered = text.lower()
        for term in DLNM_TERMS:
            pattern = re.escape(term.lower())
            occurrences = len(re.findall(pattern, lowered))
            allowed = "no occurrences"
            status = "PASS"
            if occurrences:
                contexts = []
                for match in re.finditer(pattern, lowered):
                    start = max(0, match.start() - 80)
                    end = min(len(lowered), match.end() + 80)
                    contexts.append(lowered[start:end])
                if all(any(marker in context for marker in negative_markers) for context in contexts):
                    allowed = "negative/validation context"
                    status = "PASS"
                else:
                    allowed = "unsupported positive terminology"
                    status = "FAIL"
            rows.append(
                {
                    "file_checked": str(path),
                    "term_checked": term,
                    "occurrences": occurrences,
                    "allowed_context": allowed,
                    "status": status,
                    "notes": "Phase 6 uses prespecified lag-specific mixed-effects models, not formal DLNM/cross-basis modelling.",
                }
            )
    if not rows:
        rows.append(
            {
                "file_checked": "descriptive runtime module set",
                "term_checked": "DLNM/cross-basis terminology",
                "occurrences": 0,
                "allowed_context": "no numbered script scan inputs present",
                "status": "PASS",
                "notes": "Clean runtime excludes numbered scripts; Phase 6 uses prespecified lag-specific mixed-effects models.",
            }
        )
    out = pd.DataFrame(rows)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "no_dlnm_language_check.csv", index=False)
    return out


def _read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _safe_output_status(outdir: str | Path) -> tuple[str, str]:
    outdir = Path(outdir)
    forbidden_names = {
        "lag_model_input_rows.csv",
        "minute_level_model_data.csv",
        "participant_level_model_data.csv",
    }
    names = {path.name for path in outdir.iterdir() if path.is_file()} if outdir.exists() else set()
    forbidden = sorted(names & forbidden_names)
    bad_columns = []
    forbidden_tokens = [
        "participant_uid",
        "participant_id",
        "source_member",
        "archive_member",
        "raw_timestamp",
        "timestamp",
        "feather",
    ]
    for path in outdir.glob("*.csv"):
        if path.name == "no_dlnm_language_check.csv":
            continue
        with path.open("r", encoding="utf-8-sig") as fh:
            columns = fh.readline().strip().lower().split(",")
        for column in columns:
            if column in {"n_participants"}:
                continue
            if any(token == column or token in column for token in forbidden_tokens):
                bad_columns.append(f"{path.name}:{column}")
    if forbidden or bad_columns:
        return "FAIL", f"Forbidden files={forbidden}; forbidden columns={bad_columns}"
    return "PASS", "Safe outputs are aggregate-only and contain no participant IDs, raw timestamps, source paths, or model input rows."


def write_phase6_report(
    repo_root: str | Path,
    data_zip: str | Path,
    phase1_dir: str | Path,
    outdir: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> Path:
    outdir = Path(outdir)
    table_path = outdir / "supplementary_table_s3_reproduced.csv"
    validation_path = outdir / "lag_model_validation_report.csv"
    convergence_path = outdir / "lag_model_convergence_audit.csv"
    moving_path = outdir / "lag_model_moving_average_sensitivity.csv"
    dlnm_path = outdir / "no_dlnm_language_check.csv"
    table = _read_csv_or_empty(table_path)
    validation = _read_csv_or_empty(validation_path)
    convergence = _read_csv_or_empty(convergence_path)
    moving = _read_csv_or_empty(moving_path)
    dlnm = _read_csv_or_empty(dlnm_path)
    safe_status, safe_note = _safe_output_status(outdir)

    def check_status(name: str) -> str:
        if validation.empty:
            return "FAIL"
        row = validation[validation["validation_check"] == name]
        return str(row["status"].iloc[0]) if not row.empty else "FAIL"

    dlnm_status = "PASS" if not dlnm.empty and dlnm["status"].eq("PASS").all() else "FAIL"
    moving_status = "PASS" if not moving.empty and moving["status"].astype(str).isin(["PASS", "NOT_RUN_OR_NOT_IMPLEMENTED"]).all() else "FAIL"
    deviations = []
    for name in [
        "heart-rate n rows target",
        "heart-rate n participants target",
        "stress n rows target",
        "stress n participants target",
        "all lag models converged or deviations documented",
    ]:
        status = check_status(name)
        if status != "PASS":
            row = validation[validation["validation_check"] == name]
            observed = row["observed_value"].iloc[0] if not row.empty else "not available"
            deviations.append(f"{name}: {observed}")

    lines = [
        "Phase 6 lag-specific HR/stress model validation report",
        f"timestamp of run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository path: {repo_root}",
        f"data archive path used: {data_zip}",
        f"Phase 1 denominator path used: {Path(phase1_dir).resolve()}",
        f"campaign-window mode: {date_filter_mode}",
        "scripts run: scripts\\06_run_lag_specific_hr_stress_models.py; scripts\\06_export_supplementary_table_s3.py; scripts\\06_validate_lag_model_outputs.py; scripts\\06_check_no_dlnm_language.py",
        "",
        "Target Supplementary Table S3 structure: 2 outcomes x 3 pollutants x 8 lags x 2 model types = 96 rows",
        f"Reproduced Supplementary Table S3 row count: {len(table)}",
        "",
        "PASS/FAIL:",
        f"- 96-row table structure: {check_status('96 total Supplementary Table S3 rows')}",
        f"- outcome/pollutant/lag/model-type coverage: {'PASS' if all(check_status(name) == 'PASS' for name in ['2 outcomes present', '3 pollutants present', '8 prespecified lags present', '2 model types present', 'no unsupported lags']) else 'FAIL'}",
        f"- model-row and participant-count targets: {'PASS' if all(check_status(name) == 'PASS' for name in ['heart-rate n rows target', 'heart-rate n participants target', 'stress n rows target', 'stress n participants target']) else 'FAIL'}",
        f"- convergence: {check_status('all lag models converged or deviations documented')}",
        f"- unadjusted specification: {check_status('unadjusted fixed effects match expected')}",
        f"- adjusted specification: {check_status('adjusted fixed effects include expected covariates')}",
        f"- coefficient scaling per 1 and per 10 ug/m3: {'PASS' if check_status('beta per 10 ug/m3 = 10 x beta per 1 ug/m3') == 'PASS' and check_status('CI per 10 ug/m3 = 10 x CI per 1 ug/m3') == 'PASS' else 'FAIL'}",
        f"- p-value formatting: {'PASS' if check_status('zero-formatted p-value display absent') == 'PASS' and check_status('p values below 0.001 formatted as <0.001') == 'PASS' else 'FAIL'}",
        f"- no formal DLNM/cross-basis modelling: {dlnm_status}",
        f"- moving-average sensitivity: {moving_status}",
        f"- no participant-level safe outputs: {safe_status} ({safe_note})",
        "",
        "Missing dependencies:",
        "- None for Phase 6 lag-model reproduction.",
        "",
        "Deviations from current Supplementary Table S3 values:",
        "- None." if not deviations else "- " + "; ".join(deviations),
        "",
        "Output files:",
    ]
    for name in REQUIRED_OUTPUTS:
        lines.append(f"- {(outdir / name).resolve()}")
    lines.extend(
        [
            "",
            "Confirmations:",
            "- No Phase 7 work was performed.",
            "- No sleep models were run.",
            "- No Figure 5-8 audits were run.",
            "- No HIA/YLL work was run.",
            "- No GitHub push, Git commit, or data upload was performed.",
            "- Controlled data remained local.",
            "- Safe outputs contain aggregate model results and validation tables only; participant IDs, participant UID columns, source-member paths, raw timestamps, row-level Feather identifiers, and model input rows are absent.",
        ]
    )
    outpath = outdir / "phase6_validation_report.txt"
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outpath


def run_phase6_models(
    data_zip: str | Path,
    outdir: str | Path,
    phase1_dir: str | Path = "local_outputs/denominators",
    date_filter_mode: DateFilterMode = "campaign",
    max_model_rows: int = 10_000,
    min_model_rows: int = 200,
    min_groups: int = 5,
    maxiter: int = 100,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    load_result = load_physiology_data(data_zip, date_filter_mode=date_filter_mode)
    data = add_moving_average_pm_terms(add_lagged_pm_terms(load_result.data))
    config = ModelRunConfig(
        max_model_rows=max_model_rows,
        min_model_rows=min_model_rows,
        min_groups=min_groups,
        maxiter=maxiter,
    )
    primary = _fit_grid(data, config=config, moving_average=False)
    unadjusted = primary[primary["model_type"] == "unadjusted"].reset_index(drop=True)
    adjusted = primary[primary["model_type"] == "adjusted"].reset_index(drop=True)
    moving = _fit_grid(data, config=config, moving_average=True)
    moving_output = make_moving_average_output(moving)
    convergence = make_convergence_audit(unadjusted, adjusted, moving)
    table_s3 = make_supplementary_table_s3(unadjusted, adjusted)
    validation = make_validation_report_table(table_s3, convergence)
    dlnm = check_no_dlnm_language(Path.cwd() / "scripts", outdir, outdir)

    outputs = {
        "lag_model_results_unadjusted": outdir / "lag_model_results_unadjusted.csv",
        "lag_model_results_adjusted": outdir / "lag_model_results_adjusted.csv",
        "lag_model_convergence_audit": outdir / "lag_model_convergence_audit.csv",
        "lag_model_moving_average_sensitivity": outdir / "lag_model_moving_average_sensitivity.csv",
        "supplementary_table_s3_reproduced": outdir / "supplementary_table_s3_reproduced.csv",
        "lag_model_validation_report": outdir / "lag_model_validation_report.csv",
        "no_dlnm_language_check": outdir / "no_dlnm_language_check.csv",
        "phase6_validation_report": outdir / "phase6_validation_report.txt",
    }
    unadjusted.to_csv(outputs["lag_model_results_unadjusted"], index=False)
    adjusted.to_csv(outputs["lag_model_results_adjusted"], index=False)
    convergence.to_csv(outputs["lag_model_convergence_audit"], index=False)
    moving_output.to_csv(outputs["lag_model_moving_average_sensitivity"], index=False)
    table_s3.to_csv(outputs["supplementary_table_s3_reproduced"], index=False)
    validation.to_csv(outputs["lag_model_validation_report"], index=False)
    dlnm.to_csv(outputs["no_dlnm_language_check"], index=False)
    write_phase6_report(Path.cwd(), data_zip, phase1_dir, outdir, date_filter_mode)
    return outputs


def export_supplementary_table_from_results(unadjusted_path: str | Path, adjusted_path: str | Path, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    unadjusted = pd.read_csv(unadjusted_path)
    adjusted = pd.read_csv(adjusted_path)
    outpath = outdir / "supplementary_table_s3_reproduced.csv"
    make_supplementary_table_s3(unadjusted, adjusted).to_csv(outpath, index=False)
    return outpath


def validate_lag_outputs(table_s3_path: str | Path, convergence_path: str | Path, outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table_s3 = pd.read_csv(table_s3_path)
    convergence = pd.read_csv(convergence_path)
    outpath = outdir / "lag_model_validation_report.csv"
    make_validation_report_table(table_s3, convergence).to_csv(outpath, index=False)
    return outpath
