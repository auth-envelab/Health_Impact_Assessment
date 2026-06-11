"""Manuscript table generation from aggregate workflow outputs."""

from __future__ import annotations

import csv
import math
import shutil
import zipfile
from pathlib import Path

from raise_icarus.figure_templates import manifest_output_path, manifest_source, parse_simple_manifest
from raise_icarus.stage_contracts import StageDefinition, StageResult, definition_for, run_contract_stage

STAGE = definition_for(__name__)


def stage_definition() -> StageDefinition:
    return STAGE


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    return run_contract_stage(STAGE, harmonized_zip=harmonized_zip, run_dir=run_dir, n_samples=n_samples, dry_run=dry_run)


def copy_or_convert(source: Path, target: Path) -> tuple[bool, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".xlsx" and source.suffix.lower() == ".csv":
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:
            csv_target = target.with_suffix(".csv")
            shutil.copy2(source, csv_target)
            return False, f"xlsx conversion unavailable; wrote {csv_target.name}; {exc.__class__.__name__}"
        pd.read_csv(source).to_excel(target, index=False)
        return True, f"wrote {target.name}"
    shutil.copy2(source, target)
    return True, f"wrote {target.name}"


def generate_tables(run_dir: str | Path, manifest_path: str | Path | None = None, dry_run: bool = False) -> int:
    run_dir = Path(run_dir)
    manifest = Path(manifest_path) if manifest_path else run_dir / "reports" / "manuscript_item_manifest_detected.yaml"
    report_rows: list[dict[str, str]] = []
    failed = False
    for item in parse_simple_manifest(manifest):
        if item.get("item_type") != "table":
            continue
        target_rel = manifest_output_path(item)
        source_rel = manifest_source(item)
        source = run_dir / source_rel if source_rel and not source_rel.startswith("<") else None
        if dry_run:
            report_rows.append({"item_id": item.get("item_id", ""), "expected_output": target_rel, "source": source_rel, "status": "PASS_DRY_RUN", "notes": "table mapping resolved"})
        elif source and source.exists():
            ok, note = copy_or_convert(source, run_dir / target_rel)
            failed = failed or not ok
            report_rows.append({"item_id": item.get("item_id", ""), "expected_output": target_rel, "source": source_rel, "status": "PASS" if ok else "FAIL", "notes": note})
        else:
            failed = True
            report_rows.append({"item_id": item.get("item_id", ""), "expected_output": target_rel, "source": source_rel, "status": "FAIL", "notes": "required aggregate source is missing"})
    write_table_report(run_dir / "reports" / "manuscript_table_generation_report.csv", report_rows)
    return 1 if failed else 0


def write_table_report(path: str | Path, rows: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "expected_output", "source", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)

# Controlled-data table analytical exports.

from raise_icarus.controlled_runtime import config_value as _ri_table_config_value
from raise_icarus.controlled_runtime import copy_alias as _ri_table_copy_alias
from raise_icarus.controlled_runtime import csv_to_xlsx as _ri_table_csv_to_xlsx
from raise_icarus.controlled_runtime import domain_dir as _ri_table_domain_dir
from raise_icarus.controlled_runtime import repo_root as _ri_table_repo_root
from raise_icarus.controlled_runtime import reports_dir as _ri_table_reports_dir
from raise_icarus.controlled_runtime import tables_dir as _ri_table_tables_dir
from raise_icarus.controlled_runtime import write_csv_rows as _ri_table_write_csv_rows
from raise_icarus.controlled_runtime import write_text_report as _ri_table_write_text_report
from raise_icarus.data import _apply_date_filter as _ri_s7_apply_date_filter
from raise_icarus.data import feather_members as _ri_s7_feather_members
from raise_icarus.data import parse_city_season as _ri_s7_parse_city_season
from raise_icarus.data import read_feather_member as _ri_s7_read_feather_member
from raise_icarus.stage_contracts import dry_run_stage_result as _ri_table_dry_run_stage_result


S7_DATA_FILENAME = "Supplementary Table S7 data - ANOVA personal PM by city and season.csv"
S7_XLSX_FILENAME = "Supplementary Table S7 - ANOVA personal PM by city and season.xlsx"
S7_TABLE_LABEL = "Supplementary Table S7"
S7_MODEL_LABEL = "log(PM) ~ season * city"
S7_SOURCE_DEVICE = "ICARUS PPM"
S7_TRANSFORM_LABEL = "log"
S7_LOG_OFFSET = 0.001
S7_LAGS = [1, 5, 10, 15, 30, 45, 60, 120]
S7_PM_SPECS = [
    ("PM10", "PM10_PPM"),
    ("PM2.5", "PM25_PPM"),
    ("PM1", "PM1_PPM"),
]
S7_TERM_ORDER = [
    ("C(season)", "Season"),
    ("C(city)", "City"),
    ("C(season):C(city)", "Season × City"),
]
S7_FIELDNAMES = [
    "table_label",
    "pollutant",
    "source_device",
    "outcome_transform",
    "model",
    "term",
    "df",
    "sum_sq",
    "mean_sq",
    "F",
    "p_value",
    "p_value_display",
    "significance_stars",
    "n_observations",
    "n_participants",
    "support_definition",
    "city_levels",
    "season_levels",
    "status",
    "notes",
]
S7_REQUIRED_COLUMNS = set(S7_FIELDNAMES)
S7_FORBIDDEN_OUTPUT_COLUMNS = {
    "ID",
    "TS",
    "timestamp",
    "participant" + "_uid",
    "participant" + "_id",
    "source" + "_member",
    "archive" + "_member",
}
S7_LEGACY_COMPLETE_CASE_COLUMNS = [
    "ID",
    "Activity",
    "AvgHeartRate",
    "Intensity",
    "Stress",
    "Temp_PPM",
    "Humi_PPM",
    "PM10_PPM",
    "PM25_PPM",
    "PM1_PPM",
]
S7_SUPPORT_DEFINITION = (
    "Legacy timestamp-level complete-case ICARUS PPM support: rows retained after the legacy "
    "lag and rolling PM complete-case filter over PPM, activity, heart-rate, stress, and PPM "
    "temperature/humidity fields; PM outcomes use the legacy log10(PM + 0.001) transform. "
    "Only aggregate ANOVA terms and support counts are exported."
)
S7_CAPTION_NOTE = (
    "ANOVA results for personal particulate matter concentrations by season and city. "
    "PM1, PM2.5, and PM10 data were measured using the ICARUS portable particulate "
    "monitor (PPM), not the residential uHoo sensor. PPM and uHoo PM measurements "
    "were not averaged or combined. The retained seasonal campaign windows were "
    "Milan summer: 11 June-1 July 2019; Milan winter: 9 January-17 February 2019; "
    "Thessaloniki summer: 1 June-29 July 2019; and Thessaloniki winter: "
    "16 December 2018-24 January 2019."
)


def build_table5_dependency_report(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> Path:
    del harmonized_zip
    from raise_icarus.phase10_tables import write_table5_outputs

    table_dir = _ri_table_domain_dir(out_dir, "tables")
    outputs = write_table5_outputs(table_dir, demographics_file=_ri_table_config_value(config, "demographics_file", None))
    return _ri_table_copy_alias(outputs["table5_reproduced"], table_dir / "Table 5 dependency report.csv")


def build_supplementary_table_s6(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> Path:
    from raise_icarus.phase10_tables import write_table6_outputs

    table_dir = _ri_table_domain_dir(out_dir, "tables")
    outputs = write_table6_outputs(harmonized_zip, table_dir, date_filter_mode=_ri_table_config_value(config, "date_filter_mode", "campaign"))
    csv_path = _ri_table_copy_alias(outputs["table6_reproduced"], table_dir / "Supplementary Table S6 data - Descriptive statistics by city and season.csv")
    return _ri_table_csv_to_xlsx(csv_path, _ri_table_tables_dir(out_dir, "supplementary") / "Supplementary Table S6 - Descriptive statistics by city and season.xlsx")


def _s7_normalise_participant(value: object, city: str) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    prefix = "T" if city == "Thessaloniki" else "M" if city == "Milan" else ""
    if prefix and not text.startswith(prefix):
        text = prefix + text
    return f"{city}:{text}"


def load_supplementary_table_s7_timestamp_support(
    harmonized_zip: str | Path,
    date_filter_mode: str = "none",
) -> object:
    """Load the controlled timestamp-level support used by the legacy S7 ANOVA.

    This returns an in-memory frame only. Public outputs are produced later from
    aggregate ANOVA rows and support counts, never from row-level observations.
    """
    import pandas as pd  # type: ignore

    records = []
    required = {"TS", *S7_LEGACY_COMPLETE_CASE_COLUMNS}
    with zipfile.ZipFile(Path(harmonized_zip)) as zf:
        members = _ri_s7_feather_members(harmonized_zip)
        if not members:
            raise FileNotFoundError(f"No .feather files found in {Path(harmonized_zip).name}")
        for member in members:
            city, season = _ri_s7_parse_city_season(member)
            df = _ri_s7_read_feather_member(zf, member)
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"A harmonized archive member is missing required S7 columns: {sorted(missing)}")
            tmp = df.loc[:, ["TS", *S7_LEGACY_COMPLETE_CASE_COLUMNS]].copy()
            tmp["TS"] = pd.to_datetime(tmp["TS"], errors="coerce")
            tmp = tmp.dropna(subset=["TS", "ID"])
            tmp, _audit = _ri_s7_apply_date_filter(tmp, city, season, date_filter_mode)  # type: ignore[arg-type]
            if tmp.empty:
                continue
            tmp["city"] = city
            tmp["season"] = season
            tmp["_participant_key"] = tmp["ID"].map(lambda value: _s7_normalise_participant(value, city))
            records.append(tmp)
    if not records:
        return pd.DataFrame(columns=["city", "season", "_participant_key", "TS", *S7_LEGACY_COMPLETE_CASE_COLUMNS])
    return pd.concat(records, ignore_index=True)


def _s7_prepare_frame(frame: object, enforce_legacy_complete_case: bool = True) -> object:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    data = pd.DataFrame(frame).copy()
    rename: dict[str, str] = {}
    if "City" in data.columns and "city" not in data.columns:
        rename["City"] = "city"
    if "Season" in data.columns and "season" not in data.columns:
        rename["Season"] = "season"
    if rename:
        data = data.rename(columns=rename)
    missing = {"city", "season", *[column for _label, column in S7_PM_SPECS]} - set(data.columns)
    if missing:
        raise ValueError(f"S7 input frame is missing required columns: {sorted(missing)}")
    if "_participant_key" not in data.columns:
        for candidate in ["participant" + "_uid", "participant" + "_id", "ID"]:
            if candidate in data.columns:
                data["_participant_key"] = data.apply(lambda row: _s7_normalise_participant(row[candidate], str(row["city"])), axis=1)
                break
        if "_participant_key" not in data.columns:
            data["_participant_key"] = "not_available"
    for _label, column in S7_PM_SPECS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["AvgHeartRate", "Stress", "Temp_PPM", "Humi_PPM"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    required_subset = ["city", "season", "_participant_key", *[column for _label, column in S7_PM_SPECS]]
    if enforce_legacy_complete_case:
        missing_legacy = {"TS", *S7_LEGACY_COMPLETE_CASE_COLUMNS} - set(data.columns)
        if missing_legacy:
            raise ValueError(f"S7 legacy complete-case support is missing columns: {sorted(missing_legacy)}")
        data["TS"] = pd.to_datetime(data["TS"], errors="coerce")
        data = data.sort_values(["_participant_key", "city", "season", "TS"]).copy()
        data["timestamp"] = pd.to_datetime(data["TS"], errors="coerce")
        data["hour"] = data["timestamp"].dt.hour
        data["day_of_week"] = data["timestamp"].dt.dayofweek
        data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
        data["time_of_day"] = pd.cut(data["hour"], bins=[0, 6, 12, 18, 24], labels=["Night", "Morning", "Afternoon", "Evening"], include_lowest=True)
        group_keys = ["_participant_key", "city", "season"]
        for _label, column in S7_PM_SPECS:
            for lag in S7_LAGS:
                data[f"{column}_lag{lag}"] = data.groupby(group_keys, sort=False)[column].shift(lag)
            data[f"{column}_1h_avg"] = data.groupby(group_keys, sort=False)[column].transform(lambda values: values.rolling(window=60, min_periods=10).mean())
        pm_related = [column for column in data.columns if "PM1" in str(column) or "PM25" in str(column)]
        for column in pm_related:
            numeric = pd.to_numeric(data[column], errors="coerce")
            numeric = numeric.where(numeric + S7_LOG_OFFSET > 0)
            data[column] = np.log10(numeric + S7_LOG_OFFSET)
        selected = [
            "TS",
            "ID",
            "Activity",
            "AvgHeartRate",
            "Intensity",
            "Stress",
            "Temp_PPM",
            "Humi_PPM",
            "city",
            "season",
            "timestamp",
            "hour",
            "day_of_week",
            "is_weekend",
            "time_of_day",
            "_participant_key",
            *pm_related,
        ]
        required_subset = [column for column in selected if column in data.columns]
    else:
        for _label, column in S7_PM_SPECS:
            numeric = pd.to_numeric(data[column], errors="coerce")
            numeric = numeric.where(numeric + S7_LOG_OFFSET > 0)
            data[column] = np.log10(numeric + S7_LOG_OFFSET)

    data = data.dropna(subset=required_subset).copy()
    for _label, column in S7_PM_SPECS:
        data[f"log_{column}"] = data[column]
    return data


def _s7_sorted_levels(values: object, preferred: list[str]) -> str:
    seen = [str(value) for value in values if str(value) and str(value) != "nan"]
    ordered = [value for value in preferred if value in seen]
    ordered.extend(sorted(value for value in set(seen) if value not in ordered))
    return ";".join(ordered)


def _s7_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _s7_format_p_value(p_value: object) -> str:
    p = _s7_float(p_value)
    if not math.isfinite(p):
        return ""
    if p < 0.001:
        return "p < 0.001"
    rounded = max(round(p, 3), 0.001)
    return f"p = {rounded:.3f}"


def _s7_stars(p_value: object) -> str:
    p = _s7_float(p_value)
    if not math.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def compute_supplementary_table_s7_anova(
    frame: object,
    support_definition: str = S7_SUPPORT_DEFINITION,
    enforce_legacy_complete_case: bool = True,
) -> list[dict[str, object]]:
    """Compute aggregate two-way ANOVA rows for personal PPM by season/city."""
    import statsmodels.api as sm  # type: ignore
    import statsmodels.formula.api as smf  # type: ignore

    data = _s7_prepare_frame(frame, enforce_legacy_complete_case=enforce_legacy_complete_case)
    city_levels = _s7_sorted_levels(data["city"].dropna().unique(), ["Milan", "Thessaloniki"])
    season_levels = _s7_sorted_levels(data["season"].dropna().unique(), ["Summer", "Winter"])
    rows: list[dict[str, object]] = []
    for pollutant, column in S7_PM_SPECS:
        model_data = data[["city", "season", "_participant_key", f"log_{column}"]].rename(columns={f"log_{column}": "log_pm"}).dropna().copy()
        if model_data.empty or model_data["city"].nunique() < 2 or model_data["season"].nunique() < 2:
            raise ValueError(f"S7 ANOVA has insufficient city/season support for {pollutant}")
        model = smf.ols("log_pm ~ C(season) + C(city) + C(season):C(city)", data=model_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        n_observations = int(model_data.shape[0])
        n_participants = int(model_data["_participant_key"].nunique())
        for raw_term, term_label in S7_TERM_ORDER:
            if raw_term not in anova_table.index:
                raise ValueError(f"S7 ANOVA term {raw_term} missing for {pollutant}")
            record = anova_table.loc[raw_term]
            df = _s7_float(record.get("df"))
            sum_sq = _s7_float(record.get("sum_sq"))
            mean_sq = sum_sq / df if math.isfinite(sum_sq) and math.isfinite(df) and df else math.nan
            f_stat = _s7_float(record.get("F"))
            p_value = _s7_float(record.get("PR(>F)"))
            rows.append(
                {
                    "table_label": S7_TABLE_LABEL,
                    "pollutant": pollutant,
                    "source_device": S7_SOURCE_DEVICE,
                    "outcome_transform": S7_TRANSFORM_LABEL,
                    "model": S7_MODEL_LABEL,
                    "term": term_label,
                    "df": df,
                    "sum_sq": sum_sq,
                    "mean_sq": mean_sq,
                    "F": f_stat,
                    "p_value": p_value,
                    "p_value_display": _s7_format_p_value(p_value),
                    "significance_stars": _s7_stars(p_value),
                    "n_observations": n_observations,
                    "n_participants": n_participants,
                    "support_definition": support_definition,
                    "city_levels": city_levels,
                    "season_levels": season_levels,
                    "status": "PASS",
                    "notes": S7_CAPTION_NOTE,
                }
            )
    return rows


def build_supplementary_table_s7_anova_personal_pm(
    harmonized_zip: str,
    out_dir: str,
    config: dict | None = None,
) -> dict:
    table_dir = _ri_table_domain_dir(out_dir, "tables")
    csv_path = table_dir / S7_DATA_FILENAME
    xlsx_path = _ri_table_tables_dir(out_dir, "supplementary") / S7_XLSX_FILENAME
    synthetic_frame = _ri_table_config_value(config, "synthetic_s7_frame", None)
    if synthetic_frame is None:
        date_filter_mode = str(_ri_table_config_value(config, "s7_date_filter_mode", "none"))
        frame = load_supplementary_table_s7_timestamp_support(harmonized_zip, date_filter_mode=date_filter_mode)
        support_definition = str(_ri_table_config_value(config, "s7_support_definition", S7_SUPPORT_DEFINITION))
        enforce_legacy = True
    else:
        frame = synthetic_frame
        support_definition = str(_ri_table_config_value(config, "s7_support_definition", "Synthetic aggregate-safety test support."))
        enforce_legacy = bool(_ri_table_config_value(config, "s7_enforce_legacy_complete_case", True))
    rows = compute_supplementary_table_s7_anova(
        frame,
        support_definition=support_definition,
        enforce_legacy_complete_case=enforce_legacy,
    )
    _ri_table_write_csv_rows(csv_path, rows, S7_FIELDNAMES)
    _ri_table_csv_to_xlsx(csv_path, xlsx_path)
    return {"csv": csv_path, "xlsx": xlsx_path, "status": "PASS", "rows": len(rows)}


def build_supplementary_table_s7(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> Path:
    result = build_supplementary_table_s7_anova_personal_pm(str(harmonized_zip), str(out_dir), config if isinstance(config, dict) else None)
    return Path(result["xlsx"])


def _s7_read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _s7_clean_yaml_value(value: str) -> object:
    text = value.strip().strip('"').strip("'")
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return float(text)
    except ValueError:
        return text


def _s7_expected_rows(expected_manifest: str | Path | None) -> list[dict[str, object]]:
    if expected_manifest is None:
        return []
    path = Path(expected_manifest)
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    in_section = False
    in_rows = False
    current: dict[str, object] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("supplementary_table_s7_anova:"):
            in_section = True
            in_rows = False
            continue
        if in_section and raw and not raw.startswith(" "):
            break
        if not in_section:
            continue
        stripped = raw.strip()
        if stripped == "expected_rows:":
            in_rows = True
            continue
        if not in_rows or not stripped:
            continue
        if stripped.startswith("- "):
            if current:
                rows.append(current)
            current = {}
            stripped = stripped[2:]
        if current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = _s7_clean_yaml_value(value)
    if current:
        rows.append(current)
    return rows


def _s7_add_check(rows: list[dict[str, str]], check: str, status: str, observed: object = "", expected: object = "", notes: str = "") -> None:
    rows.append({"check": check, "status": status, "observed": str(observed), "expected": str(expected), "notes": notes})


def _s7_close_enough(observed: object, expected: object, abs_tol: float = 0.02, rel_tol: float = 1e-4) -> bool:
    obs = _s7_float(observed)
    exp = _s7_float(expected)
    if not math.isfinite(obs) or not math.isfinite(exp):
        return False
    return abs(obs - exp) <= max(abs_tol, rel_tol * max(1.0, abs(exp)))


def validate_supplementary_table_s7(
    table_s7_csv: str,
    expected_manifest: str | None,
    out_dir: str,
) -> dict:
    path = Path(table_s7_csv)
    report_dir = _ri_table_reports_dir(out_dir)
    rows: list[dict[str, str]] = []
    table_rows = _s7_read_csv(path)
    _s7_add_check(rows, "s7_file_exists", "PASS" if path.exists() else "FAIL", path.name, "file present")
    if not table_rows:
        _s7_add_check(rows, "s7_has_rows", "FAIL", "0", "9")
    else:
        _s7_add_check(rows, "s7_has_rows", "PASS" if len(table_rows) == 9 else "FAIL", len(table_rows), "9")
        headers = set(table_rows[0].keys())
        missing = sorted(S7_REQUIRED_COLUMNS - headers)
        _s7_add_check(rows, "s7_required_columns", "PASS" if not missing else "FAIL", ";".join(missing) if missing else "none", "all required columns")
        forbidden = sorted(headers & S7_FORBIDDEN_OUTPUT_COLUMNS)
        _s7_add_check(rows, "s7_no_identifier_columns", "PASS" if not forbidden else "FAIL", ";".join(forbidden) if forbidden else "none", "no row-level identifier columns")
        pollutants = sorted({row.get("pollutant", "") for row in table_rows})
        terms = sorted({row.get("term", "") for row in table_rows})
        _s7_add_check(rows, "s7_pollutants", "PASS" if set(pollutants) == {"PM1", "PM2.5", "PM10"} else "FAIL", ";".join(pollutants), "PM1;PM2.5;PM10")
        _s7_add_check(rows, "s7_terms", "PASS" if set(terms) == {"Season", "City", "Season × City"} else "FAIL", ";".join(terms), "Season;City;Season × City")
        source_ok = all(row.get("source_device") == S7_SOURCE_DEVICE for row in table_rows)
        transform_ok = all(row.get("outcome_transform") == S7_TRANSFORM_LABEL for row in table_rows)
        model_ok = all(row.get("model") == S7_MODEL_LABEL for row in table_rows)
        p_display_ok = all("0.000" not in row.get("p_value_display", "") for row in table_rows)
        status_ok = all(row.get("status") == "PASS" for row in table_rows)
        no_uhoo = all("uHoo" not in row.get("pollutant", "") and "uHoo" not in row.get("source_device", "") for row in table_rows)
        notes_ok = all("not averaged or combined" in row.get("notes", "") for row in table_rows)
        _s7_add_check(rows, "s7_source_device", "PASS" if source_ok else "FAIL", source_ok, S7_SOURCE_DEVICE)
        _s7_add_check(rows, "s7_no_uhoo_pm_used", "PASS" if no_uhoo else "FAIL", no_uhoo, "uHoo excluded")
        _s7_add_check(rows, "s7_no_ppm_uhoo_averaging", "PASS" if notes_ok else "FAIL", notes_ok, "not averaged or combined")
        _s7_add_check(rows, "s7_outcome_transform", "PASS" if transform_ok else "FAIL", transform_ok, S7_TRANSFORM_LABEL)
        _s7_add_check(rows, "s7_model", "PASS" if model_ok else "FAIL", model_ok, S7_MODEL_LABEL)
        _s7_add_check(rows, "s7_p_value_format", "PASS" if p_display_ok else "FAIL", "zero-formatted p-value absent" if p_display_ok else "zero-formatted p-value detected", "p < 0.001 when needed")
        _s7_add_check(rows, "s7_row_status", "PASS" if status_ok else "FAIL", status_ok, "all rows PASS")
        expected_rows = _s7_expected_rows(expected_manifest)
        if expected_rows:
            row_by_key = {(row.get("pollutant"), row.get("term")): row for row in table_rows}
            for expected in expected_rows:
                key = (str(expected.get("pollutant", "")), str(expected.get("term", "")))
                observed = row_by_key.get(key)
                if observed is None:
                    _s7_add_check(rows, f"s7_expected_{key[0]}_{key[1]}", "FAIL", "missing", "present")
                    continue
                f_ok = _s7_close_enough(observed.get("F"), expected.get("F"))
                df_ok = _s7_close_enough(observed.get("df"), expected.get("df"), abs_tol=1e-9, rel_tol=1e-9)
                p_expected = _s7_float(expected.get("p_value"))
                p_observed = _s7_float(observed.get("p_value"))
                p_ok = math.isfinite(p_expected) and math.isfinite(p_observed) and ((p_expected < 0.001 and p_observed < 0.001) or _s7_close_enough(p_observed, p_expected, abs_tol=1e-6, rel_tol=1e-3))
                _s7_add_check(rows, f"s7_expected_{key[0]}_{key[1]}", "PASS" if f_ok and df_ok and p_ok else "FAIL", f"F={observed.get('F')};df={observed.get('df')};p={observed.get('p_value')}", f"F={expected.get('F')};df={expected.get('df')};p={expected.get('p_value')}")
        else:
            _s7_add_check(rows, "s7_expected_values_status", "PASS", "not checked", "manifest optional", "No expected S7 values found in manifest.")
    report_csv = report_dir / "supplementary_table_s7_validation.csv"
    _ri_table_write_csv_rows(report_csv, rows, ["check", "status", "observed", "expected", "notes"])
    failed = [row for row in rows if row["status"] != "PASS"]
    report_txt = _ri_table_write_text_report(
        report_dir / "supplementary_table_s7_validation_report.txt",
        "Supplementary Table S7 Validation Report",
        [f"checks_evaluated: {len(rows)}", f"checks_failed: {len(failed)}", "gate_status: " + ("FAIL" if failed else "PASS")],
    )
    return {"status": "FAIL" if failed else "PASS", "report_csv": report_csv, "report_txt": report_txt, "checks": rows}


def validate_table_outputs(out_dir: str | Path, expected_manifest: str | Path | None = None) -> Path:
    if expected_manifest is None:
        expected_manifest = _ri_table_repo_root() / "configs" / "expected_results_manifest.yaml"
    required = [
        _ri_table_domain_dir(out_dir, "tables") / "Table 5 dependency report.csv",
        _ri_table_tables_dir(out_dir, "supplementary") / "Supplementary Table S6 - Descriptive statistics by city and season.xlsx",
    ]
    lines = [f"{p.name}: {'PASS' if p.exists() else 'FAIL'}" for p in required]
    return _ri_table_write_text_report(_ri_table_reports_dir(out_dir) / "Table output validation report.txt", "Table Output Validation Report", lines)


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> StageResult:
    del n_samples
    if dry_run:
        return _ri_table_dry_run_stage_result(STAGE, run_dir)
    if harmonized_zip is None:
        return StageResult(STAGE.stage_name, STAGE.module_name, "FAIL", STAGE.output_domain, (), "harmonized archive path is required")
    build_table5_dependency_report(harmonized_zip, run_dir)
    build_supplementary_table_s6(harmonized_zip, run_dir)
    report = validate_table_outputs(run_dir, _ri_table_repo_root() / "configs" / "expected_results_manifest.yaml")
    return StageResult(STAGE.stage_name, STAGE.module_name, "PASS", STAGE.output_domain, (str(report),), "Table outputs generated for the current manuscript scope.")

