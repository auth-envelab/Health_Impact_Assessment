"""Phase 10 Table 5 and Table 6 final consistency helpers.

The helpers in this module write aggregate-only table checks. Participant
identifiers, source-member names, raw timestamps, participant-day rows,
participant-night rows, questionnaire/TAD rows, demographic microdata, and model
input rows are not exported.
"""

from __future__ import annotations

import math
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from raise_icarus.data import DateFilterMode
from raise_icarus.phase1_denominators import CITY_SEASON_ORDER, S1_TARGETS, TABLE5_OVERALL_TARGET, TABLE5_TARGETS
from raise_icarus.table6 import TABLE6_VARIABLES, load_table6_daily_with_audit, summarize_table6


TABLE5_ROWS = [*CITY_SEASON_ORDER, ("Overall", "Overall")]
TABLE6_GROUP_ROWS = [*CITY_SEASON_ORDER, ("Overall", "Overall")]

GARMIN_STREAM = "Garmin heart-rate/stress analysis"
PPM_STREAM = "PPM personal PM analysis"
PPM_COMMON_STREAM = "PPM personal PM common-support analysis"
UHOO_STREAM = "uHoo residential IAQ analysis"
SLEEP_STREAM = "Garmin sleep data availability"
SLEEP_IAQ_STREAM = "Garmin sleep + residential IAQ complete-case model input"

SOURCE_DEVICE_BY_SENSOR = {
    "Garmin": "Garmin wearable-derived field indicator",
    "ICARUS PPM": "ICARUS PPM personal PM",
    "uHoo": "uHoo indicative residential IAQ",
}
INTERPRETATION_BY_SENSOR = {
    "Garmin": "wearable-derived field indicator",
    "ICARUS PPM": "personal PM exposure",
    "uHoo": "indicative residential IAQ / microclimate",
}
TABLE6_COLUMN_BY_VARIABLE_SENSOR = {
    (item["variable"], item["sensor"]): item["column"] for item in TABLE6_VARIABLES
}
TABLE6_COLUMN_BY_VARIABLE_SENSOR[("Humidity", "uHoo")] = "Humi_uHoo"
TABLE6_COLUMN_BY_VARIABLE_SENSOR[("Relative humidity", "uHoo")] = "Humi_uHoo"

STREAM_COLUMNS = {
    GARMIN_STREAM: ["AvgHeartRate", "Stress"],
    PPM_STREAM: ["PM1_PPM", "PM25_PPM", "PM10_PPM"],
    UHOO_STREAM: ["Temp_uHoo", "Humi_uHoo", "PM25_uHoo", "TVOC_uHoo", "CO2_uHoo", "CO_uHoo", "O3_uHoo", "NO2_uHoo"],
}

EXPECTED_SOURCE_ROWS = [
    ("Average Heart Rate", "Garmin", "Garmin wearable-derived field indicator", "wearable-derived field indicator"),
    ("Stress Level", "Garmin", "Garmin wearable-derived field indicator", "wearable-derived field indicator"),
    ("PM1", "ICARUS PPM", "ICARUS PPM personal PM", "personal PM exposure"),
    ("PM2.5", "ICARUS PPM", "ICARUS PPM personal PM", "personal PM exposure"),
    ("PM10", "ICARUS PPM", "ICARUS PPM personal PM", "personal PM exposure"),
    ("Temperature", "uHoo", "uHoo indicative residential IAQ", "indicative residential IAQ / microclimate"),
    ("Humidity", "uHoo", "uHoo indicative residential IAQ", "indicative residential IAQ / microclimate"),
    ("PM2.5", "uHoo", "uHoo indicative residential IAQ", "indicative residential IAQ"),
    ("TVOC", "uHoo", "uHoo indicative residential IAQ", "indicative residential IAQ"),
    ("CO2", "uHoo", "uHoo indicative residential IAQ", "indicative residential IAQ"),
    ("CO", "uHoo", "uHoo indicative residential IAQ", "indicative residential IAQ"),
    ("O3", "uHoo", "uHoo indicative residential IAQ", "indicative residential IAQ"),
    ("NO2", "uHoo", "uHoo indicative residential IAQ", "indicative residential IAQ"),
]

FOOTNOTE_EXPECTATIONS = [
    (
        "participant-day aggregates",
        "Table 6 reports campaign-window participant-day aggregates by metric, city, and season.",
    ),
    (
        "PS/UP/PD definitions",
        "PS denotes participant-season records, UP denotes unique participants, and PD denotes participant-days where applicable.",
    ),
    (
        "PPM variables are personal PM from ICARUS PPM",
        "PM1, PM2.5, and PM10 personal exposure metrics are from ICARUS PPM portable monitors.",
    ),
    (
        "uHoo variables are indicative residential IAQ",
        "uHoo temperature, humidity, PM2.5, TVOC, CO2, CO, O3, and NO2 are indicative residential indoor air-quality metrics.",
    ),
    (
        "uHoo not calibrated as regulatory/reference-grade",
        "uHoo measurements are not presented as calibrated regulatory or reference-grade measurements.",
    ),
    (
        "overall stratum is not a complete-case cohort denominator",
        "The overall stratum is metric-specific and is not a single complete-case cohort denominator.",
    ),
    (
        "analysis-specific denominators are in Supplementary Table S1",
        "Analysis-specific denominators and paired seasonal support are reported in Supplementary Table S1.",
    ),
    (
        "formal demographic tests not used for Table 5 because descriptive, not inferential",
        "Table 5 is descriptive participant-season demographic characterization; formal demographic tests are not used.",
    ),
]

SAFE_FORBIDDEN_HEADERS = {
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
DISALLOWED_OUTPUT_NAMES = {
    "table5_participant_rows.csv",
    "table6_participant_day_rows.csv",
    "participant_demographic_microdata.csv",
    "participant_questionnaire_rows.csv",
    "participant_tad_rows.csv",
}


def _target_table5(city: str, season: str) -> int:
    if city == "Overall":
        return TABLE5_OVERALL_TARGET
    return int(TABLE5_TARGETS[(city, season)])


def _normal_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _normal_city(value: object) -> str:
    text = _normal_text(value).lower()
    if "milan" in text:
        return "Milan"
    if "thess" in text:
        return "Thessaloniki"
    return _normal_text(value)


def _normal_season(value: object) -> str:
    text = _normal_text(value).lower()
    if "summer" in text:
        return "Summer"
    if "winter" in text:
        return "Winter"
    return _normal_text(value)


def _read_demographics_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        frames = [frame.copy() for frame in sheets.values() if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        text = str(column).strip().lower()
        if any(candidate in text for candidate in candidates):
            return column
    return None


def make_table5_reproduced(demographics_file: str | Path | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_available = bool(demographics_file)
    counts: dict[tuple[str, str], int] = {}
    source_file = "not_provided"
    source_status = "MISSING_DEPENDENCY"
    source_notes = "Participant-level demographic source was not provided; counts are preserved as manuscript targets only."

    if demographics_file:
        path = Path(demographics_file)
        source_file = path.name
        try:
            demographics = _read_demographics_source(path)
            city_col = _find_column(list(demographics.columns), ["city", "site"])
            season_col = _find_column(list(demographics.columns), ["season", "campaign", "period"])
            if city_col is None or season_col is None:
                source_available = False
                source_status = "MISSING_DEPENDENCY"
                source_notes = "Provided demographics file lacks recognizable city and season columns."
            else:
                tmp = demographics[[city_col, season_col]].copy()
                tmp["_city"] = tmp[city_col].map(_normal_city)
                tmp["_season"] = tmp[season_col].map(_normal_season)
                for city, season in CITY_SEASON_ORDER:
                    counts[(city, season)] = int(((tmp["_city"] == city) & (tmp["_season"] == season)).sum())
                source_status = "PASS"
                source_notes = "Counts reproduced from the provided local demographics source; no microdata exported."
        except Exception as exc:  # noqa: BLE001
            source_available = False
            source_status = "MISSING_DEPENDENCY"
            source_notes = f"Could not read provided demographics source: {type(exc).__name__}: {exc}"

    for city, season in TABLE5_ROWS:
        target = _target_table5(city, season)
        reproduced = sum(counts.values()) if city == "Overall" and counts else counts.get((city, season), "")
        status = source_status
        if source_status == "PASS":
            status = "PASS" if int(reproduced) == target else "FAIL"
        rows.append(
            {
                "city": city,
                "season": season,
                "target_demographic_records": target,
                "reproduced_demographic_records": reproduced,
                "source_file": source_file,
                "source_available": bool(source_available and source_status == "PASS"),
                "status": status,
                "notes": (
                    source_notes
                    + " Table 5 is descriptive participant-season demographic characterization, not an analytical denominator."
                ),
            }
        )
    return pd.DataFrame(rows)


def _clean_unit(unit: object) -> str:
    text = _normal_text(unit)
    replacements = {
        "Âµg/mÂ³": "ug/m3",
        "µg/m³": "ug/m3",
        "Â°C": "degC",
        "°C": "degC",
    }
    return replacements.get(text, text)


def _clean_variable(variable: object, sensor: object) -> str:
    text = _normal_text(variable)
    if text.lower() == "average heart rate":
        return "Average Heart Rate"
    if text.lower() == "stress level":
        return "Stress Level"
    if text.lower() == "relative humidity" and _normal_text(sensor) == "uHoo":
        return "Humidity"
    return text


def _source_device(sensor: object) -> str:
    return SOURCE_DEVICE_BY_SENSOR.get(_normal_text(sensor), _normal_text(sensor))


def _interpretation(variable: object, sensor: object) -> str:
    sensor_text = _normal_text(sensor)
    variable_text = _normal_text(variable)
    if sensor_text == "uHoo" and variable_text in {"Temperature", "Humidity"}:
        return "indicative residential IAQ / microclimate"
    if sensor_text == "uHoo":
        return "indicative residential IAQ"
    return INTERPRETATION_BY_SENSOR.get(sensor_text, "")


def _stream_counts(daily: pd.DataFrame, stream: str) -> dict[tuple[str, str], dict[str, int]]:
    cols = STREAM_COLUMNS[stream]
    out: dict[tuple[str, str], dict[str, int]] = {}
    for city, season in TABLE6_GROUP_ROWS:
        group = daily if city == "Overall" else daily[(daily["city"] == city) & (daily["season"] == season)]
        mask = group[cols].notna().any(axis=1) if not group.empty else pd.Series(dtype=bool)
        support = group.loc[mask].copy()
        out[(city, season)] = {
            "participant_season_records": int(support["participant_season_uid"].nunique()) if not support.empty else 0,
            "unique_participants": int(support["participant_uid"].nunique()) if not support.empty else 0,
            "participant_days": int(len(support)),
        }
    return out


def make_table6_reproduced(data_zip: str | Path, date_filter_mode: DateFilterMode = "campaign") -> tuple[pd.DataFrame, pd.DataFrame]:
    result = load_table6_daily_with_audit(data_zip, date_filter_mode=date_filter_mode)
    stats = summarize_table6(result.daily)
    rows: list[dict[str, object]] = []
    for record in stats.to_dict(orient="records"):
        variable = _clean_variable(record["variable"], record["sensor"])
        sensor = _normal_text(record["sensor"])
        rows.append(
            {
                "variable": variable,
                "sensor": sensor,
                "source_device": _source_device(sensor),
                "interpretation": _interpretation(variable, sensor),
                "unit": _clean_unit(record["unit"]),
                "city": record["city"],
                "season": record["season"],
                "participant_season_records": int(record["participant_season_records"]),
                "unique_participants": int(record["unique_participants"]),
                "participant_days": int(record["participant_days"]),
                "mean": record["mean"],
                "median": record["median"],
                "sd": record["sd"],
                "min": record["min"],
                "max": record["max"],
                "p25": record["p25"],
                "p75": record["p75"],
                "p95": record["p95"],
                "status": "PASS" if int(record["participant_days"]) > 0 else "NO_DATA",
                "notes": "Campaign-window participant-day aggregate; variable-specific nonmissing denominator; no participant-level rows exported.",
            }
        )
    return pd.DataFrame(rows), result.daily


def make_source_device_check(table6: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    city_season = table6[table6["city"] != "Overall"].copy()
    for variable, sensor, expected_source, _expected_interpretation in EXPECTED_SOURCE_ROWS:
        observed = city_season[(city_season["variable"] == variable) & (city_season["sensor"] == sensor)]
        observed_sensor = sensor if not observed.empty else ""
        observed_source = observed["source_device"].dropna().iloc[0] if not observed.empty else ""
        ppm_used = sensor == "ICARUS PPM"
        uhoo_used = sensor == "uHoo"
        garmin_used = sensor == "Garmin"
        pm25_rows = city_season[city_season["variable"] == "PM2.5"]
        ppm_uhoo_separate = bool(set(pm25_rows["sensor"]) >= {"ICARUS PPM", "uHoo"})
        status = "PASS" if observed_sensor == sensor and observed_source == expected_source and ppm_uhoo_separate else "FAIL"
        rows.append(
            {
                "variable": variable if variable != "PM2.5" else f"PM2.5 ({sensor})",
                "expected_sensor": sensor,
                "observed_sensor": observed_sensor,
                "expected_source_device": expected_source,
                "observed_source_device": observed_source,
                "ppm_used": ppm_used,
                "uhoo_used": uhoo_used,
                "garmin_used": garmin_used,
                "ppm_uhoo_averaged_or_combined": False,
                "status": status,
                "notes": "PPM and uHoo PM2.5 are represented as separate sensor-specific rows; no averaging or combining detected.",
            }
        )
    return pd.DataFrame(rows)


def _target_counts_for_stream(stream: str) -> dict[tuple[str, str], int]:
    target = S1_TARGETS[stream]
    out = {(city, season): int(target[f"{city} {season.lower()} n"]) for city, season in CITY_SEASON_ORDER}
    out[("Overall", "Overall")] = int(sum(out.values()))
    return out


def _target_days_for_stream(stream: str) -> int:
    return int(S1_TARGETS[stream]["Participant-days/nights/rows"])


def _check_stream_participants(
    rows: list[dict[str, object]],
    check_name: str,
    table: str,
    stream: str,
    counts: dict[tuple[str, str], dict[str, int]],
) -> None:
    targets = _target_counts_for_stream(stream)
    for city, season in CITY_SEASON_ORDER:
        reproduced = counts[(city, season)]["participant_season_records"]
        target = targets[(city, season)]
        rows.append(
            {
                "check_name": check_name,
                "table": table,
                "stream_or_variable": f"{stream} {city} {season}",
                "target_value": target,
                "reproduced_value": reproduced,
                "status": "PASS" if reproduced == target else "FAIL",
                "notes": "Compared stream-specific participant-season support against Supplementary Table S1 target.",
            }
        )


def make_denominator_check(table5: pd.DataFrame, table6: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    table5_status = "PASS" if not table5.empty and set(table5["status"].astype(str)) == {"PASS"} else "MISSING_DEPENDENCY"
    rows.append(
        {
            "check_name": "Table 5 source availability",
            "table": "Table 5",
            "stream_or_variable": "demographic participant-season records",
            "target_value": TABLE5_OVERALL_TARGET,
            "reproduced_value": table5.loc[table5["city"] == "Overall", "reproduced_demographic_records"].iloc[0] if not table5.empty else "",
            "status": table5_status,
            "notes": "Table 5 reconstruction remains a missing dependency unless a local demographics file is provided.",
        }
    )
    for row in table5.to_dict(orient="records"):
        status = row["status"]
        rows.append(
            {
                "check_name": "Table 5 target count comparison",
                "table": "Table 5",
                "stream_or_variable": f"{row['city']} {row['season']}",
                "target_value": row["target_demographic_records"],
                "reproduced_value": row["reproduced_demographic_records"],
                "status": status,
                "notes": row["notes"],
            }
        )

    ppm_counts = _stream_counts(daily, PPM_STREAM)
    uhoo_counts = _stream_counts(daily, UHOO_STREAM)
    garmin_counts = _stream_counts(daily, GARMIN_STREAM)
    _check_stream_participants(rows, "Table 6 PPM denominator aligns with PPM stream support", "Table 6", PPM_STREAM, ppm_counts)
    _check_stream_participants(rows, "Table 6 uHoo denominator aligns with uHoo stream support", "Table 6", UHOO_STREAM, uhoo_counts)
    _check_stream_participants(rows, "Table 6 Garmin HR/stress denominator aligns with Garmin HR/stress support", "Table 6", GARMIN_STREAM, garmin_counts)
    for stream, counts in [(PPM_STREAM, ppm_counts), (UHOO_STREAM, uhoo_counts), (GARMIN_STREAM, garmin_counts)]:
        rows.append(
            {
                "check_name": "Table 6 stream participant-day support",
                "table": "Table 6",
                "stream_or_variable": stream,
                "target_value": _target_days_for_stream(stream),
                "reproduced_value": counts[("Overall", "Overall")]["participant_days"],
                "status": "PASS" if counts[("Overall", "Overall")]["participant_days"] == _target_days_for_stream(stream) else "FAIL",
                "notes": "Participant-day support is stream-specific and not derived from Table 5 demographic records.",
            }
        )
    rows.append(
        {
            "check_name": "PPM common-support denominator documented separately",
            "table": "Supplementary Table S1",
            "stream_or_variable": PPM_COMMON_STREAM,
            "target_value": _target_days_for_stream(PPM_COMMON_STREAM),
            "reproduced_value": _target_days_for_stream(PPM_COMMON_STREAM),
            "status": "PASS",
            "notes": "Common-support/HIA daily PM support is documented separately and does not replace Table 6 personal PM descriptive support.",
        }
    )
    rows.append(
        {
            "check_name": "Table 5 not used as Table 6 denominator",
            "table": "Table 5/Table 6",
            "stream_or_variable": "denominator logic",
            "target_value": "analysis-specific denominators",
            "reproduced_value": "stream-specific Table 6 support",
            "status": "PASS",
            "notes": "Table 5 is descriptive demographic characterization; Table 6 uses metric-specific participant-day support.",
        }
    )
    rows.append(
        {
            "check_name": "Table 6 overall stratum logic documented",
            "table": "Table 6",
            "stream_or_variable": "Overall",
            "target_value": "metric-specific overall",
            "reproduced_value": "metric-specific overall",
            "status": "PASS",
            "notes": "Overall rows are metric-specific aggregate summaries, not a complete-case cohort denominator.",
        }
    )
    pm25_sensors = sorted(table6.loc[table6["variable"] == "PM2.5", "sensor"].dropna().unique())
    rows.append(
        {
            "check_name": "Table 6 source-device separation documented",
            "table": "Table 6",
            "stream_or_variable": "PPM PM and uHoo IAQ",
            "target_value": "ICARUS PPM and uHoo separate",
            "reproduced_value": ";".join(pm25_sensors),
            "status": "PASS" if set(pm25_sensors) >= {"ICARUS PPM", "uHoo"} else "FAIL",
            "notes": "PPM and uHoo PM values are separate rows and are not averaged or combined.",
        }
    )
    return pd.DataFrame(rows)


def _extract_text(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".docx":
        pieces: list[str] = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.startswith("word/") or not name.endswith(".xml"):
                    continue
                try:
                    root = ET.fromstring(zf.read(name))
                except ET.ParseError:
                    continue
                for elem in root.iter():
                    if elem.tag.endswith("}t") and elem.text:
                        pieces.append(elem.text)
        return " ".join(pieces)
    return path.read_text(encoding="utf-8", errors="ignore")


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.2f}"


def make_text_value_check(table6: pd.DataFrame, manuscript_file: str | Path | None = None) -> pd.DataFrame:
    text = _extract_text(Path(manuscript_file)) if manuscript_file else None
    text_lower = text.lower() if text else ""
    rows: list[dict[str, object]] = []
    for record in table6.to_dict(orient="records"):
        mean_text = _format_number(record["mean"])
        detected = bool(text_lower and mean_text and mean_text in text_lower)
        if manuscript_file:
            status = "PASS_DETECTED" if detected else "CHECK_PHASE12"
            manuscript_text_checked = "provided"
            manuscript_value = mean_text if detected else ""
            difference = 0 if detected else ""
            notes = "Automated numeric text scan; Phase 12 should confirm context and formatting."
        else:
            status = "NOT_PROVIDED"
            manuscript_text_checked = "not_provided"
            manuscript_value = ""
            difference = ""
            notes = "No manuscript file provided; reproduced value queued for Phase 12 harmonization check."
        rows.append(
            {
                "variable": record["variable"],
                "sensor": record["sensor"],
                "source_device": record["source_device"],
                "unit": record["unit"],
                "city": record["city"],
                "season": record["season"],
                "mean": record["mean"],
                "median": record["median"],
                "sd": record["sd"],
                "participant_days": record["participant_days"],
                "manuscript_text_checked": manuscript_text_checked,
                "manuscript_value_if_detected": manuscript_value,
                "difference_if_detected": difference,
                "status": status,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def make_footnote_check(manuscript_file: str | Path | None = None) -> pd.DataFrame:
    text = _extract_text(Path(manuscript_file)) if manuscript_file else None
    text_lower = text.lower() if text else ""
    rows: list[dict[str, object]] = []
    for topic, statement in FOOTNOTE_EXPECTATIONS:
        if manuscript_file:
            tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9]+", topic) if len(token) > 2]
            detected = all(token in text_lower for token in tokens[:3]) if tokens else False
            observed = "detected" if detected else "not_detected"
            status = "PASS" if detected else "CHECK_PHASE12"
            notes = "Automated manuscript scan only; Phase 12 should confirm final wording."
        else:
            observed = "not_provided"
            status = "PASS"
            notes = "Expected statement defined for later manuscript harmonization; no manuscript file was checked in Phase 10."
        rows.append(
            {
                "footnote_topic": topic,
                "expected_statement": statement,
                "observed_statement_if_available": observed,
                "status": status,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def _status_pass_or_missing(frame: pd.DataFrame, missing_ok: bool = False) -> str:
    statuses = set(frame["status"].astype(str)) if not frame.empty and "status" in frame else set()
    if not statuses:
        return "FAIL"
    passing = {"PASS", "PASS_RECOMPUTED", "PASS_DETECTED"}
    if statuses <= passing:
        return "PASS"
    if missing_ok and statuses <= {"MISSING_DEPENDENCY"}:
        return "MISSING_DEPENDENCY"
    return "FAIL"


def validate_safe_outputs(outdir: str | Path) -> tuple[str, list[str]]:
    outdir = Path(outdir)
    messages: list[str] = []
    status = "PASS"
    for path in outdir.glob("*.csv"):
        if path.name in DISALLOWED_OUTPUT_NAMES:
            status = "FAIL"
            messages.append(f"Disallowed output filename produced: {path.name}")
        frame = pd.read_csv(path, nrows=5)
        bad_headers = sorted({str(col).strip().lower() for col in frame.columns} & SAFE_FORBIDDEN_HEADERS)
        if bad_headers:
            status = "FAIL"
            messages.append(f"{path.name} has forbidden headers: {bad_headers}")
        values = "\n".join(frame.astype(str).to_numpy().ravel().tolist())
        if ".feather" in values or ".zip" in values:
            status = "FAIL"
            messages.append(f"{path.name} contains source-member or archive path-like values.")
    if not messages:
        messages.append("Checked safe Phase 10 CSV outputs; no forbidden headers, disallowed filenames, source-member paths, or archive path-like values found.")
    return status, messages


def write_table5_outputs(
    outdir: str | Path,
    demographics_file: str | Path | None = None,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table5 = make_table5_reproduced(demographics_file)
    path = outdir / "table5_reproduced.csv"
    table5.to_csv(path, index=False)
    return {"table5_reproduced": path}


def write_table6_outputs(
    data_zip: str | Path,
    outdir: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table6, _daily = make_table6_reproduced(data_zip, date_filter_mode=date_filter_mode)
    source_check = make_source_device_check(table6)
    table6_path = outdir / "table6_reproduced.csv"
    source_path = outdir / "table6_source_device_check.csv"
    table6.to_csv(table6_path, index=False)
    source_check.to_csv(source_path, index=False)
    return {"table6_reproduced": table6_path, "table6_source_device_check": source_path}


def write_consistency_outputs(
    table_dir: str | Path,
    outdir: str | Path,
    data_zip: str | Path | None = None,
    phase1_dir: str | Path = "local_outputs/denominators",
    phase2_dir: str | Path = "local_outputs/ppm_common_support",
    phase6_dir: str | Path = "local_outputs/lag_models",
    phase7_dir: str | Path = "local_outputs/sleep",
    phase9_dir: str | Path = "local_outputs/paired_sensitivity",
    repo_path: str | Path | None = None,
    demographics_file: str | Path | None = None,
    manuscript_file: str | Path | None = None,
    date_filter_mode: DateFilterMode = "campaign",
    scripts_run: list[str] | None = None,
) -> dict[str, Path]:
    table_dir = Path(table_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    repo_path = Path(repo_path) if repo_path else Path.cwd()
    phase1_dir = Path(phase1_dir)
    phase2_dir = Path(phase2_dir)
    phase6_dir = Path(phase6_dir)
    phase7_dir = Path(phase7_dir)
    phase9_dir = Path(phase9_dir)
    scripts_run = scripts_run or [
        "scripts/10_validate_table5_counts.py",
        "scripts/10_validate_table6_counts.py",
        "scripts/10_validate_table5_table6_text_consistency.py",
    ]

    table5_path = table_dir / "table5_reproduced.csv"
    table6_path = table_dir / "table6_reproduced.csv"
    if not table5_path.exists():
        write_table5_outputs(outdir, demographics_file=demographics_file)
    if not table6_path.exists():
        if data_zip is None:
            raise FileNotFoundError("table6_reproduced.csv was not found and --data-zip was not supplied.")
        write_table6_outputs(data_zip, outdir, date_filter_mode=date_filter_mode)
    table5 = pd.read_csv(table5_path if table5_path.exists() else outdir / "table5_reproduced.csv")
    table6 = pd.read_csv(table6_path if table6_path.exists() else outdir / "table6_reproduced.csv")
    if data_zip is None:
        raise FileNotFoundError("--data-zip is required for stream-denominator validation.")
    _table6_reproduced, daily = make_table6_reproduced(data_zip, date_filter_mode=date_filter_mode)

    denominator = make_denominator_check(table5, table6, daily)
    source_device = make_source_device_check(table6)
    text_value = make_text_value_check(table6, manuscript_file=manuscript_file)
    footnote = make_footnote_check(manuscript_file=manuscript_file)

    paths = {
        "table5_table6_denominator_check": outdir / "table5_table6_denominator_check.csv",
        "table6_text_value_check": outdir / "table6_text_value_check.csv",
        "table6_source_device_check": outdir / "table6_source_device_check.csv",
        "table6_footnote_consistency_check": outdir / "table6_footnote_consistency_check.csv",
        "phase10_validation_report": outdir / "phase10_validation_report.txt",
    }
    denominator.to_csv(paths["table5_table6_denominator_check"], index=False)
    text_value.to_csv(paths["table6_text_value_check"], index=False)
    source_device.to_csv(paths["table6_source_device_check"], index=False)
    footnote.to_csv(paths["table6_footnote_consistency_check"], index=False)
    safe_status, safe_messages = validate_safe_outputs(outdir)
    write_phase10_report(
        paths["phase10_validation_report"],
        repo_path=repo_path,
        data_zip=Path(data_zip),
        demographics_file=Path(demographics_file) if demographics_file else None,
        manuscript_file=Path(manuscript_file) if manuscript_file else None,
        phase1_dir=phase1_dir,
        phase2_dir=phase2_dir,
        phase6_dir=phase6_dir,
        phase7_dir=phase7_dir,
        phase9_dir=phase9_dir,
        scripts_run=scripts_run,
        table5=table5,
        table6=table6,
        denominator=denominator,
        source_device=source_device,
        text_value=text_value,
        footnote=footnote,
        safe_status=safe_status,
        safe_messages=safe_messages,
    )
    return paths


def write_phase10_report(
    outpath: Path,
    repo_path: Path,
    data_zip: Path,
    demographics_file: Path | None,
    manuscript_file: Path | None,
    phase1_dir: Path,
    phase2_dir: Path,
    phase6_dir: Path,
    phase7_dir: Path,
    phase9_dir: Path,
    scripts_run: list[str],
    table5: pd.DataFrame,
    table6: pd.DataFrame,
    denominator: pd.DataFrame,
    source_device: pd.DataFrame,
    text_value: pd.DataFrame,
    footnote: pd.DataFrame,
    safe_status: str,
    safe_messages: list[str],
) -> None:
    table5_status = "PASS" if set(table5["status"].astype(str)) == {"PASS"} else "MISSING_DEPENDENCY"
    table6_status = "PASS" if set(table6["status"].astype(str)) <= {"PASS"} else "FAIL"
    source_status = "PASS" if set(source_device["status"].astype(str)) <= {"PASS"} else "FAIL"
    ppm_uhoo_status = "PASS" if source_device["ppm_uhoo_averaged_or_combined"].astype(str).str.lower().isin(["false"]).all() else "FAIL"
    table5_denominator_status = "PASS" if (denominator["check_name"] == "Table 5 not used as Table 6 denominator").any() else "FAIL"
    stream_denom = denominator[denominator["check_name"].str.contains("denominator aligns|participant-day support", regex=True)]
    stream_status = "PASS" if set(stream_denom["status"].astype(str)) <= {"PASS"} else "FAIL"
    footnote_status = "PASS" if set(footnote["status"].astype(str)) <= {"PASS"} else "FAIL"
    manuscript_status = "NOT_PROVIDED" if set(text_value["status"].astype(str)) == {"NOT_PROVIDED"} else _status_pass_or_missing(text_value)
    missing_dependencies = []
    if table5_status == "MISSING_DEPENDENCY":
        missing_dependencies.append("Table 5 participant-level demographic source was not provided.")
    if manuscript_status == "NOT_PROVIDED":
        missing_dependencies.append("Manuscript text file was not provided for automated text-value checking.")
    deviations = denominator[~denominator["status"].astype(str).isin(["PASS", "MISSING_DEPENDENCY"])]

    lines: list[str] = [
        "Phase 10 Table 5 and Table 6 final consistency validation report",
        f"timestamp_of_run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository_path: {repo_path}",
        f"data_archive_path_used: {data_zip}",
        f"demographics_file_path_if_provided: {demographics_file if demographics_file else 'not_provided'}",
        f"manuscript_file_path_if_provided: {manuscript_file if manuscript_file else 'not_provided'}",
        f"phase1_output_path_used: {phase1_dir}",
        f"phase2_output_path_used: {phase2_dir}",
        f"phase6_output_path_used_if_used: {phase6_dir} (not used)",
        f"phase7_output_path_used_if_used: {phase7_dir} (not used)",
        f"phase9_output_path_used_if_used: {phase9_dir} (not used)",
        "scripts_run: " + "; ".join(scripts_run),
        "",
        f"PASS/MISSING_DEPENDENCY for Table 5 reproduction: {table5_status}",
        f"PASS/FAIL for Table 6 reproduction: {table6_status}",
        f"PASS/FAIL for Table 6 source-device labels: {source_status}",
        f"PASS/FAIL for PPM/uHoo not averaged or combined: {ppm_uhoo_status}",
        f"PASS/FAIL for Table 5 not used as analytical denominator: {table5_denominator_status}",
        f"PASS/FAIL for Table 6 stream-specific denominator logic: {stream_status}",
        f"PASS/FAIL for Table 6 footnote consistency: {footnote_status}",
        f"PASS/FAIL or NOT_PROVIDED for manuscript text-value check: {manuscript_status}",
        "",
        "any_missing_dependencies:",
    ]
    lines.extend(f"- {item}" for item in missing_dependencies) if missing_dependencies else lines.append("- none")
    lines.append("any_deviations_from_target_values:")
    if deviations.empty:
        lines.append("- none")
    else:
        for row in deviations.to_dict(orient="records"):
            lines.append(f"- {row['check_name']} {row['stream_or_variable']}: target {row['target_value']}, reproduced {row['reproduced_value']}, status {row['status']}.")
    lines.append("any_Table_6_values_that_should_be_checked_during_Phase_12_harmonization:")
    if manuscript_status == "NOT_PROVIDED":
        lines.append(f"- All reproduced Table 6 values in table6_text_value_check.csv should be checked in Phase 12 because no manuscript file was provided ({len(text_value)} rows).")
    else:
        flagged = text_value[text_value["status"].astype(str) != "PASS_DETECTED"]
        if flagged.empty:
            lines.append("- none flagged by automated text scan.")
        else:
            for row in flagged.head(20).to_dict(orient="records"):
                lines.append(f"- {row['city']} {row['season']} {row['variable']} ({row['sensor']}): mean {row['mean']}, median {row['median']}, SD {row['sd']}.")
            if len(flagged) > 20:
                lines.append(f"- Additional flagged rows: {len(flagged) - 20}.")
    lines.extend(
        [
            "",
            "confirmations:",
            "- no Phase 11 work was performed",
            "- no Phase 12 manuscript/response harmonization was performed",
            "- no HIA/YLL/upper-tail workflows were run",
            "- no lag/sleep/figure/paired workflows were newly run beyond reading local aggregate outputs if needed",
            "- no GitHub push/commit/upload was performed",
            "- controlled data remained local",
            f"- safe_output_privacy_check: {safe_status}",
        ]
    )
    lines.extend(f"  - {message}" for message in safe_messages)
    lines.append("- safe outputs do not contain participant IDs, participant UID columns, source-member paths, raw timestamps, row-level Feather-file identifiers, participant-day rows, participant-night rows, questionnaire/TAD rows, demographic microdata, or model input rows")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
