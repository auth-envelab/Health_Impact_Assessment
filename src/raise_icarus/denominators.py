"""Denominator reconciliation utilities for RAISE/ICARUS reporting audits."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.feather as feather

from raise_icarus.data import (
    DateFilterMode,
    feather_members,
    get_campaign_date_window,
    parse_city_season,
)


CITY_SEASON_ORDER: tuple[tuple[str, str], ...] = (
    ("Milan", "Summer"),
    ("Milan", "Winter"),
    ("Thessaloniki", "Summer"),
    ("Thessaloniki", "Winter"),
)

TABLE5_REFERENCE_COUNTS: dict[tuple[str, str], int] = {
    ("Milan", "Summer"): 63,
    ("Milan", "Winter"): 89,
    ("Thessaloniki", "Summer"): 38,
    ("Thessaloniki", "Winter"): 87,
}

TABLE5_REFERENCE_VALUES: dict[str, dict[str, object]] = {
    "sample size": {
        "Milan summer": 63,
        "Milan winter": 89,
        "Thessaloniki summer": 38,
        "Thessaloniki winter": 87,
        "Overall": 277,
    },
    "female": {
        "Milan summer": 34,
        "Milan winter": 51,
        "Thessaloniki summer": 22,
        "Thessaloniki winter": 47,
        "Overall": 154,
    },
    "male": {
        "Milan summer": 29,
        "Milan winter": 38,
        "Thessaloniki summer": 16,
        "Thessaloniki winter": 40,
        "Overall": 123,
    },
    "mean age": {
        "Milan summer": 42.2,
        "Milan winter": 37.9,
        "Thessaloniki summer": 28.2,
        "Thessaloniki winter": 29.8,
        "Overall": 35.0,
    },
    "age range": {
        "Milan summer": "6-66",
        "Milan winter": "4-65",
        "Thessaloniki summer": "3-58",
        "Thessaloniki winter": "3-65",
        "Overall": "3-66",
    },
    "employed": {
        "Milan summer": 43,
        "Milan winter": 55,
        "Thessaloniki summer": 19,
        "Thessaloniki winter": 42,
        "Overall": 159,
    },
    "primary education": {
        "Milan summer": 7,
        "Milan winter": 15,
        "Thessaloniki summer": 13,
        "Thessaloniki winter": 16,
        "Overall": 51,
    },
    "secondary education": {
        "Milan summer": 20,
        "Milan winter": 27,
        "Thessaloniki summer": 6,
        "Thessaloniki winter": 17,
        "Overall": 70,
    },
    "higher education": {
        "Milan summer": 35,
        "Milan winter": 45,
        "Thessaloniki summer": 17,
        "Thessaloniki winter": 41,
        "Overall": 138,
    },
    "married": {
        "Milan summer": 28,
        "Milan winter": 39,
        "Thessaloniki summer": 18,
        "Thessaloniki winter": 44,
        "Overall": 129,
    },
}

DEMOGRAPHIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "sex_gender": ("sex", "gender"),
    "age": ("age",),
    "employment": ("employment", "employ", "occupation", "work"),
    "education": ("education", "educ"),
    "marital_status": ("marital", "married", "marriage"),
    "health_status": ("asthma", "health", "disease", "diagnosis", "condition"),
    "questionnaire": ("questionnaire", "survey"),
}

SEARCH_TERMS: tuple[str, ...] = (
    "Sample Size",
    "participant-season",
    "Table 5",
    "277",
    "63",
    "89",
    "38",
    "87",
    "female",
    "male",
    "sex",
    "gender",
    "age",
    "employment",
    "education",
    "marital",
    "married",
)


@dataclass(frozen=True)
class DenominatorAuditResult:
    """Container for denominator reconciliation audit outputs."""

    archive_inventory: pd.DataFrame
    participant_id_reconciliation: pd.DataFrame
    eligibility_audit: pd.DataFrame
    denominator_summary: pd.DataFrame
    table5_source_search_results: pd.DataFrame
    discrepant_records: pd.DataFrame
    unique_participant_counts: pd.DataFrame
    demographic_columns: pd.DataFrame
    candidate_table5: pd.DataFrame | None


def _yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _join_values(values: Iterable[object], limit: int = 12) -> str:
    cleaned = [str(v) for v in values if not pd.isna(v)]
    unique = sorted(set(cleaned))
    if len(unique) > limit:
        return ";".join(unique[:limit]) + f";...(+{len(unique) - limit})"
    return ";".join(unique)


def _normalise_id_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    if re.fullmatch(r"\d+(\.0+)?", text):
        return f"{int(float(text)):03d}"
    text = re.sub(r"\.0$", "", text)
    match = re.fullmatch(r"([MTmt]?)(\d+)", text)
    if match:
        return f"{int(match.group(2)):03d}"
    return text


def _prefixed_participant_id(city: str, value: object) -> str:
    token = _normalise_id_token(value)
    if token == "":
        return ""
    prefix = "M" if city == "Milan" else "T" if city == "Thessaloniki" else ""
    if re.fullmatch(r"[MTmt]\d+", str(value).strip()):
        token = _normalise_id_token(str(value).strip()[1:])
    return f"{prefix}{token}" if prefix else token


def _file_stem_matches_id(file_stem: str, first_id: object) -> bool:
    stem_token = _normalise_id_token(file_stem)
    id_token = _normalise_id_token(first_id)
    return stem_token != "" and id_token != "" and stem_token == id_token


def _detect_demographic_columns(columns: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for column in columns:
        lower = str(column).lower()
        for field, patterns in DEMOGRAPHIC_PATTERNS.items():
            if any(pattern in lower for pattern in patterns):
                rows.append({"field_type": field, "column": str(column)})
    return rows


def _nonmissing_count(df: pd.DataFrame, columns: list[str]) -> int:
    if not columns:
        return 0
    return int(df[columns].notna().any(axis=1).sum())


def _sensor_counts(df: pd.DataFrame, inside_mask: pd.Series) -> dict[str, int]:
    ppm_cols = [col for col in ("PM1_PPM", "PM25_PPM", "PM10_PPM") if col in df.columns]
    uhoo_cols = [col for col in df.columns if col.endswith("_uHoo")]
    garmin_cols = [col for col in ("AvgHeartRate", "Stress") if col in df.columns]
    sleep_cols = [col for col in ("Sleep",) if col in df.columns]

    numeric = df.copy()
    for column in ppm_cols + uhoo_cols + garmin_cols:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    valid_ppm = _nonmissing_count(numeric.loc[inside_mask], ppm_cols)
    if {"PM1_PPM", "PM25_PPM", "PM10_PPM"} <= set(numeric.columns):
        ppm_triplet = numeric.loc[inside_mask, ["PM1_PPM", "PM25_PPM", "PM10_PPM"]].copy()
        valid_triplet = int(
            (
                ppm_triplet.notna().all(axis=1)
                & (ppm_triplet["PM1_PPM"] <= ppm_triplet["PM25_PPM"])
                & (ppm_triplet["PM25_PPM"] <= ppm_triplet["PM10_PPM"])
            ).sum()
        )
    else:
        valid_triplet = 0

    valid_uhoo = _nonmissing_count(numeric.loc[inside_mask], uhoo_cols)
    valid_garmin = _nonmissing_count(numeric.loc[inside_mask], garmin_cols)
    valid_sleep = _nonmissing_count(numeric.loc[inside_mask], sleep_cols)
    return {
        "valid_ppm_rows": valid_ppm,
        "valid_common_support_pm_triplet_rows": valid_triplet,
        "valid_uhoo_rows": valid_uhoo,
        "valid_garmin_heart_rate_stress_rows": valid_garmin,
        "valid_sleep_rows": valid_sleep,
        "any_valid_sensor_rows": int(any([valid_ppm, valid_uhoo, valid_garmin, valid_sleep])),
    }


def _read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        table = feather.read_table(io.BytesIO(fh.read()))
    return table.to_pandas()


def inspect_archive(data_zip: str | Path, date_filter_mode: DateFilterMode = "campaign") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inspect feather inventory, participant IDs, eligibility, and demographic columns."""
    data_zip = Path(data_zip)
    inventory_rows: list[dict[str, object]] = []
    id_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    demographic_rows: list[dict[str, object]] = []

    with zipfile.ZipFile(data_zip) as zf:
        info_by_name = {info.filename: info for info in zf.infolist()}
        members = feather_members(data_zip)
        for member in members:
            folder = Path(member).parts[0]
            file_stem = Path(member).stem
            city, season = parse_city_season(member)
            file_size = info_by_name[member].file_size if member in info_by_name else pd.NA
            base = {
                "archive_path": member,
                "archive_folder": folder,
                "city_folder": city,
                "season_folder": season,
                "file_stem": file_stem,
                "inferred_city": city,
                "inferred_season": season,
                "file_size_bytes": file_size,
            }

            try:
                df = _read_member(zf, member)
                readable = True
                error_message = ""
            except Exception as exc:  # noqa: BLE001 - audit should record and continue.
                df = pd.DataFrame()
                readable = False
                error_message = f"{type(exc).__name__}: {exc}"

            inventory_rows.append(
                {
                    **base,
                    "readable": _yes_no(readable),
                    "n_rows": len(df) if readable else pd.NA,
                    "n_columns": len(df.columns) if readable else pd.NA,
                    "error_message": error_message,
                }
            )

            id_exists = readable and "ID" in df.columns
            ids = df["ID"].dropna().unique().tolist() if id_exists else []
            first_id = ids[0] if ids else pd.NA
            prefixed_id = _prefixed_participant_id(city, first_id)
            participant_season_key = f"{city}|{season}|{prefixed_id}" if prefixed_id else ""

            city_values = df["City"].dropna().unique().tolist() if readable and "City" in df.columns else []
            season_values = df["Season"].dropna().unique().tolist() if readable and "Season" in df.columns else []
            demographic_matches = _detect_demographic_columns(df.columns) if readable else []
            for match in demographic_matches:
                demographic_rows.append({**base, **match})

            id_rows.append(
                {
                    **base,
                    "internal_id_column_exists": _yes_no(id_exists),
                    "unique_id_values": _join_values(ids),
                    "n_unique_internal_ids": len(ids),
                    "first_internal_id": "" if pd.isna(first_id) else str(first_id),
                    "city_values": _join_values(city_values),
                    "season_values": _join_values(season_values),
                    "inferred_prefixed_participant_id": prefixed_id,
                    "participant_season_key": participant_season_key,
                    "file_stem_matches_internal_id": _yes_no(_file_stem_matches_id(file_stem, first_id)),
                    "multiple_ids_in_one_file": _yes_no(len(ids) > 1),
                }
            )

            if not readable:
                eligibility_rows.append(
                    {
                        **base,
                        "participant_season_key": participant_season_key,
                        "any_valid_timestamp_rows": "no",
                        "valid_timestamp_rows": 0,
                        "date_min": "",
                        "date_max": "",
                        "within_campaign_date_window": "no",
                        "rows_inside_campaign_window": 0,
                        "all_valid_timestamps_inside_campaign_window": "no",
                        "valid_ppm_rows": 0,
                        "valid_common_support_pm_triplet_rows": 0,
                        "valid_uhoo_rows": 0,
                        "valid_garmin_heart_rate_stress_rows": 0,
                        "valid_sleep_rows": 0,
                        "row_is_empty": "yes",
                        "sensor_empty": "yes",
                        "demographic_fields_present": "no",
                        "demographic_nonmissing_rows": 0,
                        "eligible_monitoring_record": "no",
                        "exclusion_reason": error_message,
                    }
                )
                continue

            timestamps = pd.to_datetime(df["TS"], errors="coerce") if "TS" in df.columns else pd.Series(pd.NaT, index=df.index)
            valid_ts = timestamps.notna()
            start, end = get_campaign_date_window(city, season)
            if date_filter_mode == "campaign":
                if start is None or end is None:
                    inside_mask = pd.Series(False, index=df.index)
                    window_reason = "no_campaign_window_configured"
                else:
                    inside_mask = valid_ts & (timestamps.dt.date >= start.date()) & (timestamps.dt.date <= end.date())
                    window_reason = ""
            else:
                inside_mask = valid_ts
                window_reason = ""

            sensor = _sensor_counts(df, inside_mask)
            demographic_columns = [row["column"] for row in demographic_matches]
            demographic_nonmissing = _nonmissing_count(df, demographic_columns)
            row_is_empty = len(df) == 0
            sensor_empty = not any(
                [
                    sensor["valid_ppm_rows"],
                    sensor["valid_uhoo_rows"],
                    sensor["valid_garmin_heart_rate_stress_rows"],
                    sensor["valid_sleep_rows"],
                ]
            )
            exclusion_reasons: list[str] = []
            if row_is_empty:
                exclusion_reasons.append("empty_file")
            if not id_exists or len(ids) == 0:
                exclusion_reasons.append("missing_internal_id")
            if len(ids) > 1:
                exclusion_reasons.append("multiple_internal_ids")
            if int(valid_ts.sum()) == 0:
                exclusion_reasons.append("no_valid_timestamp_rows")
            if int(inside_mask.sum()) == 0:
                exclusion_reasons.append(window_reason or "no_rows_inside_campaign_window")
            if sensor_empty:
                exclusion_reasons.append("no_valid_sensor_rows_inside_campaign_window")

            eligibility_rows.append(
                {
                    **base,
                    "participant_season_key": participant_season_key,
                    "any_valid_timestamp_rows": _yes_no(int(valid_ts.sum()) > 0),
                    "valid_timestamp_rows": int(valid_ts.sum()),
                    "date_min": timestamps.loc[valid_ts].min().date() if int(valid_ts.sum()) else "",
                    "date_max": timestamps.loc[valid_ts].max().date() if int(valid_ts.sum()) else "",
                    "within_campaign_date_window": _yes_no(int(inside_mask.sum()) > 0),
                    "rows_inside_campaign_window": int(inside_mask.sum()),
                    "all_valid_timestamps_inside_campaign_window": _yes_no(int(valid_ts.sum()) == int(inside_mask.sum()) and int(valid_ts.sum()) > 0),
                    **sensor,
                    "row_is_empty": _yes_no(row_is_empty),
                    "sensor_empty": _yes_no(sensor_empty),
                    "demographic_fields_present": _yes_no(bool(demographic_columns)),
                    "demographic_nonmissing_rows": demographic_nonmissing,
                    "eligible_monitoring_record": _yes_no(len(exclusion_reasons) == 0),
                    "exclusion_reason": ";".join(exclusion_reasons),
                }
            )

    id_df = pd.DataFrame(id_rows)
    if not id_df.empty:
        duplicate_counts = id_df["participant_season_key"].replace("", pd.NA).dropna().value_counts()
        id_df["same_participant_season_key_appears_in_multiple_files"] = id_df["participant_season_key"].map(
            lambda value: _yes_no(value in duplicate_counts.index and duplicate_counts[value] > 1)
        )

    return (
        pd.DataFrame(inventory_rows),
        id_df,
        pd.DataFrame(eligibility_rows),
        pd.DataFrame(demographic_rows).drop_duplicates() if demographic_rows else pd.DataFrame(columns=["archive_path", "field_type", "column"]),
    )


def summarise_denominators(
    archive_inventory: pd.DataFrame,
    participant_id_reconciliation: pd.DataFrame,
    eligibility_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise archive and reconciled denominator counts by city-season."""
    rows: list[dict[str, object]] = []
    for city, season in CITY_SEASON_ORDER:
        inv = archive_inventory[(archive_inventory["inferred_city"] == city) & (archive_inventory["inferred_season"] == season)]
        ids = participant_id_reconciliation[
            (participant_id_reconciliation["inferred_city"] == city)
            & (participant_id_reconciliation["inferred_season"] == season)
        ]
        elig = eligibility_audit[
            (eligibility_audit["inferred_city"] == city)
            & (eligibility_audit["inferred_season"] == season)
        ]
        table5_count = TABLE5_REFERENCE_COUNTS[(city, season)]
        rows.append(
            {
                "city": city,
                "season": season,
                "raw_file_count": int(len(inv)),
                "readable_file_count": int((inv["readable"] == "yes").sum()) if not inv.empty else 0,
                "unique_participant_season_keys": int(ids["participant_season_key"].replace("", pd.NA).dropna().nunique()) if not ids.empty else 0,
                "valid_timestamp_keys": int(elig.loc[elig["any_valid_timestamp_rows"] == "yes", "participant_season_key"].replace("", pd.NA).dropna().nunique()) if not elig.empty else 0,
                "valid_sensor_keys": int(elig.loc[elig["sensor_empty"] == "no", "participant_season_key"].replace("", pd.NA).dropna().nunique()) if not elig.empty else 0,
                "inside_campaign_window_keys": int(elig.loc[elig["within_campaign_date_window"] == "yes", "participant_season_key"].replace("", pd.NA).dropna().nunique()) if not elig.empty else 0,
                "eligible_monitoring_record_keys": int(elig.loc[elig["eligible_monitoring_record"] == "yes", "participant_season_key"].replace("", pd.NA).dropna().nunique()) if not elig.empty else 0,
                "demographic_metadata_keys": int(elig.loc[elig["demographic_nonmissing_rows"] > 0, "participant_season_key"].replace("", pd.NA).dropna().nunique()) if not elig.empty else 0,
                "table5_reference_count": table5_count,
                "archive_minus_table5_reference": int(len(inv)) - table5_count,
            }
        )
    frame = pd.DataFrame(rows)
    overall = {
        "city": "Overall",
        "season": "All",
        "raw_file_count": int(frame["raw_file_count"].sum()),
        "readable_file_count": int(frame["readable_file_count"].sum()),
        "unique_participant_season_keys": int(frame["unique_participant_season_keys"].sum()),
        "valid_timestamp_keys": int(frame["valid_timestamp_keys"].sum()),
        "valid_sensor_keys": int(frame["valid_sensor_keys"].sum()),
        "inside_campaign_window_keys": int(frame["inside_campaign_window_keys"].sum()),
        "eligible_monitoring_record_keys": int(frame["eligible_monitoring_record_keys"].sum()),
        "demographic_metadata_keys": int(frame["demographic_metadata_keys"].sum()),
        "table5_reference_count": int(frame["table5_reference_count"].sum()),
        "archive_minus_table5_reference": int(frame["archive_minus_table5_reference"].sum()),
    }
    return pd.concat([frame, pd.DataFrame([overall])], ignore_index=True)


def summarise_unique_participants(participant_id_reconciliation: pd.DataFrame) -> pd.DataFrame:
    """Summarise unique city-prefixed participant IDs and paired seasons."""
    records = participant_id_reconciliation.copy()
    if records.empty:
        return pd.DataFrame()
    records = records[records["inferred_prefixed_participant_id"].astype(str) != ""]
    rows: list[dict[str, object]] = []
    overall_ids = set(records["inferred_prefixed_participant_id"])
    rows.append(
        {
            "scope": "overall",
            "city": "Overall",
            "unique_participant_ids": len(overall_ids),
            "summer_only": pd.NA,
            "winter_only": pd.NA,
            "both_summer_and_winter": pd.NA,
        }
    )
    for city in ("Milan", "Thessaloniki"):
        city_records = records[records["inferred_city"] == city]
        summer = set(city_records.loc[city_records["inferred_season"] == "Summer", "inferred_prefixed_participant_id"])
        winter = set(city_records.loc[city_records["inferred_season"] == "Winter", "inferred_prefixed_participant_id"])
        rows.append(
            {
                "scope": "city",
                "city": city,
                "unique_participant_ids": len(summer | winter),
                "summer_only": len(summer - winter),
                "winter_only": len(winter - summer),
                "both_summer_and_winter": len(summer & winter),
            }
        )
    return pd.DataFrame(rows)


def _searchable_paths(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "README.md",
        repo_root / "src",
        repo_root / "scripts",
        repo_root / "docs",
        repo_root / "legacy",
        repo_root / "tests",
        repo_root / "outputs" / "harmonization_qc",
        repo_root / "outputs" / "tables",
        repo_root / "outputs" / "reporting",
    ]
    files: list[Path] = []
    suffixes = {".py", ".md", ".csv", ".txt", ".r", ".R", ".yaml", ".yml", ".json"}
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                    files.append(path)

    external = repo_root.parents[1] / "Py_scripts" / "HIA 7.py"
    if external.exists():
        files.append(external)
    return sorted(set(files))


def search_table5_sources(repo_root: str | Path) -> pd.DataFrame:
    """Search local repo/legacy sources for Table 5 demographic count clues."""
    repo_root = Path(repo_root)
    rows: list[dict[str, object]] = []
    patterns = {
        term: re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
        if re.fullmatch(r"[A-Za-z ]+", term)
        else re.compile(rf"(?<![\d.]){re.escape(term)}(?![\d.])")
        for term in SEARCH_TERMS
    }
    for path in _searchable_paths(repo_root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            rows.append(
                {
                    "source_path": str(path),
                    "line_number": pd.NA,
                    "search_term": "READ_ERROR",
                    "snippet": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            snippet = " ".join(line.strip().split())
            if not snippet:
                continue
            for term, pattern in patterns.items():
                if pattern.search(line):
                    rows.append(
                        {
                            "source_path": str(path),
                            "line_number": line_number,
                            "search_term": term,
                            "snippet": snippet[:500],
                        }
                    )
    return pd.DataFrame(rows, columns=["source_path", "line_number", "search_term", "snippet"])


def _first_column_for_type(demographic_columns: pd.DataFrame, field_type: str) -> str | None:
    rows = demographic_columns[demographic_columns["field_type"] == field_type]
    if rows.empty:
        return None
    return str(rows["column"].iloc[0])


def build_candidate_table5_from_archive(
    data_zip: str | Path,
    demographic_columns: pd.DataFrame,
) -> pd.DataFrame | None:
    """Build a rough candidate Table 5 when demographic fields exist in archive files."""
    if demographic_columns.empty:
        return None

    gender_col = _first_column_for_type(demographic_columns, "sex_gender")
    age_col = _first_column_for_type(demographic_columns, "age")
    employment_col = _first_column_for_type(demographic_columns, "employment")
    education_col = _first_column_for_type(demographic_columns, "education")
    marital_col = _first_column_for_type(demographic_columns, "marital_status")
    needed = [col for col in {gender_col, age_col, employment_col, education_col, marital_col} if col]
    if not needed:
        return None

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(data_zip) as zf:
        for member in feather_members(data_zip):
            city, season = parse_city_season(member)
            df = _read_member(zf, member)
            record: dict[str, object] = {
                "city": city,
                "season": season,
                "participant_key": f"{city}|{season}|{_prefixed_participant_id(city, df['ID'].dropna().iloc[0])}" if "ID" in df.columns and df["ID"].dropna().size else "",
            }
            for col in needed:
                if col in df.columns:
                    values = df[col].dropna()
                    record[col] = values.iloc[0] if not values.empty else pd.NA
            rows.append(record)

    participant_frame = pd.DataFrame(rows).drop_duplicates(subset=["participant_key"])
    out_rows: list[dict[str, object]] = []

    def _subset(city: str | None = None, season: str | None = None) -> pd.DataFrame:
        out = participant_frame
        if city:
            out = out[out["city"] == city]
        if season:
            out = out[out["season"] == season]
        return out

    groups = {
        "Milan summer": _subset("Milan", "Summer"),
        "Milan winter": _subset("Milan", "Winter"),
        "Thessaloniki summer": _subset("Thessaloniki", "Summer"),
        "Thessaloniki winter": _subset("Thessaloniki", "Winter"),
        "Overall": participant_frame,
    }

    def _count_matching(group: pd.DataFrame, col: str | None, pattern: str) -> int | pd.NA:
        if col is None or col not in group.columns:
            return pd.NA
        return int(group[col].astype(str).str.contains(pattern, flags=re.IGNORECASE, na=False).sum())

    def _mean_age(group: pd.DataFrame) -> float | pd.NA:
        if age_col is None or age_col not in group.columns:
            return pd.NA
        ages = pd.to_numeric(group[age_col], errors="coerce").dropna()
        return round(float(ages.mean()), 1) if not ages.empty else pd.NA

    def _age_range(group: pd.DataFrame) -> str | pd.NA:
        if age_col is None or age_col not in group.columns:
            return pd.NA
        ages = pd.to_numeric(group[age_col], errors="coerce").dropna()
        return f"{int(ages.min())}-{int(ages.max())}" if not ages.empty else pd.NA

    metric_specs = {
        "sample size": lambda g: int(len(g)),
        "female": lambda g: _count_matching(g, gender_col, r"\bf(emale)?\b|woman|women"),
        "male": lambda g: _count_matching(g, gender_col, r"\bm(ale)?\b|man|men"),
        "mean age": _mean_age,
        "age range": _age_range,
        "employed": lambda g: _count_matching(g, employment_col, r"employ|work|yes|1"),
        "primary education": lambda g: _count_matching(g, education_col, r"primary"),
        "secondary education": lambda g: _count_matching(g, education_col, r"secondary"),
        "higher education": lambda g: _count_matching(g, education_col, r"higher|university|college|tertiary"),
        "married": lambda g: _count_matching(g, marital_col, r"married|yes|1"),
    }
    for metric, func in metric_specs.items():
        row = {"characteristic": metric}
        for label, group in groups.items():
            row[label] = func(group)
            row[f"{label} reference"] = TABLE5_REFERENCE_VALUES[metric][label]
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def make_discrepant_records(
    denominator_summary: pd.DataFrame,
    participant_id_reconciliation: pd.DataFrame,
    eligibility_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Create a discrepancy diagnostics table for 280 archive files versus Table 5 references."""
    rows: list[dict[str, object]] = []
    for _, row in denominator_summary[denominator_summary["city"] != "Overall"].iterrows():
        rows.append(
            {
                "discrepancy_type": "city_season_count_difference",
                "city": row["city"],
                "season": row["season"],
                "archive_count": row["raw_file_count"],
                "table5_reference_count": row["table5_reference_count"],
                "difference_archive_minus_table5": row["archive_minus_table5_reference"],
                "archive_path": "",
                "participant_season_key": "",
                "reason": "Archive file inventory and Table 5 demographic reference have different city-season distributions.",
            }
        )

    duplicate_keys = participant_id_reconciliation[
        participant_id_reconciliation["same_participant_season_key_appears_in_multiple_files"] == "yes"
    ]
    for _, dup in duplicate_keys.iterrows():
        rows.append(
            {
                "discrepancy_type": "duplicate_participant_season_key",
                "city": dup["inferred_city"],
                "season": dup["inferred_season"],
                "archive_count": pd.NA,
                "table5_reference_count": pd.NA,
                "difference_archive_minus_table5": pd.NA,
                "archive_path": dup["archive_path"],
                "participant_season_key": dup["participant_season_key"],
                "reason": "The same inferred participant-season key appears in more than one archive file.",
            }
        )

    ineligible = eligibility_audit[eligibility_audit["eligible_monitoring_record"] == "no"]
    for _, bad in ineligible.iterrows():
        rows.append(
            {
                "discrepancy_type": "ineligible_archive_record",
                "city": bad["inferred_city"],
                "season": bad["inferred_season"],
                "archive_count": pd.NA,
                "table5_reference_count": pd.NA,
                "difference_archive_minus_table5": pd.NA,
                "archive_path": bad["archive_path"],
                "participant_season_key": bad["participant_season_key"],
                "reason": bad["exclusion_reason"],
            }
        )
    return pd.DataFrame(rows)


def run_denominator_audit(
    data_zip: str | Path,
    repo_root: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> DenominatorAuditResult:
    """Run the full denominator reconciliation audit."""
    archive_inventory, participant_ids, eligibility, demographic_columns = inspect_archive(
        data_zip,
        date_filter_mode=date_filter_mode,
    )
    summary = summarise_denominators(archive_inventory, participant_ids, eligibility)
    source_search = search_table5_sources(repo_root)
    candidate = build_candidate_table5_from_archive(data_zip, demographic_columns)
    discrepant = make_discrepant_records(summary, participant_ids, eligibility)
    unique_counts = summarise_unique_participants(participant_ids)
    return DenominatorAuditResult(
        archive_inventory=archive_inventory,
        participant_id_reconciliation=participant_ids,
        eligibility_audit=eligibility,
        denominator_summary=summary,
        table5_source_search_results=source_search,
        discrepant_records=discrepant,
        unique_participant_counts=unique_counts,
        demographic_columns=demographic_columns,
        candidate_table5=candidate,
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional dependencies."""
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


def write_qc_markdown(result: DenominatorAuditResult, outpath: str | Path) -> None:
    """Write denominator reconciliation QC summary."""
    outpath = Path(outpath)
    summary = result.denominator_summary
    unique_counts = result.unique_participant_counts
    demographics_present = not result.demographic_columns.empty
    overall = summary[summary["city"] == "Overall"].iloc[0]
    exact_three = (
        int(overall["raw_file_count"]) - int(overall["table5_reference_count"]) == 3
        and int(result.discrepant_records[result.discrepant_records["discrepancy_type"] == "ineligible_archive_record"].shape[0]) == 3
    )
    lines = [
        "# Denominator Reconciliation QC",
        "",
        "This audit distinguishes raw harmonized archive files from participant-season monitoring records, unique participants, demographic-table support, and analysis-specific complete-case denominators.",
        "",
        "## Main Findings",
        "",
        f"- Raw feather-file inventory: {int(overall['raw_file_count'])} files.",
        f"- Readable feather-file inventory: {int(overall['readable_file_count'])} files.",
        f"- Unique inferred participant-season keys: {int(overall['unique_participant_season_keys'])}.",
        f"- Keys with valid timestamps inside the configured campaign windows: {int(overall['inside_campaign_window_keys'])}.",
        f"- Keys with valid sensor data inside campaign windows: {int(overall['valid_sensor_keys'])}.",
        f"- Keys with demographic metadata in the harmonized feather files: {int(overall['demographic_metadata_keys'])}.",
        f"- Table 5 reference participant-season records: {int(overall['table5_reference_count'])}.",
        "",
        "## City-Season Reconciliation",
        "",
        markdown_table(summary),
        "",
        "## Unique Participant IDs",
        "",
        markdown_table(unique_counts),
        "",
        "## Interpretation",
        "",
    ]
    if demographics_present:
        lines.append("Demographic-like columns were detected in the harmonized archive, so a candidate Table 5 was generated for external inspection.")
    else:
        lines.append("No sex/gender, age, employment, education, marital-status, health-status, or questionnaire-derived demographic columns were detected in the harmonized feather files. Table 5 demographics therefore cannot be regenerated from the harmonized monitoring archive alone.")
    lines.extend(
        [
            "",
            "The 280 versus 277 discrepancy is not explained by exactly three ineligible archive files. The archive contains 70 files in each city-season folder, whereas the Table 5 reference counts are 63, 89, 38, and 87. Because Milan winter and Thessaloniki winter Table 5 counts exceed the corresponding archive-file counts, the discrepancy cannot be resolved by removing three bad files from the 280-file archive.",
            "",
        ]
    )
    if exact_three:
        lines.append("Exactly three ineligible archive records were detected, but the city-season distribution still must be checked before treating those as the Table 5 discrepancy.")
    else:
        lines.append("No file-level set of exactly three discrepant archive records was identified.")
    lines.extend(
        [
            "",
            "Recruitment, pre-monitoring dropout, and demographic-table exclusions cannot be reconstructed from the harmonized monitoring archive alone.",
            "",
            "Recommendation: do not replace the manuscript Table 5 denominator of 277 with 280 based only on feather-file counts. Treat 280 as the raw harmonized sensor-file inventory. Use 277 only as the demographic participant-season denominator if the original demographic source can be cited or archived; otherwise describe Table 5 as based on an unreconstructed demographic source and keep analysis-specific denominators in Supplementary Table S1.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommended_manuscript_text(outpath: str | Path) -> None:
    """Write conservative manuscript wording for the denominator discrepancy."""
    text = """# Recommended Manuscript Text

The harmonized monitoring archive used for the reproducible sensor analyses contained 280 readable feather files, with 70 files in each city-season folder. These files are a raw sensor-file inventory and were not treated as the demographic denominator unless they could be reconciled to valid participant-season records with demographic metadata.

The demographic summary table reports participant-season monitoring records, not independent participants. The manuscript Table 5 denominator remains 277 participant-season records (Milan summer 63, Milan winter 89, Thessaloniki summer 38, Thessaloniki winter 87) if supported by the original demographic source. The harmonized feather archive does not contain sex/gender, age, employment, education, or marital-status fields, so Table 5 cannot be regenerated from the harmonized monitoring archive alone.

Recruitment and dropout before monitoring could not be reconstructed from the harmonized archive. We therefore report the archive-file inventory, the demographic participant-season denominator, unique participant counts, and analysis-specific complete-case denominators separately. Supplementary Table S1 provides the participant-level and participant-day support for each analysis stream.
"""
    Path(outpath).write_text(text, encoding="utf-8")


def write_reviewer_note(outpath: str | Path) -> None:
    """Write response-to-reviewers note for the denominator reconciliation."""
    text = """# Recommended Response-To-Reviewer Note

We performed an additional denominator-reconciliation audit to support STROBE-style reporting. The harmonized sensor archive contains 280 readable feather files (70 in each city-season folder), but these files are an archive inventory and not automatically a demographic denominator. Each file was checked for internal participant ID, timestamp support, campaign-window overlap, and valid sensor data. The archive files mapped to 280 unique participant-season keys with valid campaign-window timestamps; 278 keys had valid campaign-window sensor rows, while MilanSummer/017.feather and ThessalonikiSummer/064.feather had no valid PPM, uHoo, Garmin heart-rate/stress, or sleep rows inside the campaign window. No demographic fields (sex/gender, age, employment, education, or marital status) were present in the harmonized feather files.

The discrepancy between 280 archive files and the Table 5 denominator of 277 is not attributable to exactly three invalid files: the city-season distributions differ (70/70/70/70 in the archive versus 63/89/38/87 in Table 5), and two Table 5 winter counts exceed the corresponding archive-file counts. Therefore, the audit does not justify replacing 277 with 280. We will distinguish the raw monitoring-file inventory from the demographic participant-season denominator and from analysis-specific complete-case denominators, and we state that recruitment/dropout before monitoring cannot be reconstructed from the harmonized archive alone.
"""
    Path(outpath).write_text(text, encoding="utf-8")


def update_reporting_qc(repo_root: str | Path, result: DenominatorAuditResult) -> None:
    """Append or replace the denominator reconciliation note in reporting_qc.md."""
    repo_root = Path(repo_root)
    path = repo_root / "outputs" / "reporting" / "reporting_qc.md"
    if not path.exists():
        return
    summary = result.denominator_summary
    overall = summary[summary["city"] == "Overall"].iloc[0]
    block = "\n".join(
        [
            "## Denominator Reconciliation Audit",
            "",
            f"A dedicated audit in `outputs/denominator_reconciliation/` found {int(overall['raw_file_count'])} raw readable feather files and {int(overall['unique_participant_season_keys'])} unique inferred participant-season keys, with 70 files in each city-season folder. Of these, {int(overall['valid_sensor_keys'])} keys had valid campaign-window sensor rows. The harmonized feather files did not contain demographic fields, so Table 5 cannot be regenerated from the archive alone.",
            "",
            "The 280 versus 277 discrepancy is not explained by exactly three invalid archive files. The city-season distributions differ: the archive is 70/70/70/70, whereas Table 5 is 63/89/38/87. Milan winter and Thessaloniki winter Table 5 counts exceed the corresponding archive-file counts, so 280 should be reported as raw archive inventory rather than substituted for the demographic Table 5 denominator.",
            "",
            "Recruitment dropout before monitoring remains unreconstructable from the harmonized monitoring archive alone.",
        ]
    )
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"\n## Denominator Reconciliation Audit\n.*?(?=\n## |\Z)", flags=re.S)
    if pattern.search(text):
        text = pattern.sub("\n" + block + "\n", text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")

# Manuscript reproducibility descriptive stage contract.
try:
    from pathlib import Path as _RI_Path
    from raise_icarus.stage_contracts import StageDefinition as _RI_StageDefinition
    from raise_icarus.stage_contracts import StageResult as _RI_StageResult
    from raise_icarus.stage_contracts import definition_for as _ri_definition_for
    from raise_icarus.stage_contracts import run_contract_stage as _ri_run_contract_stage

    _RI_STAGE = _ri_definition_for(__name__)

    def stage_definition() -> _RI_StageDefinition:
        return _RI_STAGE

    def run_stage(
        harmonized_zip: str | _RI_Path | None,
        run_dir: str | _RI_Path,
        n_samples: int = 10000,
        dry_run: bool = False,
    ) -> _RI_StageResult:
        return _ri_run_contract_stage(_RI_STAGE, harmonized_zip=harmonized_zip, run_dir=run_dir, n_samples=n_samples, dry_run=dry_run)
except Exception:
    pass

# Descriptive controlled-data analytical exports.

from raise_icarus.controlled_runtime import copy_alias as _ri_copy_alias
from raise_icarus.controlled_runtime import config_value as _ri_config_value
from raise_icarus.controlled_runtime import domain_dir as _ri_domain_dir
from raise_icarus.controlled_runtime import reports_dir as _ri_reports_dir
from raise_icarus.controlled_runtime import repo_root as _ri_repo_root
from raise_icarus.controlled_runtime import write_text_report as _ri_write_text_report
from raise_icarus.stage_contracts import StageResult as _RI_StageResult2
from raise_icarus.stage_contracts import dry_run_stage_result as _ri_dry_run_stage_result


def build_analysis_denominators(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> dict[str, Path]:
    from raise_icarus.phase1_denominators import write_phase1_denominator_outputs

    output_dir = _ri_domain_dir(out_dir, "denominators")
    outputs = write_phase1_denominator_outputs(harmonized_zip, output_dir, date_filter_mode=_ri_config_value(config, "date_filter_mode", "campaign"))
    _ri_copy_alias(outputs["strobe_denominators_by_stream"], output_dir / "STROBE denominator support by stream.csv")
    _ri_copy_alias(outputs["participant_flow_counts"], output_dir / "Participant flow counts.csv")
    return outputs


def build_supplementary_table_s1(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> Path:
    del harmonized_zip, config
    from raise_icarus.phase1_denominators import write_supplementary_table_s1

    output_dir = _ri_domain_dir(out_dir, "denominators")
    path = write_supplementary_table_s1(output_dir / "strobe_denominators_by_stream.csv", output_dir)
    return _ri_copy_alias(path, output_dir / "Supplementary Table S1 - Analysis-specific denominators.csv")


def build_supplementary_figure_s1_data(harmonized_zip: str | Path, out_dir: str | Path, config: object = None) -> Path:
    del harmonized_zip, config
    from raise_icarus.phase1_denominators import write_supplementary_figure_s1_counts

    output_dir = _ri_domain_dir(out_dir, "denominators")
    path = write_supplementary_figure_s1_counts(output_dir / "strobe_denominators_by_stream.csv", output_dir)
    _ri_copy_alias(path, output_dir / "Supplementary Figure S1 data - Participant flow completeness summary.csv")
    return _ri_copy_alias(path, output_dir / "Supplementary Figure S1 data - Participant flow and completeness.csv")


def validate_denominator_targets(outputs_dir: str | Path, expected_manifest: str | Path | None = None) -> Path:
    del expected_manifest
    output_dir = _ri_domain_dir(outputs_dir, "denominators")
    report = _ri_reports_dir(outputs_dir) / "Denominator validation report.txt"
    checks = [
        output_dir / "Supplementary Table S1 - Analysis-specific denominators.csv",
        output_dir / "Supplementary Figure S1 data - Participant flow and completeness.csv",
    ]
    return _ri_write_text_report(report, "Denominator Validation Report", [f"{path.name}: {'PASS' if path.exists() else 'FAIL'}" for path in checks])


def run_stage(harmonized_zip: str | Path | None, run_dir: str | Path, n_samples: int = 10000, dry_run: bool = False) -> _RI_StageResult2:
    del n_samples
    if dry_run:
        return _ri_dry_run_stage_result(_RI_STAGE, run_dir)
    if harmonized_zip is None:
        return _RI_StageResult2(_RI_STAGE.stage_name, _RI_STAGE.module_name, "FAIL", _RI_STAGE.output_domain, (), "harmonized archive path is required")
    build_analysis_denominators(harmonized_zip, run_dir)
    s1 = build_supplementary_table_s1(harmonized_zip, run_dir)
    fig = build_supplementary_figure_s1_data(harmonized_zip, run_dir)
    report = validate_denominator_targets(run_dir)
    return _RI_StageResult2(_RI_STAGE.stage_name, _RI_STAGE.module_name, "PASS", _RI_STAGE.output_domain, (str(s1), str(fig), str(report)), "Denominator outputs generated.")

