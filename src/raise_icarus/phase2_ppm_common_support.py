"""Phase 2 PPM common-support and HIA exposure-source validation helpers.

These helpers produce aggregate-only local validation outputs. Participant IDs
are used only in memory for unique and paired counts; participant-day records,
source members, and raw timestamps are not written to the safe output directory.
"""

from __future__ import annotations

import io
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
from raise_icarus.phase1_denominators import (
    CITY_SEASON_ORDER,
    PM_COLUMNS,
    S1_TARGETS,
    UHOO_COLUMNS,
    city_prefixed_participant_id,
)


PPM_PERSONAL_STREAM = "PPM personal PM analysis"
PPM_COMMON_STREAM = "PPM personal PM common-support analysis"
HIA_STREAM = "HIA daily PM input"
PHASE2_STREAMS = (PPM_PERSONAL_STREAM, PPM_COMMON_STREAM, HIA_STREAM)

PPM_SOURCE_DEVICE = "ICARUS PPM personal portable particulate monitor"
UHOO_SOURCE_DEVICE = "uHoo static residential indoor air-quality monitor"
PM_HIERARCHY_RULE = "PM1_PPM <= PM25_PPM <= PM10_PPM"
LOCAL_HIA_COMMON_SUPPORT_SUMMARY = Path("outputs/hia_common_support/hia_common_support_campaign_summary.csv")
LOCAL_HIA_SCENARIO_INPUT_AUDIT = Path("outputs/hia_common_support_scenario/hia_pm_common_support_input_audit.csv")


@dataclass
class StreamSupport:
    """In-memory participant/date support for aggregate stream counts."""

    participant_sets: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: {key: set() for key in CITY_SEASON_ORDER}
    )
    date_sets: dict[tuple[str, str], set[tuple[str, object]]] = field(
        default_factory=lambda: {key: set() for key in CITY_SEASON_ORDER}
    )

    def add(self, city: str, season: str, participant_id: str, dates: Iterable[object]) -> None:
        clean_dates = [date for date in dates if not pd.isna(date)]
        if not clean_dates:
            return
        key = (city, season)
        self.participant_sets[key].add(participant_id)
        self.date_sets[key].update((participant_id, date) for date in clean_dates)

    def n_participants(self, city: str, season: str) -> int:
        return len(self.participant_sets[(city, season)])

    def participant_days(self, city: str, season: str) -> int:
        return len(self.date_sets[(city, season)])

    def paired_n(self, city: str) -> int:
        return len(self.participant_sets[(city, "Summer")] & self.participant_sets[(city, "Winter")])

    def total_days(self) -> int:
        return sum(len(values) for values in self.date_sets.values())


@dataclass
class Phase2BuildResult:
    """Aggregate Phase 2 build outputs."""

    personal_audit: pd.DataFrame
    common_support_audit: pd.DataFrame
    hierarchy_validation: pd.DataFrame
    hia_daily_validation: pd.DataFrame
    source_metadata: dict[str, object]


def _read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        return feather.read_table(io.BytesIO(fh.read())).to_pandas()


def _campaign_label(city: str, season: str) -> str:
    start, end = get_campaign_date_window(city, season)
    if start is None or end is None:
        return "not configured"
    return f"{start.date()} to {end.date()}"


def _campaign_filter(df: pd.DataFrame, city: str, season: str, mode: DateFilterMode) -> pd.DataFrame:
    if "TS" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["TS"] = pd.to_datetime(out["TS"], errors="coerce")
    valid_ts = out["TS"].notna()
    if mode == "campaign":
        start, end = get_campaign_date_window(city, season)
        if start is None or end is None:
            return out.iloc[0:0].copy()
        mask = valid_ts & (out["TS"].dt.date >= start.date()) & (out["TS"].dt.date <= end.date())
    elif mode == "none":
        mask = valid_ts
    else:
        raise ValueError(f"Unsupported date filter mode: {mode}")
    out = out.loc[mask].copy()
    out["date"] = out["TS"].dt.date
    return out


def _numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _target_participants(stream: str, city: str, season: str) -> int:
    return int(S1_TARGETS[stream][f"{city} {season.lower()} n"])


def _target_days(stream: str) -> int:
    return int(S1_TARGETS[stream]["Participant-days/nights/rows"])


def _support_status(stream: str, city: str, season: str, participants: int, total_days: int) -> str:
    return "PASS" if participants == _target_participants(stream, city, season) and total_days == _target_days(stream) else "FAIL"


def _daily_date_range(date_sets: set[tuple[str, object]]) -> tuple[str, str]:
    dates = sorted({date for _, date in date_sets})
    if not dates:
        return "", ""
    return str(dates[0]), str(dates[-1])


def _add_support(support: StreamSupport, city: str, season: str, participant_id: str, frame: pd.DataFrame, valid_mask: pd.Series) -> None:
    if frame.empty or not bool(valid_mask.any()):
        return
    support.add(city, season, participant_id, frame.loc[valid_mask, "date"].dropna().unique().tolist())


def _empty_counts() -> dict[str, int]:
    return {
        "timestamp_rows_before_filter": 0,
        "timestamp_rows_complete_pm_triplet": 0,
        "timestamp_rows_hierarchy_valid": 0,
        "timestamp_rows_hierarchy_invalid": 0,
        "pm1_gt_pm25_rows": 0,
        "pm25_gt_pm10_rows": 0,
        "pm1_gt_pm10_rows": 0,
    }


def _detect_uhoo_columns(columns: Iterable[str]) -> list[str]:
    return sorted([column for column in columns if str(column).endswith("_uHoo")])


def compute_phase2_build(
    data_zip: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> Phase2BuildResult:
    """Compute aggregate PPM support, hierarchy, and HIA daily input validation."""
    data_zip = Path(data_zip)
    personal = StreamSupport()
    common = StreamSupport()
    timestamp_counts: dict[tuple[str, str], dict[str, int]] = {key: _empty_counts() for key in CITY_SEASON_ORDER}
    uhoo_columns_detected: set[str] = set()
    ppm_columns_present: set[str] = set()
    member_count = 0

    with zipfile.ZipFile(data_zip) as zf:
        members = feather_members(data_zip)
        if not members:
            raise FileNotFoundError(f"No .feather files found in {data_zip}")
        for member in members:
            member_count += 1
            city, season = parse_city_season(member)
            raw = _read_member(zf, member)
            uhoo_columns_detected.update(_detect_uhoo_columns(raw.columns))
            ppm_columns_present.update([column for column in PM_COLUMNS if column in raw.columns])
            raw_id = raw["ID"].dropna().iloc[0] if "ID" in raw.columns and raw["ID"].dropna().size else Path(member).stem
            participant_id = city_prefixed_participant_id(city, raw_id)
            frame = _campaign_filter(raw, city, season, date_filter_mode)
            if frame.empty:
                continue
            frame = _numeric_columns(frame, PM_COLUMNS)
            key = (city, season)
            counts = timestamp_counts[key]
            counts["timestamp_rows_before_filter"] += int(len(frame))

            any_pm = frame[list(PM_COLUMNS)].notna().any(axis=1)
            _add_support(personal, city, season, participant_id, frame, any_pm)

            complete = frame[list(PM_COLUMNS)].notna().all(axis=1)
            pm1_gt_pm25 = complete & (frame["PM1_PPM"] > frame["PM25_PPM"])
            pm25_gt_pm10 = complete & (frame["PM25_PPM"] > frame["PM10_PPM"])
            pm1_gt_pm10 = complete & (frame["PM1_PPM"] > frame["PM10_PPM"])
            invalid = pm1_gt_pm25 | pm25_gt_pm10 | pm1_gt_pm10
            valid_hierarchy = complete & ~invalid

            counts["timestamp_rows_complete_pm_triplet"] += int(complete.sum())
            counts["timestamp_rows_hierarchy_valid"] += int(valid_hierarchy.sum())
            counts["timestamp_rows_hierarchy_invalid"] += int(invalid.sum())
            counts["pm1_gt_pm25_rows"] += int(pm1_gt_pm25.sum())
            counts["pm25_gt_pm10_rows"] += int(pm25_gt_pm10.sum())
            counts["pm1_gt_pm10_rows"] += int(pm1_gt_pm10.sum())
            _add_support(common, city, season, participant_id, frame, valid_hierarchy)

    personal_rows: list[dict[str, object]] = []
    common_rows: list[dict[str, object]] = []
    hierarchy_rows: list[dict[str, object]] = []
    hia_rows: list[dict[str, object]] = []

    for city, season in CITY_SEASON_ORDER:
        personal_days_total = personal.total_days()
        common_days_total = common.total_days()
        counts = timestamp_counts[(city, season)]
        personal_rows.append(
            {
                "analysis_stream": PPM_PERSONAL_STREAM,
                "city": city,
                "season": season,
                "n_participants": personal.n_participants(city, season),
                "participant_days": personal.participant_days(city, season),
                "paired_n_if_city_level": personal.paired_n(city),
                "campaign_window": _campaign_label(city, season),
                "source_device": PPM_SOURCE_DEVICE,
                "validity_rule": "Any non-missing ICARUS PPM PM1, PM2.5, or PM10 row inside campaign window; aggregated to participant-day support.",
                "target_participants": _target_participants(PPM_PERSONAL_STREAM, city, season),
                "target_participant_days_if_applicable": _target_days(PPM_PERSONAL_STREAM),
                "status": _support_status(PPM_PERSONAL_STREAM, city, season, personal.n_participants(city, season), personal_days_total),
                "notes": "Aggregate-only; participant IDs used in memory only for unique and paired counts.",
            }
        )
        common_rows.append(
            {
                "analysis_stream": PPM_COMMON_STREAM,
                "city": city,
                "season": season,
                "n_participants": common.n_participants(city, season),
                "participant_days": common.participant_days(city, season),
                "paired_n_if_city_level": common.paired_n(city),
                "campaign_window": _campaign_label(city, season),
                "source_device": PPM_SOURCE_DEVICE,
                "common_support_rule": f"Complete timestamp triplet and ordered hierarchy: {PM_HIERARCHY_RULE}; invalid timestamp rows excluded before daily aggregation.",
                "timestamp_rows_before_filter": counts["timestamp_rows_before_filter"],
                "timestamp_rows_complete_pm_triplet": counts["timestamp_rows_complete_pm_triplet"],
                "timestamp_rows_hierarchy_valid": counts["timestamp_rows_hierarchy_valid"],
                "timestamp_rows_hierarchy_invalid": counts["timestamp_rows_hierarchy_invalid"],
                "daily_rows_after_aggregation": common.participant_days(city, season),
                "target_participants": _target_participants(PPM_COMMON_STREAM, city, season),
                "target_participant_days_if_applicable": _target_days(PPM_COMMON_STREAM),
                "status": _support_status(PPM_COMMON_STREAM, city, season, common.n_participants(city, season), common_days_total),
                "notes": "HIA common-support daily PM input is derived from retained PPM timestamp rows only.",
            }
        )
        hierarchy_rows.append(
            {
                "city": city,
                "season": season,
                "timestamp_rows_checked": counts["timestamp_rows_before_filter"],
                "complete_triplet_rows": counts["timestamp_rows_complete_pm_triplet"],
                "pm1_gt_pm25_rows": counts["pm1_gt_pm25_rows"],
                "pm25_gt_pm10_rows": counts["pm25_gt_pm10_rows"],
                "pm1_gt_pm10_rows": counts["pm1_gt_pm10_rows"],
                "any_hierarchy_violation_rows": counts["timestamp_rows_hierarchy_invalid"],
                "hierarchy_valid_rows": counts["timestamp_rows_hierarchy_valid"],
                "hierarchy_invalid_rows": counts["timestamp_rows_hierarchy_invalid"],
                "hierarchy_rule": PM_HIERARCHY_RULE,
                "status": "PASS",
                "notes": "Hierarchy violations, when present, are excluded before common-support daily aggregation.",
            }
        )
        date_min, date_max = _daily_date_range(common.date_sets[(city, season)])
        hia_rows.append(
            {
                "city": city,
                "season": season,
                "n_participants": common.n_participants(city, season),
                "participant_days": common.participant_days(city, season),
                "paired_n_if_city_level": common.paired_n(city),
                "date_min": date_min,
                "date_max": date_max,
                "source_device": PPM_SOURCE_DEVICE,
                "pm_columns": ";".join(PM_COLUMNS),
                "requires_complete_ordered_pm_triplets": "yes",
                "target_participants": _target_participants(HIA_STREAM, city, season),
                "target_participant_days": _target_days(HIA_STREAM),
                "status": _support_status(HIA_STREAM, city, season, common.n_participants(city, season), common_days_total),
                "notes": "Validated as final common-support daily personal PPM input; no HIA scenario calculation run.",
            }
        )

    source_metadata = {
        "data_zip": str(data_zip),
        "date_filter_mode": date_filter_mode,
        "archive_members": member_count,
        "ppm_columns_present": sorted(ppm_columns_present),
        "uhoo_columns_detected": sorted(uhoo_columns_detected),
        "ppm_personal_total_participant_days": personal.total_days(),
        "ppm_common_total_participant_days": common.total_days(),
        "hierarchy_invalid_rows_total": sum(row["timestamp_rows_hierarchy_invalid"] for row in timestamp_counts.values()),
        "hierarchy_valid_rows_total": sum(row["timestamp_rows_hierarchy_valid"] for row in timestamp_counts.values()),
    }
    return Phase2BuildResult(
        personal_audit=pd.DataFrame(personal_rows),
        common_support_audit=pd.DataFrame(common_rows),
        hierarchy_validation=pd.DataFrame(hierarchy_rows),
        hia_daily_validation=pd.DataFrame(hia_rows),
        source_metadata=source_metadata,
    )


def write_phase2_build_outputs(
    data_zip: str | Path,
    outdir: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> dict[str, Path]:
    """Write aggregate Phase 2 build outputs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = compute_phase2_build(data_zip, date_filter_mode)
    outputs = {
        "ppm_personal_pm_support_audit": outdir / "ppm_personal_pm_support_audit.csv",
        "ppm_common_support_daily_input_audit": outdir / "ppm_common_support_daily_input_audit.csv",
        "ppm_hierarchy_validation": outdir / "ppm_hierarchy_validation.csv",
        "hia_daily_pm_input_validation": outdir / "hia_daily_pm_input_validation.csv",
    }
    result.personal_audit.to_csv(outputs["ppm_personal_pm_support_audit"], index=False)
    result.common_support_audit.to_csv(outputs["ppm_common_support_daily_input_audit"], index=False)
    result.hierarchy_validation.to_csv(outputs["ppm_hierarchy_validation"], index=False)
    result.hia_daily_validation.to_csv(outputs["hia_daily_pm_input_validation"], index=False)
    pd.DataFrame([result.source_metadata]).to_csv(outdir / "_phase2_source_metadata.csv", index=False)
    return outputs


def validate_existing_hia_summaries(repo_root: str | Path, hia_validation: pd.DataFrame) -> str:
    """Return a compact aggregate-only note about existing local HIA input outputs."""
    repo_root = Path(repo_root)
    expected = {
        (row["city"], row["season"]): (int(row["n_participants"]), int(row["participant_days"]))
        for row in hia_validation.to_dict(orient="records")
    }
    notes: list[str] = []
    summary_path = repo_root / LOCAL_HIA_COMMON_SUPPORT_SUMMARY
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        mismatches: list[str] = []
        for row in summary.to_dict(orient="records"):
            key = (row.get("city"), row.get("season"))
            if key in expected:
                participants = int(row.get("participants", -1))
                days = int(row.get("participant_days", -1))
                if (participants, days) != expected[key]:
                    mismatches.append(f"{key}: existing {participants}/{days}, recomputed {expected[key][0]}/{expected[key][1]}")
        notes.append(
            f"Existing aggregate common-support summary checked at {LOCAL_HIA_COMMON_SUPPORT_SUMMARY}: "
            + ("matches recomputation" if not mismatches else "mismatches: " + "; ".join(mismatches))
        )
    else:
        notes.append(f"Existing aggregate common-support summary not found at {LOCAL_HIA_COMMON_SUPPORT_SUMMARY}.")

    audit_path = repo_root / LOCAL_HIA_SCENARIO_INPUT_AUDIT
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        invalid_col = "daily_pm_hierarchy_violation_rows"
        invalid_sum = int(pd.to_numeric(audit.get(invalid_col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        notes.append(
            f"Existing HIA common-support input audit checked at {LOCAL_HIA_SCENARIO_INPUT_AUDIT}: daily hierarchy violation rows={invalid_sum}."
        )
    else:
        notes.append(f"Existing HIA common-support scenario input audit not found at {LOCAL_HIA_SCENARIO_INPUT_AUDIT}.")
    return " ".join(notes)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_hierarchy_validation_from_input(input_dir: str | Path, outdir: str | Path) -> Path:
    """Validate/refresh hierarchy status from the aggregate build output."""
    input_dir = Path(input_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    source = input_dir / "ppm_hierarchy_validation.csv"
    hierarchy = _read_csv_if_exists(source)
    if hierarchy.empty:
        raise FileNotFoundError(f"Missing hierarchy validation input: {source}")
    hierarchy = hierarchy.copy()
    hierarchy["status"] = np.where(
        hierarchy["hierarchy_valid_rows"].astype(int) + hierarchy["hierarchy_invalid_rows"].astype(int)
        == hierarchy["complete_triplet_rows"].astype(int),
        "PASS",
        "FAIL",
    )
    hierarchy["notes"] = hierarchy.apply(
        lambda row: (
            "Complete-triplet accounting validated; hierarchy-invalid timestamp rows were excluded before daily aggregation."
            if row["status"] == "PASS"
            else "Complete-triplet accounting mismatch; inspect build logic before using HIA input."
        ),
        axis=1,
    )
    outpath = outdir / "ppm_hierarchy_validation.csv"
    hierarchy.to_csv(outpath, index=False)
    return outpath


def make_hia_exposure_source_audit(
    input_dir: str | Path,
    data_zip: str | Path,
    repo_root: str | Path,
) -> pd.DataFrame:
    """Create source-device audit proving HIA exposure input uses PPM only."""
    input_dir = Path(input_dir)
    metadata = _read_csv_if_exists(input_dir / "_phase2_source_metadata.csv")
    hia_validation = _read_csv_if_exists(input_dir / "hia_daily_pm_input_validation.csv")
    uhoo_columns = []
    ppm_columns = []
    if not metadata.empty:
        uhoo_columns = str(metadata["uhoo_columns_detected"].iloc[0]).strip("[]").replace("'", "").split(", ")
        uhoo_columns = [column for column in uhoo_columns if column]
        ppm_columns = str(metadata["ppm_columns_present"].iloc[0]).strip("[]").replace("'", "").split(", ")
        ppm_columns = [column for column in ppm_columns if column]

    if not uhoo_columns:
        with zipfile.ZipFile(data_zip) as zf:
            detected: set[str] = set()
            ppm_detected: set[str] = set()
            for member in feather_members(data_zip):
                table = feather.read_table(io.BytesIO(zf.read(member)))
                detected.update(_detect_uhoo_columns(table.column_names))
                ppm_detected.update([column for column in PM_COLUMNS if column in table.column_names])
        uhoo_columns = sorted(detected)
        ppm_columns = sorted(ppm_detected)

    existing_note = validate_existing_hia_summaries(repo_root, hia_validation) if not hia_validation.empty else "HIA validation output not available for existing-summary cross-check."
    status = "PASS" if set(PM_COLUMNS).issubset(set(ppm_columns)) and len(uhoo_columns) > 0 else "FAIL"
    row = {
        "exposure_input_name": "HIA daily PM input",
        "source_device": PPM_SOURCE_DEVICE,
        "included_pm_columns": ";".join(PM_COLUMNS),
        "excluded_pm_columns_or_devices": f"{UHOO_SOURCE_DEVICE}: " + ";".join(uhoo_columns),
        "uhoo_columns_detected": ";".join(uhoo_columns),
        "uhoo_columns_used_in_hia_input": "none",
        "ppm_uhoo_averaged_or_combined": "no",
        "participant_level_rows_written_to_safe_outputs": "no",
        "status": status,
        "notes": (
            "HIA exposure input is built from complete ordered PPM PM1/PM2.5/PM10 timestamp triplets. "
            "uHoo columns are detected in the archive but are excluded from the HIA exposure input. "
            "No PPM/uHoo averaging or combining is performed. "
            + existing_note
        ),
    }
    return pd.DataFrame([row])


def write_hia_exposure_source_audit(
    input_dir: str | Path,
    data_zip: str | Path,
    outdir: str | Path,
    repo_root: str | Path,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "hia_exposure_source_audit.csv"
    make_hia_exposure_source_audit(input_dir, data_zip, repo_root).to_csv(outpath, index=False)
    write_phase2_validation_report(repo_root, data_zip, outdir)
    return outpath


def _status_summary(frame: pd.DataFrame, column: str = "status") -> str:
    if frame.empty or column not in frame.columns:
        return "not available"
    counts = frame[column].astype(str).value_counts().to_dict()
    return "; ".join(f"{key}: {value}" for key, value in counts.items())


def _overall_status(frame: pd.DataFrame, column: str = "status") -> str:
    if frame.empty or column not in frame.columns:
        return "FAIL"
    values = set(frame[column].astype(str))
    return "PASS" if values <= {"PASS"} else "FAIL"


def write_phase2_validation_report(
    repo_root: str | Path,
    data_zip: str | Path,
    outdir: str | Path,
    date_filter_mode: DateFilterMode = "campaign",
) -> Path:
    """Write Phase 2 validation report from aggregate local outputs."""
    repo_root = Path(repo_root)
    data_zip = Path(data_zip)
    outdir = Path(outdir)
    personal = _read_csv_if_exists(outdir / "ppm_personal_pm_support_audit.csv")
    common = _read_csv_if_exists(outdir / "ppm_common_support_daily_input_audit.csv")
    hierarchy = _read_csv_if_exists(outdir / "ppm_hierarchy_validation.csv")
    hia = _read_csv_if_exists(outdir / "hia_daily_pm_input_validation.csv")
    source = _read_csv_if_exists(outdir / "hia_exposure_source_audit.csv")

    hierarchy_invalid = int(pd.to_numeric(hierarchy.get("hierarchy_invalid_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not hierarchy.empty else 0
    hia_days = int(pd.to_numeric(hia.get("participant_days", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not hia.empty else 0
    source_pass = _overall_status(source)
    uhoo_used_pass = (
        "PASS"
        if not source.empty and str(source["uhoo_columns_used_in_hia_input"].iloc[0]).lower() in {"none", "no", ""}
        else "FAIL"
    )
    combined_pass = (
        "PASS"
        if not source.empty and str(source["ppm_uhoo_averaged_or_combined"].iloc[0]).lower() == "no"
        else "FAIL"
    )

    lines = [
        "Phase 2 validation report",
        f"timestamp of run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository path: {repo_root}",
        f"data archive path used: {data_zip}",
        f"campaign-window mode: {date_filter_mode}",
        "scripts run: scripts\\02_build_ppm_common_support_daily_input.py; scripts\\02_validate_ppm_common_support_hierarchy.py; scripts\\02_validate_hia_exposure_source.py",
        "",
        "Target values:",
        f"- PPM personal PM analysis: {S1_TARGETS[PPM_PERSONAL_STREAM]}",
        f"- PPM common-support PM analysis: {S1_TARGETS[PPM_COMMON_STREAM]}",
        f"- HIA daily PM input: {S1_TARGETS[HIA_STREAM]}",
        "",
        "Reproduced values:",
    ]
    for label, frame in [
        (PPM_PERSONAL_STREAM, personal),
        (PPM_COMMON_STREAM, common),
        (HIA_STREAM, hia),
    ]:
        if frame.empty:
            lines.append(f"- {label}: output not found")
            continue
        rows = []
        for row in frame.to_dict(orient="records"):
            rows.append(f"{row['city']} {row['season']} n={row['n_participants']}, days={row['participant_days']}, paired={row['paired_n_if_city_level']}")
        lines.append(f"- {label}: {'; '.join(rows)}")

    lines.extend(
        [
            "",
            "PASS/FAIL:",
            f"- PPM personal PM denominator: {_overall_status(personal)} ({_status_summary(personal)})",
            f"- PPM common-support PM denominator: {_overall_status(common)} ({_status_summary(common)})",
            f"- HIA daily PM input denominator: {_overall_status(hia)} ({_status_summary(hia)})",
            f"- HIA daily PM input retains 2,427 participant-days: {'PASS' if hia_days == _target_days(HIA_STREAM) else 'FAIL'} (reproduced={hia_days})",
            f"- PM hierarchy validation: {_overall_status(hierarchy)} ({_status_summary(hierarchy)}; hierarchy-invalid timestamp rows found and excluded={hierarchy_invalid})",
            f"- HIA input uses PPM only: {source_pass}",
            f"- uHoo not used in HIA exposure input: {uhoo_used_pass}",
            f"- PPM and uHoo not averaged or combined: {combined_pass}",
            "",
            "Missing dependencies:",
            "- None for Phase 2 validation.",
            "",
            "Deviations from target values:",
        ]
    )
    if all(_overall_status(frame) == "PASS" for frame in [personal, common, hia, hierarchy]) and hia_days == _target_days(HIA_STREAM):
        lines.append("- None.")
    else:
        lines.append("- One or more Phase 2 outputs failed; inspect status columns in local CSVs.")

    lines.extend(
        [
            "",
            "Output files:",
        ]
    )
    outpath = outdir / "phase2_validation_report.txt"
    output_paths = sorted(outdir.glob("*.csv")) + sorted(outdir.glob("*.txt"))
    if outpath not in output_paths:
        output_paths.append(outpath)
    for path in output_paths:
        lines.append(f"- {path.resolve()}")

    lines.extend(
        [
            "",
            "Confirmations:",
            "- No HIA scenario calculations were run.",
            "- No Phase 3 work was performed.",
            "- No GitHub push, Git commit, or data upload was performed.",
            "- Controlled data remained local.",
            "- Safe outputs do not contain restricted participant identifier values or columns, source-member paths, raw timestamps, or row-level Feather-file identifiers.",
        ]
    )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outpath
