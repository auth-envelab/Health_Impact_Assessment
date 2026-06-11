"""Phase 4 upper-tail HIA sensitivity helpers.

The Phase 4 workflow caps daily personal PPM PM2.5/PM10 values in memory,
reruns the same primary HIA scenarios as Phase 3, and writes aggregate-only
local outputs. It does not write participant-day inputs, iteration-level
samples, YLL outputs, figures, manuscript files, or response-letter files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from raise_icarus.data import DateFilterMode
from raise_icarus.hia import HIAConfig, calculate_hia_scenarios
from raise_icarus.phase2_ppm_common_support import HIA_STREAM, PPM_SOURCE_DEVICE, S1_TARGETS
from raise_icarus.phase3_hia_primary import (
    EXPOSURE_METRIC,
    _augment_summary,
    _common_support_daily_pm,
)


REQUIRED_SAFE_OUTPUTS = [
    "hia_upper_tail_cap_audit.csv",
    "hia_upper_tail_sensitivity_summary.csv",
    "hia_upper_tail_percent_change.csv",
    "hia_upper_tail_largest_changes.csv",
    "phase4_validation_report.txt",
]
FORBIDDEN_SAFE_FILENAMES = {
    "hia_daily_personal_pm_input.csv",
    "hia_daily_personal_pm_input_capped.csv",
    "hia_scenario_samples.csv",
}
METRICS = ["cases_mean", "cases_median", "pm_mean_of_samples", "exposure_contrast_mean"]
SUMMARY_METRICS = [
    "cases_mean",
    "cases_median",
    "cases_p025",
    "cases_p975",
    "pm_mean_of_samples",
    "exposure_contrast_mean",
    "share_samples_above_counterfactual",
]


def cap_daily_pm_by_upper_tail(
    daily_pm: pd.DataFrame,
    cap_quantile: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap PM2.5 and PM10 by city-season-pollutant quantile in memory."""
    if not 0 < cap_quantile <= 1:
        raise ValueError("cap_quantile must be in the interval (0, 1].")

    capped = daily_pm.copy()
    rows: list[dict[str, object]] = []
    pollutant_columns = [("PM2.5", "pm25"), ("PM10", "pm10")]

    for (city, season), idx in capped.groupby(["city", "season"]).groups.items():
        idx = list(idx)
        for pollutant, column in pollutant_columns:
            before = daily_pm.loc[idx, column].astype(float)
            cap_value = float(before.quantile(cap_quantile))
            after = before.clip(upper=cap_value)
            capped.loc[idx, column] = after
            above_cap = before > cap_value
            mean_before = float(before.mean())
            mean_after = float(after.mean())
            percent_mean_change = np.nan
            if mean_before != 0:
                percent_mean_change = 100.0 * (mean_after - mean_before) / mean_before
            status = "PASS" if len(before) > 0 and float(after.max()) <= cap_value + 1e-9 else "FAIL"
            rows.append(
                {
                    "city": city,
                    "season": season,
                    "pollutant": pollutant,
                    "cap_quantile": cap_quantile,
                    "cap_value_ug_m3": cap_value,
                    "participant_days": int(before.notna().sum()),
                    "values_above_cap": int(above_cap.sum()),
                    "share_values_above_cap": float(above_cap.mean()),
                    "mean_before_cap": mean_before,
                    "mean_after_cap": mean_after,
                    "median_before_cap": float(before.median()),
                    "median_after_cap": float(after.median()),
                    "max_before_cap": float(before.max()),
                    "max_after_cap": float(after.max()),
                    "absolute_mean_change": mean_after - mean_before,
                    "percent_mean_change": percent_mean_change,
                    "source_device": PPM_SOURCE_DEVICE,
                    "exposure_metric": EXPOSURE_METRIC,
                    "status": status,
                    "notes": "Aggregate cap audit; participant-day records were kept in memory only.",
                }
            )

    cap_audit = pd.DataFrame(rows).sort_values(["city", "season", "pollutant"]).reset_index(drop=True)
    return capped, cap_audit


def _merge_primary_capped(primary: pd.DataFrame, capped: pd.DataFrame, cap_quantile: float) -> pd.DataFrame:
    base_cols = [
        "scenario_key",
        "city",
        "season",
        "pollutant",
        "endpoint",
        "framework",
        "output_scale",
        "counterfactual",
        "crf_mapping",
        "source_pollutant",
    ]
    keep_primary = base_cols + SUMMARY_METRICS
    keep_capped = ["scenario_key"] + SUMMARY_METRICS
    merged = primary[keep_primary].merge(
        capped[keep_capped],
        on="scenario_key",
        how="left",
        suffixes=("_primary", "_upper_tail_capped"),
        validate="one_to_one",
    )
    merged["upper_tail_cap_quantile"] = cap_quantile
    metric_cols = [f"{metric}_upper_tail_capped" for metric in SUMMARY_METRICS]
    merged["status"] = np.where(merged[metric_cols].notna().all(axis=1), "PASS", "FAIL")
    merged["notes"] = np.where(
        merged["status"].eq("PASS"),
        "Primary and capped HIA rows merged by scenario_key; PM2.5 and PM10 remain separate.",
        "Missing capped metric after scenario_key merge.",
    )
    return merged


def make_percent_change(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary[
        [
            "scenario_key",
            "city",
            "season",
            "pollutant",
            "endpoint",
            "framework",
            "output_scale",
        ]
    ].copy()
    for metric in METRICS:
        primary = summary[f"{metric}_primary"].astype(float)
        capped = summary[f"{metric}_upper_tail_capped"].astype(float)
        absolute = capped - primary
        percent = np.where(
            primary != 0,
            100.0 * absolute / primary,
            np.where(absolute == 0, 0.0, np.nan),
        )
        out[f"{metric}_absolute_change"] = absolute
        out[f"{metric}_percent_change"] = percent

    out["sensitivity_rank_by_abs_cases_mean_change"] = (
        out["cases_mean_absolute_change"].abs().rank(method="min", ascending=False).astype(int)
    )
    percent_rank_source = out["cases_mean_percent_change"].abs().replace([np.inf, -np.inf], np.nan).fillna(-1)
    out["sensitivity_rank_by_percent_cases_mean_change"] = percent_rank_source.rank(
        method="min", ascending=False
    ).astype(int)
    out["thessaloniki_summer_pm10_flag"] = np.where(
        (out["city"] == "Thessaloniki") & (out["season"] == "Summer") & (out["pollutant"] == "PM10"),
        "yes",
        "no",
    )
    change_cols = [column for column in out.columns if column.endswith("_absolute_change") or column.endswith("_percent_change")]
    finite_changes = np.isfinite(out[change_cols].astype(float).to_numpy()).all()
    out["status"] = "PASS" if finite_changes else "FAIL"
    out["notes"] = np.where(
        out["thessaloniki_summer_pm10_flag"].eq("yes"),
        "Explicit Thessaloniki summer PM10 sensitivity row.",
        "Upper-tail percent-change comparison row.",
    )
    return out.sort_values("sensitivity_rank_by_abs_cases_mean_change").reset_index(drop=True)


def make_largest_changes(percent_change: pd.DataFrame, cap_quantile: float, top_n: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    specs = [
        ("cases_mean_absolute_change", "absolute"),
        ("cases_mean_percent_change", "percent"),
    ]
    for metric_column, mode in specs:
        ranked = percent_change.copy()
        ranked["_rank_value"] = ranked[metric_column].abs()
        ranked = ranked.sort_values("_rank_value", ascending=False).head(top_n).reset_index(drop=True)
        for rank, row in enumerate(ranked.to_dict(orient="records"), start=1):
            metric = "cases_mean"
            primary = float(row[f"{metric}_absolute_change"])
            percent = float(row[f"{metric}_percent_change"])
            primary_value = np.nan
            capped_value = np.nan
            # Reconstruct values from absolute and percent changes only when possible.
            if np.isfinite(percent) and percent != 0:
                primary_value = primary / (percent / 100.0)
                capped_value = primary_value + primary
            flag = "thessaloniki_summer_pm10" if row["thessaloniki_summer_pm10_flag"] == "yes" else f"largest_{mode}_cases_mean_change"
            rows.append(
                {
                    "rank": rank,
                    "scenario_key": row["scenario_key"],
                    "city": row["city"],
                    "season": row["season"],
                    "pollutant": row["pollutant"],
                    "endpoint": row["endpoint"],
                    "framework": row["framework"],
                    "metric": metric_column,
                    "primary_value": primary_value,
                    "capped_value": capped_value,
                    "absolute_change": primary,
                    "percent_change": percent,
                    "upper_tail_cap_quantile": cap_quantile,
                    "interpretation_flag": flag,
                    "notes": "Ranked by absolute magnitude of the listed cases_mean change metric.",
                }
            )

    return pd.DataFrame(rows)


def _group_sensitivity(percent_change: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        percent_change.assign(
            abs_cases_mean_change=percent_change["cases_mean_absolute_change"].astype(float).abs(),
            abs_cases_mean_percent_change=percent_change["cases_mean_percent_change"].astype(float).abs(),
        )
        .groupby(["city", "season", "pollutant"], as_index=False)
        .agg(
            max_abs_cases_mean_change=("abs_cases_mean_change", "max"),
            max_abs_cases_mean_percent_change=("abs_cases_mean_percent_change", "max"),
            median_abs_cases_mean_percent_change=("abs_cases_mean_percent_change", "median"),
        )
    )
    grouped["rank_by_abs_cases_mean_change"] = grouped["max_abs_cases_mean_change"].rank(
        method="min", ascending=False
    ).astype(int)
    grouped["rank_by_percent_cases_mean_change"] = grouped["max_abs_cases_mean_percent_change"].rank(
        method="min", ascending=False
    ).astype(int)
    return grouped.sort_values("rank_by_percent_cases_mean_change").reset_index(drop=True)


def thessaloniki_summer_pm10_result(percent_change: pd.DataFrame) -> tuple[str, str]:
    grouped = _group_sensitivity(percent_change)
    row = grouped[
        (grouped["city"] == "Thessaloniki")
        & (grouped["season"] == "Summer")
        & (grouped["pollutant"] == "PM10")
    ]
    if row.empty:
        return "not supported", "Thessaloniki summer PM10 group was not present."
    item = row.iloc[0]
    percent_rank = int(item["rank_by_percent_cases_mean_change"])
    abs_rank = int(item["rank_by_abs_cases_mean_change"])
    detail = (
        "Thessaloniki summer PM10 max absolute cases_mean change="
        f"{item['max_abs_cases_mean_change']:.6g}; max absolute percent change="
        f"{item['max_abs_cases_mean_percent_change']:.3f}%; ranks abs={abs_rank}, percent={percent_rank}."
    )
    if percent_rank == 1 or abs_rank == 1:
        return "supported", detail
    if percent_rank <= 3 or abs_rank <= 3:
        return "partially supported", detail
    return "not supported", detail


def validate_cap_audit(cap_audit: pd.DataFrame, expected_quantile: float = 0.95) -> pd.DataFrame:
    out = cap_audit.copy()
    out["status"] = np.where(
        np.isclose(out["cap_quantile"].astype(float), expected_quantile)
        & (out["participant_days"].astype(int) > 0)
        & (out["max_after_cap"].astype(float) <= out["cap_value_ug_m3"].astype(float) + 1e-9),
        "PASS",
        "FAIL",
    )
    out["notes"] = np.where(
        out["status"].eq("PASS"),
        "Cap quantile and capped maximum validated.",
        "Cap audit failed; inspect quantile, participant-days, or capped maximum.",
    )
    return out


def _source_device_status(summary: pd.DataFrame) -> str:
    if "source_device" not in summary.columns:
        return "FAIL"
    return "PASS" if summary["source_device"].astype(str).eq(PPM_SOURCE_DEVICE).all() else "FAIL"


def _uhoo_status(phase2_dir: str | Path) -> str:
    source_path = Path(phase2_dir) / "hia_exposure_source_audit.csv"
    if not source_path.exists():
        return "FAIL"
    source = pd.read_csv(source_path)
    if source.empty or "uhoo_columns_used_in_hia_input" not in source.columns:
        return "FAIL"
    return "PASS" if source["uhoo_columns_used_in_hia_input"].astype(str).str.lower().eq("none").all() else "FAIL"


def _safe_output_status(outdir: str | Path) -> tuple[str, str]:
    outdir = Path(outdir)
    if not outdir.exists():
        return "FAIL", "Output directory does not exist."
    names = {path.name for path in outdir.iterdir() if path.is_file()}
    forbidden = sorted(names & FORBIDDEN_SAFE_FILENAMES)
    csv_headers = {}
    for path in outdir.glob("*.csv"):
        with path.open("r", encoding="utf-8-sig") as fh:
            csv_headers[path.name] = fh.readline().strip().lower().split(",")
    forbidden_tokens = [
        "participant_uid",
        "participant_id",
        "source_member",
        "raw_timestamp",
        "timestamp",
        "feather",
        "iteration",
        "yll",
    ]
    bad_columns = []
    for name, columns in csv_headers.items():
        for column in columns:
            if column in {"participant_days", "participants"}:
                continue
            if any(token == column or token in column for token in forbidden_tokens):
                bad_columns.append(f"{name}:{column}")
    if forbidden or bad_columns:
        return "FAIL", f"Forbidden files={forbidden}; forbidden columns={bad_columns}"
    return "PASS", "Safe outputs are aggregate-only and contain no forbidden filenames or row-level/sample columns."


def write_phase4_validation_report(
    repo_root: str | Path,
    data_zip: str | Path,
    phase2_dir: str | Path,
    phase3_dir: str | Path,
    outdir: str | Path,
    cap_quantile: float,
    date_filter_mode: DateFilterMode = "campaign",
) -> Path:
    outdir = Path(outdir)
    cap_audit = pd.read_csv(outdir / "hia_upper_tail_cap_audit.csv") if (outdir / "hia_upper_tail_cap_audit.csv").exists() else pd.DataFrame()
    summary = pd.read_csv(outdir / "hia_upper_tail_sensitivity_summary.csv") if (outdir / "hia_upper_tail_sensitivity_summary.csv").exists() else pd.DataFrame()
    percent = pd.read_csv(outdir / "hia_upper_tail_percent_change.csv") if (outdir / "hia_upper_tail_percent_change.csv").exists() else pd.DataFrame()
    primary_path = Path(phase3_dir) / "hia_primary_scenario_summary.csv"
    primary = pd.read_csv(primary_path) if primary_path.exists() else pd.DataFrame()

    cap_status = "PASS" if len(cap_audit) == 8 and cap_audit.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").all() else "FAIL"
    capped_40_status = "PASS" if len(summary) == 40 else "FAIL"
    primary_40_status = "PASS" if len(primary) == 40 else "FAIL"
    merge_status = "PASS" if len(summary) == 40 and summary.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").all() else "FAIL"
    percent_status = "PASS" if not percent.empty and percent.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").all() else "FAIL"
    pm_sum_cols = [column for column in summary.columns if "total_pm" in column.lower() or "summed_pm" in column.lower()]
    pm_separation_status = "PASS" if not pm_sum_cols and set(summary.get("pollutant", [])) == {"PM2.5", "PM10"} else "FAIL"
    source_status = _source_device_status(primary)
    uhoo_status = _uhoo_status(phase2_dir)
    safe_status, safe_note = _safe_output_status(outdir)
    claim_status, claim_detail = thessaloniki_summer_pm10_result(percent) if not percent.empty else ("not supported", "Percent-change output is missing.")

    largest_abs = percent.sort_values(
        by="cases_mean_absolute_change",
        key=lambda s: s.astype(float).abs(),
        ascending=False,
    ).head(5) if not percent.empty else pd.DataFrame()
    largest_pct = percent.sort_values(
        by="cases_mean_percent_change",
        key=lambda s: s.astype(float).abs(),
        ascending=False,
    ).head(5) if not percent.empty else pd.DataFrame()
    hia_days = int(cap_audit["participant_days"].astype(int).max()) if not cap_audit.empty else 0
    total_days = int(cap_audit[cap_audit["pollutant"] == "PM2.5"]["participant_days"].sum()) if not cap_audit.empty else 0

    lines = [
        "Phase 4 upper-tail HIA sensitivity validation report",
        f"timestamp of run: {datetime.now().isoformat(timespec='seconds')}",
        f"repository path: {repo_root}",
        f"data archive path used: {data_zip}",
        f"Phase 2 output path used: {Path(phase2_dir).resolve()}",
        f"Phase 3 output path used: {Path(phase3_dir).resolve()}",
        f"campaign-window mode: {date_filter_mode}",
        "scripts run: scripts\\04_run_hia_upper_tail_sensitivity.py; scripts\\04_validate_upper_tail_caps.py; scripts\\04_compare_primary_vs_capped_hia.py",
        "",
        "Target values inherited from Phase 2 and Phase 3:",
        "- Phase 2 HIA daily PM input participant-days: 2,427",
        "- Phase 3 primary HIA scenario rows: 40",
        "- Phase 3 primary HIA has no zero-counterfactual rows and uses WHO 2021 counterfactuals",
        "- Phase 3 primary HIA separates PM2.5 and PM10 and uses ICARUS PPM only",
        "",
        f"Cap quantile used: {cap_quantile}",
        f"Cap rows reproduced: {len(cap_audit)}",
        f"Participant-days represented by cap audit: {total_days}",
        "",
        "PASS/FAIL:",
        f"- city-season-pollutant cap calculation: {cap_status}",
        f"- 40-row capped HIA output: {capped_40_status}",
        f"- 40-row primary HIA input: {primary_40_status}",
        f"- primary-versus-capped merge by scenario_key: {merge_status}",
        f"- percent-change calculation: {percent_status}",
        f"- PM2.5/PM10 separation and no summing: {pm_separation_status}",
        f"- HIA source device = ICARUS PPM only: {source_status}",
        f"- uHoo not used in HIA exposure input: {uhoo_status}",
        f"- no participant-level safe outputs: {safe_status} ({safe_note})",
        "",
        "Largest changes by cases_mean absolute change:",
    ]
    for row in largest_abs.to_dict(orient="records"):
        lines.append(
            f"- {row['scenario_key']}: abs={float(row['cases_mean_absolute_change']):.6g}, "
            f"pct={float(row['cases_mean_percent_change']):.3f}%"
        )
    lines.append("")
    lines.append("Largest changes by cases_mean percent change:")
    for row in largest_pct.to_dict(orient="records"):
        lines.append(
            f"- {row['scenario_key']}: pct={float(row['cases_mean_percent_change']):.3f}%, "
            f"abs={float(row['cases_mean_absolute_change']):.6g}"
        )
    lines.extend(
        [
            "",
            f"Thessaloniki summer PM10 sensitivity: {claim_detail}",
            f"Manuscript/response claim support: {claim_status}.",
            "",
            "Missing dependencies:",
            "- None for Phase 4 validation.",
            "",
            "Deviations from target values:",
            "- None." if cap_status == "PASS" and capped_40_status == "PASS" and merge_status == "PASS" and total_days == 2427 else "- One or more validation checks failed; inspect local CSVs.",
            "",
            "Output files:",
        ]
    )
    for name in REQUIRED_SAFE_OUTPUTS:
        lines.append(f"- {(outdir / name).resolve()}")
    lines.extend(
        [
            "",
            "Confirmations:",
            "- YLL was not calculated.",
            "- No Phase 5 work was performed.",
            "- No GitHub push, Git commit, or data upload was performed.",
            "- Controlled data remained local.",
            "- Safe outputs contain aggregate sensitivity and validation tables only; participant IDs, participant UID columns, source-member paths, raw timestamps, row-level Feather identifiers, participant-day rows, and Monte Carlo iteration-level samples are absent.",
        ]
    )
    outpath = outdir / "phase4_validation_report.txt"
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outpath


def run_upper_tail_sensitivity(
    data_zip: str | Path,
    phase2_dir: str | Path,
    phase3_dir: str | Path,
    outdir: str | Path,
    n_samples: int = 10_000,
    cap_quantile: float = 0.95,
    seed: int = 20260430,
    date_filter_mode: DateFilterMode = "campaign",
) -> dict[str, Path]:
    """Run Phase 4 upper-tail sensitivity and write aggregate local outputs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    daily_pm = _common_support_daily_pm(data_zip, date_filter_mode)
    capped_daily, cap_audit = cap_daily_pm_by_upper_tail(daily_pm, cap_quantile)
    cap_audit = validate_cap_audit(cap_audit, cap_quantile)

    config = HIAConfig(n_samples=n_samples, seed=seed, upper_tail_sensitivity_quantile=cap_quantile)
    _, capped_summary, _, _ = calculate_hia_scenarios(capped_daily, config=config, keep_samples=False)
    capped_augmented = _augment_summary(capped_summary, config)

    primary_path = Path(phase3_dir) / "hia_primary_scenario_summary.csv"
    if not primary_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 primary HIA summary: {primary_path}")
    primary = pd.read_csv(primary_path)
    sensitivity = _merge_primary_capped(primary, capped_augmented, cap_quantile)
    percent = make_percent_change(sensitivity)
    largest = make_largest_changes(percent, cap_quantile)

    outputs = {
        "hia_upper_tail_cap_audit": outdir / "hia_upper_tail_cap_audit.csv",
        "hia_upper_tail_sensitivity_summary": outdir / "hia_upper_tail_sensitivity_summary.csv",
        "hia_upper_tail_percent_change": outdir / "hia_upper_tail_percent_change.csv",
        "hia_upper_tail_largest_changes": outdir / "hia_upper_tail_largest_changes.csv",
        "phase4_validation_report": outdir / "phase4_validation_report.txt",
    }
    cap_audit.to_csv(outputs["hia_upper_tail_cap_audit"], index=False)
    sensitivity.to_csv(outputs["hia_upper_tail_sensitivity_summary"], index=False)
    percent.to_csv(outputs["hia_upper_tail_percent_change"], index=False)
    largest.to_csv(outputs["hia_upper_tail_largest_changes"], index=False)
    write_phase4_validation_report(
        repo_root=Path.cwd(),
        data_zip=data_zip,
        phase2_dir=phase2_dir,
        phase3_dir=phase3_dir,
        outdir=outdir,
        cap_quantile=cap_quantile,
        date_filter_mode=date_filter_mode,
    )
    return outputs


def validate_cap_audit_file(cap_audit: str | Path, outdir: str | Path, cap_quantile: float = 0.95) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(cap_audit)
    out = validate_cap_audit(frame, cap_quantile)
    outpath = outdir / "hia_upper_tail_cap_audit.csv"
    out.to_csv(outpath, index=False)
    return outpath


def compare_primary_vs_capped_files(primary: str | Path, capped: str | Path, outdir: str | Path) -> dict[str, Path]:
    """Regenerate percent-change and largest-change tables from sensitivity summary."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    primary_frame = pd.read_csv(primary)
    capped_frame = pd.read_csv(capped)
    if len(primary_frame) != 40:
        raise ValueError(f"Expected 40 primary rows, observed {len(primary_frame)}")
    if len(capped_frame) != 40:
        raise ValueError(f"Expected 40 capped/merged rows, observed {len(capped_frame)}")
    missing = set(primary_frame["scenario_key"]) - set(capped_frame["scenario_key"])
    if missing:
        raise ValueError(f"Capped summary missing scenario_key values: {sorted(missing)}")
    cap_quantile = float(capped_frame["upper_tail_cap_quantile"].iloc[0])
    percent = make_percent_change(capped_frame)
    largest = make_largest_changes(percent, cap_quantile)
    outputs = {
        "hia_upper_tail_percent_change": outdir / "hia_upper_tail_percent_change.csv",
        "hia_upper_tail_largest_changes": outdir / "hia_upper_tail_largest_changes.csv",
    }
    percent.to_csv(outputs["hia_upper_tail_percent_change"], index=False)
    largest.to_csv(outputs["hia_upper_tail_largest_changes"], index=False)
    return outputs
