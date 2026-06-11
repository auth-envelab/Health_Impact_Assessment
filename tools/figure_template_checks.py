"""Template-fidelity checks for regenerated manuscript figures."""

from __future__ import annotations

import csv
from pathlib import Path


def placeholder_or_missing(value: str) -> bool:
    cleaned = (value or "").strip()
    return cleaned == "" or cleaned.startswith("<") or cleaned.upper() in {"MISSING", "TBD", "NOT_IDENTIFIED"}


def validate_template_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        failures: list[str] = []
        if placeholder_or_missing(record.get("template_reference", "")):
            failures.append("TEMPLATE_REFERENCE_STATUS = MISSING")
        if placeholder_or_missing(record.get("expected_output_path", "")):
            failures.append("regenerated output path is missing")
        rows.append(
            {
                "figure_id": record.get("item_id", ""),
                "figure_label": record.get("manuscript_label", ""),
                "accepted_template_source": record.get("template_reference", ""),
                "source_plotting_script": record.get("source_script", ""),
                "regenerated_output_path": record.get("expected_output_path", ""),
                "dimensions": record.get("dimensions", "to_be_checked"),
                "aspect_ratio": record.get("aspect_ratio", "to_be_checked"),
                "font_family": record.get("font_family", "Times New Roman or approved manuscript font required"),
                "font_sizes": record.get("font_sizes", "to_be_checked"),
                "legend_location": record.get("legend_location", "to_be_checked"),
                "color_palette": record.get("color_palette", "to_be_checked"),
                "panel_layout": record.get("panel_layout", "to_be_checked"),
                "axis_label_style": record.get("axis_label_style", "to_be_checked"),
                "line_bar_box_scatter_style": record.get("line_bar_box_scatter_style", "to_be_checked"),
                "annotation_style": record.get("annotation_style", "to_be_checked"),
                "p_value_significance_formatting_status": record.get("p_value_status", "to_be_checked"),
                "pm_subscript_status": record.get("pm_subscript_status", "to_be_checked"),
                "ppm_uhoo_source_label_status": record.get("source_label_status", "to_be_checked"),
                "hia_yll_significance_marker_status": record.get("hia_yll_marker_status", "to_be_checked"),
                "status": "FAIL" if failures else "PASS",
                "notes": "; ".join(failures) if failures else "template reference present",
            }
        )
    return rows


def write_template_check_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "figure_id", "figure_label", "accepted_template_source", "source_plotting_script",
        "regenerated_output_path", "dimensions", "aspect_ratio", "font_family", "font_sizes",
        "legend_location", "color_palette", "panel_layout", "axis_label_style",
        "line_bar_box_scatter_style", "annotation_style", "p_value_significance_formatting_status",
        "pm_subscript_status", "ppm_uhoo_source_label_status", "hia_yll_significance_marker_status", "status", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_template_text_report(path: Path, rows: list[dict[str, str]]) -> None:
    failed = [row for row in rows if row.get("status") != "PASS"]
    lines = [
        "Figure Template Fidelity Report", "", f"figures_checked: {len(rows)}",
        f"figures_failed: {len(failed)}", "gate_status: " + ("FAIL" if failed else "PASS"), "",
    ]
    for row in failed:
        lines.append(f"- {row.get('figure_label', '')} {row.get('figure_id', '')}: {row.get('notes', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
