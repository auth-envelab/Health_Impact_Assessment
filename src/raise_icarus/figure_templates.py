"""Manifest and template-reference helpers for manuscript figures."""

from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_TEMPLATE_MANIFEST = Path("local_outputs/manuscript_reproducibility/figure_template_reference_manifest.csv")
MANUSCRIPT_ITEM_MANIFEST = Path("local_outputs/manuscript_reproducibility/manuscript_item_manifest_detected.yaml")
FIGURE_PROVENANCE_DIR = Path("local_outputs/manuscript_reproducibility/figure_provenance_inventory")
FIGURE_TEMPLATE_CANDIDATE_MAP = FIGURE_PROVENANCE_DIR / "figure_template_candidate_map.csv"
REJECTED_FIGURE_CANDIDATES = FIGURE_PROVENANCE_DIR / "rejected_figure_candidates.csv"

EXPECTED_FIGURE_IDS = [
    "seasonal_personal_pm_distributions",
    "activity_stratified_personal_pm_exposure",
    "residential_uhoo_iaq_correlation_matrix",
    "uhoo_iaq_garmin_sleep_correlation_matrix",
    "hia_attributable_cases",
    "yll_scenario_outputs",
    "participant_flow_completeness_summary",
    "diurnal_pm_hr_stress",
    "seasonal_garmin_sleep_distributions",
    "ppm_garmin_correlation_matrix",
    "descriptive_ols_pm_hr_lag",
    "ols_vs_lag_mixed_comparison",
    "iaq_sleep_regression_panels",
    "stress_sleep_exploratory_panels",
]

SEED_ITEMS = [
    ("concentration_response_functions", "Table", "Concentration-response functions", "table", "main"),
    ("hia_equations_summary", "Table", "Summary of equations", "table", "main"),
    ("baseline_disease_rates", "Table", "Baseline disease rates", "table", "main"),
    ("yll_life_table_inputs", "Table", "Age-specific mortality rates and remaining life expectancy", "table", "main"),
    ("demographic_characteristics", "Table", "Demographic and socioeconomic characteristics", "table", "main"),
    ("seasonal_personal_pm_distributions", "Figure", "Seasonal distributions of personal PM", "figure", "main"),
    ("activity_stratified_personal_pm_exposure", "Figure", "Personal PM by Garmin-derived activity", "figure", "main"),
    ("residential_uhoo_iaq_correlation_matrix", "Figure", "Residential uHoo IAQ correlation", "figure", "main"),
    ("uhoo_iaq_garmin_sleep_correlation_matrix", "Figure", "uHoo IAQ Garmin sleep correlation", "figure", "main"),
    ("hia_attributable_cases", "Figure", "HIA attributable cases", "figure", "main"),
    ("yll_scenario_outputs", "Figure", "YLL scenario outputs", "figure", "main"),
    ("analysis_specific_denominators", "Supplementary Table", "Analysis-specific denominators", "table", "supplementary"),
    ("strobe_checklist", "Supplementary Table", "STROBE checklist", "table", "supplementary"),
    ("lag_specific_hr_stress_models", "Supplementary Table", "Lag-specific heart-rate and stress models", "table", "supplementary"),
    ("participant_flow_completeness_summary", "Supplementary Figure", "Participant flow and completeness", "figure", "supplementary"),
    ("diurnal_pm_hr_stress", "Supplementary Figure", "Diurnal PM heart-rate stress distributions", "figure", "supplementary"),
    ("seasonal_garmin_sleep_distributions", "Supplementary Figure", "Seasonal Garmin sleep distributions", "figure", "supplementary"),
    ("ppm_garmin_correlation_matrix", "Supplementary Figure", "PPM Garmin correlation matrix", "figure", "supplementary"),
    ("descriptive_ols_pm_hr_lag", "Supplementary Figure", "Descriptive OLS PM heart-rate lag scatterplots", "figure", "supplementary"),
    ("ols_vs_lag_mixed_comparison", "Supplementary Figure", "OLS vs lag-specific mixed-effects comparison", "figure", "supplementary"),
    ("iaq_sleep_regression_panels", "Supplementary Figure", "IAQ sleep regression panels", "figure", "supplementary"),
    ("stress_sleep_exploratory_panels", "Supplementary Figure", "Stress sleep exploratory panels", "figure", "supplementary"),
]


def parse_simple_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("- "):
            if current:
                rows.append(current)
            current = {}
            raw = raw[2:]
        if current is not None and ":" in raw:
            key, value = raw.strip().split(":", 1)
            current[key] = clean_yaml_value(value)
    if current:
        rows.append(current)
    return rows


def clean_yaml_value(value: str) -> str:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('\\"', '"')
    return value


def load_template_reference_manifest(path: str | Path | None = None) -> list[dict[str, str]]:
    resolved = Path(path) if path else REPO_ROOT / FIGURE_TEMPLATE_MANIFEST
    if not resolved.exists() and path is None:
        resolved = REPO_ROOT / "configs" / "figure_template_reference_manifest.csv"
    if not resolved.exists():
        return []
    with resolved.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def load_figure_template_candidate_map(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load private provenance rows that bind figure ids to accepted templates."""
    resolved = Path(path) if path else REPO_ROOT / FIGURE_TEMPLATE_CANDIDATE_MAP
    if not resolved.exists() and path is None:
        resolved = REPO_ROOT / "configs" / "figure_template_candidate_map.csv"
    if not resolved.exists():
        return []
    with resolved.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def load_rejected_figure_candidates(path: str | Path | None = None) -> list[dict[str, str]]:
    resolved = Path(path) if path else REPO_ROOT / REJECTED_FIGURE_CANDIDATES
    if not resolved.exists() and path is None:
        resolved = REPO_ROOT / "configs" / "rejected_figure_candidates.csv"
    if not resolved.exists():
        return []
    with resolved.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def _normalise_path_text(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve()).lower()
    except Exception:
        return text.replace("/", "\\").lower()


def _candidate_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("figure_id", ""): row for row in rows if row.get("figure_id")}


def reject_superseded_figure_candidates(
    figure_id: str,
    candidate_file: str | Path | None,
    rejected_candidates: list[dict[str, str]] | None = None,
) -> None:
    """Raise if a figure attempts to use an explicitly rejected/superseded file."""
    if candidate_file is None or str(candidate_file).strip() == "":
        return
    rejected = rejected_candidates if rejected_candidates is not None else load_rejected_figure_candidates()
    candidate = _normalise_path_text(candidate_file)
    for row in rejected:
        rejected_path = _normalise_path_text(row.get("figure_file", ""))
        if candidate and rejected_path and candidate == rejected_path:
            label = row.get("likely_label", "") or row.get("likely_manuscript_or_supplement_label", "")
            raise ValueError(f"BLOCKED_REJECTED_CANDIDATE_USED: {figure_id or label}")


def resolve_accepted_template_reference(
    figure_id: str,
    candidate_map: list[dict[str, str]] | None = None,
    template_rows: list[dict[str, str]] | None = None,
    require_exists: bool = True,
) -> dict[str, str]:
    """Resolve one authoritative accepted template reference for a figure."""
    candidate_rows = candidate_map if candidate_map is not None else load_figure_template_candidate_map()
    row = _candidate_by_id(candidate_rows).get(figure_id, {}).copy()
    template_map = template_reference_by_id(template_rows if template_rows is not None else load_template_reference_manifest())
    template_row = template_map.get(figure_id, {})
    accepted = row.get("accepted_template_candidate_file") or template_row.get("template_source_path_or_placeholder", "")
    if not accepted:
        raise FileNotFoundError(f"BLOCKED_TEMPLATE_REFERENCE_MISSING: {figure_id}")
    reject_superseded_figure_candidates(figure_id, accepted)
    accepted_path = Path(accepted)
    if require_exists and not accepted_path.exists():
        raise FileNotFoundError(f"BLOCKED_TEMPLATE_REFERENCE_MISSING: {figure_id}")
    row.setdefault("figure_id", figure_id)
    row["accepted_template_reference"] = str(accepted_path)
    row["accepted_template_source"] = str(accepted_path)
    row["accepted_template_source_type"] = template_row.get("template_source_type", "accepted_current_manuscript_sm_image")
    row["current_label"] = row.get("current_or_final_label") or template_row.get("current_label", "")
    row["main_or_supplement"] = template_row.get("main_or_supplement", "")
    row["image_dimensions"] = template_row.get("image_dimensions", "")
    row["aspect_ratio"] = template_row.get("aspect_ratio", "")
    row["font_family_detected_if_possible"] = template_row.get("font_family_detected_if_possible", "")
    row["legend_position_detected_if_possible"] = template_row.get("legend_position_detected_if_possible", "")
    row["panel_layout_detected_if_possible"] = template_row.get("panel_layout_detected_if_possible", "")
    return row


def template_reference_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("figure_id", ""): row for row in rows if row.get("figure_id")}


def manifest_output_path(row: dict[str, str]) -> str:
    if row.get("expected_output_path"):
        return row["expected_output_path"]
    if row.get("item_type") == "figure":
        label = row.get("current_label") or row.get("manuscript_label") or "Figure"
        item_id = re.sub(r"[^A-Za-z0-9]+", "_", row.get("item_id", "").strip()).strip("_")
        filename = f"{label} - {item_id}.png" if item_id else f"{label}.png"
        root = "figures"
        subdir = "supplementary" if row.get("main_or_supplement") == "supplementary" else "main"
        return f"{root}/{subdir}/{filename}"
    filename = row.get("expected_public_filename") or row.get("public_filename") or f"{row.get('current_label', row.get('manuscript_label', 'Figure TBD'))}.png"
    root = "figures" if row.get("item_type") == "figure" else "tables"
    subdir = "supplementary" if row.get("main_or_supplement") == "supplementary" else "main"
    return f"{root}/{subdir}/{filename}"


def manifest_label(row: dict[str, str]) -> str:
    return row.get("current_label") or row.get("manuscript_label") or ""


def manifest_source(row: dict[str, str]) -> str:
    return row.get("source_aggregate_data") or row.get("source_data") or ""


def accepted_template_source(row: dict[str, str], template_row: dict[str, str] | None = None) -> str:
    for key in ("accepted_template_source", "template_reference"):
        if row.get(key) and row[key] != "not_applicable":
            return row[key]
    return template_row.get("template_source_path_or_placeholder", "") if template_row else ""


def build_template_check_records(manifest_rows: list[dict[str, str]], template_rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    template_map = template_reference_by_id(template_rows or [])
    candidate_map = load_figure_template_candidate_map()
    records: list[dict[str, str]] = []
    for row in manifest_rows:
        if row.get("item_type") != "figure":
            continue
        template_row = template_map.get(row.get("item_id", ""), {})
        try:
            accepted = resolve_accepted_template_reference(row.get("item_id", ""), candidate_map, template_rows or [])
        except Exception:
            accepted = {}
        records.append({
            "item_id": row.get("item_id", ""),
            "manuscript_label": manifest_label(row),
            "template_reference": accepted.get("accepted_template_reference") or accepted_template_source(row, template_row),
            "accepted_template_source_type": accepted.get("accepted_template_source_type", template_row.get("template_source_type", "")),
            "source_script": accepted.get("source_script_candidate") or row.get("source_script", "scripts/generate_manuscript_figures.py"),
            "expected_output_path": manifest_output_path(row),
            "dimensions": template_row.get("image_dimensions", "to_be_checked"),
            "aspect_ratio": template_row.get("aspect_ratio", "to_be_checked"),
            "font_family": template_row.get("font_family_detected_if_possible", "Times New Roman or approved manuscript font required"),
            "legend_location": template_row.get("legend_position_detected_if_possible", "to_be_checked"),
            "panel_layout": template_row.get("panel_layout_detected_if_possible", "to_be_checked"),
            "source_data": accepted.get("source_data_candidate") or manifest_source(row),
            "rejected_candidate_files": accepted.get("rejected_candidate_files", ""),
            "template_lock_status": accepted.get("template_lock_status", ""),
            "generation_status": "TEMPLATE_LOCK_INPUT_READY" if accepted.get("accepted_template_reference") else "BLOCKED_TEMPLATE_REFERENCE_MISSING",
        })
    return records


def template_dimensions(record: dict[str, str]) -> tuple[float, float]:
    raw = record.get("dimensions") or record.get("image_dimensions") or ""
    match = re.match(r"^(\d+)x(\d+)$", raw.strip())
    if match:
        width = max(int(match.group(1)), 1)
        height = max(int(match.group(2)), 1)
        scale = 900.0 / width
        return max(width * scale / 100, 5.0), max(height * scale / 100, 3.0)
    try:
        aspect = float(record.get("aspect_ratio", "1.6"))
    except Exception:
        aspect = 1.6
    return 9.0, max(9.0 / max(aspect, 0.4), 3.0)


def accepted_template_export_options(template_metadata: dict[str, str] | None = None) -> dict[str, object]:
    """Return conservative PNG export settings for accepted-template renderers."""
    metadata = dict(template_metadata or {})
    try:
        dpi = int(float(metadata.get("export_dpi", 300) or 300))
    except (TypeError, ValueError):
        dpi = 300
    return {
        "dpi": dpi,
        "bbox_inches": "tight",
        "pad_inches": 0.02,
        "facecolor": "white",
        "edgecolor": "none",
    }


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        data = zf.read("word/document.xml")
    root = ET.fromstring(data)
    ns = {"w": "http" + "://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return "\n".join(node.text or "" for node in root.findall(".//w:t", ns))


def identify_current_docs(project_root: Path) -> tuple[Path | None, list[Path], str]:
    roots = [project_root / "Revision", project_root / "Manuscript", REPO_ROOT]
    docs: list[Path] = []
    for root in roots:
        if root.exists():
            docs.extend(p for p in root.rglob("*.docx") if not p.name.startswith("~$"))
    docs = sorted(docs, key=lambda p: p.stat().st_mtime, reverse=True)
    main = [p for p in docs if "Manuscript Tracked Changes4" in p.name] or [p for p in docs if "Manuscript Tracked Changes" in p.name]
    supp = [p for p in docs if re.match(r"SM\d+_", p.name)]
    if main and supp:
        return main[0], supp, "DETECTED_WITH_LOCAL_REVISION_FILES"
    return None, [], "BLOCKED_CURRENT_DOCS_NOT_IDENTIFIED"


def label_from_text(text: str, kind: str, title: str) -> str:
    escaped = re.escape(title)
    if kind == "Figure":
        pattern = rf"Figure\s+(\d+)\b[^\n]{{0,260}}{escaped}"
    elif kind == "Table":
        pattern = rf"Table\s+(\d+)\b[^\n]{{0,260}}{escaped}"
    elif kind == "Supplementary Figure":
        pattern = rf"Supplementary\s+Figure\s+(S\d+)\b[^\n]{{0,260}}{escaped}"
    else:
        pattern = rf"Supplementary\s+Table\s+(S\d+)\b[^\n]{{0,260}}{escaped}"
    found = re.findall(pattern, text, flags=re.IGNORECASE)
    return f"{kind} {found[0]}" if found else f"{kind} TBD"


def quote(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["item_id", "current_label", "final_label_status", "item_type", "main_or_supplement", "current_document_source", "accepted_template_source_status", "accepted_template_source", "expected_public_filename", "source_script", "source_aggregate_data", "validation_rule", "include_in_public_candidate", "status", "notes"]
    lines: list[str] = []
    for row in rows:
        lines.append(f"- item_id: {row['item_id']}")
        for field in fields[1:]:
            value = row.get(field, "")
            lines.append(f"  {field}: {value}" if field == "include_in_public_candidate" else f"  {field}: {quote(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest_rows(project_root: Path) -> tuple[list[dict[str, str]], str]:
    main_doc, supp_docs, status = identify_current_docs(project_root)
    corpus = docx_text(main_doc) if main_doc else ""
    for path in supp_docs:
        corpus += "\n" + docx_text(path)
    template_map = template_reference_by_id(load_template_reference_manifest())
    rows: list[dict[str, str]] = []
    for item_id, kind, title, item_type, section in SEED_ITEMS:
        label = label_from_text(corpus, kind, title) if corpus else f"{kind} TBD"
        ext = ".png" if item_type == "figure" else ".xlsx"
        source_domain = "validation"
        if item_id in {"hia_attributable_cases", "concentration_response_functions", "hia_equations_summary", "baseline_disease_rates"}:
            source_domain = "hia"
        elif item_id in {"yll_scenario_outputs", "yll_life_table_inputs"}:
            source_domain = "yll"
        elif item_id in {"lag_specific_hr_stress_models", "descriptive_ols_pm_hr_lag", "ols_vs_lag_mixed_comparison"}:
            source_domain = "lag_models"
        elif item_id in {"seasonal_personal_pm_distributions", "activity_stratified_personal_pm_exposure"}:
            source_domain = "exposure"
        elif item_id in {"seasonal_garmin_sleep_distributions", "iaq_sleep_regression_panels", "stress_sleep_exploratory_panels", "uhoo_iaq_garmin_sleep_correlation_matrix"}:
            source_domain = "sleep"
        template = template_map.get(item_id, {})
        template_source = template.get("template_source_path_or_placeholder", "") if item_type == "figure" else "not_applicable"
        template_status = template.get("status", "MISSING") if item_type == "figure" else "not_applicable"
        rows.append({
            "item_id": item_id,
            "current_label": label,
            "final_label_status": "RESOLVED" if "TBD" not in label else "BLOCKED_UNRESOLVED_ITEM_LABEL",
            "item_type": item_type,
            "main_or_supplement": section,
            "current_document_source": str(main_doc) if main_doc else "not_identified",
            "accepted_template_source_status": template_status,
            "accepted_template_source": template_source,
            "expected_public_filename": f"{label} - {title}{ext}",
            "source_script": "scripts/generate_manuscript_figures.py" if item_type == "figure" else "scripts/generate_manuscript_tables.py",
            "source_aggregate_data": f"results/{source_domain}/{label} data - {title}.csv",
            "validation_rule": "file_exists; no_p_value_000; template_fidelity" if item_type == "figure" else "file_exists; no_p_value_000",
            "include_in_public_candidate": "true",
            "status": "READY_FOR_USER_RUN" if "TBD" not in label else "BLOCKED_UNRESOLVED_ITEM_LABEL",
            "notes": "detected_from_current_documents" if "TBD" not in label else status,
        })
    return rows, status


def create_manuscript_item_manifest(project_root: str | Path, out_dir: str | Path, status_only: bool = False) -> int:
    out_dir = Path(out_dir)
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows, status = build_manifest_rows(Path(project_root))
    manifest_path = out_dir / "manuscript_item_manifest_detected.yaml" if out_dir.name == "manuscript_reproducibility" else report_dir / "manuscript_item_manifest_detected.yaml"
    write_manifest(manifest_path, rows)
    blocked = [row["item_id"] for row in rows if "TBD" in row.get("current_label", "")]
    lines = ["Manuscript Item Manifest Detection Report", "", f"status: {status}", f"items_recorded: {len(rows)}", f"unresolved_labels: {len(blocked)}", "mapping_policy: use detected final labels only"]
    if status != "DETECTED_WITH_LOCAL_REVISION_FILES":
        lines.append("MANUSCRIPT_ITEM_MAPPING_STATUS = BLOCKED_CURRENT_DOCS_NOT_IDENTIFIED")
    if blocked:
        lines.append("MANUSCRIPT_ITEM_MAPPING_STATUS = BLOCKED_UNRESOLVED_ITEM_LABELS")
    (report_dir / "manuscript_item_manifest_detection_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if status_only or (status == "DETECTED_WITH_LOCAL_REVISION_FILES" and not blocked) else 2

# Template-fidelity validation exports.


def load_template_reference(figure_id: str, path: str | Path | None = None) -> dict[str, str]:
    for row in load_template_reference_manifest(path):
        if row.get("figure_id") == figure_id:
            return row
    return {}


def load_template_metadata(template_reference: str | Path | dict[str, str]) -> dict[str, str]:
    if isinstance(template_reference, dict):
        row = template_reference
    else:
        path = Path(template_reference)
        row = {"template_source_path_or_placeholder": str(path)}
    source = Path(row.get("accepted_template_reference") or row.get("accepted_template_source") or row.get("template_source_path_or_placeholder", ""))
    metadata = {
        "template_source": str(source),
        "dimensions": row.get("image_dimensions", ""),
        "aspect_ratio": row.get("aspect_ratio", ""),
        "font_family": normalise_font_family(row.get("font_family_detected_if_possible", "Times New Roman")),
        "legend_location": row.get("legend_position_detected_if_possible", "REQUIRES_VISUAL_REVIEW"),
        "panel_layout": row.get("panel_layout_detected_if_possible", "REQUIRES_VISUAL_REVIEW"),
        "export_dpi": row.get("export_dpi", "300"),
        "template_lock_status": row.get("template_lock_status", ""),
        "accepted_template_source_type": row.get("accepted_template_source_type", ""),
    }
    if source.exists():
        try:
            from PIL import Image  # type: ignore

            with Image.open(source) as image:
                metadata["dimensions"] = f"{image.width}x{image.height}"
                metadata["aspect_ratio"] = f"{image.width / image.height:.3f}" if image.height else ""
        except Exception:
            metadata["image_read_status"] = "REQUIRES_VISUAL_REVIEW"
    return metadata


def normalise_font_family(value: str | None) -> str:
    text = (value or "").strip()
    unresolved = {"", "not_detected_from_raster", "requires_visual_template_review", "requires_visual_review", "to_be_checked"}
    return "Times New Roman" if text.lower() in unresolved else text


def extract_template_metadata(template_reference: str | Path | dict[str, str]) -> dict[str, str]:
    return load_template_metadata(template_reference)


def apply_template_style(template_metadata: dict[str, str] | None = None) -> dict[str, object]:
    """Apply the machine-readable portion of the accepted manuscript style."""
    metadata = dict(template_metadata or {})
    try:
        import matplotlib.pyplot as plt  # type: ignore

        font_family = normalise_font_family(str(metadata.get("font_family", "")))
        metadata["font_family"] = font_family
        plt.rcParams.update(
            {
                "font.family": font_family,
                "axes.unicode_minus": False,
                "figure.dpi": int(float(metadata.get("export_dpi", 300) or 300)),
                "savefig.dpi": int(float(metadata.get("export_dpi", 300) or 300)),
            }
        )
    except Exception:
        metadata["style_application_status"] = "REQUIRES_VISUAL_REVIEW"
    metadata.setdefault("font_family", "Times New Roman")
    metadata.setdefault("style_application_status", "APPLIED_MACHINE_READABLE_TEMPLATE_STYLE")
    return metadata


def validate_template_lock_inputs(
    figure_id: str,
    output_path: str | Path | None,
    template_reference: str | Path | dict[str, str] | None = None,
    candidate_map: list[dict[str, str]] | None = None,
) -> dict[str, str]:
    rows = candidate_map if candidate_map is not None else load_figure_template_candidate_map()
    try:
        resolved = template_reference if isinstance(template_reference, dict) and template_reference.get("accepted_template_reference") else resolve_accepted_template_reference(figure_id, rows)
        reference = resolved.get("accepted_template_reference", "") if isinstance(resolved, dict) else str(resolved)
        reject_superseded_figure_candidates(figure_id, reference)
        reject_superseded_figure_candidates(figure_id, output_path)
    except FileNotFoundError as exc:
        return {"figure_id": figure_id, "status": "BLOCKED_TEMPLATE_REFERENCE_MISSING", "notes": str(exc)}
    except ValueError as exc:
        return {"figure_id": figure_id, "status": "BLOCKED_REJECTED_CANDIDATE_USED", "notes": str(exc)}
    if not Path(reference).exists():
        return {"figure_id": figure_id, "status": "BLOCKED_TEMPLATE_REFERENCE_MISSING", "notes": reference}
    return {"figure_id": figure_id, "status": "READY_FOR_USER_RUN", "accepted_template_reference": reference, "notes": "accepted template reference resolved"}


def compare_figure_to_template(figure_path: str | Path, template_reference: str | Path | dict[str, str]) -> list[dict[str, str]]:
    figure = Path(figure_path)
    template = load_template_metadata(template_reference)
    rows: list[dict[str, str]] = []
    figure_meta = load_template_metadata(figure)
    for check in ["dimensions", "aspect_ratio"]:
        observed = figure_meta.get(check, "")
        expected = template.get(check, "")
        rows.append({"check": check, "observed": observed, "expected": expected, "status": "PASS" if observed and expected and observed == expected else "REQUIRES_VISUAL_REVIEW", "notes": ""})
    for check in ["font_family", "legend_location", "color_palette", "panel_layout", "axis_label_style", "line_bar_box_scatter_style", "annotation_style", "significance_p_value_formatting", "pm_subscript_source_labels", "hia_yll_no_significance_markers"]:
        rows.append({"check": check, "observed": "", "expected": template.get(check, ""), "status": "REQUIRES_VISUAL_REVIEW", "notes": "Image-level automated check is not sufficient for this property."})
    return rows


def validate_template_fidelity(figure_path: str | Path, template_reference: str | Path | dict[str, str], figure_id: str = "") -> dict[str, str]:
    rows = compare_figure_to_template(figure_path, template_reference)
    failing = [row for row in rows if row["status"] == "FAIL"]
    review = [row for row in rows if row["status"] == "REQUIRES_VISUAL_REVIEW"]
    return {
        "figure_id": figure_id,
        "figure_path": str(figure_path),
        "status": "FAIL" if failing else "REQUIRES_VISUAL_REVIEW" if review else "PASS",
        "checks_failed": str(len(failing)),
        "checks_requiring_visual_review": str(len(review)),
        "notes": "REQUIRES_VISUAL_REVIEW" if review else "",
    }


def write_template_lock_report(path: str | Path, rows: list[dict[str, str]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocked = [row for row in rows if str(row.get("status", "")).startswith("BLOCKED")]
    review = [row for row in rows if row.get("post_run_visual_review_required") == "yes"]
    lines = [
        "Figure Template Lock Report",
        "",
        f"figures_checked: {len(rows)}",
        f"figures_blocked: {len(blocked)}",
        f"figures_requiring_post_run_visual_review: {len(review)}",
        "gate_status: " + ("BLOCKED_BEFORE_USER_RUN" if blocked else "READY_FOR_USER_POWERSHELL_RUN"),
        "POST_RUN_MANUAL_CHECK_REQUIRED = FIGURE_TEMPLATE_VISUAL_REVIEW",
    ]
    for row in blocked:
        lines.append(f"- {row.get('figure_id', '')}: {row.get('status', '')}; {row.get('notes', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_figure_template_fidelity_report(path: str | Path, rows: list[dict[str, str]]) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["figure_id", "figure_path", "status", "checks_failed", "checks_requiring_visual_review", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path

