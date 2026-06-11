#!/usr/bin/env python3
"""Run the manuscript reproducibility workflow from the controlled harmonized dataset."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap_repo_imports

REPO_ROOT = bootstrap_repo_imports()

from raise_icarus.manuscript_runtime import ensure_workflow_layout, run_controlled_workflow


def py(script: str, *args: object) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "scripts" / script), *[str(arg) for arg in args]]


def write_step_report(report_dir: Path, rows: list[dict[str, str]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "workflow_steps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "status", "return_code", "log_file", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def write_text_report(report_dir: Path, rows: list[dict[str, str]], status: str) -> None:
    failed = [row for row in rows if row["status"] != "PASS"]
    lines = ["Manuscript Reproducibility Workflow Report", "", f"run_status: {status}", f"steps_recorded: {len(rows)}", f"steps_failed: {len(failed)}", "workflow_scope: controlled harmonized dataset to manuscript outputs and public aggregate outputs", ""]
    if failed:
        lines.append("Failed stages:")
        lines.extend(f"- {row['stage']}: {row['notes']}" for row in failed)
    else:
        lines.append("All recorded stages passed.")
    (report_dir / "manuscript_reproducibility_workflow_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(stage: str, command: list[str], layout: dict[str, Path], rows: list[dict[str, str]]) -> int:
    log_path = layout["logs"] / f"{stage}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    rows.append({"stage": stage, "status": "PASS" if completed.returncode == 0 else "FAIL", "return_code": str(completed.returncode), "log_file": str(log_path), "notes": "Completed" if completed.returncode == 0 else "See local log"})
    write_step_report(layout["reports"], rows)
    return completed.returncode


def run_controlled(args: argparse.Namespace, layout: dict[str, Path]) -> int:
    data_zip = Path(args.harmonized_zip)
    if not data_zip.exists():
        raise SystemExit("Controlled harmonized dataset archive was not found")
    if args.debug_traceback:
        os.environ["RAISE_ICARUS_DEBUG_TRACEBACK"] = "1"
    run_dir = layout["logs"].parent
    code, stage_results = run_controlled_workflow(data_zip, run_dir, n_samples=args.n_samples, dry_run=args.dry_run)
    rows = [{"stage": result.stage_name, "status": "PASS" if result.status.startswith("PASS") else "FAIL", "return_code": "0" if result.status.startswith("PASS") else "3", "log_file": str(layout["reports"] / "controlled_stage_report.txt"), "notes": result.notes} for result in stage_results]
    write_step_report(layout["reports"], rows)
    if code == 0 and args.build_public_candidate and not args.dry_run:
        build_code = run_command(
            "public_release_candidate_build",
            py("build_public_release_candidate.py", "--run-dir", run_dir, "--candidate-dir", layout["candidate"]),
            layout,
            rows,
        )
        code = build_code
    write_text_report(layout["reports"], rows, "PASS" if code == 0 else "FAIL")
    return code


def run_public_aggregate(args: argparse.Namespace, layout: dict[str, Path]) -> int:
    if not args.candidate_dir:
        raise SystemExit("--candidate-dir is required in public aggregate mode")
    rows: list[dict[str, str]] = []
    for stage, command in [
        ("public_aggregate_reproduction", py("reproduce_from_public_aggregate_outputs.py", "--candidate-dir", args.candidate_dir, "--out-dir", layout["logs"].parent)),
        ("public_aggregate_validation", py("validate_public_release_candidate.py", "--candidate-dir", args.candidate_dir)),
    ]:
        code = run_command(stage, command, layout, rows)
        if code != 0:
            write_text_report(layout["reports"], rows, "FAIL")
            return code
    write_text_report(layout["reports"], rows, "PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run manuscript reproducibility workflow from a controlled harmonized dataset.")
    parser.add_argument("--harmonized-zip", default=None, help="Path to the controlled harmonized dataset archive.")
    parser.add_argument("--out-dir", required=True, help="Local output directory for this run.")
    parser.add_argument("--n-samples", type=int, default=10000, help="Monte Carlo or bootstrap sample count.")
    parser.add_argument("--build-public-candidate", action="store_true", help="Build a validated public aggregate candidate after validation passes.")
    parser.add_argument("--public-aggregate-only", action="store_true", help="Run only from public aggregate outputs.")
    parser.add_argument("--candidate-dir", default=None, help="Public aggregate candidate directory for public aggregate mode.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve stage contracts without reading controlled data.")
    parser.add_argument("--debug-traceback", action="store_true", help="Include full Python tracebacks in controlled-stage failure notes.")
    args = parser.parse_args()
    layout = ensure_workflow_layout(Path(args.out_dir))
    if args.public_aggregate_only:
        return run_public_aggregate(args, layout)
    if not args.harmonized_zip:
        raise SystemExit("--harmonized-zip is required unless public aggregate mode is selected")
    return run_controlled(args, layout)


if __name__ == "__main__":
    raise SystemExit(main())
