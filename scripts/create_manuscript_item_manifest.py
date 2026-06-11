#!/usr/bin/env python3
"""Create or verify the manuscript item manifest used by the analysis workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap_repo_imports

REPO_ROOT = bootstrap_repo_imports()

from raise_icarus.figure_templates import REPO_ROOT, create_manuscript_item_manifest

PROJECT_ROOT = REPO_ROOT.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify manuscript item manifest mapping.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project folder containing current manuscript files.")
    parser.add_argument("--out-dir", required=True, help="Workflow run directory or local development output directory.")
    parser.add_argument("--status-only", action="store_true", help="Create a status report without failing on unresolved labels.")
    args = parser.parse_args()
    return create_manuscript_item_manifest(args.project_root, args.out_dir, status_only=args.status_only)

if __name__ == "__main__":
    raise SystemExit(main())
