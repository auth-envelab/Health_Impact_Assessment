#!/usr/bin/env python3
"""Generate manuscript figures from aggregate workflow outputs and accepted template references."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap_repo_imports

REPO_ROOT = bootstrap_repo_imports()

from raise_icarus.figures import generate_figures


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate manuscript figures with template-fidelity validation.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--template-manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return generate_figures(args.run_dir, manifest_path=args.manifest, template_manifest_path=args.template_manifest, dry_run=args.dry_run)

if __name__ == "__main__":
    raise SystemExit(main())
