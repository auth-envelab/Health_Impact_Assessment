#!/usr/bin/env python3
"""Generate manuscript table files from aggregate workflow outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap_repo_imports

REPO_ROOT = bootstrap_repo_imports()

from raise_icarus.tables import generate_tables


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate manuscript table files from aggregate outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return generate_tables(args.run_dir, manifest_path=args.manifest, dry_run=args.dry_run)

if __name__ == "__main__":
    raise SystemExit(main())
