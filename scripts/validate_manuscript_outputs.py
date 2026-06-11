#!/usr/bin/env python3
"""Validate manuscript outputs produced by the analysis workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap_repo_imports

REPO_ROOT = bootstrap_repo_imports()

from raise_icarus.validation import validate_manuscript_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate manuscript and supplementary outputs.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    return validate_manuscript_outputs(args.run_dir)

if __name__ == "__main__":
    raise SystemExit(main())
