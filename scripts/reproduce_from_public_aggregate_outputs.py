#!/usr/bin/env python3
"""Reproduce supported outputs from public aggregate files only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap_repo_imports

REPO_ROOT = bootstrap_repo_imports()

from raise_icarus.public_candidate import reproduce_public_aggregate_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce supported public aggregate outputs.")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    return reproduce_public_aggregate_outputs(args.candidate_dir, args.out_dir)

if __name__ == "__main__":
    raise SystemExit(main())
