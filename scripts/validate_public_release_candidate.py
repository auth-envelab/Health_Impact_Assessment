#!/usr/bin/env python3
"""Validate a public aggregate release candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap_repo_imports

REPO_ROOT = bootstrap_repo_imports()

from raise_icarus.public_candidate import validate_public_release_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate public aggregate release candidate safety and completeness.")
    parser.add_argument("--candidate-dir", required=True)
    args = parser.parse_args()
    return validate_public_release_candidate(args.candidate_dir)

if __name__ == "__main__":
    raise SystemExit(main())
