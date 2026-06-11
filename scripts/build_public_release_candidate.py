#!/usr/bin/env python3
"""Build a local public aggregate release candidate from validated workflow outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap_repo_imports

REPO_ROOT = bootstrap_repo_imports()

from raise_icarus.public_candidate import build_public_release_candidate


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a public aggregate release candidate from a validated run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    args = parser.parse_args()
    return build_public_release_candidate(args.run_dir, args.candidate_dir)

if __name__ == "__main__":
    raise SystemExit(main())
