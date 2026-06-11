"""Repository-local import bootstrap for public entrypoint scripts."""

from __future__ import annotations

from pathlib import Path
import sys


def bootstrap_repo_imports() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    for path in (str(src_dir), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_root
