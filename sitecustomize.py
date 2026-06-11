"""Repository-local Python path setup for direct commands run from this folder."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir():
    src_path = str(_SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
