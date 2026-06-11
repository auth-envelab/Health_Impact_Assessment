"""Import shim for direct repository-root Python commands."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "raise_icarus"
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))
