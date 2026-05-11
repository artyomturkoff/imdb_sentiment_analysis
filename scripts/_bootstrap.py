"""Make script imports work from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    root = str(root_dir)
    if root not in sys.path:
        sys.path.insert(0, root)
