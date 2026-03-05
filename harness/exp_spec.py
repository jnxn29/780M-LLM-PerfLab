from __future__ import annotations

from pathlib import Path
from typing import Any

from expctl.spec import load_exp_spec


def load_spec(path: Path) -> dict[str, Any]:
    return load_exp_spec(path)

