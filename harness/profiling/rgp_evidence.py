from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ts(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def collect_rgp_evidence(*, enabled: bool, root_glob: str, base_dir: Path) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "file_count": 0, "total_bytes": 0, "files": []}
    root = Path(root_glob)
    pattern = str(root if root.is_absolute() else (base_dir / root))
    files = []
    total = 0
    for path in sorted(base_dir.glob(root_glob)) if not root.is_absolute() else sorted(root.parent.glob(root.name)):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        files.append({"path": str(path.resolve()), "size_bytes": size, "mtime_utc": _ts(path.stat().st_mtime)})
    return {
        "enabled": True,
        "root_glob": pattern,
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }

