from __future__ import annotations

import json
import threading
from pathlib import Path

from core import now_utc_iso

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


class ResourceSampler:
    def __init__(self, out_file: Path, enabled: bool, interval_sec: float) -> None:
        self.out_file = out_file
        self.enabled = enabled
        self.interval_sec = max(0.5, float(interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ResourceSampler":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _snapshot(self) -> dict[str, object]:
        if psutil is None:
            return {"status": "psutil_unavailable"}
        vm = psutil.virtual_memory()
        return {
            "status": "ok",
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "ram_used_bytes": int(vm.used),
            "ram_available_bytes": int(vm.available),
        }

    def _loop(self) -> None:
        with self.out_file.open("a", encoding="utf-8") as fh:
            while not self._stop.is_set():
                row = {"timestamp_utc": now_utc_iso(), **self._snapshot()}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                self._stop.wait(self.interval_sec)
