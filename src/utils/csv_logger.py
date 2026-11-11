"""CSV telemetry writer: one row per detected object per frame."""

from __future__ import annotations

from pathlib import Path
class CsvLogger:
    """One row per detected object per frame, appended to logs/detections.csv."""

    HEADER = ("timestamp,track_id,class_name,confidence,x_center,y_center,"
              "distance_m,speed_mps,speed_kmh")

    def __init__(self, path: str = "logs/detections.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        if fresh:
            self._fh.write(self.HEADER + "\n")
        self.rows = 0

    def log(self, ts, track_id, name, conf, cx, cy, distance, speed) -> None:
        if self._fh is None:
            return
        tid = "" if track_id is None else str(int(track_id))
        dist = "" if distance is None else f"{distance:.2f}"
        sp = "" if speed is None else f"{speed:.2f}"
        spk = "" if speed is None else f"{speed * 3.6:.2f}"
        self._fh.write(f"{ts:.3f},{tid},{name},{conf:.3f},{cx:.1f},{cy:.1f},"
                       f"{dist},{sp},{spk}\n")
        self.rows += 1

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:                                 # noqa: BLE001
                pass
            self._fh = None
