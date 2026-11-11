"""Speed estimation in m/s from the 3D position of each track.

The box centre is unprojected with the current distance, so lateral and
towards-the-camera motion are measured the same way."""

from __future__ import annotations

import math
from utils.constants import *
class SpeedEstimator:
    """Metres per second for every track, from its 3D position over time.

    The box centre is unprojected into camera space with the current distance:

        X = (cx - frame_cx) * Z / focal
        Y = (cy - frame_cy) * Z / focal

    so someone walking sideways (X changes, Z constant) and someone walking
    towards the camera (Z changes) are both measured correctly - the speed is
    the length of the full 3D delta, not just the depth change.

    Timestamps come from the frame, so a dropped frame or a slow pass does not
    corrupt the result; gaps longer than max_gap simply restart the baseline.
    """

    def __init__(self, focal: float, smoothing: float = 0.55, max_gap: float = 1.5):
        self.focal = float(focal)
        self.smoothing = smoothing
        self.max_gap = max_gap
        self._history: dict[int, tuple[float, float, float, float]] = {}
        self._speeds: dict[int, float] = {}
        self._seen: set[int] = set()

    def begin_frame(self) -> None:
        self._seen.clear()

    def update(self, track_id, box, distance_m, now: float, frame_shape):
        """Returns the smoothed speed in m/s, or None if it cannot be computed."""
        if track_id is None or distance_m is None:
            return None
        tid = int(track_id)
        self._seen.add(tid)

        h, w = frame_shape[:2]
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        x_m = (cx - (w - 1) / 2.0) * distance_m / self.focal
        y_m = (cy - (h - 1) / 2.0) * distance_m / self.focal

        previous = self._history.get(tid)
        self._history[tid] = (now, x_m, y_m, distance_m)
        if previous is None:
            return None

        dt = now - previous[0]
        if dt <= 1e-3 or dt > self.max_gap:
            return None

        dx = x_m - previous[1]
        dy = y_m - previous[2]
        dz = distance_m - previous[3]
        raw = math.sqrt(dx * dx + dy * dy + dz * dz) / dt

        old = self._speeds.get(tid)
        value = raw if old is None else old * (1.0 - self.smoothing) + raw * self.smoothing
        self._speeds[tid] = value
        return value

    def end_frame(self) -> None:
        for tid in list(self._history):
            if tid not in self._seen:
                del self._history[tid]
                self._speeds.pop(tid, None)

    def speed_of(self, track_id):
        return self._speeds.get(int(track_id)) if track_id is not None else None
