"""Per-track motion trails drawn as a fading tail."""

from __future__ import annotations

from collections import deque
import cv2
import numpy as np
from utils.constants import *
class TrailTracker:
    """Per-ID history of box centres, drawn as a fading tail.

    Colours are derived from the track id, so person #1 keeps the same colour
    for the whole session and two overlapping trails never look alike.
    """

    PALETTE = (
        (255, 128, 0), (0, 200, 255), (255, 0, 200), (0, 255, 160),
        (255, 200, 0), (200, 0, 255), (0, 160, 255), (120, 255, 0),
    )

    def __init__(self, maxlen: int = 30):
        self.maxlen = maxlen
        self.trails: dict[int, deque] = {}
        self._seen: set[int] = set()

    def color_for(self, track_id: int):
        return self.PALETTE[int(track_id) % len(self.PALETTE)]

    def begin_frame(self) -> None:
        self._seen.clear()

    def update(self, track_id, centre) -> None:
        if track_id is None or centre is None:
            return
        tid = int(track_id)
        self._seen.add(tid)
        trail = self.trails.get(tid)
        if trail is None:
            trail = deque(maxlen=self.maxlen)
            self.trails[tid] = trail
        trail.append((int(centre[0]), int(centre[1])))

    def end_frame(self) -> None:
        """Forget trails of objects that are gone, otherwise memory grows forever."""
        for tid in list(self.trails):
            if tid not in self._seen:
                del self.trails[tid]

    def draw(self, frame) -> None:
        for tid, pts in self.trails.items():
            if len(pts) < 2:
                continue
            colour = self.color_for(tid)
            points = list(pts)
            n = len(points)
            for i in range(1, n):
                # older segments are thin and dim, the newest one is bold
                age = i / (n - 1)
                thick = max(1, int(round(1 + 2 * age)))
                seg = np.array([points[i - 1], points[i]], np.int32)
                faded = tuple(int(c * (0.35 + 0.65 * age)) for c in colour)
                cv2.polylines(frame, [seg], False, faded, thick, cv2.LINE_AA)
            cv2.circle(frame, points[-1], 3, colour, -1, cv2.LINE_AA)

    def clear(self) -> None:
        self.trails.clear()
