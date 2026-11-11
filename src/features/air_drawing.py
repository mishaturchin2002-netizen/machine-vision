"""Air drawing: a trail of the last N fingertip (or nose) positions."""

from __future__ import annotations

from collections import deque
import cv2
from utils.constants import *
class AirDrawer:
    """Trail of the last N points, drawn as pink segments."""

    def __init__(self, max_points: int = 50, thickness: int = 3):
        self.points: deque = deque(maxlen=max_points)
        self.thickness = thickness

    def add(self, point):
        if point is None:
            return
        x, y = int(point[0]), int(point[1])
        if self.points and self.points[-1] == (x, y):
            return
        self.points.append((x, y))

    def draw(self, frame):
        for p1, p2 in zip(self.points, list(self.points)[1:]):
            cv2.line(frame, p1, p2, COLOR_DRAW, self.thickness, cv2.LINE_AA)
        if self.points:
            cv2.circle(frame, self.points[-1], 4, COLOR_DRAW, -1, cv2.LINE_AA)

    def clear(self):
        self.points.clear()
