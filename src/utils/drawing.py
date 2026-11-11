"""Drawing primitives shared by every overlay: labels, status panel, skeleton.

Kept dependency-free on purpose - they only need cv2/numpy and the colour
constants, so any feature module can draw without importing the main loop."""

from __future__ import annotations

import cv2
import numpy as np
from utils.constants import *
def draw_label(frame, text, x, y, colour, scale=0.5, thickness=1):
    """Filled label above (x, y), flipped inside the frame when there is no room."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    top = y - th - base - 4
    if top < 2:
        top = y + 2
    left = min(max(x, 0), max(frame.shape[1] - tw - 7, 0))
    cv2.rectangle(frame, (left, top), (left + tw + 6, top + th + base + 5), colour, -1)
    cv2.putText(frame, text, (left + 3, top + th + 3), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_panel(frame, lines, origin=(8, 8), scale=0.6, colour=(40, 40, 40),
               align_right=False):
    """Dark panel with several coloured text lines. Returns (height, width).

    align_right puts it in the top-right corner: object labels sit at the top-left
    of their boxes, so a panel up there ended up hiding "person #1 0.92 | 2.4m".
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    widths, heights = [], []
    for text, _ in lines:
        (w, h), b = cv2.getTextSize(text, font, scale, 2)
        widths.append(w)
        heights.append(h + b + 6)
    pw = max(widths + [10]) + 16
    ph = sum(heights) + 10
    if align_right:
        x = max(frame.shape[1] - pw - 8, 0)
        y = origin[1]
    else:
        x, y = origin
    cv2.rectangle(frame, (x, y), (x + pw, y + ph), colour, -1)
    cv2.rectangle(frame, (x, y), (x + pw, y + ph), (90, 90, 90), 1)
    cursor = y + 6
    for (text, col), hh in zip(lines, heights):
        cursor += hh - 6
        cv2.putText(frame, text, (x + 8, cursor), font, scale, col, 2, cv2.LINE_AA)
        cursor += 6
    return ph, pw


def draw_skeleton(frame, xy, conf):
    pts = np.asarray(xy, dtype=np.float32)
    cf = np.asarray(conf, dtype=np.float32) if conf is not None else np.ones(len(pts), np.float32)
    visible = cf >= KEYPOINT_CONF
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            p1 = (int(pts[a][0]), int(pts[a][1]))
            p2 = (int(pts[b][0]), int(pts[b][1]))
            if (p1[0] <= 0 and p1[1] <= 0) or (p2[0] <= 0 and p2[1] <= 0):
                continue
            cv2.line(frame, p1, p2, COLOR_BONE, 2, cv2.LINE_AA)
    for i in range(len(pts)):
        if visible[i]:
            x, y = int(pts[i][0]), int(pts[i][1])
            if x > 0 or y > 0:
                cv2.circle(frame, (x, y), 3, COLOR_KEYPOINT, -1, cv2.LINE_AA)
