"""Classic geometry that needs no neural network.

* detect_action()  - HANDS UP / SITTING / LYING from COCO-17 keypoints
* find_shapes()    - Canny + approxPolyDP -> Triangle / Square-Rect / Circle
* containment helpers used by the shape detector"""

from __future__ import annotations

import math
import cv2
import numpy as np
from utils.constants import *
def detect_action(xy, conf, box) -> str:
    """Return the strongest action label for one person.

    Rules (image coordinates: y grows downwards):
      LYING    torso more horizontal than vertical, or the person box is wide
      HANDS UP both wrists above the nose
      SITTING  hips below the knees
      STANDING fallback when hips and knees are both visible
    """
    pts = np.asarray(xy, dtype=np.float32)
    cf = np.asarray(conf, dtype=np.float32) if conf is not None else np.ones(len(pts), np.float32)
    vis = cf >= KEYPOINT_CONF

    def pt(i):
        return pts[i] if vis[i] else None

    x1, y1, x2, y2 = box
    width, height = max(x2 - x1, 1), max(y2 - y1, 1)

    # --- LYING: needs visible hips, otherwise a head-and-shoulders close-up
    # (wide, short box) would be reported as lying down
    hips_visible = pt(KP_L_HIP) is not None and pt(KP_R_HIP) is not None
    if hips_visible and width / float(height) > 1.35:
        return "LYING"
    sho = [p for p in (pt(KP_L_SHO), pt(KP_R_SHO)) if p is not None]
    hip = [p for p in (pt(KP_L_HIP), pt(KP_R_HIP)) if p is not None]
    if len(sho) == 2 and len(hip) == 2:
        sx = float(np.mean([p[0] for p in sho]))
        sy = float(np.mean([p[1] for p in sho]))
        hx = float(np.mean([p[0] for p in hip]))
        hy = float(np.mean([p[1] for p in hip]))
        dx, dy = abs(hx - sx), abs(hy - sy)
        if dx > dy * 1.6 and dx > 0.35 * width:
            return "LYING"

    # --- HANDS UP: both wrists above the nose
    nose = pt(KP_NOSE)
    lw, rw = pt(KP_L_WRI), pt(KP_R_WRI)
    if nose is not None and lw is not None and rw is not None:
        if lw[1] < nose[1] and rw[1] < nose[1]:
            return "HANDS UP"

    # --- SITTING: hips lower than the knees
    sitting = False
    known = False
    for h, k in ((pt(KP_L_HIP), pt(KP_L_KNE)), (pt(KP_R_HIP), pt(KP_R_KNE))):
        if h is not None and k is not None:
            known = True
            if h[1] > k[1] + 6:
                sitting = True
    if sitting:
        return "SITTING"
    return "STANDING" if known else ""




def _iou(ax, ay, aw, ah, bx, by, bw, bh):
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0


def overlaps_yolo(bx, by, bw, bh, yolo_boxes):
    cx, cy = bx + bw / 2.0, by + bh / 2.0
    for x1, y1, x2, y2 in yolo_boxes:
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
        ix1, iy1 = max(bx, x1), max(by, y1)
        ix2, iy2 = min(bx + bw, x2), min(by + bh, y2)
        if ix2 > ix1 and iy2 > iy1:
            if (ix2 - ix1) * (iy2 - iy1) / float(bw * bh) > SHAPE_YOLO_OVERLAP_MAX:
                return True
    return False


def find_shapes(frame, yolo_boxes):
    h, w = frame.shape[:2]
    scale = SHAPE_SCALE if 0.0 < SHAPE_SCALE < 1.0 else 1.0
    small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
    sh, sw = small.shape[:2]
    area_ref = float(sh * sw)
    inv = 1.0 / scale

    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), SHAPE_MEDIAN_BLUR)
    edges = cv2.Canny(gray, SHAPE_CANNY_LOW, SHAPE_CANNY_HIGH)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             np.ones((SHAPE_CLOSE_KERNEL, SHAPE_CLOSE_KERNEL), np.uint8),
                             iterations=SHAPE_CLOSE_ITER)
    # RETR_LIST: nested loops (a pane inside a frame) disappear with RETR_EXTERNAL
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < SHAPE_MIN_AREA_RATIO * area_ref or area > SHAPE_MAX_AREA_RATIO * area_ref:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or area / hull_area < SHAPE_MIN_SOLIDITY:
            continue

        approx = cv2.approxPolyDP(contour, SHAPE_EPSILON * perimeter, True)
        sides = len(approx)
        x, y, bw, bh = cv2.boundingRect(approx)
        if bw < 12 * scale or bh < 12 * scale:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        big_enough = area >= SHAPE_CIRCLE_MIN_AREA_RATIO * area_ref
        if sides == 3:
            label = "Triangle"
        elif sides == 4:
            if min(bw, bh) / float(max(bw, bh)) < SHAPE_MAX_ELONGATION:
                continue
            label = "Square/Rect"
        elif sides >= SHAPE_CIRCLE_MIN_SIDES and circularity >= SHAPE_CIRCLE_MIN_CIRCULARITY and big_enough:
            label = "Circle"
        else:
            continue

        fx, fy = int(x * inv), int(y * inv)
        fbw, fbh = int(bw * inv), int(bh * inv)
        if overlaps_yolo(fx, fy, fbw, fbh, yolo_boxes):
            continue
        if any(_iou(fx, fy, fbw, fbh, o[1], o[2], o[3], o[4]) > SHAPE_DUPLICATE_IOU for o in found):
            continue
        found.append((label, fx, fy, fbw, fbh, area))

    found.sort(key=lambda item: -item[5])
    return [(lb, x, y, bw, bh) for lb, x, y, bw, bh, _ in found[:SHAPE_MAX_COUNT]]
