"""Pinhole distance estimation and focal-length calibration.

distance = PERSON_HEIGHT * focal / box_height_px
The focal length can be calibrated on the user's own framing with
--calibrate and is persisted to focal_config.txt."""

from __future__ import annotations

import logging

import math
import time
from pathlib import Path

import cv2
import numpy as np

from utils.constants import *

log = logging.getLogger(__name__)

def estimate_distance(box_height_px: float, focal: float | None = None,
                      real_height_m: float | None = None) -> float:
    """distance = real_height * focal / pixel_height, in metres.

    Defaults are resolved at call time (not bound at import) so config.yaml
    can override PERSON_HEIGHT / CALIBRATION_FOCAL without reimporting.
    """
    focal = CALIBRATION_FOCAL if focal is None else focal
    real_height_m = PERSON_HEIGHT if real_height_m is None else real_height_m
    if box_height_px <= 1:
        return float("inf")
    return real_height_m * focal / box_height_px


def person_distance(box, frame_h: int, focal: float,
                    calibrated_framing: bool = False) -> float | None:
    """Distance to one person in metres, or None when it cannot be trusted.

    Without calibration a box clipped by the frame edge is useless: its pixel
    height is shorter than the person, so the pinhole formula returns a value
    that is too large. Such boxes are dropped and the label keeps just the
    class/confidence.

    With calibration (--calibrate, or a focal_config.txt) the focal length was
    measured on EXACTLY this kind of framing - typically someone sitting at a
    desk whose legs are out of frame - so the clipping is already baked into the
    constant. Edge checks are skipped and the number is shown; otherwise the
    calibration would be pointless for the person who calibrated it.
    """
    x1, y1, x2, y2 = box
    height = y2 - y1
    if height < 20:
        return None
    if not calibrated_framing and (y1 <= 1 or y2 >= frame_h - 2):
        return None
    metres = estimate_distance(height, focal)
    if not math.isfinite(metres) or metres > 50.0:
        return None
    return metres


def load_focal_config(path: str | None = None) -> dict | None:
    """Read focal_config.txt written by --calibrate. Returns fields or None."""
    p = Path(path or FOCAL_CONFIG)
    if not p.exists():
        return None
    data: dict[str, str] = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            data[key.strip()] = value.strip()
        focal = float(data.get("focal", ""))
    except (OSError, ValueError):
        return None
    if focal <= 0:
        return None
    data["focal_value"] = focal
    return data


def save_focal_config(focal: float, distance_m: float, pixel_height: float,
                      path: str | None = None) -> None:
    p = Path(path or FOCAL_CONFIG)
    p.write_text(
        "# written by:  realtime_vision.py --calibrate %.2f\n"
        "# delete this file to fall back to CALIBRATION_FOCAL\n"
        "focal=%.2f\n"
        "person_height=%.2f\n"
        "calibrated_distance_m=%.2f\n"
        "pixel_height=%.1f\n"
        "saved=%s\n"
        % (distance_m, focal, PERSON_HEIGHT, distance_m, pixel_height,
           time.strftime("%Y-%m-%d %H:%M:%S")),
        encoding="utf-8",
    )




def run_calibration(distance_m: float, detector, det_kwargs, cap, mirror: bool,
                    frames: int = 25) -> int:
    """Take the current person box, compute the focal length for a known
    distance and persist it, so every later run shows that distance.

    The box height is taken as the median over several frames: a single frame
    can catch a half-open mouth or a merge with a chair and be 20 px off.
    """
    heights: list[float] = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        res = detector.predict(frame, **det_kwargs)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        names = res.names
        tallest = 0.0
        for xyxy, cls_id in zip(boxes.xyxy.cpu().numpy(),
                                boxes.cls.cpu().numpy().astype(int)):
            name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            if name != "person":
                continue
            tallest = max(tallest, float(xyxy[3] - xyxy[1]))
        if tallest >= 20:
            heights.append(tallest)

    if not heights:
        log.warning("[!] calibration failed: no person detected in %d frames." % frames)
        log.info("    sit in front of the camera, wait for 'person 0.xx' and try again.")
        return 1

    px_height = float(np.median(heights))
    focal = distance_m * px_height / PERSON_HEIGHT

    log.info("=" * 64)
    log.info("CALIBRATION")
    log.info(f"  known distance      : {distance_m:.2f} m")
    log.info(f"  person height       : {PERSON_HEIGHT:.2f} m")
    log.info(f"  box height (median) : {px_height:.1f} px over {len(heights)} frame(s)")
    log.info(f"  focal = D * px / H  : {distance_m:.2f} * {px_height:.1f} / "
             f"{PERSON_HEIGHT:.2f} = {focal:.2f} px")
    try:
        save_focal_config(focal, distance_m, px_height)
        log.info(f"  saved to            : {FOCAL_CONFIG}")
        log.info("  next runs pick it up automatically.")
    except OSError as exc:
        log.warning(f"  [!] could not write {FOCAL_CONFIG}: {exc}")
        return 1
    if not (120 <= focal <= 4000):
        log.warning("  [!] that focal looks implausible - check the distance you passed.")
    log.info("=" * 64)
    return 0
