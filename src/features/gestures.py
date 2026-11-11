"""Hand landmarks, gesture vocabulary and the media-key controller.

FIST -> play/pause, POINTING swipe -> previous/next track. Key presses go
through pyautogui and are edge-triggered with a cooldown."""

from __future__ import annotations

import logging

import math
import time
from pathlib import Path

import cv2

from utils.constants import *
from utils.mp_utils import mp_modules

log = logging.getLogger(__name__)

class HandTracker:
    """Hand landmarks + a small gesture vocabulary (FIST / POINTING / OPEN_PALM).

    MediaPipe 1.x has no mp.solutions.hands either, so this is the Tasks
    HandLandmarker with hand_landmarker.task (21 landmarks, same numbering).
    """

    WRIST = 0
    THUMB_TIP, THUMB_IP = 4, 3
    INDEX_TIP, INDEX_PIP = 8, 6
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP, RING_PIP = 16, 14
    PINKY_TIP, PINKY_PIP = 20, 18

    def __init__(self, model_path: str):
        mp, vision, BaseOptions = mp_modules()
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} missing - download it from\n"
                "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task")
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.perf_counter()
        self.gesture = ""
        self.tip = None            # normalised (x, y) of the index fingertip

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def _classify(self, lm) -> str:
        """Finger states from wrist-relative distances (rotation tolerant)."""
        wrist = lm[self.WRIST]
        extended = []
        for tip_i, pip_i in ((self.INDEX_TIP, self.INDEX_PIP),
                             (self.MIDDLE_TIP, self.MIDDLE_PIP),
                             (self.RING_TIP, self.RING_PIP),
                             (self.PINKY_TIP, self.PINKY_PIP)):
            extended.append(self._dist(lm[tip_i], wrist) > self._dist(lm[pip_i], wrist) * 1.12)
        thumb = self._dist(lm[self.THUMB_TIP], wrist) > self._dist(lm[self.THUMB_IP], wrist) * 1.15
        count = sum(extended)

        if count == 0 and not thumb:
            return "FIST"
        if count == 4:
            return "OPEN_PALM"
        if extended[0] and not any(extended[1:]):
            return "POINTING"
        if count == 0 and thumb:
            return "THUMB_UP"
        return ""

    def update(self, frame):
        mp = self._mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = int((time.perf_counter() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(image, ts)

        if not result.hand_landmarks:
            self.gesture = ""
            self.tip = None
            return None

        lm = result.hand_landmarks[0]
        self.gesture = self._classify(lm)
        self.tip = (lm[self.INDEX_TIP].x, lm[self.INDEX_TIP].y)
        return lm, self.gesture, self.tip

    @staticmethod
    def draw(frame, lm):
        if not lm:
            return
        h, w = frame.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 4),
                     (0, 5), (5, 6), (6, 7), (7, 8),
                     (5, 9), (9, 10), (10, 11), (11, 12),
                     (9, 13), (13, 14), (14, 15), (15, 16),
                     (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)):
            cv2.line(frame, pts[a], pts[b], COLOR_HAND, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 2, COLOR_HAND, -1, cv2.LINE_AA)
        cv2.circle(frame, pts[HandTracker.INDEX_TIP], 6, COLOR_DRAW, 2, cv2.LINE_AA)

    def close(self):
        try:
            self._landmarker.close()
        except Exception:                                     # noqa: BLE001
            pass


class GestureController:
    """Turns gestures into media keys.

    Both actions are edge-triggered with a cooldown, so holding a fist does not
    spam play/pause: the key fires once when the fist appears.
    """

    FIST_COOLDOWN = 1.0
    SWIPE_COOLDOWN = 0.6
    SWIPE_DX = 0.025        # normalised x travel per frame that counts as a swipe

    def __init__(self, active: bool):
        self.active = active
        self._pg = None
        if active:
            try:
                import pyautogui
                pyautogui.FAILSAFE = True
                self._pg = pyautogui
            except Exception as exc:                          # noqa: BLE001
                log.warning(f"[!] pyautogui unavailable, gestures are display-only: {exc}")
                self.active = False
        self.prev_gesture = ""
        self.last_fist = 0.0
        self.last_swipe = 0.0
        self.prev_x = None
        self.last_action = ""

    def update(self, gesture: str, tip) -> str:
        """Returns the action fired this frame ('' if none)."""
        fired = ""
        now = time.time()

        if gesture == "FIST" and self.prev_gesture != "FIST":
            if now - self.last_fist >= self.FIST_COOLDOWN:
                self.last_fist = now
                fired = "playpause"
                if self._pg is not None:
                    self._pg.press("playpause")

        if gesture == "POINTING" and tip is not None and self.prev_gesture == "POINTING":
            x = tip[0]
            if self.prev_x is not None:
                dx = x - self.prev_x
                if abs(dx) > self.SWIPE_DX and now - self.last_swipe >= self.SWIPE_COOLDOWN:
                    fired = "right" if dx > 0 else "left"
                    self.last_swipe = now
                    if self._pg is not None:
                        self._pg.press(fired)
            self.prev_x = x
        else:
            self.prev_x = None

        self.prev_gesture = gesture
        if fired:
            self.last_action = fired
        return fired
