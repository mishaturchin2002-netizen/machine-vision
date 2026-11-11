"""Gaze/attention tracking and facial emotions from MediaPipe FaceLandmarker.

One 478-landmark pass yields both the iris position (gaze) and, when
enabled, the 52 blendshape scores used for HAPPY / ANGRY / SURPRISED."""

from __future__ import annotations

import math
import time
from pathlib import Path
import cv2
import numpy as np
from utils.constants import *
from utils.mp_utils import mp_modules
class EyeTracker:
    """Gaze from the iris position inside the eye opening.

    MediaPipe 1.x dropped mp.solutions.face_mesh, so this uses the Tasks API
    (FaceLandmarker + face_landmarker.task): same 478 landmarks, iris included.

    With `with_blendshapes=True` the same pass also returns the 52 ARKit-style
    blendshape scores, which is what the emotion classifier reads. One face model
    run feeds both gaze and emotions, so enabling --emotions next to --eyes costs
    nothing extra.
    """

    IRIS_RIGHT = (469, 470, 471, 472)
    IRIS_LEFT = (474, 475, 476, 477)
    LEFT_EYE = {"a": 33, "b": 133, "top": 159, "bottom": 145}
    RIGHT_EYE = {"a": 362, "b": 263, "top": 386, "bottom": 374}

    def __init__(self, model_path: str, with_blendshapes: bool = False):
        mp, vision, BaseOptions = mp_modules()
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} missing - download it from\n"
                "  https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task")
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=with_blendshapes,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.with_blendshapes = with_blendshapes
        self._t0 = time.perf_counter()
        self.looking = None
        self.gaze = (0.0, 0.0)
        self.eye_point = None
        self.blendshapes: dict[str, float] = {}

    @staticmethod
    def _iris_centre(lm, ids):
        return (float(np.mean([lm[i].x for i in ids])),
                float(np.mean([lm[i].y for i in ids])))

    @staticmethod
    def _eye_ratio(lm, eye, iris):
        a, b = lm[eye["a"]], lm[eye["b"]]
        top, bot = lm[eye["top"]], lm[eye["bottom"]]
        if abs(b.x - a.x) < 1e-6 or abs(bot.y - top.y) < 1e-6:
            return None
        lo, hi = (a.x, b.x) if a.x < b.x else (b.x, a.x)
        return ((iris[0] - lo) / (hi - lo), (iris[1] - top.y) / (bot.y - top.y))

    def update(self, frame):
        """Run the face mesh; returns (looking, (dx, dy), eye_centre) or None.

        dx/dy are computed in the *already mirrored* frame the caller passes in,
        so +dx always means "to the right on screen".
        """
        h, w = frame.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = int((time.perf_counter() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(image, ts)

        # blendshapes come from the same pass; keep them even without landmarks
        if self.with_blendshapes:
            self.blendshapes = (
                {c.category_name: float(c.score)
                 for c in result.face_blendshapes[0]}
                if result.face_blendshapes else {}
            )
        else:
            self.blendshapes = {}

        if not result.face_landmarks:
            self.looking = None
            self.eye_point = None
            return None

        lm = result.face_landmarks[0]
        try:
            iris_r = self._iris_centre(lm, self.IRIS_RIGHT)
            iris_l = self._iris_centre(lm, self.IRIS_LEFT)
            ratios = []
            for eye in (self.LEFT_EYE, self.RIGHT_EYE):
                mid = ((lm[eye["a"]].x + lm[eye["b"]].x) / 2.0,
                       (lm[eye["a"]].y + lm[eye["b"]].y) / 2.0)
                d_r = (iris_r[0] - mid[0]) ** 2 + (iris_r[1] - mid[1]) ** 2
                d_l = (iris_l[0] - mid[0]) ** 2 + (iris_l[1] - mid[1]) ** 2
                r = self._eye_ratio(lm, eye, iris_r if d_r <= d_l else iris_l)
                if r is not None:
                    ratios.append(r)
            if not ratios:
                return None
            rx = float(np.mean([r[0] for r in ratios]))
            ry = float(np.mean([r[1] for r in ratios]))
        except (IndexError, ZeroDivisionError):
            return None

        dx = float(np.clip((rx - 0.5) * 2.0, -1.5, 1.5))
        dy = float(np.clip((ry - 0.5) * 2.0, -1.5, 1.5))
        self.gaze = (dx, dy)
        self.looking = math.hypot(dx, dy) < GAZE_CENTRE_THRESHOLD

        eye_centre = (((lm[self.LEFT_EYE["a"]].x + lm[self.RIGHT_EYE["b"]].x) / 2.0) * w,
                      ((lm[self.LEFT_EYE["top"]].y + lm[self.RIGHT_EYE["top"]].y) / 2.0) * h)

        # iris centroid in pixels - the actual "where am I looking" point that
        # the heatmap accumulates (both irises averaged)
        self.eye_point = (((iris_r[0] + iris_l[0]) / 2.0) * w,
                          ((iris_r[1] + iris_l[1]) / 2.0) * h)
        return self.looking, self.gaze, eye_centre

    @staticmethod
    def draw(frame, data):
        if data is None:
            return
        looking, (dx, dy), (ex, ey) = data
        norm = math.hypot(dx, dy)
        if norm < 0.12:
            cv2.circle(frame, (int(ex), int(ey)), 10, COLOR_GAZE, 2, cv2.LINE_AA)
        else:
            length = 25.0 + 55.0 * min(norm, 1.0)
            tip = (int(ex + dx / norm * length), int(ey + dy / norm * length))
            cv2.arrowedLine(frame, (int(ex), int(ey)), tip, COLOR_GAZE, 2, cv2.LINE_AA, tipLength=0.35)

    def close(self):
        try:
            self._landmarker.close()
        except Exception:                                     # noqa: BLE001
            pass




def detect_emotion(blendshapes: dict[str, float]) -> str:
    """Map a few blendshape pairs onto three basic emotions.

    Thresholds are deliberately read off pairs (left AND right) so a half-turned
    head, where one side is barely visible, does not produce a bogus label.
    """
    if not blendshapes:
        return ""

    def s(name: str) -> float:
        return blendshapes.get(name, 0.0)

    if s("mouthSmileLeft") > EMOTION_SMILE and s("mouthSmileRight") > EMOTION_SMILE:
        return "HAPPY"
    if s("browDownLeft") > EMOTION_BROW and s("browDownRight") > EMOTION_BROW:
        return "ANGRY"
    if s("eyeWideLeft") > EMOTION_EYE_WIDE and s("eyeWideRight") > EMOTION_EYE_WIDE:
        return "SURPRISED"
    return "NEUTRAL"
