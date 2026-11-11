"""Project-wide constants: colours, thresholds, model names, tuning knobs.

Everything here used to be hard-coded in the main module. Values that you
are expected to tune (focal length, depth range, confidence overrides)
live in config.yaml instead - this file holds defaults and lookup tables.
"""

from __future__ import annotations

WINDOW_NAME = "YOLOv8 Lab: OpenVINO + Pose + Actions"
DEPTH_WINDOW = "Depth Map"


COLOR_OBJECT = (0, 255, 0)      # green   - YOLO objects
COLOR_SHAPE = (255, 0, 0)       # blue    - geometric shapes
COLOR_KEYPOINT = (0, 0, 255)    # red     - joints
COLOR_BONE = (0, 0, 255)        # red     - skeleton lines
COLOR_ACTION = (0, 255, 255)    # yellow  - action status
COLOR_GAZE = (255, 0, 255)      # magenta - gaze arrow
COLOR_DRAW = (255, 0, 255)      # magenta - air drawing (bright pink)
COLOR_WARN = (60, 60, 230)      # red     - low-confidence marks
COLOR_HAND = (0, 255, 0)        # green   - hand landmarks


SHAPE_MEDIAN_BLUR = 9
SHAPE_SCALE = 1.0
SHAPE_CANNY_LOW = 40
SHAPE_CANNY_HIGH = 120
SHAPE_CLOSE_KERNEL = 3
SHAPE_CLOSE_ITER = 1
SHAPE_EPSILON = 0.02
SHAPE_MIN_AREA_RATIO = 0.003
SHAPE_MAX_AREA_RATIO = 0.50
SHAPE_CIRCLE_MIN_AREA_RATIO = 0.006
SHAPE_MIN_SOLIDITY = 0.62
SHAPE_MAX_ELONGATION = 0.10
SHAPE_CIRCLE_MIN_CIRCULARITY = 0.70
SHAPE_CIRCLE_MIN_SIDES = 7
SHAPE_MAX_COUNT = 12
SHAPE_EVERY = 2                 # refresh geometry every N-th frame, reuse between
SHAPE_YOLO_OVERLAP_MAX = 0.15
SHAPE_DUPLICATE_IOU = 0.85

# COCO-17 skeleton topology
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
KEYPOINT_CONF = 0.50

# COCO-17 indices used by the action rules
KP_NOSE, KP_L_SHO, KP_R_SHO = 0, 5, 6
KP_L_WRI, KP_R_WRI = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNE, KP_R_KNE = 13, 14

EYES_EVERY = 2                  # MediaPipe face mesh every N-th frame
HANDS_EVERY = 2                 # MediaPipe hand mesh every N-th frame
# Ultralytics tracker config for --track. bytetrack is noticeably lighter than
# botsort (which also runs a ReID branch) and costs ~5-8 ms less per frame.
TRACKER_CONFIG = "bytetrack.yaml"
GAZE_CENTRE_THRESHOLD = 0.28    # |iris offset| below this == looking at camera

# emotion thresholds (see detect_emotion)
EMOTION_SMILE = 0.50
EMOTION_BROW = 0.40
EMOTION_EYE_WIDE = 0.50

# gaze heatmap
HEATMAP_SPOT = 3                # side of the gaussian spot stamped per frame (3x3)
HEATMAP_ALPHA = 0.45            # live overlay opacity
HEATMAP_VIZ_EVERY = 10          # rebuild the coloured overlay every N frames
HEATMAP_LIVE_BLUR = 15          # blur used for the live preview (cheap)
HEATMAP_FINAL_BLUR = 51         # blur used for the saved PNG (smooth, pretty)
HEATMAP_OUTPUT = "gaze_heatmap.png"
HEATMAP_CSV = "logs/gaze_points.csv"

# Distance estimation (pinhole camera model)
#
#     Distance = Real_Height * Focal_Length / Pixel_Height
#
# КАК ОТКАЛИБРОВАТЬ (автоматически, рекомендуемый способ):
#   start_vision.bat --calibrate 0.6
#   Сядьте/встаньте на известное расстояние, запустите с этим флагом - скрипт
#   сам замерит высоту вашей рамки в пикселях, посчитает focal и сохранит его
#   в focal_config.txt. Дальнейшие запуски подхватят файл автоматически.
#
# ВРУЧНУЮ: встаньте на 2 метра, посмотрите, что показывает скрипт, и подгоните
#   CALIBRATION_FOCAL так, чтобы выходило ровно 2.0 м.
#   Пример: при 700 пишет 1.7 м -> нужно 700 * 2.0 / 1.7 ≈ 823.
#   focal = D_известное * pixel_height / PERSON_HEIGHT
#
# ВАЖНО ПРО ОБРЕЗАННУЮ РАМКУ: если вы сидите за столом и YOLO видит вас только
#   по пояс, "высота человека в пикселях" - это высота не всего тела. Формула
#   считает по полной высоте рамки, поэтому такие числа завышены. Именно это и
#   лечит --calibrate: он подгоняет focal под ВАШЕ типичное кадрирование, то
#   есть встроенная поправка на обрезку. Побочный эффект: калибровка верна
#   только для похожей позы/кадрирования; встанете в полный рост - поплывёт.
PERSON_HEIGHT = 1.90            # metres - the user's real height
CALIBRATION_FOCAL = 700.0       # px, see the calibration note above
FOCAL_CONFIG = "focal_config.txt"   # written by --calibrate, read on start

# Per-class confidence overrides
#
# yolov8m is COCO-trained, so some classes are systematically noisy in a room:
# a sofa is very often called "bed", an office chair "chair" with low certainty.
# Raising the bar for those two filters out the sofa->bed confusion, which is what
# was producing "bed 0.38" boxes on a couch. They are NOT lost: the model runs at
# a lower floor (--conf) and this filter is applied afterwards.
#
# HEADPHONES DO NOT EXIST IN COCO. No threshold will ever make yolov8m name them;
# they show up as "cell phone" or nothing at all. The honest fix is a custom
# model, and for that the Active Learning folder (dataset/low_confidence/) already
# collects exactly the frames worth labelling: those where the model was unsure.
# Label ~200 of them for the missing classes, fine-tune yolov8m, export again.
CLASS_CONF_OVERRIDES = {
    "bed": 0.40,
    "chair": 0.40,
    "couch": 0.40,
}
