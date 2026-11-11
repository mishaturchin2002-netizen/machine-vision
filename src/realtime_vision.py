"""realtime_vision.py - real-time computer-vision lab (entry point).

Pipeline:  VideoStream -> YOLOv8m/OpenVINO detection -> tracker ->
           pose/actions, gaze, gestures, depth, shapes, trails, speed,
           background segmentation -> overlay -> window.

Every heavy feature is opt-in via argparse; with no flags you get the
light profile (detection + pose + actions + distance).

Run:  python src/realtime_vision.py [flags]
See README.md for the flag table and docs/BENCHMARKS.md for numbers."""

from __future__ import annotations

import logging

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from utils.constants import *  # noqa: F401,F403
from utils import config as app_config
from utils.calibration import (  # noqa: F401
    estimate_distance, load_focal_config, person_distance,
    run_calibration, save_focal_config,
)
from utils.csv_logger import CsvLogger
from utils.drawing import draw_label, draw_panel, draw_skeleton
from utils.geometry import detect_action, find_shapes
from utils.log import setup_logging
from utils.models import load_model, open_source, ov_dir_for
from features.active_learning import LowConfidenceSaver
from features.air_drawing import AirDrawer
from features.background import Segmenter
from features.depth import DepthEstimator
from features.gaze import EyeTracker, detect_emotion
from features.gestures import GestureController, HandTracker
from features.heatmap import GazeHeatmap
from features.ocr import OcrEngine
from features.speed import SpeedEstimator
from features.trails import TrailTracker
from video_stream import VideoStream

log = logging.getLogger(__name__)

def parse_args(config: dict | None = None) -> argparse.Namespace:
    """Build the CLI. Values from config.yaml become argparse defaults, so any
    flag given on the command line still wins over the file."""
    ap = argparse.ArgumentParser(
        description="YOLOv8 lab: OpenVINO + pose/actions + optional eyes/gestures/"
                    "drawing/distance/shapes/depth")
    ap.add_argument("--source", default="0", help="camera index or video file (default: 0)")
    ap.add_argument("--det-model", default="yolov8m.pt")
    ap.add_argument("--pose-model", default="yolov8n-pose.pt")
    ap.add_argument("--backend", choices=("auto", "openvino", "torch"), default="auto",
                    help="auto = OpenVINO if the exported model folder exists, else torch")
    ap.add_argument("--ov-device", default="intel:gpu",
                    help="OpenVINO device: intel:gpu (Arc), intel:cpu, intel:npu")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="torch inference size (OpenVINO bakes its size at export)")
    ap.add_argument("--pose-imgsz", type=int, default=640, help="torch pose inference size")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--pose-conf", type=float, default=0.50)
    ap.add_argument("--threads", type=int, default=0, help="torch threads per model (0 = auto)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="do not flip the preview (raw camera orientation)")

    # --- feature flags: all off by default, all cheap to skip ---------------
    ap.add_argument("--shapes", action="store_true", help="OpenCV geometry detection")
    ap.add_argument("--eyes", action="store_true", help="MediaPipe gaze / attention")
    ap.add_argument("--emotions", action="store_true",
                    help="face blendshapes -> HAPPY / ANGRY / SURPRISED / NEUTRAL")
    ap.add_argument("--track", action="store_true",
                    help="Ultralytics tracker: stable IDs, labels become 'person #1 0.95'")
    ap.add_argument("--ocr", action="store_true",
                    help="EasyOCR on the centre of the frame (heavy: background thread)")
    ap.add_argument("--heatmap", action="store_true",
                    help="accumulate a gaze heatmap, live overlay + gaze_heatmap.png on exit")
    ap.add_argument("--ocr-every", type=int, default=20,
                    help="run OCR on every N-th frame (default 20, it is slow)")
    ap.add_argument("--gestures", action="store_true",
                    help="hand gestures -> media keys via pyautogui")
    ap.add_argument("--draw", action="store_true", help="air drawing (fingertip or nose)")
    ap.add_argument("--distance", action="store_true",
                    help="deprecated: distance is now always shown for every person")
    ap.add_argument("--depth", action="store_true",
                    help="MiDaS depth heatmap (downloads a model, costs a lot of FPS)")
    ap.add_argument("--depth-every", type=int, default=3)
    ap.add_argument("--all", action="store_true",
                    help="enable eyes + gestures + draw + distance at once (max load)")
    ap.add_argument("--no-pose", action="store_true", help="disable pose/skeleton/actions")

    # --- architecture / new features ---------------------------------------
    ap.add_argument("--async-grab", action="store_true",
                    help="read frames in a background thread (VideoStream) so "
                         "inference never waits for the camera")
    ap.add_argument("--speed", action="store_true",
                    help="estimate each track's speed in m/s and km/h "
                         "(needs --track)")
    ap.add_argument("--bg-blur", action="store_true",
                    help="blur everything that is not a person (yolov8n-seg)")
    ap.add_argument("--bg-remove", action="store_true",
                    help="replace the background with chroma green (yolov8n-seg)")
    ap.add_argument("--seg-model", default="yolov8n-seg.pt")
    ap.add_argument("--seg-every", type=int, default=3,
                    help="run segmentation every N frames (default 3)")
    ap.add_argument("--export-csv", action="store_true",
                    help="append detections to logs/detections.csv")
    ap.add_argument("--csv-path", default="logs/detections.csv")

    # --- gesture safety -----------------------------------------------------
    ap.add_argument("--no-gesture-actions", action="store_true",
                    help="detect gestures but never press keys (safe demo mode)")

    # --- active learning ----------------------------------------------------
    ap.add_argument("--no-lowconf", action="store_true", help="do not save low-conf frames")
    ap.add_argument("--lowconf", type=float, default=0.6)
    ap.add_argument("--lowconf-dir", default="dataset/low_confidence")
    ap.add_argument("--lowconf-cooldown", type=float, default=2.0)

    ap.add_argument("--focal", type=float, default=None,
                    help="override the focal length in px (default: focal_config.txt "
                         f"if present, else CALIBRATION_FOCAL={CALIBRATION_FOCAL:.0f})")
    ap.add_argument("--calibrate", type=float, default=None, metavar="METRES",
                    help="measure the current person box, compute the focal length for "
                         "this distance, save it to focal_config.txt and exit "
                         "(e.g. --calibrate 0.6)")
    ap.add_argument("--depth-map", action="store_true",
                    help="MiDaS depth shown as a MAGMA map in a separate 'Depth Map' window")
    ap.add_argument("--trails", action="store_true",
                    help="draw a motion trail behind every tracked object")
    ap.add_argument("--headless", type=int, default=0,
                    help="process N frames with no window, print stats, exit")
    ap.add_argument("--benchmark", action="store_true",
                    help="measure torch vs OpenVINO FPS on the camera, then exit")

    if config:
        known = {action.dest for action in ap._actions}
        ap.set_defaults(**{k: v for k, v in config.items() if k in known})
    return ap.parse_args()



def run_benchmark(args) -> int:
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads or max(2, min(6, (os.cpu_count() or 4) // 2)))

    cap = open_source(args.source, not args.no_mirror)
    if cap is None:
        return 1
    frames = []
    for _ in range(8):
        ok, f = cap.read()
        if ok:
            frames.append(cv2.flip(f, 1) if not args.no_mirror else f)
    cap.release()
    if not frames:
        log.warning("[!] no frames for benchmark")
        return 1

    def measure(det, pose, kw_det, kw_pose, n=12):
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            for _ in range(3):
                a = pool.submit(det.predict, frames[0], **kw_det)
                b = pool.submit(pose.predict, frames[0], **kw_pose)
                a.result()
                b.result()
            times = []
            for i in range(n):
                t = time.perf_counter()
                a = pool.submit(det.predict, frames[i % len(frames)], **kw_det)
                b = pool.submit(pose.predict, frames[i % len(frames)], **kw_pose)
                a.result()
                b.result()
                times.append((time.perf_counter() - t) * 1000.0)
            times.sort()
            return times[len(times) // 2]
        except Exception as exc:                              # noqa: BLE001
            return f"FAILED ({type(exc).__name__})"
        finally:
            pool.shutdown()

    kw_det = dict(conf=args.conf, iou=0.45, max_det=50, verbose=False, imgsz=args.imgsz)
    kw_pose = dict(conf=args.pose_conf, verbose=False, imgsz=args.pose_imgsz)
    rows = []

    log.info("=" * 64)
    log.info(f"BENCHMARK - {args.det_model} + {args.pose_model}, parallel, median of 12")
    log.info("=" * 64)

    det, pose = YOLO(args.det_model), YOLO(args.pose_model)
    rows.append(("torch CPU", measure(det, pose, kw_det, kw_pose)))
    del det, pose

    det_ov, pose_ov = ov_dir_for(args.det_model), ov_dir_for(args.pose_model)
    if det_ov.exists() and pose_ov.exists():
        for dev in ("intel:cpu", "intel:gpu"):
            det, pose = YOLO(str(det_ov)), YOLO(str(pose_ov))
            rows.append((f"OpenVINO {dev}",
                         measure(det, pose, dict(kw_det, device=dev), dict(kw_pose, device=dev))))
            del det, pose
    else:
        log.warning("[!] OpenVINO folders missing, skipping OV rows")

    log.info(f"{'backend':24s} {'frame time':>12s} {'FPS':>8s}")
    for name, ms in rows:
        if isinstance(ms, str):
            log.info(f"{name:24s} {ms:>12s}")
        else:
            log.info(f"{name:24s} {ms:8.0f} ms {1000.0 / ms:7.1f}")
    log.info("=" * 64)
    return 0



def main() -> int:
    setup_logging()
    # config.yaml -> argparse defaults -> CLI flags override them
    app_cfg = app_config.load()
    app_config.apply_runtime(app_cfg)
    args = parse_args(app_cfg)

    if args.benchmark:
        return run_benchmark(args)

    # --all turns on the four medium-weight extras
    if args.all:
        args.eyes = args.gestures = args.draw = True

    # ---- focal length: --focal > focal_config.txt > CALIBRATION_FOCAL ----
    cfg = load_focal_config(FOCAL_CONFIG)
    if args.focal is not None:
        focal_source = "--focal"
    elif cfg is not None:
        args.focal = float(cfg["focal_value"])
        focal_source = (f"{FOCAL_CONFIG} (calibrated at "
                        f"{cfg.get('calibrated_distance_m', '?')} m)")
    else:
        args.focal = CALIBRATION_FOCAL
        focal_source = f"CALIBRATION_FOCAL default ({CALIBRATION_FOCAL:.0f})"
    # a focal that was measured on this exact framing already accounts for the
    # legs being out of frame, so the bottom-clip guard can be relaxed
    calibrated = cfg is not None or focal_source == "--focal"

    import torch
    torch.set_num_threads(args.threads or max(2, min(6, (os.cpu_count() or 4) // 2)))

    detector, det_label, det_ov = load_model(args.det_model, args.backend)
    log.info(f"[i] detector: {det_label}")

    pose_model = None
    pose_ov = False
    if not args.no_pose:
        pose_model, pose_label, pose_ov = load_model(args.pose_model, args.backend)
        log.info(f"[i] pose    : {pose_label}")

    # the model runs at the lowest threshold we care about, per-class filtering
    # happens afterwards (see CLASS_CONF_OVERRIDES)
    model_conf = min([args.conf] + list(CLASS_CONF_OVERRIDES.values()))
    det_kwargs = dict(conf=model_conf, iou=0.45, max_det=50, verbose=False, imgsz=args.imgsz)
    pose_kwargs = dict(conf=args.pose_conf, verbose=False, imgsz=args.pose_imgsz)
    if det_ov:
        det_kwargs["device"] = args.ov_device
    if pose_ov:
        pose_kwargs["device"] = args.ov_device

    mirror = not args.no_mirror
    cap = open_source(args.source, mirror)
    if cap is None:
        return 1

    # ---- camera speed probe ------------------------------------------
    # The webcam is a hard ceiling: if it delivers 15 fps, no amount of GPU
    # tuning will push the loop past 15. Measure it and say so out loud,
    # because "FPS dropped" is almost always the camera, not the model.
    try:
        probe_n = 12
        t_probe = time.perf_counter()
        for _ in range(probe_n):
            cap.read()
        cam_ms = (time.perf_counter() - t_probe) / probe_n * 1000.0
        cam_fps = 1000.0 / cam_ms if cam_ms > 0 else 0.0
        log.info(f"[i] camera: {cam_ms:.1f} ms/frame (~{cam_fps:.0f} fps) "
                 f"- hard ceiling for the pipeline")
        if cam_fps < 20:
            log.warning(f"[!] camera is slow ({cam_fps:.0f} fps). Dim light or USB "
                        f"bandwidth caps the whole app; add light or lower the "
                        f"resolution. The GPU is not the limit here.")
    except Exception as exc:                                  # noqa: BLE001
        log.warning(f"[!] camera probe failed: {exc}")
    if args.async_grab:
        # the grab thread takes over the blocking read(); .read() then just
        # returns whatever frame is current, so inference never waits
        cap = VideoStream(cap, name=Path(str(args.source)).name or "cam")
        log.info("[i] async grab: ON (background thread, queue(maxsize=1))")

    # the heatmap needs the frame size up front, not on the first loop iteration
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    heatmap = None
    if args.heatmap:
        try:
            heatmap = GazeHeatmap(src_w, src_h)
            log.info(f"[i] heatmap: accumulating {src_w}x{src_h} float32, "
                  f"csv -> {heatmap.csv_path}, png -> {HEATMAP_OUTPUT} on exit")
        except Exception as exc:                              # noqa: BLE001
            log.warning(f"[!] heatmap disabled: {type(exc).__name__}: {exc}")

    dummy = np.zeros((args.imgsz, args.imgsz, 3), np.uint8)

    def warmup(model, kwargs, name):
        try:
            model.predict(dummy, **kwargs)
        except Exception as exc:                              # noqa: BLE001
            if "device" in kwargs and "gpu" in str(kwargs["device"]):
                log.warning(f"[!] {name}: {kwargs['device']} failed ({str(exc)[:80]}), "
                      f"falling back to intel:cpu")
                kwargs["device"] = "intel:cpu"
                model.predict(dummy, **kwargs)
            else:
                raise

    warmup(detector, det_kwargs, "detector")
    if pose_model is not None:
        warmup(pose_model, pose_kwargs, "pose")
    log.info(f"[i] inference device: {det_kwargs.get('device', 'torch/cpu')}")

    # ---- explicit device report --------------------------------------
    # silent CPU fallback is the classic reason for "why is it slow", so make
    # the active device and the available hardware impossible to miss.
    if det_ov:
        actual = str(det_kwargs.get("device", "?"))
        try:
            import openvino as ov
            available = ov.Core().available_devices
        except Exception:                                     # noqa: BLE001
            available = []
        log.info(f"[i] OpenVINO devices available: {available or 'unknown'}")
        if "cpu" in actual.lower():
            log.warning(f"[!] running on {actual.upper()} - expect 3-5x slower than "
                        f"the iGPU. Check --ov-device / config.yaml inference.device")
        else:
            log.info(f"[i] MODEL RUNNING ON: {actual} "
                     f"(ultralytics picks GPU.0 when 'intel:gpu' is requested)")

    # ---- calibration run: measure, save, exit ---------------------------
    if args.calibrate is not None:
        if args.calibrate <= 0:
            log.warning("[!] --calibrate needs a positive distance in metres, e.g. --calibrate 0.6")
            cap.release()
            return 1
        rc = run_calibration(args.calibrate, detector, det_kwargs, cap, mirror)
        cap.release()
        return rc

    # ---- optional modules ---------------------------------------------
    # one FaceLandmarker instance serves gaze and emotions: with --emotions it is
    # created with output_face_blendshapes=True, so the extra 52 scores come from
    # the same inference pass and cost nothing
    eye_tracker = None
    want_face = args.eyes or args.emotions or args.heatmap
    if want_face:
        try:
            eye_tracker = EyeTracker("models/face_landmarker.task",
                                     with_blendshapes=args.emotions)
            log.info(f"[i] face: MediaPipe FaceLandmarker ready "
                  f"(gaze={args.eyes}, blendshapes={args.emotions})")
        except Exception as exc:                              # noqa: BLE001
            log.warning(f"[!] face disabled: {type(exc).__name__}: {exc}")

    ocr = None
    if args.ocr:
        try:
            ocr = OcrEngine(every=args.ocr_every)
            log.info(f"[i] ocr: ready, every {ocr.every} frame(s)")
        except Exception as exc:                              # noqa: BLE001
            log.warning(f"[!] ocr disabled: {type(exc).__name__}: {exc}")

    hand_tracker = None
    gestures = None
    if args.gestures or args.draw:
        try:
            hand_tracker = HandTracker("models/hand_landmarker.task")
            log.info("[i] hands: MediaPipe HandLandmarker ready")
        except Exception as exc:                              # noqa: BLE001
            log.warning(f"[!] hands disabled: {type(exc).__name__}: {exc}")

    if args.gestures and hand_tracker is not None:
        # key presses are disabled in headless runs: they would land in whatever
        # window happens to be focused, which is not something a test should do
        can_press = not (args.no_gesture_actions or args.headless)
        gestures = GestureController(active=can_press)
        if not can_press:
            log.info("[i] gestures: detection ON, key presses OFF (safe mode)")
        else:
            log.info("[i] gestures: fist -> playpause, pointing swipe -> left/right")

    drawer = AirDrawer(max_points=50, thickness=3) if args.draw else None
    trails = TrailTracker(maxlen=30) if args.trails else None

    # --depth overlays MiDaS on the main frame; --depth-map opens a second
    # window with the MAGMA map. Both need the same (heavy) model, so it is
    # loaded once and shared.
    depth = None
    if args.depth or args.depth_map:
        try:
            depth = DepthEstimator()
            what = []
            if args.depth:
                what.append("overlay on the main frame")
            if args.depth_map:
                what.append("separate 'Depth Map' window (MAGMA)")
            log.info(f"[i] depth: MiDaS_small ready, every {args.depth_every} frame(s), "
                  f"{' + '.join(what)}. WARNING: MiDaS is heavy, FPS will drop.")
        except Exception as exc:                              # noqa: BLE001
            log.warning(f"[!] depth disabled: {type(exc).__name__}: {exc}")

    saver = None
    if not args.no_lowconf:
        saver = LowConfidenceSaver(args.lowconf_dir, args.lowconf, args.lowconf_cooldown)
        log.info(f"[i] active learning: conf < {args.lowconf} -> {saver.folder}")

    # ---- speed / csv / segmentation --------------------------------------
    speed_est = SpeedEstimator(args.focal) if args.speed else None
    if speed_est is not None:
        log.info("[i] speed: on (3D delta per track, needs --track for ids)")

    csv_log = None
    if args.export_csv:
        try:
            csv_log = CsvLogger(args.csv_path)
            log.info(f"[i] csv: appending to {csv_log.path}")
        except OSError as exc:
            log.warning(f"[!] csv disabled: {exc}")

    segmenter = None
    bg_mode = "remove" if args.bg_remove else ("blur" if args.bg_blur else None)
    if bg_mode:
        try:
            segmenter = Segmenter(args.seg_model, args.backend, args.ov_device,
                                  every=args.seg_every)
            log.info(f"[i] segmentation: {segmenter.label}, every {segmenter.every} frame(s), "
                  f"mode={bg_mode}")
        except Exception as exc:                              # noqa: BLE001
            log.warning(f"[!] segmentation disabled: {type(exc).__name__}: {exc}")
            bg_mode = None

    # distance is now always on for every person detection
    log.info(f"[i] distance: always on, focal={args.focal:.1f}px ({focal_source}), "
          f"height {PERSON_HEIGHT:.2f} m, "
          f"edge guard={'off (calibrated framing)' if calibrated else 'on'}")
    if args.trails:
        log.info("[i] trails: on (last 30 box centres per track id)")

    show_objects = True
    show_pose = not args.no_pose
    show_shapes = args.shapes
    show_eyes = args.eyes and eye_tracker is not None
    show_emotions = args.emotions and eye_tracker is not None
    show_hands = hand_tracker is not None and (args.gestures or args.draw)
    show_timing = False
    emotion_now = ""

    if args.headless:
        log.info(f"[i] headless mode, {args.headless} frames")
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 900, 675)
        log.info("[i] running - [q] quit, [1] obj [2] pose [3] shapes [4] eyes [5] hands "
              "[6] draw, c clear, [d] timings, [s] shot")

    pool = ThreadPoolExecutor(max_workers=2)
    # MediaPipe face mesh runs here so it overlaps with GPU inference.
    # Single worker on purpose: FaceLandmarker is a stateful VIDEO-mode tracker.
    face_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face")

    # --track: model.track(persist=True) keeps the ID state inside the model, so
    # the same instance must be reused for every frame (it is, it is `detector`).
    det_track = None
    if args.track:
        try:
            tracker_cfg = TRACKER_CONFIG    # bytetrack = lighter than botsort

            def _track_fn(frame, **kwargs):
                return detector.track(frame, persist=True, tracker=tracker_cfg, **kwargs)

            det_track = _track_fn
            log.info(f"[i] tracking: Ultralytics tracker enabled "
                     f"({tracker_cfg}, persist=True)")
        except Exception as exc:                          # noqa: BLE001
            log.warning(f"[!] tracking disabled: {exc}")
            det_track = None

    fps = 0.0
    t_prev = time.perf_counter()
    frame_idx = 0
    shape_cache = None
    shape_ms = net_ms = 0.0
    last_eyes = None
    last_hand = None
    frame_times: list[float] = []
    saved_flash = 0.0
    last_action = ""
    gesture_now = ""
    distance_note = ""
    person_metres = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("[!] frame grab failed - stopping")
                break
            frame_ts = time.time()          # wall clock for speed / CSV rows

            # ---- THE single mirror point -------------------------------
            # Everything below (models, drawing, gaze maths) works on this
            # mirrored copy, so no coordinate can be flipped twice or not at all.
            if mirror:
                frame = cv2.flip(frame, 1)

            # OCR reads the clean frame; the result arrives later from a worker
            if ocr is not None:
                ocr.maybe_submit(frame, frame_idx)

            # segmentation also runs off-thread; the mask may be a frame old
            if segmenter is not None:
                segmenter.maybe_submit(frame, frame_idx)

            t0 = time.perf_counter()
            # --track swaps predict() for track(persist=True): same OpenVINO model,
            # plus a BoT-SORT/ByteTrack pass that keeps IDs stable across frames.
            det_fn = det_track if det_track is not None else detector.predict

            # The MediaPipe face mesh (~15-20 ms on CPU) is submitted to its own
            # single-worker thread so it overlaps with the GPU detection instead
            # of being added on top of it. One worker only: FaceLandmarker is a
            # stateful VIDEO-mode tracker and must not be called concurrently.
            track_face = eye_tracker is not None and (
                show_eyes or show_emotions or heatmap is not None)
            face_due = track_face and (frame_idx % EYES_EVERY == 0 or last_eyes is None)
            fut_face = face_pool.submit(eye_tracker.update, frame) if face_due else None

            if pose_model is not None:
                fut_det = pool.submit(det_fn, frame, **det_kwargs)
                fut_pose = pool.submit(pose_model.predict, frame, **pose_kwargs)
                det_res = fut_det.result()[0]
                pose_res = fut_pose.result()[0]
            else:
                det_res = det_fn(frame, **det_kwargs)[0]
                pose_res = None
            if fut_face is not None:
                last_eyes = fut_face.result()
                if show_emotions:
                    emotion_now = detect_emotion(eye_tracker.blendshapes)
            net_ms = (time.perf_counter() - t0) * 1000.0

            fh, fw = frame.shape[:2]
            yolo_boxes, objects = [], []
            boxes = det_res.boxes
            if boxes is not None and len(boxes) > 0:
                names = det_res.names
                ids = None
                if args.track and getattr(boxes, "id", None) is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                for i, (xyxy, conf, cls_id) in enumerate(zip(boxes.xyxy.cpu().numpy(),
                                                             boxes.conf.cpu().numpy(),
                                                             boxes.cls.cpu().numpy().astype(int))):
                    x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
                    x2, y2 = min(fw - 1, int(xyxy[2])), min(fh - 1, int(xyxy[3]))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    if float(conf) < CLASS_CONF_OVERRIDES.get(name, args.conf):
                        continue
                    track_id = int(ids[i]) if ids is not None and i < len(ids) else None
                    yolo_boxes.append((x1, y1, x2, y2))
                    objects.append((name, float(conf), x1, y1, x2, y2, track_id))

            # ---- pose / skeleton / actions -----------------------------
            skeletons, actions = [], []
            if pose_res is not None:
                kpts = pose_res.keypoints
                if kpts is not None and kpts.xy is not None and len(kpts.xy) > 0:
                    xy_all = kpts.xy.cpu().numpy()
                    conf_all = (kpts.conf.cpu().numpy()
                                if getattr(kpts, "conf", None) is not None else None)
                    for i in range(len(xy_all)):
                        cf = conf_all[i] if conf_all is not None else None
                        kp = xy_all[i]
                        skeletons.append((kp, cf))
                        bx1 = int(max(0, kp[:, 0].min()))
                        by1 = int(max(0, kp[:, 1].min()))
                        bx2 = int(min(fw - 1, kp[:, 0].max()))
                        by2 = int(min(fh - 1, kp[:, 1].max()))
                        actions.append(detect_action(kp, cf, (bx1, by1, bx2, by2)))

            # ---- geometry on the clean frame ---------------------------
            if show_shapes:
                if shape_cache is None or frame_idx % SHAPE_EVERY == 0:
                    t1 = time.perf_counter()
                    shape_cache = find_shapes(frame, yolo_boxes)
                    shape_ms = (time.perf_counter() - t1) * 1000.0
                shapes = shape_cache
            else:
                shape_cache = None
                shapes = []

            # ---- face: gaze and/or emotions -----------------------------
            # already computed in face_pool above (overlapped with detection);
            # eye_tracker is None when the MediaPipe task file is missing.

            # ---- gaze heatmap accumulation -----------------------------
            if heatmap is not None and eye_tracker is not None:
                if eye_tracker.eye_point is not None:
                    z = None
                    if (args.depth or args.depth_map) and depth is not None \
                            and getattr(depth, "last_norm", None) is not None:
                        ex, ey = eye_tracker.eye_point
                        px = int(np.clip(ex, 0, fw - 1))
                        py = int(np.clip(ey, 0, fh - 1))
                        try:
                            z = float(depth.last_norm[py, px]) / 255.0
                        except (IndexError, TypeError):
                            z = None
                    if z is None and person_metres is not None:
                        z = person_metres
                    heatmap.add(eye_tracker.eye_point, z)
                if frame_idx % heatmap.viz_every == 0:
                    heatmap.refresh_overlay()

            # ---- hands / gestures --------------------------------------
            if show_hands and (frame_idx % HANDS_EVERY == 0 or last_hand is None):
                last_hand = hand_tracker.update(frame)
                if gestures is not None:
                    g = last_hand[1] if last_hand else ""
                    tip = last_hand[2] if last_hand else None
                    fired = gestures.update(g, tip)
                    if fired:
                        # make it obvious when nothing was actually sent to Windows
                        suffix = "" if gestures.active else " (dry)"
                        last_action = f"{g} -> {fired}{suffix}"
            gesture_now = last_hand[1] if last_hand else ""

            # ---- air drawing -------------------------------------------
            if drawer is not None:
                point = None
                if show_hands and last_hand is not None:
                    tip = last_hand[2]
                    if tip is not None:
                        point = (tip[0] * fw, tip[1] * fh)
                elif skeletons:
                    # no hand in view: fall back to the nose keypoint
                    kp, cf = skeletons[0]
                    if cf is None or cf[KP_NOSE] >= KEYPOINT_CONF:
                        nx, ny = float(kp[KP_NOSE][0]), float(kp[KP_NOSE][1])
                        if nx > 0 or ny > 0:
                            point = (nx, ny)
                drawer.add(point)

            # ---- background blur / removal ------------------------------
            # applied before any overlay so boxes and labels stay crisp
            if segmenter is not None and segmenter.mask is not None:
                frame = segmenter.apply(frame, bg_mode)

            # ---- draw ---------------------------------------------------
            distance_note = ""
            if trails is not None:
                trails.begin_frame()
            if speed_est is not None:
                speed_est.begin_frame()
            if show_objects:
                for name, conf, x1, y1, x2, y2, track_id in objects:
                    col = COLOR_OBJECT if conf >= args.lowconf else COLOR_WARN
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    # with --track the ID becomes part of the label: person #12 0.95
                    tag = f"{name} #{track_id}" if track_id is not None else name
                    text = f"{tag} {conf:.2f}"
                    metres = None
                    speed = None
                    # distance is computed for every person, always - it is free.
                    # A clipped box (touching the frame edge) yields None and the
                    # label simply stays as "person #1 0.92" - unless the focal was
                    # calibrated on this same framing, see person_distance().
                    if name == "person":
                        metres = person_distance((x1, y1, x2, y2), fh, args.focal,
                                                 calibrated_framing=calibrated)
                        if metres is not None:
                            text += f" | {metres:.1f}m"
                            person_metres = metres
                            distance_note = f"{metres:.1f} m"
                            if speed_est is not None:
                                speed = speed_est.update(track_id, (x1, y1, x2, y2),
                                                         metres, frame_ts, (fh, fw))
                                if speed is not None:
                                    text += f" | {speed * 3.6:.1f} km/h"
                    draw_label(frame, text, x1, y1, col)

                    if trails is not None and track_id is not None:
                        trails.update(track_id, ((x1 + x2) // 2, (y1 + y2) // 2))
                    if csv_log is not None:
                        csv_log.log(frame_ts, track_id, name, conf,
                                    (x1 + x2) / 2.0, (y1 + y2) / 2.0, metres, speed)
            if trails is not None:
                trails.end_frame()
                trails.draw(frame)
            if speed_est is not None:
                speed_est.end_frame()
            if show_pose:
                for kp, cf in skeletons:
                    draw_skeleton(frame, kp, cf)
                for (kp, cf), act in zip(skeletons, actions):
                    if act:
                        draw_label(frame, act,
                                   int(np.clip(kp[:, 0].min(), 0, fw - 1)),
                                   int(np.clip(kp[:, 1].max(), 0, fh - 1)), COLOR_ACTION)
            for label, x, y, bw, bh in shapes:
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_SHAPE, 2)
                draw_label(frame, label, x, y, COLOR_SHAPE)
            if show_eyes:
                eye_tracker.draw(frame, last_eyes)
            if show_hands and last_hand is not None:
                HandTracker.draw(frame, last_hand[0])

            # ---- depth: overlay on the frame and/or a separate window ----
            if depth is not None and frame_idx % max(args.depth_every, 1) == 0:
                try:
                    norm = depth.infer(frame)
                    if args.depth:
                        heat = DepthEstimator.colorize(norm, cv2.COLORMAP_INFERNO)
                        frame = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
                    if args.depth_map:
                        depth_vis = DepthEstimator.colorize(norm, cv2.COLORMAP_MAGMA)
                        if not args.headless:
                            cv2.imshow(DEPTH_WINDOW, depth_vis)
                except Exception as exc:                      # noqa: BLE001
                    log.warning(f"[!] depth frame failed: {exc}")
                    depth = None

            # drawing sits on top so the depth blend cannot swallow it
            if drawer is not None:
                drawer.draw(frame)

            # ---- active learning ---------------------------------------
            # (before the heatmap blend, so the saved frames stay clean)
            if saver is not None and saver.maybe_save(frame, objects):
                saved_flash = time.perf_counter()

            # ---- gaze heatmap overlay ----------------------------------
            # the coloured map is rebuilt every viz_every frames; the blend itself
            # is cheap and runs every frame so the blob follows the live video
            if heatmap is not None:
                frame = heatmap.blend(frame)

            # ---- status panel ------------------------------------------
            primary = ""
            for want in ("LYING", "HANDS UP", "SITTING", "STANDING"):
                if want in actions:
                    primary = want
                    break
            if primary:
                lines = [(primary, COLOR_ACTION)]
            elif skeletons:
                lines = [("TRACKING", (190, 190, 190))]
            else:
                lines = [("NO PERSON", (190, 190, 190))]
            if show_eyes:
                if last_eyes is None:
                    lines.append(("NO FACE", (180, 180, 180)))
                else:
                    lines.append(("LOOKING AT CAMERA" if last_eyes[0] else "LOOKING AWAY",
                                  (0, 255, 0) if last_eyes[0] else (0, 165, 255)))
            if show_emotions:
                # big, top-left, first line of the panel
                emo_colour = {
                    "HAPPY": (0, 255, 0),
                    "ANGRY": (0, 0, 255),
                    "SURPRISED": (0, 220, 255),
                    "NEUTRAL": (210, 210, 210),
                }.get(emotion_now, (180, 180, 180))
                lines.insert(0, (emotion_now or "NEUTRAL", emo_colour))
            if show_hands and gesture_now:
                lines.append((gesture_now, COLOR_HAND))
            if distance_note:
                lines.append((f"DIST {distance_note}", (150, 220, 255)))
            if depth is not None:
                lines.append(("DEPTH ON", (200, 120, 255)))
            # top-right, so it never covers the object labels in the top-left
            panel_h, _ = draw_panel(frame, lines, scale=0.62, align_right=True)

            # ---- OCR text, top-right below the panel --------------------
            if ocr is not None:
                ocr.draw(frame, top_offset=panel_h + 14)

            # ---- HUD ----------------------------------------------------
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

            parts = [f"FPS {fps:4.1f}", f"obj {len(objects)}"]
            if pose_res is not None:
                parts.append(f"pose {len(skeletons)}")
            if show_shapes:
                parts.append(f"shapes {len(shapes)}")
            if show_eyes:
                parts.append("eyes")
            if show_emotions:
                parts.append("emo")
            if args.track:
                parts.append("track")
            if ocr is not None:
                parts.append(f"ocr {ocr.runs}")
            if heatmap is not None:
                parts.append(f"heat {heatmap.count}")
            if show_hands:
                parts.append("hands")
            if drawer is not None:
                parts.append(f"draw {len(drawer.points)}")
            if depth is not None:
                parts.append("depth" if args.depth else "depth-map")
            if trails is not None:
                parts.append(f"trails {len(trails.trails)}")
            if speed_est is not None:
                parts.append("speed")
            if segmenter is not None:
                parts.append(bg_mode)
            if csv_log is not None:
                parts.append(f"csv {csv_log.rows}")
            if isinstance(cap, VideoStream) and cap.dropped:
                parts.append(f"drop {cap.dropped}")
            if saver is not None:
                parts.append(f"saved {saver.saved}")
            parts.append("OpenVINO" if det_ov else "torch")
            hud = "   ".join(parts)
            hint = ("[q] quit  [1] obj [2] pose [3] shapes [4] eyes [5] hands "
                    "[6] draw  [c] clear  [d] ms  [s] shot")
            cv2.putText(frame, hud, (8, fh - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hud, (8, fh - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, hint, (8, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hint, (8, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            if show_timing:
                cv2.putText(frame, f"nets {net_ms:.0f}ms  shapes {shape_ms:.1f}ms",
                            (8, fh - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            if last_action:
                cv2.putText(frame, f"ACTION: {last_action}", (8, fh - 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HAND, 2, cv2.LINE_AA)
            if time.perf_counter() - saved_flash < 0.6:
                cv2.putText(frame, "SAVED LOW-CONF FRAME", (8, fh - 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARN, 2, cv2.LINE_AA)

            if args.headless:
                frame_times.append(dt * 1000.0)
                frame_idx += 1
                if frame_idx >= args.headless:
                    med = float(np.median(frame_times))
                    ids = [o[6] for o in objects if o[6] is not None]
                    dists = [round(m, 2) for m in
                             (person_distance((o[2], o[3], o[4], o[5]), fh, args.focal,
                                              calibrated_framing=calibrated)
                              for o in objects if o[0] == "person")
                             if m is not None]
                    trail_n = len(trails.trails) if trails else 0
                    log.info(f"[i] headless done: median={med:.0f}ms ({1000.0 / med:.1f} FPS) "
                          f"objs={len(objects)} ids={ids} dists={dists} people={len(skeletons)} "
                          f"shapes={len(shapes)} actions={actions} trails={trail_n} "
                          f"face={'yes' if last_eyes else 'no'} emotion={emotion_now or '-'} "
                          f"gesture={gesture_now or '-'} drawpts={len(drawer.points) if drawer else 0} "
                          f"ocr={ocr.runs if ocr else 0}runs text={repr(ocr.text) if ocr else ''} "
                          f"heat={heatmap.count if heatmap else 0} "
                          f"saved={saver.saved if saver else 0}")
                    break
                continue

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                log.info("[i] quit key pressed")
                break
            if key == ord("1"):
                show_objects = not show_objects
            elif key == ord("2"):
                show_pose = not show_pose
            elif key == ord("3"):
                show_shapes = not show_shapes
            elif key == ord("4"):
                if eye_tracker is not None:
                    show_eyes = not show_eyes
            elif key == ord("5"):
                if hand_tracker is not None:
                    show_hands = not show_hands
            elif key == ord("6"):
                if drawer is not None:
                    drawer.clear()
            elif key == ord("c"):
                if drawer is not None:
                    drawer.clear()
                if heatmap is not None:
                    heatmap.clear()
                log.info("[i] cleared: drawing + heatmap")
            elif key == ord("d"):
                show_timing = not show_timing
            elif key == ord("s"):
                shot = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(shot, frame)
                log.info(f"[i] saved {shot}")

            frame_idx += 1
            if frame_idx % 15 == 0:
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        log.info("[i] window closed")
                        break
                except cv2.error:
                    break
    except KeyboardInterrupt:
        log.info("[i] interrupted")
    finally:
        pool.shutdown(wait=False)
        if saver is not None:
            saver.close()
        if ocr is not None:
            ocr.close()
        if eye_tracker is not None:
            eye_tracker.close()
        if hand_tracker is not None:
            hand_tracker.close()
        if segmenter is not None:
            segmenter.close()
        if csv_log is not None:
            csv_log.close()
            if csv_log.rows:
                log.info(f"[i] csv: {csv_log.rows} row(s) -> {csv_log.path}")
        if isinstance(cap, VideoStream):
            log.info(f"[i] async grab: {cap.frames} frame(s) grabbed, "
                  f"{cap.dropped} dropped")
        # save the final heatmap here so it is written even on Ctrl+C / exception,
        # and still before the windows go away
        if heatmap is not None:
            saved = heatmap.save(HEATMAP_OUTPUT)
            if saved:
                log.info(f"[i] heatmap: {heatmap.count} gaze point(s) -> {saved}")
            else:
                log.info("[i] heatmap: no gaze points collected, nothing saved")
            heatmap.close()
        cap.release()
        cv2.destroyAllWindows()

    if saver is not None and saver.saved:
        log.info(f"[i] active learning: {saver.saved} frame(s) in {saver.folder}")
    log.info("[i] stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
