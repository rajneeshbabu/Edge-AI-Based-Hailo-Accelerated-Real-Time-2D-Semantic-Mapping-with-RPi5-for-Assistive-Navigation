#!/usr/bin/env python3
"""
Edge-SLAM Navigation System
RPi5 + Hailo-8 AI HAT + Nicla Vision IMU + Raspberry Pi Camera

Main features:
- Hailo HEF inference for YOLOv8n-pruned model
- Picamera2 camera capture
- Nicla Vision yaw angle over USB serial
- Robust IMU reader: does not depend on NICLA_IMU_READY handshake
- Terminal logging of objects, confidence, bbox, FPS, IMU angle
- Semantic map with angle, distance estimate, confidence, frame count
- Optional OpenCV display window
"""

import os
import sys
import cv2
import time
import queue
import serial
import threading
import numpy as np
import pyttsx3

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
from picamera2 import Picamera2


# ============================================================
# CONFIG
# ============================================================

HEF_PATH = os.path.expanduser("~/edge_nav/models/yolov8n_pruned.hef")

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200

IMG_SIZE = 640
CONF_THRESH = 0.40
IOU_THRESH = 0.45

CAM_WIDTH = 640
CAM_HEIGHT = 480

SHOW_WINDOW = True          # Set False if running headless over SSH
ENABLE_AUDIO = True         # Set False if pyttsx3/audio gives issues

MAP_PRINT_INTERVAL = 30     # Print semantic map every N frames
LOG_INTERVAL = 10           # Print FPS/status every N frames
ANNOUNCE_INTERVAL = 5.0
MAX_OBJ_AGE = 10.0

# Horizontal camera FOV approximation used for semantic angle estimate.
# angle_offset = normalized_x_offset * HALF_FOV_DEG
HALF_FOV_DEG = 30.0


COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
    'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# Classes useful for navigation.
# Remove this filtering in YOLOv8DFLDecoder if you want all 80 classes.
VI_CLASSES = {
    0,    # person
    24,   # backpack
    26,   # handbag
    28,   # suitcase
    39,   # bottle
    41,   # cup
    42,   # fork
    43,   # knife
    44,   # spoon
    45,   # bowl
    56,   # chair
    57,   # couch
    59,   # bed
    60,   # dining table
    63,   # laptop
    64,   # mouse
    65,   # remote
    66,   # keyboard
    67,   # cell phone
    73,   # book
}


# ============================================================
# YOLOv8 DFL POST-PROCESSING
# ============================================================

def softmax(x, axis=-1):
    x = x.astype(np.float32)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(x):
    x = x.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-x))


class YOLOv8DFLDecoder:
    """
    Decodes raw YOLOv8 native head outputs from Hailo.

    Expected HEF outputs:
    - 3 box outputs: channels = 64  = 4 sides × 16 DFL bins
    - 3 cls outputs: channels = 80  = COCO classes
    """

    def __init__(self, img_size=640, num_classes=80, reg_max=16, strides=(8, 16, 32)):
        self.img_size = img_size
        self.nc = num_classes
        self.reg_max = reg_max
        self.strides = strides

        self.grids = []
        for stride in strides:
            grid_size = img_size // stride
            yv, xv = np.meshgrid(
                np.arange(grid_size),
                np.arange(grid_size),
                indexing="ij"
            )
            grid = np.stack([xv, yv], axis=-1).astype(np.float32)
            self.grids.append(grid)

        self.bins = np.arange(reg_max).astype(np.float32)

    def decode_one_scale(self, box_raw, cls_raw, grid, stride):
        """
        box_raw: [H, W, 64]
        cls_raw: [H, W, 80]
        """

        H, W, _ = box_raw.shape
        n = H * W

        box_dfl = box_raw.reshape(H, W, 4, self.reg_max)
        box_dfl = softmax(box_dfl, axis=-1)
        box_dist = (box_dfl * self.bins).sum(axis=-1)

        cx = (grid[..., 0] + 0.5) * stride
        cy = (grid[..., 1] + 0.5) * stride

        x1 = cx - box_dist[..., 0] * stride
        y1 = cy - box_dist[..., 1] * stride
        x2 = cx + box_dist[..., 2] * stride
        y2 = cy + box_dist[..., 3] * stride

        boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(n, 4)
        scores = sigmoid(cls_raw).reshape(n, self.nc)

        return boxes, scores

    def __call__(self, raw_outputs, conf_thr, iou_thr, orig_h, orig_w):
        sorted_by_size = sorted(
            raw_outputs.items(),
            key=lambda kv: kv[1].shape[1],
            reverse=True,
        )

        box_outs = []
        cls_outs = []

        for name, tensor in sorted_by_size:
            channels = tensor.shape[-1]

            if channels == 4 * self.reg_max:
                box_outs.append((name, tensor))
            elif channels == self.nc:
                cls_outs.append((name, tensor))

        if len(box_outs) != 3 or len(cls_outs) != 3:
            raise RuntimeError(
                f"Expected 3 box outputs and 3 class outputs, "
                f"got {len(box_outs)} box and {len(cls_outs)} class outputs."
            )

        all_boxes = []
        all_scores = []

        for i, stride in enumerate(self.strides):
            box_t = box_outs[i][1][0]
            cls_t = cls_outs[i][1][0]
            grid = self.grids[i]

            boxes, scores = self.decode_one_scale(box_t, cls_t, grid, stride)
            all_boxes.append(boxes)
            all_scores.append(scores)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), class_ids]

        # Use this line for selected navigation classes only:
        mask = (confs > conf_thr) & np.isin(class_ids, list(VI_CLASSES))

        # Use this line instead if you want all 80 COCO classes:
        # mask = (confs > conf_thr)

        if not mask.any():
            return []

        boxes = boxes[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        sx = orig_w / self.img_size
        sy = orig_h / self.img_size

        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)

        detections = []

        for cid in np.unique(class_ids):
            m = class_ids == cid

            xywh = np.stack([
                boxes[m, 0],
                boxes[m, 1],
                boxes[m, 2] - boxes[m, 0],
                boxes[m, 3] - boxes[m, 1],
            ], axis=1).tolist()

            keep = cv2.dnn.NMSBoxes(
                xywh,
                confs[m].tolist(),
                conf_thr,
                iou_thr
            )

            if len(keep):
                for k in np.array(keep).flatten():
                    detections.append([
                        float(boxes[m][k, 0]),
                        float(boxes[m][k, 1]),
                        float(boxes[m][k, 2]),
                        float(boxes[m][k, 3]),
                        float(confs[m][k]),
                        int(cid),
                    ])

        return detections


# ============================================================
# HAILO INFERENCE ENGINE
# ============================================================

class HailoDetector:
    def __init__(self, hef_path):
        from hailo_platform import (
            HEF,
            VDevice,
            HailoStreamInterface,
            InferVStreams,
            ConfigureParams,
            InputVStreamParams,
            OutputVStreamParams,
            FormatType,
        )

        hef_path = os.path.expanduser(hef_path)
        if not Path(hef_path).exists():
            raise FileNotFoundError(f"HEF file not found: {hef_path}")

        self._InferVStreams = InferVStreams

        self.hef = HEF(hef_path)
        self.device = VDevice()

        cfg = ConfigureParams.create_from_hef(
            self.hef,
            interface=HailoStreamInterface.PCIe
        )

        self.ng = self.device.configure(self.hef, cfg)[0]
        self.ng_params = self.ng.create_params()

        self.in_params = InputVStreamParams.make(
            self.ng,
            format_type=FormatType.UINT8
        )

        self.out_params = OutputVStreamParams.make(
            self.ng,
            format_type=FormatType.FLOAT32
        )

        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_infos = self.hef.get_output_vstream_infos()

        print(f"[Hailo] HEF: {hef_path}", flush=True)
        print(f"[Hailo] Input:  {self.in_info.name}  shape={self.in_info.shape}", flush=True)
        print(f"[Hailo] Outputs ({len(self.out_infos)}):", flush=True)
        for o in self.out_infos:
            print(f"          {o.name}  shape={o.shape}", flush=True)

        self.decoder = YOLOv8DFLDecoder(
            img_size=IMG_SIZE,
            num_classes=80,
            reg_max=16,
        )

    def preprocess(self, bgr):
        img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0).astype(np.uint8)
        return img

    def infer(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        inp = self.preprocess(bgr_frame)

        with self._InferVStreams(self.ng, self.in_params, self.out_params) as pipe:
            with self.ng.activate(self.ng_params):
                raw = pipe.infer({self.in_info.name: inp})

        detections = self.decoder(
            raw,
            CONF_THRESH,
            IOU_THRESH,
            h,
            w,
        )

        return detections


# ============================================================
# ROBUST NICLA IMU READER
# ============================================================

class NiclaIMUReader:
    """
    Reads yaw angle from Nicla Vision over USB serial.

    Expected Nicla line:
        A:<angle_deg>

    Example:
        A:156.68

    Important:
    - This does NOT depend on NICLA_IMU_READY.
    - Handshake is optional.
    - If Nicla resets after serial open, this waits and keeps reading.
    - If the serial port disconnects, it tries to reconnect.
    """

    def __init__(self, port="/dev/ttyACM0", baud=115200, debug=True):
        self.port = port
        self.baud = baud
        self.debug = debug

        self.angle = 0.0
        self.lines_seen = 0
        self.valid_lines = 0
        self.last_update_t = 0.0
        self.connected = False

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        print(f"[IMU] Starting reader on {self.port} @ {self.baud}", flush=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def get_angle(self):
        with self._lock:
            return self.angle

    def status(self):
        with self._lock:
            age = None
            if self.last_update_t > 0:
                age = time.time() - self.last_update_t

            return {
                "connected": self.connected,
                "angle": self.angle,
                "lines_seen": self.lines_seen,
                "valid_lines": self.valid_lines,
                "last_update_age_s": age,
            }

    def _open_serial(self):
        ser = serial.Serial(
            self.port,
            self.baud,
            timeout=0.5,
            write_timeout=0.5,
        )

        # For USB CDC devices this can help stabilize after opening.
        try:
            ser.dtr = True
            ser.rts = False
        except Exception:
            pass

        return ser

    def _parse_line(self, line):
        line = line.strip()

        if not line:
            return False

        if line == "NICLA_IMU_READY":
            print("[IMU] Nicla handshake seen", flush=True)
            return False

        if line.startswith("A:"):
            try:
                value = float(line[2:])
            except ValueError:
                return False

            with self._lock:
                self.angle = value
                self.valid_lines += 1
                self.last_update_t = time.time()

            return True

        return False

    def _worker(self):
        raw_print_count = 0

        while not self._stop.is_set():
            ser = None

            try:
                ser = self._open_serial()

                with self._lock:
                    self.connected = True

                print("[IMU] Serial opened successfully", flush=True)
                print("[IMU] Waiting for A:<angle> stream from Nicla...", flush=True)

                # Opening serial may reset the Nicla.
                # Your Nicla code calibrates for around 3 seconds.
                # So do not declare failure early.
                start_t = time.time()

                while not self._stop.is_set():
                    try:
                        raw = ser.readline()
                    except serial.SerialException as e:
                        print(f"[IMU] Serial read error: {e}", flush=True)
                        break

                    try:
                        line = raw.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue

                    if not line:
                        # Print a warning only if no valid data for long time.
                        if time.time() - start_t > 8.0:
                            st = self.status()
                            if st["valid_lines"] == 0 and raw_print_count < 5:
                                print("[IMU] Still waiting for valid A:<angle> data...", flush=True)
                                raw_print_count += 1
                                start_t = time.time()
                        continue

                    with self._lock:
                        self.lines_seen += 1

                    if self.debug and raw_print_count < 15:
                        print(f"[IMU RAW] {line!r}", flush=True)
                        raw_print_count += 1

                    ok = self._parse_line(line)

                    if ok:
                        st = self.status()

                        if st["valid_lines"] <= 5:
                            print(f"[IMU] First angle update: {st['angle']:.2f} deg", flush=True)

                        if st["valid_lines"] % 50 == 0:
                            print(f"[IMU] angle={st['angle']:.2f} deg, valid_lines={st['valid_lines']}", flush=True)

            except serial.SerialException as e:
                with self._lock:
                    self.connected = False

                print(f"[IMU] Serial open failed: {e}", flush=True)
                print("[IMU] Retrying in 2 seconds...", flush=True)
                time.sleep(2.0)

            finally:
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass

                with self._lock:
                    self.connected = False


# ============================================================
# SEMANTIC MAP
# ============================================================

@dataclass
class ObjectEntry:
    angle: float
    confidence: float
    distance_m: float
    timestamp: float
    frame_count: int = 0


class SemanticMap:
    REFERENCE_PX_AT_1M = {
        "person": 400,
        "chair": 200,
        "couch": 180,
        "dining table": 150,
        "bed": 160,
    }

    DEFAULT_REF_PX = 200

    def __init__(self):
        self.objects: Dict[str, ObjectEntry] = {}
        self._lock = threading.Lock()

    def update(self, class_id, conf, bbox, frame_w, frame_h, imu_angle):
        name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"class_{class_id}"

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        bbox_h = max(1.0, y2 - y1)

        offset = (cx - frame_w / 2.0) / (frame_w / 2.0)
        angle_offset = offset * HALF_FOV_DEG
        abs_angle = (imu_angle + angle_offset) % 360.0

        ref_px = self.REFERENCE_PX_AT_1M.get(name, self.DEFAULT_REF_PX)
        dist_m = max(0.3, ref_px / bbox_h)

        now = time.time()

        with self._lock:
            existing = self.objects.get(name)

            if existing:
                existing.angle = 0.7 * existing.angle + 0.3 * abs_angle
                existing.confidence = max(existing.confidence, conf)
                existing.distance_m = 0.7 * existing.distance_m + 0.3 * dist_m
                existing.timestamp = now
                existing.frame_count += 1
            else:
                self.objects[name] = ObjectEntry(
                    angle=abs_angle,
                    confidence=conf,
                    distance_m=dist_m,
                    timestamp=now,
                    frame_count=1,
                )

    def prune_stale(self):
        now = time.time()

        with self._lock:
            stale = [
                name for name, obj in self.objects.items()
                if now - obj.timestamp > MAX_OBJ_AGE
            ]

            for name in stale:
                del self.objects[name]

    def nearest_obstacle(self, current_angle):
        with self._lock:
            front_objects = {}

            for name, obj in self.objects.items():
                diff = ((obj.angle - current_angle + 180) % 360) - 180
                if abs(diff) < 45:
                    front_objects[name] = obj

            if not front_objects:
                return None

            name, closest = min(
                front_objects.items(),
                key=lambda item: item[1].distance_m
            )

            if closest.distance_m < 1.5:
                diff = ((closest.angle - current_angle + 180) % 360) - 180
                side = "right" if diff > 0 else "left"
                return f"Warning: {name} {closest.distance_m:.1f} metres to your {side}."

        return None

    def print_map(self):
        print("═══ SEMANTIC MAP ═══", flush=True)

        with self._lock:
            if not self.objects:
                print("  (empty)", flush=True)

            for name, obj in sorted(self.objects.items(), key=lambda item: item[1].distance_m):
                print(
                    f"  {name:<20} "
                    f"{obj.angle:6.1f}° | "
                    f"{obj.distance_m:4.1f} m | "
                    f"conf={obj.confidence:5.1%} | "
                    f"frames={obj.frame_count}",
                    flush=True,
                )

        print("════════════════════", flush=True)


# ============================================================
# AUDIO ANNOUNCER
# ============================================================

class AudioAnnouncer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._q = queue.Queue(maxsize=3)
        self._last_cue: Dict[str, float] = {}

        if not self.enabled:
            print("[Audio] Disabled", flush=True)
            return

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 160)
        except Exception as e:
            print(f"[Audio] TTS init failed: {e}", flush=True)
            print("[Audio] Continuing without speech output", flush=True)
            self.enabled = False
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def announce(self, text, key="default"):
        if not self.enabled:
            return

        now = time.time()
        last = self._last_cue.get(key, 0)

        if now - last < ANNOUNCE_INTERVAL:
            return

        self._last_cue[key] = now

        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass

    def _worker(self):
        while True:
            text = self._q.get()

            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass


# ============================================================
# CAMERA
# ============================================================

class PiCameraSource:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.picam2 = None

    def start(self):
        self.picam2 = Picamera2()

        config = self.picam2.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888",
            }
        )

        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1.0)

        print(f"[Camera] Picamera2 ready: {self.width}x{self.height}", flush=True)

    def read(self):
        frame_rgb = self.picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def stop(self):
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cid = det
        name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else f"class_{cid}"

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{name} {conf:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        y_text = max(y1, th + 8)

        cv2.rectangle(
            frame,
            (x1, y_text - th - 6),
            (x1 + tw + 4, y_text + baseline),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1 + 2, y_text - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

def draw_topdown_map(sem_map, current_angle, canvas_size=520, max_range_m=6.0):
    """
    Draws a simple 2D top-down semantic map.
    Center = user/RPi position.
    Object position is computed from stored object angle + estimated distance.
    0 degrees is drawn upward, angles increase clockwise.
    """

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    center = canvas_size // 2
    scale = (canvas_size * 0.42) / max_range_m

    # Background
    canvas[:] = (20, 20, 20)

    # Range rings
    for r_m in range(1, int(max_range_m) + 1):
        r_px = int(r_m * scale)
        cv2.circle(canvas, (center, center), r_px, (70, 70, 70), 1)
        cv2.putText(
            canvas,
            f"{r_m}m",
            (center + 5, center - r_px - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

    # Cross axes
    cv2.line(canvas, (center, 20), (center, canvas_size - 20), (50, 50, 50), 1)
    cv2.line(canvas, (20, center), (canvas_size - 20, center), (50, 50, 50), 1)

    # User position
    cv2.circle(canvas, (center, center), 8, (0, 255, 255), -1)
    cv2.putText(
        canvas,
        "USER",
        (center + 10, center + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )

    # Heading arrow
    theta = np.deg2rad(current_angle)
    arrow_len = 55
    hx = int(center + arrow_len * np.sin(theta))
    hy = int(center - arrow_len * np.cos(theta))
    cv2.arrowedLine(
        canvas,
        (center, center),
        (hx, hy),
        (0, 255, 255),
        2,
        tipLength=0.25,
    )

    cv2.putText(
        canvas,
        f"Heading: {current_angle:.1f} deg",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # Draw objects
    with sem_map._lock:
        items = list(sem_map.objects.items())

    for name, obj in items:
        dist = min(obj.distance_m, max_range_m)
        angle = obj.angle

        theta = np.deg2rad(angle)
        r_px = dist * scale

        x = int(center + r_px * np.sin(theta))
        y = int(center - r_px * np.cos(theta))

        # Color by distance
        if obj.distance_m < 1.0:
            color = (0, 0, 255)       # close object
        elif obj.distance_m < 2.5:
            color = (0, 165, 255)     # medium
        else:
            color = (0, 255, 0)       # far

        cv2.circle(canvas, (x, y), 7, color, -1)
        cv2.line(canvas, (center, center), (x, y), color, 1)

        label = f"{name} {obj.distance_m:.1f}m {obj.confidence:.0%}"
        cv2.putText(
            canvas,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )

    # Legend
    cv2.putText(canvas, "Top-down semantic map", (15, canvas_size - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(canvas, "Red <1m | Orange <2.5m | Green far",
                (15, canvas_size - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return canvas

def print_detections(frame_count, imu_angle, inf_ms, detections):
    if not detections:
        return

    print(
        f"\n[FRAME {frame_count}] "
        f"IMU={imu_angle:.1f}° | "
        f"inference={inf_ms:.1f} ms | "
        f"detections={len(detections)}",
        flush=True,
    )

    for det in detections:
        x1, y1, x2, y2, conf, cid = det
        name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else f"class_{cid}"

        print(
            f"  [DETECTED] {name:<15} "
            f"conf={conf*100:5.1f}% "
            f"bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
            flush=True,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Initialising Edge-SLAM Navigation System...", flush=True)

    print("[System] Starting IMU first...", flush=True)
    imu = NiclaIMUReader(SERIAL_PORT, SERIAL_BAUD, debug=True)
    imu.start()

    # Give the IMU thread time to open serial and catch initial data.
    # It will continue updating in the background even after this.
    time.sleep(8.0)
    st = imu.status()
    print(
        f"[IMU] Status after init: "
        f"connected={st['connected']}, "
        f"angle={st['angle']:.2f}°, "
        f"lines_seen={st['lines_seen']}, "
        f"valid_lines={st['valid_lines']}",
        flush=True,
    )

    print("[System] Initialising Hailo detector...", flush=True)
    detector = HailoDetector(HEF_PATH)
    print("✅ Hailo detector ready", flush=True)

    print("[System] Starting camera...", flush=True)
    camera = PiCameraSource(CAM_WIDTH, CAM_HEIGHT)
    camera.start()

    sem_map = SemanticMap()
    audio = AudioAnnouncer(enabled=ENABLE_AUDIO)

    fps_log = []
    frame_count = 0

    audio.announce("Navigation system ready.", "startup")

    try:
        while True:
            frame = camera.read()
            h, w = frame.shape[:2]

            imu_angle = imu.get_angle()

            t0 = time.time()
            detections = detector.infer(frame)
            inf_ms = (time.time() - t0) * 1000.0

            print_detections(frame_count, imu_angle, inf_ms, detections)

            for det in detections:
                x1, y1, x2, y2, conf, cid = det
                sem_map.update(cid, conf, [x1, y1, x2, y2], w, h, imu_angle)

            frame_count += 1

            if frame_count % MAP_PRINT_INTERVAL == 0:
                sem_map.prune_stale()
                sem_map.print_map()

            warn = sem_map.nearest_obstacle(imu_angle)
            if warn:
                audio.announce(warn, "obstacle")

            fps_log.append(1000.0 / max(inf_ms, 1e-6))
            fps = float(np.mean(fps_log[-20:]))

            if frame_count % LOG_INTERVAL == 0:
                imu_st = imu.status()
                age = imu_st["last_update_age_s"]
                age_str = "no updates" if age is None else f"{age:.1f}s ago"

                print(
                    f"[RUNNING] frame={frame_count} "
                    f"inf={inf_ms:.1f} ms "
                    f"fps={fps:.2f} "
                    f"imu={imu_angle:.1f}° "
                    f"imu_valid={imu_st['valid_lines']} "
                    f"last_imu={age_str} "
                    f"detections={len(detections)}",
                    flush=True,
                )

            draw_detections(frame, detections)
            map_img = draw_topdown_map(sem_map, imu_angle, canvas_size=520, max_range_m=6.0)

            cv2.putText(
                frame,
                f"FPS:{fps:.1f} | IMU:{imu_angle:.0f} deg | objs:{len(detections)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )

            if SHOW_WINDOW:
                cv2.imshow("Edge-SLAM Navigation", frame)
                cv2.imshow("2D Semantic Map", map_img)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("m"):
                    sem_map.print_map()
                elif key == ord("i"):
                    print(f"[IMU manual] {imu.status()}", flush=True)

    except KeyboardInterrupt:
        print("\n[System] Ctrl+C received", flush=True)

    finally:
        print("[System] Shutting down...", flush=True)
        camera.stop()
        imu.stop()

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        print("System shutdown.", flush=True)


if __name__ == "__main__":
    main()