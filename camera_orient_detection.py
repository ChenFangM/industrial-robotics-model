#!/usr/bin/env python3
"""
USB Camera Feed with Orientation Detection (Angle of Rectangle)

This script captures video from a USB camera and, instead of object detection,
detects the orientation (angle of rotation) of the largest rectangular object
in view. The UI remains the same: one annotated frame is shown every
`interval_seconds` with a countdown overlay.

Usage:
    python3 camera_orient_detection.py

Requirements:
    - OpenCV (pip install opencv-python)
    - NumPy (pip install numpy)
    - USB camera connected to the device

Press 'q' to quit the application.
"""

import cv2
import numpy as np
import sys
import time
import threading
import os
import urllib.request
from typing import Tuple, Optional, List
try:
    import onnxruntime as ort
except ImportError:
    ort = None


RECT_COLOR = (0, 200, 255)
MODEL_URL = "https://github.com/yangxuce/YOLOv5-OBB/releases/download/v1.0/yolov5s-obb.onnx"  # Example URL (replace if different)
MODEL_FILENAME = "yolov5s-obb.onnx"


def download_obb_model(model_path: str) -> str:
    """Download YOLOv5-OBB ONNX model if missing."""
    if os.path.exists(model_path):
        return model_path
    print(f"Downloading YOLOv5-OBB model to {model_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Proceeding without model (will fallback to contour orientation).")
        return ""
    print("Model downloaded.")
    return model_path


def create_session(model_path: str):
    """Create ONNX Runtime inference session."""
    if ort is None or not model_path:
        return None
    providers = ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
        return sess
    except Exception as e:
        print(f"Failed to create ONNX Runtime session: {e}")
        return None


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding."""
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def obb_nms(dets: List[Tuple[int, float, float, float, float, float, float]], iou_thresh=0.45) -> List[int]:
    """Very simple NMS on axis-aligned bbox of oriented boxes.
    dets: list of (idx, score, cx, cy, w, h, angle_deg)
    Returns indices kept.
    """
    if not dets:
        return []
    boxes = []
    scores = []
    for i, score, cx, cy, w, h, ang in dets:
        # Axis-aligned box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
    boxes = np.array(boxes)
    scores = np.array(scores)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def detect_yolov5_obb(frame: np.ndarray, sess, conf_thresh=0.25, max_det=50) -> List[Tuple[int, float, float, float, float, float]]:
    """Run YOLOv5-OBB inference and return oriented boxes.
    Returns list of (class_id, confidence, cx, cy, w, h, angle_deg).
    If session is None, returns empty list.
    """
    if sess is None:
        return []
    input_name = sess.get_inputs()[0].name
    img0 = frame
    img, r, (pad_w, pad_h) = letterbox(img0, new_shape=(640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_float = np.transpose(img_float, (2, 0, 1))  # CHW
    img_float = np.expand_dims(img_float, 0)

    outputs = sess.run(None, {input_name: img_float})
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]

    num_cols = pred.shape[1]
    # Assume format: cx, cy, w, h, angle, obj_conf, class_probs...
    # So base = 6 then classes = num_cols - 6
    if num_cols < 7:
        return []
    num_classes = num_cols - 6
    results_tmp = []
    for row in pred:
        cx, cy, w, h, angle_raw, obj_conf = row[:6]
        class_scores = row[6:]
        if obj_conf < conf_thresh:
            continue
        c_id = int(np.argmax(class_scores))
        cls_conf = class_scores[c_id]
        conf = obj_conf * cls_conf
        if conf < conf_thresh:
            continue
        # Scale back to original image coordinates
        cx = (cx - pad_w) / r
        cy = (cy - pad_h) / r
        w /= r
        h /= r
        # Angle: if appears radian (|angle| < 6.3) convert to degrees
        angle_deg = angle_raw if abs(angle_raw) > 6.3 else angle_raw * 180.0 / np.pi
        results_tmp.append((c_id, conf, float(cx), float(cy), float(w), float(h), float(angle_deg)))
        if len(results_tmp) >= max_det:
            break

    # NMS
    dets_indexed = [(i, d[1], d[2], d[3], d[4], d[5], d[6]) for i, d in enumerate(results_tmp)]
    keep_indices = obb_nms(dets_indexed)
    final = [results_tmp[i] for i in keep_indices]
    return final


def oriented_box_points(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    rect = ((cx, cy), (w, h), angle_deg)
    box = cv2.boxPoints(rect)
    return box.astype(np.int32)


def draw_obb_detections(frame: np.ndarray, detections: List[Tuple[int, float, float, float, float, float, float]]) -> np.ndarray:
    for class_id, conf, cx, cy, w, h, ang in detections:
        box = oriented_box_points(cx, cy, w, h, ang)
        cv2.drawContours(frame, [box], 0, RECT_COLOR, 2)
        x_coords = box[:, 0]
        y_coords = box[:, 1]
        tx = int(np.min(x_coords))
        ty = int(np.min(y_coords)) - 10
        ty = max(15, ty)
        label = f"Angle: {ang:.1f} deg | Conf: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(frame, (tx - 4, ty - lh - 4), (tx + lw + 4, ty + baseline + 4), RECT_COLOR, cv2.FILLED)
        cv2.putText(frame, label, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return frame


def main():
    """Run the camera feed with rectangle orientation detection."""
    print("=" * 60)
    print("USB Camera Feed with YOLOv5-OBB Orientation Detection")
    print("=" * 60)

    # Initialize the camera
    # Try different camera indices if default doesn't work
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        print("Trying alternative camera indices...")

        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera found at index {i}")
                break
            cap.release()
        else:
            print("Error: Could not find any available camera")
            print("Please check that your USB camera is connected")
            sys.exit(1)

    # Set camera properties for better performance on Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Get actual camera properties
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera resolution: {int(actual_width)}x{int(actual_height)} @ {actual_fps} FPS")

    # Create window
    window_name = "Orientation Detection - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nStarting camera feed...")
    print("Press 'q' to quit")
    print("-" * 60)
    # Capture one annotated frame every `interval_seconds` and show it for the whole interval
    interval_seconds = 3.0

    # Load YOLOv5-OBB model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)
    model_path = download_obb_model(model_path)
    sess = create_session(model_path)
    if sess is None:
        print("Warning: YOLOv5-OBB session unavailable. Falling back to contour-based orientation (largest rectangle).")

    # Wrap capture in a background reader to minimize latency from driver buffers
    async_cap = VideoCaptureAsync(cap).start()

    # Wait briefly for the background reader to populate the first frame
    startup_wait = 2.0
    t0 = time.time()
    while True:
        ok, _ = async_cap.read()
        if ok:
            break
        if time.time() - t0 > startup_wait:
            print("Warning: no frames received from camera after startup wait; continuing and retrying.")
            break
        time.sleep(0.05)

    try:
        while True:
            # Capture the most recent frame from background reader
            ret, frame = async_cap.read()

            # test_frame = np.full((400, 400, 3), 255, dtype=np.uint8)
            # cv2.rectangle(test_frame, (50, 50), (350, 200), (0, 0, 0), thickness=cv2.FILLED)
            # cv2.imwrite("test_rect.png", test_frame)
            # ret = True
            # frame = test_frame

            if not ret:
                # don't abort immediately; retry a few times to handle transient driver delays
                print("Warning: failed to grab frame; retrying...")
                time.sleep(0.1)
                continue

            # Run YOLOv5-OBB; fallback to contour if no session or no detections
            detections = detect_yolov5_obb(frame, sess)
            if not detections:
                # fallback single detection via contours
                fallback_angle, fallback_box = None, None
                # Simple fallback using minAreaRect largest rectangle
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                edges = cv2.Canny(blur, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (w, h), a = rect
                    if w > 0 and h > 0:
                        if w < h:
                            a += 90
                        fallback_angle = a % 180.0
                        fallback_box = cv2.boxPoints(rect).astype(np.int32)
                annotated = frame.copy()
                if fallback_box is not None and fallback_angle is not None:
                    cv2.drawContours(annotated, [fallback_box], 0, RECT_COLOR, 2)
                    label = f"Angle: {fallback_angle:.1f} deg"
                    x_coords = fallback_box[:,0]; y_coords = fallback_box[:,1]
                    tx = int(np.min(x_coords)); ty = int(np.min(y_coords)) - 10
                    ty = max(15, ty)
                    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; thickness = 2
                    (lw, lh), base = cv2.getTextSize(label, font, font_scale, thickness)
                    cv2.rectangle(annotated, (tx-4, ty-lh-4), (tx+lw+4, ty+base+4), RECT_COLOR, cv2.FILLED)
                    cv2.putText(annotated, label, (tx, ty), font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
            else:
                annotated = draw_obb_detections(frame.copy(), detections)

            # Show the annotated frame and display a countdown until next capture
            start_t = time.time()
            while True:
                elapsed = time.time() - start_t
                remaining = round(max(0, interval_seconds - elapsed), 1)

                # Prepare display copy with countdown text
                display = annotated.copy()

                # Draw countdown (large, top-left)
                countdown_text = f"Next capture in: {remaining}s"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                text_size, baseline = cv2.getTextSize(countdown_text, font, font_scale, thickness)
                text_org = (10, text_size[1] + 10)

                # Position text in bottom-right corner
                h, w = display.shape[:2]
                x = w - text_size[0] - 10
                y = h - 10
                text_org = (x, y)

                # Background rectangle coordinates (clamped to image bounds)
                tl = (max(0, x - 5), max(0, y - text_size[1] - 5))
                br = (min(w, x + text_size[0] + 5), min(h, y + baseline + 5))

                cv2.rectangle(display, tl, br, (0, 0, 0), cv2.FILLED)
                cv2.putText(display, countdown_text, text_org, font, font_scale, (255, 255, 255), thickness)

                # Show the frame
                cv2.imshow(window_name, display)

                # Wait a short time so the window stays responsive; user can press 'q' to quit
                key = cv2.waitKey(200) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting...")
                    raise KeyboardInterrupt

                # Break when interval elapsed and capture next frame
                if elapsed >= interval_seconds:
                    break
    except KeyboardInterrupt:
        pass

    # Cleanup
    async_cap.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")


class VideoCaptureAsync:
    """Background thread that constantly reads frames from a cv2.VideoCapture
    and keeps the most recent frame available via `read()`.

    This reduces latency caused by internal driver buffers since the latest
    frame is always stored in memory and returned immediately on request.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.grabbed = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = True
        self.thread = None

    def start(self):
        if not self.stopped:
            return self
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame

    def read(self) -> Tuple[bool, any]:
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            # return a copy to avoid race conditions with the background thread
            return True, self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
    # (main loop previously here was moved into `main()` so class body stays clean)


if __name__ == "__main__":
    main()
