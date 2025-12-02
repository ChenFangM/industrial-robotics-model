#!/usr/bin/env python3
"""
USB Camera Feed with Object Detection for Raspberry Pi OS

This script captures video from a USB camera and performs real-time object detection
using OpenCV's DNN module with MobileNet SSD model. Detected objects are displayed
with bounding boxes in a window.

Usage:
    python3 camera_object_detection.py

Requirements:
    - OpenCV with DNN support (pip install opencv-python)
    - NumPy (pip install numpy)
    - USB camera connected to the device

Press 'q' to quit the application.
"""

import cv2
import numpy as np
import os
import sys
import urllib.request
from urllib.error import URLError, HTTPError
import time
import threading
from typing import Tuple


# COCO class labels that MobileNet SSD was trained on
# CLASSES = [
#     "background", "aeroplane", "bicycle", "bird", "boat",
#     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#     "sofa", "train", "tvmonitor"
# ]
CLASSES = [
    "background", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Generate random colors for each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)


def download_model_files():
    """
    Downloads the MobileNet SSD model files if they don't exist.
    Returns the paths to the prototxt and caffemodel files.
    """
    # Model file URLs
    prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

    # Local file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prototxt_path = os.path.join(script_dir, "deploy.prototxt")
    model_path = os.path.join(script_dir, "mobilenet_iter_73000.caffemodel")

    try:
        # Download prototxt if not exists
        if not os.path.exists(prototxt_path):
            print("Downloading MobileNet SSD prototxt file...")
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
            print("Downloaded deploy.prototxt")

        # Download caffemodel if not exists
        if not os.path.exists(model_path):
            print("Downloading MobileNet SSD model file (this may take a while)...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Downloaded mobilenet_iter_73000.caffemodel")
    except HTTPError as e:
        print(f"Error: HTTP error while downloading model files: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)
    except URLError as e:
        print(f"Error: Network error while downloading model files: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)
    except OSError as e:
        print(f"Error: File system error while saving model files: {e}")
        print("Please check disk space and write permissions.")
        sys.exit(1)

    return prototxt_path, model_path


def initialize_detector():
    """
    Initialize the MobileNet SSD object detector.
    Returns the neural network model.
    """
    prototxt_path, model_path = download_model_files()

    print("Loading MobileNet SSD model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Use CPU backend (compatible with Raspberry Pi)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("Model loaded successfully!")
    return net


def detect_objects(frame, net, confidence_threshold=0.5):
    """
    Perform object detection on a frame.

    Args:
        frame: Input image/frame from camera
        net: The neural network model
        confidence_threshold: Minimum confidence for detection

    Returns:
        List of detections, each containing (class_id, confidence, box)
    """
    # Get frame dimensions
    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()

    results = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])

            # Validate class_id is within valid range
            if class_id < 0 or class_id >= len(CLASSES):
                continue

            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            start_x, start_y, end_x, end_y = box.astype("int")

            # Ensure coordinates are within frame bounds
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(width, end_x)
            end_y = min(height, end_y)

            results.append((class_id, confidence, (start_x, start_y, end_x, end_y)))

    return results


def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on the frame.

    Args:
        frame: Input image/frame
        detections: List of detections from detect_objects()

    Returns:
        Frame with drawn detections
    """
    for class_id, confidence, box in detections:
        # Skip invalid class IDs
        if class_id < 0 or class_id >= len(CLASSES):
            continue

        start_x, start_y, end_x, end_y = box

        # Get color and label for this class
        color = [int(c) for c in COLORS[class_id]]
        label = f"{CLASSES[class_id]}: {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        # Draw label background
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y = max(start_y, label_size[1] + 10)
        cv2.rectangle(
            frame,
            (start_x, y - label_size[1] - 10),
            (start_x + label_size[0], y + baseline - 10),
            color,
            cv2.FILLED
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (start_x, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    return frame


def main():
    """
    Main function to run the camera feed with object detection.
    """
    print("=" * 60)
    print("USB Camera Feed with Object Detection")
    print("=" * 60)

    # Initialize the object detector
    net = initialize_detector()

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
    window_name = "Object Detection - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nStarting camera feed...")
    print("Press 'q' to quit")
    print("-" * 60)
    # Capture one annotated frame every `interval_seconds` and show it for the whole interval
    interval_seconds = 3

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
            if not ret:
                # don't abort immediately; retry a few times to handle transient driver delays
                print("Warning: failed to grab frame; retrying...")
                time.sleep(0.1)
                continue

            # Run detection on the captured frame
            detections = detect_objects(frame, net, confidence_threshold=0.5)

            # Draw detections on a copy so we can overlay countdown separately
            annotated = draw_detections(frame.copy(), detections)

            # Show the annotated frame and display a countdown until next capture
            start_t = time.time()
            while True:
                elapsed = time.time() - start_t
                remaining = int(max(0, interval_seconds - elapsed))

                # Prepare display copy with countdown text
                display = annotated.copy()

                # Draw countdown (large, top-left)
                countdown_text = f"Next capture in: {remaining + 1}s"
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
