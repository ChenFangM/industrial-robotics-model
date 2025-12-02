#!/usr/bin/env python3
"""
USB Camera Feed with Object Detection for Raspberry Pi OS

This script captures video from a USB camera and detects the orientation of the
largest rectangular object in the scene. The script shows one annotated frame
every few seconds and draws the detected rectangle with its rotation angle.

Usage:
    python3 camera_orient_detection.py

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
import time
import threading
from typing import Tuple


# Color used for drawing detected rectangle and text
RECT_COLOR = (0, 200, 255)


def detect_rectangle_orientation(frame):
    """Detect the largest rectangular-ish contour in `frame` and return
    its minimum-area bounding box and rotation angle in degrees.

    Returns: (angle_deg, box_points) or (None, None) if nothing detected.
    """
    h, w = frame.shape[:2]

    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive threshold or Canny to find edges
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Find the largest contour by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # ignore small contours/noise
            continue

        # Compute minimum area rect
        rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Normalize angle: OpenCV returns angle in range [-90,0)
        angle = rect[2]
        width, height = rect[1]
        if width < height:
            angle = angle + 90.0

        # Accept rectangle-like shapes: aspect ratio filter (optional)
        if width == 0 or height == 0:
            continue
        aspect = max(width, height) / (min(width, height) + 1e-6)
        # If contour is very elongated, maybe it's not the rectangle we want; still allow

        return float(angle), box

    return None, None


def draw_orientation(frame, angle, box):
    """Draw the rotated box and angle text onto `frame`.

    `box` should be 4 points (Nx2) from cv2.boxPoints.
    """
    if box is None or angle is None:
        return frame

    # Draw polygon
    cv2.drawContours(frame, [box], 0, RECT_COLOR, 2)

    # Put angle text near top-left of box
    x_coords = box[:, 0]
    y_coords = box[:, 1]
    tx = int(np.min(x_coords))
    ty = int(np.min(y_coords)) - 10
    ty = max(20, ty)

    label = f"Angle: {angle:.1f} deg"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    # Background
    (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(frame, (tx - 4, ty - lh - 4), (tx + lw + 4, ty + baseline + 4), RECT_COLOR, cv2.FILLED)
    cv2.putText(frame, label, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return frame


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

        # Draw label with consistent sizing and bounds checking
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        pad = 6

        # Measure text size
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        h, w = frame.shape[:2]

        # Try to place label above the box; if it would go out of frame, place it below
        x1 = int(max(0, start_x))
        y1 = int(start_y - label_h - 2 * pad)
        if y1 < 0:
            y1 = int(start_y + pad)

        x2 = int(min(w, x1 + label_w + 2 * pad))
        y2 = int(min(h, y1 + label_h + 2 * pad))

        # Draw filled background rectangle for label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)

        # Draw label text (bottom-left origin)
        text_x = x1 + pad
        text_y = y1 + label_h + pad - max(0, baseline // 2)
        cv2.putText(frame, label, (int(text_x), int(text_y)), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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

            # Run rectangle orientation detection on the captured frame
            angle, box = detect_rectangle_orientation(frame)

            # Draw detection results on a copy so we can overlay countdown separately
            annotated = draw_orientation(frame.copy(), angle, box)

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
