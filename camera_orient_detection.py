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
from typing import Tuple, Optional


RECT_COLOR = (0, 200, 255)


def detect_rectangle_orientation(frame: np.ndarray, min_area: int = 1000) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Detect the largest rectangular-ish contour and return (angle_deg, box_points).

    Angle is normalized to [0, 180). Returns (None, None) if nothing detected.
    """
    if frame is None:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        angle = float(rect[2])
        w, h = rect[1]
        # Adjust angle to be rotation from horizontal in degrees [0,180)
        if w < h:
            angle = angle + 90.0

        # normalize
        angle = angle % 180.0

        return angle, box

    return None, None


def draw_orientation(frame: np.ndarray, angle: Optional[float], box: Optional[np.ndarray]) -> np.ndarray:
    if box is None or angle is None:
        return frame

    cv2.drawContours(frame, [box], 0, RECT_COLOR, 2)

    x_coords = box[:, 0]
    y_coords = box[:, 1]
    tx = int(np.min(x_coords))
    ty = int(np.min(y_coords)) - 10
    ty = max(20, ty)

    label = f"Angle: {angle:.1f} deg"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(frame, (tx - 4, ty - lh - 4), (tx + lw + 4, ty + baseline + 4), RECT_COLOR, cv2.FILLED)
    cv2.putText(frame, label, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return frame


def main():
    """Run the camera feed with rectangle orientation detection."""
    print("=" * 60)
    print("USB Camera Feed with Orientation Detection")
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

            # Draw orientation on a copy so we can overlay countdown separately
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
