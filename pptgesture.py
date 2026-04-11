import argparse
import math
import re
import tempfile
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
WINDOW_NAME = "PPT Gesture Control"
EXPORT_WIDTH = 1280
EXPORT_HEIGHT = 720
THUMB_STABLE_SECONDS = 0.25
THUMB_COOLDOWN_SECONDS = 1.0
THUMB_BOUNCE_DISTANCE = 0.04
THUMB_BOUNCE_RETURN_RATIO = 0.35
THUMB_BOUNCE_TIMEOUT_SECONDS = 1.2


def choose_presentation():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose a PowerPoint presentation",
        filetypes=[
            ("PowerPoint presentations", "*.pptx *.ppt"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(path) if path else None


def parse_args():
    parser = argparse.ArgumentParser(description="Control PowerPoint slides with thumb gestures.")
    parser.add_argument(
        "presentation",
        nargs="?",
        help="Path to a .ppt or .pptx file. If omitted, a file picker opens.",
    )
    return parser.parse_args()


def export_presentation_slides(presentation_path, export_dir):
    try:
        import win32com.client
    except ImportError as exc:
        raise RuntimeError("Install pywin32 first: pip install pywin32") from exc

    powerpoint = None
    presentation = None

    try:
        powerpoint = win32com.client.DispatchEx("PowerPoint.Application")
        powerpoint.Visible = 1
        presentation = powerpoint.Presentations.Open(str(presentation_path), ReadOnly=1, WithWindow=0)
        presentation.Export(str(export_dir), "PNG", EXPORT_WIDTH, EXPORT_HEIGHT)
    except Exception as exc:
        raise RuntimeError(
            "Could not export the presentation. Make sure Microsoft PowerPoint is installed "
            f"and the file can be opened: {presentation_path}"
        ) from exc
    finally:
        if presentation is not None:
            presentation.Close()
        if powerpoint is not None:
            powerpoint.Quit()


def slide_number(path):
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def load_slide_images(export_dir):
    slide_paths = sorted(
        [path for path in export_dir.iterdir() if path.suffix.lower() == ".png"],
        key=slide_number,
    )
    slides = []

    for slide_path in slide_paths:
        slide = cv2.imread(str(slide_path))
        if slide is not None:
            slides.append(slide)

    if not slides:
        raise RuntimeError("PowerPoint export finished, but no slide images were readable.")

    return slides


def open_camera(camera_indexes=range(6), width=1280, height=720):
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0

    for camera_index in camera_indexes:
        cap = cv2.VideoCapture(camera_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        for _ in range(5):
            success, frame = cap.read()
            if success and frame is not None:
                print(f"Using camera index {camera_index}")
                return cap

        cap.release()

    raise RuntimeError(
        "No working camera found. Check that your webcam is connected, not being "
        "used by another app, and that Windows camera privacy permissions allow Python."
    )


def create_hand_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Download the MediaPipe HandLandmarker model "
            "and place it beside pptgesture.py: "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
    )


def fit_image(image, width, height):
    image_height, image_width = image.shape[:2]
    scale = min(width / image_width, height / image_height)
    resized_width = int(image_width * scale)
    resized_height = int(image_height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return canvas


def draw_hand_landmarks(image, landmarks, connections):
    height, width, _ = image.shape

    for connection in connections:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(image, start_point, end_point, (0, 220, 255), 2)

    for landmark in landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(image, point, 4, (0, 255, 80), cv2.FILLED)


def draw_camera_preview(canvas, camera_frame):
    preview_width = 260
    preview_height = 146
    margin = 18
    preview = cv2.resize(camera_frame, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    x1 = canvas.shape[1] - preview_width - margin
    y1 = canvas.shape[0] - preview_height - margin
    x2 = x1 + preview_width
    y2 = y1 + preview_height

    cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (20, 20, 20), cv2.FILLED)
    canvas[y1:y2, x1:x2] = preview


def draw_status(canvas, slide_index, slide_count, gesture_text):
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 54), (10, 10, 10), cv2.FILLED)
    cv2.putText(
        canvas,
        f"Slide {slide_index + 1}/{slide_count}",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        "Bounce thumbs up: next   Bounce thumbs down: previous   Esc: exit",
        (220, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (210, 210, 210),
        2,
    )

    if gesture_text:
        cv2.putText(
            canvas,
            gesture_text,
            (24, canvas.shape[0] - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )


def landmark_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def detect_thumb_pose(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    palm_width = max(landmark_distance(index_mcp, pinky_mcp), 0.08)
    thumb_extension = landmark_distance(thumb_tip, thumb_mcp)
    thumb_delta_y = thumb_tip.y - thumb_mcp.y
    thumb_delta_x = abs(thumb_tip.x - thumb_mcp.x)

    thumb_is_extended = thumb_extension > palm_width * 0.75
    thumb_is_vertical = abs(thumb_delta_y) > palm_width * 0.65 and thumb_delta_x < palm_width * 1.35

    if not thumb_is_extended or not thumb_is_vertical:
        return None

    if thumb_tip.y < thumb_ip.y and thumb_delta_y < -palm_width * 0.45:
        return "up"

    if thumb_tip.y > thumb_ip.y and thumb_delta_y > palm_width * 0.45:
        return "down"

    return None


class ThumbGestureDetector:
    def __init__(self):
        self.pose = None
        self.pose_since = 0
        self.base_y = None
        self.bounce_direction = None
        self.bounce_started_at = 0
        self.last_gesture_time = 0

    def reset(self):
        self.pose = None
        self.pose_since = 0
        self.base_y = None
        self.bounce_direction = None
        self.bounce_started_at = 0

    def start_pose(self, pose, thumb_y, now):
        self.pose = pose
        self.pose_since = now
        self.base_y = thumb_y
        self.bounce_direction = None
        self.bounce_started_at = 0

    def reset_bounce(self, thumb_y):
        self.base_y = thumb_y
        self.bounce_direction = None
        self.bounce_started_at = 0

    def update(self, landmarks):
        now = time.monotonic()
        pose = detect_thumb_pose(landmarks)
        thumb_y = landmarks[4].y

        if pose is None:
            self.reset()
            return None

        if pose != self.pose:
            self.start_pose(pose, thumb_y, now)
            return None

        if now - self.pose_since < THUMB_STABLE_SECONDS:
            return None

        if now - self.last_gesture_time < THUMB_COOLDOWN_SECONDS:
            self.reset_bounce(thumb_y)
            return None

        if self.base_y is None:
            self.reset_bounce(thumb_y)
            return None

        delta_y = thumb_y - self.base_y

        if self.bounce_direction is None:
            if abs(delta_y) >= THUMB_BOUNCE_DISTANCE:
                self.bounce_direction = 1 if delta_y > 0 else -1
                self.bounce_started_at = now
            return None

        if now - self.bounce_started_at > THUMB_BOUNCE_TIMEOUT_SECONDS:
            self.reset_bounce(thumb_y)
            return None

        returned_upward_bounce = self.bounce_direction < 0 and delta_y >= -THUMB_BOUNCE_DISTANCE * THUMB_BOUNCE_RETURN_RATIO
        returned_downward_bounce = self.bounce_direction > 0 and delta_y <= THUMB_BOUNCE_DISTANCE * THUMB_BOUNCE_RETURN_RATIO

        if not returned_upward_bounce and not returned_downward_bounce:
            return None

        self.last_gesture_time = now
        self.reset_bounce(thumb_y)
        return pose


def main():
    args = parse_args()
    presentation_path = Path(args.presentation).expanduser() if args.presentation else choose_presentation()

    if not presentation_path:
        raise SystemExit("No presentation selected.")

    presentation_path = presentation_path.resolve()
    if not presentation_path.exists():
        raise FileNotFoundError(f"Presentation not found: {presentation_path}")

    with tempfile.TemporaryDirectory(prefix="pptgesture_") as export_temp_dir:
        export_dir = Path(export_temp_dir)
        print("Exporting slides from PowerPoint...")
        export_presentation_slides(presentation_path, export_dir)
        slides = load_slide_images(export_dir)
        print(f"Loaded {len(slides)} slides.")

        hands = create_hand_landmarker()
        cap = open_camera()
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        thumb_detector = ThumbGestureDetector()
        slide_index = 0
        gesture_text = ""
        gesture_until = 0
        last_timestamp_ms = -1

        try:
            while True:
                success, camera_frame = cap.read()
                if not success:
                    raise RuntimeError("Camera stopped returning frames.")

                camera_frame = cv2.flip(camera_frame, 1)
                camera_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=camera_rgb)
                timestamp_ms = int(time.monotonic() * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms
                results = hands.detect_for_video(mp_image, timestamp_ms)

                canvas = fit_image(slides[slide_index], EXPORT_WIDTH, EXPORT_HEIGHT)

                if results.hand_landmarks:
                    landmarks = results.hand_landmarks[0]
                    thumb_tip = landmarks[4]
                    x = int(thumb_tip.x * EXPORT_WIDTH)
                    y = int(thumb_tip.y * EXPORT_HEIGHT)
                    thumb_gesture = thumb_detector.update(landmarks)

                    if thumb_gesture == "up" and slide_index < len(slides) - 1:
                        slide_index += 1
                        gesture_text = "Next slide"
                        gesture_until = time.monotonic() + 0.8
                    elif thumb_gesture == "down" and slide_index > 0:
                        slide_index -= 1
                        gesture_text = "Previous slide"
                        gesture_until = time.monotonic() + 0.8

                    cv2.circle(canvas, (x, y), 12, (0, 255, 255), cv2.FILLED)
                    draw_hand_landmarks(canvas, landmarks, connections)
                else:
                    thumb_detector.reset()

                visible_gesture_text = gesture_text if time.monotonic() < gesture_until else ""
                draw_status(canvas, slide_index, len(slides), visible_gesture_text)
                draw_camera_preview(canvas, camera_frame)
                cv2.imshow(WINDOW_NAME, canvas)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            hands.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
