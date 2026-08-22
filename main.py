"""
AI Railway Track Safety System - Standalone Desktop Application (OpenCV + YOLOv8 + SQLite).
Run this directly in VS Code with Python!
"""

import cv2
import numpy as np
import sqlite3
import winsound
import os
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
DB_FILE = "railway_safety.db"
SCREENSHOT_DIR = Path("saved_screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)
DEMO_VIDEO_FILE = "demo_railway_track.mp4"
CONFIDENCE_THRESHOLD = 0.40

# -----------------------------------------------------------------------------
# SCREENSHOT CONTROL SETTINGS
# -----------------------------------------------------------------------------
# Set ENABLE_SCREENSHOTS to False to completely turn off screenshot saving.
ENABLE_SCREENSHOTS = True

# Cooldown time in seconds between screenshots (e.g., 15.0 seconds)
SCREENSHOT_COOLDOWN = 15.0

# COCO Target Obstacle Classes
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe"
}

CLASS_CATEGORIES = {
    "person": "Person",
    "bicycle": "Vehicle", "car": "Vehicle", "motorcycle": "Vehicle", "bus": "Vehicle", "truck": "Vehicle",
    "cat": "Animal", "dog": "Animal", "horse": "Animal", "sheep": "Animal", "cow": "Animal",
    "elephant": "Animal", "bear": "Animal", "zebra": "Animal", "giraffe": "Animal"
}

CATEGORY_COLORS_BGR = {
    "Person": (0, 0, 255),       # Red
    "Vehicle": (0, 140, 255),    # Orange
    "Animal": (0, 215, 255)      # Yellow
}

# -----------------------------------------------------------------------------
# DATABASE INITIALIZATION & INCIDENT LOGGING
# -----------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            date TEXT,
            time TEXT,
            obstacle_type TEXT,
            category TEXT,
            confidence REAL,
            risk_level TEXT,
            screenshot_path TEXT
        );
    """)
    conn.commit()
    conn.close()

def log_detection(obstacle_type, category, confidence, risk_level, screenshot_path):
    now = datetime.now()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO detections (timestamp, date, time, obstacle_type, category, confidence, risk_level, screenshot_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            str(obstacle_type),
            str(category),
            float(confidence),
            str(risk_level),
            str(screenshot_path)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ DB Log Warning: {e}")

# -----------------------------------------------------------------------------
# SCREENSHOT SAVER WITH CUSTOM COOLDOWN & TOGGLE
# -----------------------------------------------------------------------------
last_saved_times = {}

def save_screenshot(frame, obstacle_name):
    if not ENABLE_SCREENSHOTS:
        return None

    now = time.time()
    if now - last_saved_times.get(obstacle_name, 0) >= SCREENSHOT_COOLDOWN:
        last_saved_times[obstacle_name] = now
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"obstacle_{obstacle_name}_{timestamp}.jpg"
        filepath = SCREENSHOT_DIR / filename
        cv2.imwrite(str(filepath), frame)
        return str(filepath)
    return None

# -----------------------------------------------------------------------------
# AUDIO ALARM SIREN (WINDOWS NATIVE)
# -----------------------------------------------------------------------------
def play_siren():
    try:
        winsound.Beep(1000, 150) # 1000Hz tone for 150ms
    except Exception:
        pass

# -----------------------------------------------------------------------------
# SYNTHETIC DEMO VIDEO GENERATOR
# -----------------------------------------------------------------------------
def generate_demo_video(filename=DEMO_VIDEO_FILE, duration=10, fps=25):
    if os.path.exists(filename):
        return filename

    print("🎥 Generating synthetic railway track test video...")
    w, h = 960, 540
    total_frames = duration * fps
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for f in range(total_frames):
        frame = np.full((h, w, 3), (35, 45, 55), dtype=np.uint8)
        vanish_x, vanish_y = w // 2, 180
        
        cv2.rectangle(frame, (0, vanish_y), (w, h), (30, 60, 40), -1)
        ballast = np.array([[vanish_x - 40, vanish_y], [vanish_x + 40, vanish_y], [w - 100, h], [100, h]], np.int32)
        cv2.fillPoly(frame, [ballast], (80, 85, 90))

        for s in range(16):
            prog = (s / 16)
            y = int(vanish_y + (h - vanish_y) * (prog ** 1.8))
            hw = int(30 + (w / 2.4 - 30) * (prog ** 1.8))
            cv2.line(frame, (vanish_x - hw, y), (vanish_x + hw, y), (40, 30, 20), max(2, int(5 * prog)))

        for side in [-1, 1]:
            cv2.line(frame, (vanish_x + side * 25, vanish_y), (vanish_x + side * (w // 3 - 20), h), (200, 210, 220), 4)

        if 20 <= f <= 110:
            prog = (f - 20) / 90
            px = int(180 + prog * 600)
            py = int(360 + np.sin(prog * np.pi) * -15)
            cv2.circle(frame, (px, py - 40), 12, (200, 180, 160), -1)
            cv2.rectangle(frame, (px - 10, py - 28), (px + 10, py + 10), (30, 40, 180), -1)
            cv2.line(frame, (px - 5, py + 10), (px - 8, py + 35), (20, 20, 20), 4)
            cv2.line(frame, (px + 5, py + 10), (px + 8, py + 35), (20, 20, 20), 4)

        elif 130 <= f <= 220:
            prog = (f - 130) / 90
            vx = int(780 - prog * 600)
            vy = 320
            cv2.rectangle(frame, (vx, vy), (vx + 90, vy + 40), (0, 140, 255), -1)
            cv2.rectangle(frame, (vx + 15, vy - 20), (vx + 75, vy), (0, 140, 255), -1)
            cv2.circle(frame, (vx + 20, vy + 40), 10, (10, 10, 10), -1)
            cv2.circle(frame, (vx + 70, vy + 40), 10, (10, 10, 10), -1)

        writer.write(frame)

    writer.release()
    return filename

# -----------------------------------------------------------------------------
# MAIN APPLICATION LOOP
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("🚆 AI RAILWAY TRACK SAFETY MONITORING SYSTEM")
    print("=" * 60)

    init_db()

    print("🧠 Loading YOLOv8 Model Architecture (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    video_source = generate_demo_video()
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"❌ Error opening video {video_source}")
        return

    print("🚀 Safety Monitoring Active! Press 'q' or 'ESC' on the OpenCV window to exit.")
    
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        h, w, _ = frame.shape

        # Yellow Track Corridor Overlay
        roi_poly = np.array([[int(w*0.25), int(h*0.40)], [int(w*0.75), int(h*0.40)], [int(w*0.95), int(h*0.95)], [int(w*0.05), int(h*0.95)]], np.int32)
        cv2.polylines(frame, [roi_poly], isClosed=True, color=(0, 255, 255), thickness=2)

        # YOLOv8 Detection
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device='cpu')[0]

        has_obstacle = False
        detected_names = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                if cls_id in TARGET_CLASSES:
                    has_obstacle = True
                    obj_name = TARGET_CLASSES[cls_id]
                    category = CLASS_CATEGORIES.get(obj_name, "Obstacle")
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    detected_names.append(obj_name.upper())

                    # Bounding Box
                    color = CATEGORY_COLORS_BGR.get(category, (0, 0, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label_text = f"{obj_name.upper()} {int(conf * 100)}%"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    # Save Screenshot & SQLite Log safely
                    risk_level = "CRITICAL" if obj_name in ["person", "car", "truck", "bus"] else "WARNING"
                    saved_path = save_screenshot(frame, obj_name)
                    if saved_path:
                        log_detection(obj_name, category, round(conf, 3), risk_level, saved_path)
                        print(f"🚨 [INCIDENT LOGGED] {obj_name.upper()} detected -> Screenshot saved to {saved_path}")

        # Render Warning Banner & Audio Siren
        if has_obstacle:
            play_siren()
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
            types_str = ", ".join(set(detected_names))
            cv2.putText(frame, f"OBSTACLE DETECTED ON TRACK! [{types_str}]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
        else:
            cv2.rectangle(frame, (0, 0), (w, 45), (0, 150, 0), -1)
            cv2.putText(frame, "TRACK CLEAR - SAFETY MONITORING ACTIVE", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show OpenCV GUI Window
        cv2.imshow("AI Railway Track Safety System - OpenCV Desktop Monitor", frame)

        # Press 'q' or ESC to stop
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
