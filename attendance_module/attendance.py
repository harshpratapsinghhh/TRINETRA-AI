import cv2
from ultralytics import YOLO
from datetime import datetime
import csv
import os

# Paths
MODEL_PATH = os.path.join("models", "yolov8s.pt")
ATTENDANCE_FILE = os.path.join("logs", "attendance.csv")

# Ensure attendance CSV exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# Load model
model = YOLO(MODEL_PATH)

# To prevent duplicate attendance
marked = set()

def mark_attendance(name):
    """Mark attendance once per person per session."""
    if name not in marked:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_now = now.strftime("%H:%M:%S")

        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time_now])

        marked.add(name)
        print(f"[ATTENDANCE] {name} marked at {time_now}")

def attendance_stream():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Run YOLO object detection
        results = model(frame, conf=0.5, verbose=False)

        verifying_idx = 1  # Track which person is being verified

        for res in results:
            boxes = res.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]  # class label from YOLO

                if name == "person":
                    # MARK ATTENDANCE
                    mark_attendance(name)

                    # Bounding box drawing
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Verification label
                    cv2.putText(frame, f"{name} (Verifying {verifying_idx})",
                                (x1, y1 - 10), font, 0.7, (0, 255, 0), 2)

                    verifying_idx += 1

        # Mode label on top-left
        cv2.putText(frame, "Mode: ATTENDANCE", (10, 30), font, 1, (0, 255, 255), 2)

        # Encode for Flask
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()
