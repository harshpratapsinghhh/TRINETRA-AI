import cv2
from ultralytics import YOLO
from datetime import datetime
import csv
import os

# Paths
MODEL_PATH = os.path.join("models", "yolov8s.pt")  # ya tumhara trained model
ATTENDANCE_FILE = os.path.join("logs", "attendance.csv")

# Ensure CSV header exists
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# Load YOLO model
model = YOLO(MODEL_PATH)

# Already marked attendance to avoid duplicates
marked = set()

def mark_attendance(name):
    if name not in marked:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, current_time])
        marked.add(name)
        print(f"[ATTENDANCE] {name} marked at {current_time}")

def attendance_stream():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Detect objects
        results = model(frame, conf=0.5, verbose=False)

        # To show which person camera is verifying (if multiple)
        verifying_idx = 1

        for res in results:
            boxes = res.boxes
            for box, cls in zip(boxes.xyxy, boxes.cls):
                name = model.names[int(cls)]
                if name == "person":  # only mark people
                    mark_attendance(name)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put name and verifying index
                    cv2.putText(frame, f"{name} (Verifying {verifying_idx})", 
                                (x1, y1 - 10), font, 0.7, (0, 255, 0), 2)
                    verifying_idx += 1

        # Display mode text
        cv2.putText(frame, "Mode: ATTENDANCE", (10, 30), font, 1, (0, 255, 255), 2)

        # Encode frame for Flask streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()
