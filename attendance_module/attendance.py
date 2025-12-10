import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os
import csv
from keras_facenet import FaceNet

# ======================================================
# GLOBALS
# ======================================================
yolo_model = None

def set_yolo_model(model):
    global yolo_model
    yolo_model = model


# ======================================================
# PATHS (AUTO HANDLED)
# ======================================================
BASE_DIR = os.path.dirname(__file__)                       # attendance_module/
MODEL_PATH = os.path.join("models", "yolov8s.pt")
STUDENT_DATA = os.path.join(BASE_DIR, "student_data.csv")
ATTENDANCE_FILE = os.path.join("logs", "attendance.csv")

# ======================================================
# LOAD YOLO
# ======================================================
yolo = YOLO(MODEL_PATH)

# ======================================================
# LOAD FaceNet
# ======================================================
embedder = FaceNet()

# ======================================================
# LOAD STUDENT DATABASE (CSV)
# ======================================================
known_encodings = []
known_ids = []
known_names = []

with open(STUDENT_DATA, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    # FIX BOM + SPACES
    reader.fieldnames = [name.strip().replace('\ufeff', '') for name in reader.fieldnames]

    for row in reader:
        student_id = row["Student_id"]
        student_name = row["Name"]
        img_path = os.path.join(BASE_DIR, row["Image_path"])
        
        img = cv2.imread(img_path)

        if img is None:
            print(f"[ERROR] Image not found: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        enc = embedder.embeddings([img_rgb])[0]

        known_encodings.append(enc)
        known_ids.append(student_id)
        known_names.append(student_name)

print("\n[STUDENTS LOADED]:", known_names)


# ======================================================
# ENSURE ATTENDANCE CSV EXISTS
# ======================================================
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Student_ID", "Name", "Date", "Time"])

# Prevent double marking in one session
marked = set()


# ======================================================
# ATTENDANCE STREAM
# ======================================================
def attendance_stream():
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        results = yolo_model(frame, classes=[0])  # Only person class

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                cls_name = yolo.names[cls]

                if cls_name == "person":

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Crop face from person box
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                    emb = embedder.embeddings([face_rgb])[0]

                    # Calculate distances
                    distances = np.linalg.norm(
                        np.array(known_encodings) - emb, axis=1
                    )
                    best_idx = np.argmin(distances)

                    if distances[best_idx] < 0.85:
                        student_id = known_ids[best_idx]
                        student_name = known_names[best_idx]
                    else:
                        student_id = "Unknown"
                        student_name = "Unknown"

                    # Mark only once
                    if student_id != "Unknown" and student_id not in marked:
                        marked.add(student_id)

                        now = datetime.now()
                        with open(ATTENDANCE_FILE, "a", newline="") as f:
                            w = csv.writer(f)
                            w.writerow([
                                student_id,
                                student_name,
                                now.strftime("%Y-%m-%d"),
                                now.strftime("%H:%M:%S")
                            ])
                        print(f"[ATTENDANCE] {student_id} - {student_name}")

                    # Draw Overlay
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{student_id} - {student_name}",
                                (x1, y1 - 10), font, 0.7, (0, 255, 0), 2)

        # Stream to Flask
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    cam.release()
