from flask import Flask, Response, render_template, jsonify
import cv2
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Load YOLO model in background
yolo_model = None

def load_yolo():
    global yolo_model
    yolo_model = YOLO("yolov8n.pt")  # FASTEST MODEL

# Start YOLO loading thread
threading.Thread(target=load_yolo).start()


# ---------- Attendance Mode -----------
def attendance_stream():
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()


# ---------- Object Detection ----------
def detection_stream():
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        if yolo_model:
            results = yolo_model(frame, classes=[0,1,2,3,5,7])  # person, car, bus, animal
            annotated = results[0].plot()
        else:
            annotated = frame

        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_attendance")
def video_attendance():
    return Response(attendance_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_detect")
def video_detect():
    return Response(detection_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
