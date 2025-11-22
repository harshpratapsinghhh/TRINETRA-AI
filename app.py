from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import threading
import os
from attendance_module.attendance import attendance_stream 

app = Flask(__name__)

# Load YOLO Model in background
yolo_model = None
MODEL_PATH = os.path.join("models", "yolov8n.pt")

def load_yolo():
    global yolo_model
    yolo_model = YOLO(MODEL_PATH)
    print("YOLO Model Loaded Successfully")

threading.Thread(target=load_yolo).start()


# Object Detection Stream
def detection_stream():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        if yolo_model:
            results = yolo_model(frame, classes=[0,1,2,3,5,7])
            annotated = results[0].plot()
        else:
            annotated = frame

        # Encode frame for Flask streaming
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()


# Routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start_object_detection")
def start_object_detection():
    return Response(detection_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start_attendance")
def start_attendance():
    return Response(attendance_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop")
def stop_camera():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
