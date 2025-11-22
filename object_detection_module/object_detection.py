from ultralytics import YOLO
import cv2
import time
import numpy as np

# Load Yolo 
model = YOLO("yolov8s.pt")

# To use GPU
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    model.to("cuda")

# Camera Starts here
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

for _ in range(5):
    cap.read()

print("Camera ready.. Press 'q' or click 'End' button to stop.")

cv2.namedWindow("Traffic Object Recognition")

# Close Button here
def on_close(event, x, y, flags, param):
    global running
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 < x < 90 and 10 < y < 40:
            running = False

cv2.setMouseCallback("Traffic Object Recognition", on_close)

running = True
font = cv2.FONT_HERSHEY_SIMPLEX

while running:
    ret, frame = cap.read()
    if not ret:
        print("Frame not captured!")
        break

    # Resize for consistency
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO inference
    results = model(frame, conf=0.45, verbose=False)  # higher conf = more accurate

    annotated_frame = results[0].plot()

    # End button
    cv2.rectangle(annotated_frame, (10, 10), (90, 40), (0, 0, 255), -1)
    cv2.putText(annotated_frame, "END", (25, 32), font, 0.8, (255, 255, 255), 2)

    cv2.imshow("Traffic Object Recognition", annotated_frame)

    # Exit by q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera Closed Successfully ")
