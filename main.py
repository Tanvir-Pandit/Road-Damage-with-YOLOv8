import cv2
import pandas as pd
from ultralytics import YOLO
from tracker.tracker import Tracker

model = YOLO('model/best.pt')

cap = cv2.VideoCapture("videos/crack 2.mp4")

with open("objects/object_list.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

frame_count = 0
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    bounding_boxes = pd.DataFrame(results[0].boxes.data.cpu().numpy()).astype("float")
    detections = []

    warning_issued = False
    for index, row in bounding_boxes.iterrows():
        x1, y1, x2, y2, _, class_id = row
        class_name = class_list[int(class_id)]
        if 'Cracks' in class_name or 'pothole' in class_name:
            detections.append([x1, y1, x2, y2])
            warning_issued = True

    tracked_objects = tracker.update(detections)

    for bbox in tracked_objects:
        x3, y3, x4, y4, obj_id = bbox
        cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (int(x3), int(y3)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if warning_issued:
        cv2.putText(frame, 'WARNING: Cracks or Potholes Detected!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()