import cv2
import supervision as sv
from ultralytics import YOLO


model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

    annotated_frame= mask_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imshow("Annotated Image", annotated_frame)

    if cv2.waitKey(1) &0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()