import cv2
import supervision as sv
from ultralytics import YOLO


model = YOLO("yolov8n-seg.pt")
custom_model = tf.keras.models.load_model('models/saved_model/model.h5')

cap = cv2.VideoCapture(0)


mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_frame= mask_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    cv2.imshow("Annotated Image", annotated_frame)

    if cv2.waitKey(1) &0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()