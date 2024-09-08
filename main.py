import cv2
import supervision as sv
from ultralytics import YOLO
import face_recognition
import mediapipe as mp
import numpy as np

model = YOLO("yolov8n-seg.pt")

face_encodings = []
face_names = []

person1_image = face_recognition.load_image_file("./data/processed/train/Nisa./face_8.jpg")
person2_image = face_recognition.load_image_file("./data/processed/train/Serhan./face_12.jpg")
person3_image = face_recognition.load_image_file("./data/processed/train/Sengul./face_8.jpg")

person1_encoding = face_recognition.face_encodings(person1_image)[0]
person2_encoding = face_recognition.face_encodings(person2_image)[0]
person3_encoding = face_recognition.face_encodings(person3_image)[0]

face_encodings.append(person1_encoding)
face_encodings.append(person2_encoding)
face_encodings.append(person3_encoding)

face_names.append("Nisa")
face_names.append("Serhan")
face_names.append("Sengul")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

clothing_image = cv2.imread("clothe.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_frame = mask_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    face_locations = face_recognition.face_locations(frame)
    face_encodings_in_frame = face_recognition.face_encodings(frame, face_locations)

    for (face_encoding, face_location) in zip(face_encodings_in_frame, face_locations):
        matches = face_recognition.compare_faces(face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = face_names[first_match_index]

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:

        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        frame_height, frame_width, _ = frame.shape
        left_shoulder_coords = (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height))
        right_shoulder_coords = (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height))

        cv2.circle(frame, left_shoulder_coords, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_shoulder_coords, 5, (0, 255, 0), -1)

        shoulder_width = right_shoulder_coords[0] - left_shoulder_coords[0]

        if shoulder_width > 0:
            scale_factor = (shoulder_width / clothing_image.shape[1])
            clothing_height = int(clothing_image.shape[0] * scale_factor)

            if clothing_height > 0:
                clothing_resized = cv2.resize(clothing_image, (shoulder_width, clothing_height))

                alpha_clothing = clothing_resized[:, :, 3] / 255.0
                for c in range(0, 3):
                    frame[left_shoulder_coords[1]:left_shoulder_coords[1] + clothing_height,
                        left_shoulder_coords[0]:left_shoulder_coords[0] + shoulder_width, c] = \
                        alpha_clothing * clothing_resized[:, :, c] + \
                        (1 - alpha_clothing) * frame[left_shoulder_coords[1]:left_shoulder_coords[1] + clothing_height,
                                                    left_shoulder_coords[0]:left_shoulder_coords[0] + shoulder_width, c]


    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
