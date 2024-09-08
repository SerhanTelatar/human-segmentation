import cv2
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model("./models/saved_model/model.keras")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

class_labels = os.listdir("./data/raw")

def preprocess_image(face_image, image_size=(400,400)):
    image = cv2.resize(face_image, image_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in faces:

        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

        face = frame[y:y+h, x:x+w]

        preprocess_face = preprocess_image(face)

        predictions = model.predict(preprocess_face)

        predicted_class_idx = np.argmax(predictions)

        confidence = np.max(predictions)

        predicted_label = class_labels[predicted_class_idx]

        label = f'{predicted_label}: {confidence*100:.2f}%'
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()