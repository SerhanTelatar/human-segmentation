import cv2
import face_recognition

face_encodings = []
face_names = []

import dlib
print(dlib.DLIB_USE_CUDA)  # True dönerse CUDA kullanılıyor demektir
print(dlib.cuda.get_num_devices()) 

# Yüz verilerini yükleyip, encoding'leri alıyoruz
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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # CUDA kullanarak CNN modelini etkinleştiriyoruz
    face_locations = face_recognition.face_locations(frame)
    face_encodings_in_frame = face_recognition.face_encodings(frame, face_locations)

    # Yüzlerin tanınması ve çerçeve içine alınması
    for (face_encoding, face_location) in zip(face_encodings_in_frame, face_locations):
        matches = face_recognition.compare_faces(face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = face_names[first_match_index]

        # Çerçeve çizimi ve isim yazımı
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Sonucu ekrana yazdır
    cv2.imshow("Video", frame)

    # 'q' tuşuna basılırsa çıkış
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

