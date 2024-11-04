import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

prev_x = None
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera acilamadi.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = hand_landmarks.landmark[1].x

            if prev_x is not None:
                if x > prev_x + 0.01:
                    counter += 1
                    print(f'Sağa hareket tespit edildi. Sayaç: {counter}')
                if x < prev_x - 0.01:
                    counter -= 1
                    print(f'Sola hareket tespit edildi. Sayaç: {counter}')

            # Mevcut x pozisyonunu sakla
            prev_x = x

    # Mevcut sayacı görüntüye yazdır
    cv2.putText(frame, f"Sayac: {counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # Sonucu göster
    cv2.imshow('El Takibi ve Sayaç', frame)

    # ESC tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Kamerayı ve pencereyi serbest bırak
cap.release()
cv2.destroyAllWindows()
