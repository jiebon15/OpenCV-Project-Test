import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inisialisasi OpenCV
cap = cv2.VideoCapture(0)  # Ganti dengan 1 jika menggunakan kamera eksternal

while cap.isOpened():
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    # flip kamera
    frame = cv2.flip(frame, 1)
    # Konversi frame ke format RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan menggunakan MediaPipe Hands
    results = hands.process(rgb_frame)

    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dapatkan posisi landmark tangan
            for i, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Tampilkan landmark pada frame
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Tentukan tangan kanan atau kiri berdasarkan landmark tertentu (misalnya, ibu jari)
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        hand_side = "Kanan" if thumb_tip.x > 0.5 else "Kiri"

        # Tampilkan sisi tangan pada frame
        cv2.putText(frame, f"Sisi Tangan: {hand_side}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Deteksi Tangan', frame)

    # Keluar dari program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya
cap.release()
cv2.destroyAllWindows()

