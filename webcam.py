import cv2

# Buka kamera
cap = cv2.VideoCapture(0)  # Angka 0 mengacu pada kamera utama (biasanya built-in webcam)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Tampilkan frame dalam jendela
    cv2.imshow('Webcam Feed', frame)

    # Hentikan loop jika tombol "q" ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya dan tutup jendela
cap.release()
cv2.destroyAllWindows()
