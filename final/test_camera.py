import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Webcam trovata su indice {i}")
        cap.release()
    else:
        print(f"❌ Nessuna webcam su indice {i}")
