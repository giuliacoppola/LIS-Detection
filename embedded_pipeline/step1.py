import cv2
import numpy as np
import tensorflow as tf

# === Carica modello palm detector ===
interpreter = tf.lite.Interpreter(model_path="models/palm_detection.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
print("Premi 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converti e normalizza
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256)).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_resized, axis=0)

    # Inferenza
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]['index'])  # classificators
    boxes = interpreter.get_tensor(output_details[1]['index'])   # regressors

    max_idx = np.argmax(scores[0])
    max_score = scores[0][max_idx][0]
    print(f"[Palm] Max score: {max_score:.3f}")

    if max_score < 0.2:
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Estrai box (cx, cy, w, h) e riconverti in coordinate originali
    cx, cy, w, h = boxes[0][max_idx][:4]
    img_h, img_w, _ = frame.shape
    xmin = int((cx - w/2) * img_w)
    ymin = int((cy - h/2) * img_h)
    xmax = int((cx + w/2) * img_w)
    ymax = int((cy + h/2) * img_h)

    # Clipping
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)

    if xmax > xmin and ymax > ymin:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
